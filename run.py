import argparse
import hashlib
import json
import os
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from problem import NO_WINDOWING
from hpo_problem import (
    SimpleMLP,
    GradientBasedHPO,
    decode_hyperparams,
    encode_hyperparams,
    project_hyperparams,
    load_mnist_arrays,
    preprocess_images,
)

RESULTS_DIR = os.path.join("results", "gradient_based_hpo")
NUM_TRAIN = 4_000
NUM_VAL = 100
INNER_BATCH = 32
LOSS_INTERVAL = 1
INNER_STEPS = 1000
OUTER_STEPS = 200
OUTER_BATCH = 4
OUTER_LR = 0.01
OUTER_OPTIMIZER = "sgd"
INIT_LR = 0.01
SEED = 0


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_json(payload: dict, filename: str, *, on_exists: str = "error") -> str | None:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        if on_exists == "skip":
            return None
        if on_exists == "error":
            raise FileExistsError(
                f"{path} already exists. Use --overwrite-results to replace or "
                "--skip-existing-results to skip."
            )
        if on_exists != "overwrite":
            raise ValueError(f"Unknown on_exists policy: {on_exists}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def should_skip_run(
    run_id: str, filenames: list[str], *, on_exists: str, run_label: str
) -> bool:
    if on_exists == "overwrite":
        return False
    existing = [
        name for name in filenames if os.path.exists(os.path.join(RESULTS_DIR, name))
    ]
    if not existing:
        return False
    if on_exists == "error":
        existing_list = ", ".join(os.path.join(RESULTS_DIR, name) for name in existing)
        raise FileExistsError(
            f"{run_label} results for run_id={run_id} already exist: {existing_list}. "
            "Use --overwrite-results to replace or --skip-existing-results to skip."
        )
    if on_exists == "skip" and len(existing) == len(filenames):
        print(
            f"Skipping {run_label} run_id={run_id}; all expected results already exist."
        )
        return True
    return False


def should_skip_file(filename: str, *, on_exists: str, run_label: str) -> bool:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return False
    if on_exists == "overwrite":
        return False
    if on_exists == "error":
        raise FileExistsError(
            f"{run_label} results already exist at {path}. "
            "Use --overwrite-results to replace or --skip-existing-results to skip."
        )
    if on_exists == "skip":
        print(f"Skipped existing {run_label} results at {path}")
        return True
    raise ValueError(f"Unknown on_exists policy: {on_exists}")


def list_result_files() -> list[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(
        os.path.join(RESULTS_DIR, name)
        for name in os.listdir(RESULTS_DIR)
        if name.endswith(".json")
    )


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def results_on_exists_policy(args: argparse.Namespace) -> str:
    if args.overwrite_results:
        return "overwrite"
    if args.skip_existing_results:
        return "skip"
    return "error"


def make_train_payload_fn(
    *,
    run_id: str,
    method: str,
    windowing: int | None,
    unroll_length: int,
    weight_scale: float,
    loss_interval: int,
    inner_batch: int,
    different_init_each_outer_step: bool,
    dummy: bool = False,
) -> Callable[[list[float], list[float], list[float], jax.Array], dict]:
    def build(
        losses: list[float],
        lrs: list[float],
        grad_norms: list[float],
        final_params: jax.Array,
    ) -> dict:
        payload = {
            "type": "train_hpo",
            "timestamp": int(time.time()),
            "run_id": run_id,
            "method": method,
            "windowing": windowing,
            "unroll_length": unroll_length,
            "outer_steps": OUTER_STEPS,
            "losses": losses,
            "lrs": lrs,
            "grad_norms": grad_norms,
            "final_lr": float(decode_hyperparams(final_params)[0]),
            "weight_scale": weight_scale,
            "loss_interval": loss_interval,
            "inner_batch": inner_batch,
            "different_init_each_outer_step": different_init_each_outer_step,
        }
        if dummy:
            payload["dummy"] = True
        return payload

    return build


def print_train_summary(losses: list[float], lrs: list[float]) -> None:
    if not losses or not lrs:
        print("No train results to summarize.")
        return
    best_idx = int(jnp.argmin(jnp.array(losses)))
    best_lr = lrs[best_idx]
    best_loss = losses[best_idx]
    last_lr = lrs[-1]
    last_loss = losses[-1]
    print(
        "Best lr is: "
        f"{best_lr:.6f} | Best validation error is: {best_loss:.6f} | "
        f"Last validation error is: {last_loss:.6f} | Last lr is: {last_lr:.6f}"
    )


def mean_grad_variance(
    problem: GradientBasedHPO,
    base_params: jax.Array,
    sigma: float,
    num_samples: int,
    *,
    key: jax.Array,
    init_params,
    data_key: jax.Array | None = None,
) -> jax.Array:
    grad_fn = problem.grad(expanded=False)
    _, grad0 = grad_fn(base_params, init_args=(init_params,), data_key=data_key)
    grad0 = grad0.reshape(-1)

    def body(carry, key_i):
        mean, m2, count = carry
        noise = jr.normal(key_i, base_params.shape) * sigma
        _, grads = grad_fn(
            base_params + noise, init_args=(init_params,), data_key=data_key
        )
        grads = grads.reshape(-1)
        count = count + 1
        delta = grads - mean
        mean = mean + delta / count
        delta2 = grads - mean
        m2 = m2 + delta * delta2
        return (mean, m2, count), None

    keys = jr.split(key, num_samples)
    mean0 = jnp.zeros_like(grad0)
    m20 = jnp.zeros_like(grad0)
    count0 = jnp.array(0, dtype=jnp.int32)
    (mean, m2, count), _ = jax.lax.scan(body, (mean0, m20, count0), keys)
    var = m2 / jnp.maximum(count, 1)
    return jnp.mean(var)


def compute_grad_variances(
    problem: GradientBasedHPO,
    base_params: jax.Array,
    init_params_batched,
    *,
    key: jax.Array,
    sigma: float,
    num_samples: int,
    data_key: jax.Array | None = None,
) -> jax.Array:
    num_inits = jax.tree_util.tree_leaves(init_params_batched)[0].shape[0]
    sample_keys = jr.split(key, num_inits)
    variance_fn = lambda p, k: mean_grad_variance(
        problem,
        base_params,
        sigma,
        num_samples,
        key=k,
        init_params=p,
        data_key=data_key,
    )
    return eqx.filter_vmap(variance_fn)(init_params_batched, sample_keys)


def run_outer_loop(
    problem: GradientBasedHPO,
    outer_params: jax.Array,
    outer_opt: optax.GradientTransformation,
    *,
    data_key: jax.Array,
    windowing: int,
    method: str,
    outer_steps: int,
    normalize_window_grads: bool = False,
    different_init_each_outer_step: bool = False,
    save_filename: str | None = None,
    save_payload_fn: Callable[[list[float], list[float], list[float], jax.Array], dict]
    | None = None,
    step_callback: Callable[[int, jax.Array], None] | None = None,
    on_exists: str = "error",
    run_label: str = "train",
) -> tuple[list[float], list[float], list[float], jax.Array] | None:
    if save_filename and should_skip_file(
        save_filename, on_exists=on_exists, run_label=run_label
    ):
        return None

    opt_state = outer_opt.init(outer_params)
    losses: list[float] = []
    lrs: list[float] = []
    grad_norms: list[float] = []

    grad_fn = problem.grad(
        windowing=windowing,
        expanded=False,
        ours_simple=(method == "ours"),
    )
    jit_grad_fn = eqx.filter_jit(grad_fn)

    for step in range(outer_steps):
        if step_callback:
            step_callback(step, outer_params)
        if different_init_each_outer_step:
            data_key, step_key, init_key = jr.split(data_key, 3)
            init_params = problem.sample_init_params(init_key)
            init_args = (init_params,)
        else:
            data_key, step_key = jr.split(data_key)
            init_args = None
        val_loss, grads = jit_grad_fn(
            outer_params,
            init_args=init_args,
            data_key=step_key,
        )
        if normalize_window_grads and windowing > 0:
            ratio = problem.max_steps / windowing
            if float(ratio).is_integer() and ratio > 0:
                grads = jax.tree.map(lambda g: g / ratio, grads)
        losses.append(float(val_loss))
        lrs.append(float(decode_hyperparams(outer_params)[0]))
        grad_norms.append(float(jnp.linalg.norm(grads)))

        updates, opt_state = outer_opt.update(grads, opt_state)
        outer_params = jax.tree.map(lambda p, u: p + u, outer_params, updates)
        outer_params = project_hyperparams(outer_params)

    if step_callback:
        step_callback(outer_steps, outer_params)

    if save_filename and save_payload_fn:
        payload = save_payload_fn(losses, lrs, grad_norms, outer_params)
        path = save_json(payload, save_filename, on_exists=on_exists)
        if path:
            print(f"Saved {run_label} results to {path}")
            print_train_summary(losses, lrs)

    return losses, lrs, grad_norms, outer_params


def _build_init_params_batched(
    num_inits: int,
    model_key: jax.Array,
    height: int,
    width: int,
    num_classes: int,
    weight_scale: float,
):
    init_keys = jr.split(model_key, num_inits)
    init_params_list = []
    for init_key in init_keys:
        init_model = SimpleMLP(
            height, width, num_classes, key=init_key, weight_scale=weight_scale
        )
        init_params, _ = eqx.partition(init_model, eqx.is_inexact_array)
        init_params_list.append(init_params)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *init_params_list)


def run_variance_experiment(args: argparse.Namespace) -> None:
    unroll_lengths = parse_int_list(args.unroll_lengths)
    config = {
        "type": "variance",
        "unroll_lengths": unroll_lengths,
        "num_inits": args.num_inits,
        "variance_sigma": args.variance_sigma,
        "variance_samples": args.variance_samples,
        "weight_scale": args.weight_scale,
        "loss_interval": args.loss_interval,
    }
    run_id = stable_json_hash(config)
    on_exists = results_on_exists_policy(args)
    filename = f"variance_{run_id}.json"
    if should_skip_run(run_id, [filename], on_exists=on_exists, run_label="variance"):
        return

    key = jr.PRNGKey(SEED)
    data_key, model_key, noise_key = jr.split(key, 3)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        NUM_TRAIN, NUM_VAL
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1
    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)
    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]
    model = SimpleMLP(
        height, width, num_classes, key=model_key, weight_scale=args.weight_scale
    )

    outer_params = encode_hyperparams(INIT_LR)
    variances = []

    init_params_batched = _build_init_params_batched(
        args.num_inits, model_key, height, width, num_classes, args.weight_scale
    )

    for steps in unroll_lengths:
        per_unroll_key = data_key
        problem = GradientBasedHPO(
            model=model,
            train_data=(train_images, train_targets),
            val_data=(val_images, val_targets),
            num_steps=steps,
            batch_size=INNER_BATCH,
            key=per_unroll_key,
            weight_scale=args.weight_scale,
            loss_interval=args.loss_interval,
        )
        noise_key, sample_key = jr.split(noise_key, 2)
        sample_keys = jr.split(sample_key, args.num_inits)
        variance_fn = lambda p, k: mean_grad_variance(
            problem,
            outer_params,
            args.variance_sigma,
            args.variance_samples,
            key=k,
            init_params=p,
        )
        variances_step = eqx.filter_vmap(variance_fn)(init_params_batched, sample_keys)
        if jnp.all(jnp.isnan(variances_step)):
            print(f"unroll={steps} encountered only NaNs; stopping.")
            break
        variances.append(variances_step)
        median_step = jnp.median(variances_step)
        std_step = jnp.std(variances_step)
        print(
            "unroll="
            f"{steps} median={float(median_step):.6e} "
            f"std={float(std_step):.6e} "
            f"min={float(jnp.min(variances_step)):.6e} "
            f"max={float(jnp.max(variances_step)):.6e}"
        )

    if not variances:
        print("No valid variance results to save.")
        return

    config["unroll_lengths"] = unroll_lengths[: len(variances)]
    payload = {
        "type": "variance",
        "timestamp": int(time.time()),
        "run_id": run_id,
        "unroll_lengths": config["unroll_lengths"],
        "variances": jnp.stack(variances, axis=1).tolist(),
        "num_inits": config["num_inits"],
        "variance_sigma": config["variance_sigma"],
        "variance_samples": config["variance_samples"],
        "weight_scale": config["weight_scale"],
        "loss_interval": config["loss_interval"],
    }
    path = save_json(payload, filename, on_exists=results_on_exists_policy(args))
    if path:
        print(f"Saved variance results to {path}")
    else:
        print(
            f"Skipped existing variance results at {os.path.join(RESULTS_DIR, filename)}"
        )


def run_variance_snapshots_experiment(args: argparse.Namespace) -> None:
    unroll_lengths = parse_int_list(args.train_unroll_lengths)
    run_config = {
        "type": "variance_snapshots",
        "unroll_lengths": unroll_lengths,
        "outer_steps": OUTER_STEPS,
        "outer_lr": OUTER_LR,
        "outer_optimizer": OUTER_OPTIMIZER,
        "init_lr": INIT_LR,
        "inner_batch": args.inner_batch,
        "inner_steps": INNER_STEPS,
        "weight_scale": args.weight_scale,
        "loss_interval": args.loss_interval,
        "variance_sigma": args.variance_sigma,
        "variance_samples": args.variance_samples,
        "num_inits": args.num_inits,
        "different_init_each_outer_step": bool(args.different_init_each_outer_step),
    }
    run_id = stable_json_hash(run_config)
    on_exists = results_on_exists_policy(args)
    filename = f"variance_snapshots_{run_id}.json"
    if should_skip_run(run_id, [filename], on_exists=on_exists, run_label="variance"):
        return

    key = jr.PRNGKey(SEED)
    data_key, model_key = jr.split(key, 2)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        NUM_TRAIN, NUM_VAL
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1
    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)
    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]
    model = SimpleMLP(
        height, width, num_classes, key=model_key, weight_scale=args.weight_scale
    )

    if OUTER_OPTIMIZER == "adam":
        outer_opt = optax.adam(OUTER_LR)
    else:
        outer_opt = optax.sgd(OUTER_LR, momentum=0.9)

    variance_key = jr.fold_in(data_key, 123)
    variance_key, init_key = jr.split(variance_key)

    init_params_batched = _build_init_params_batched(
        args.num_inits, init_key, height, width, num_classes, args.weight_scale
    )

    snapshot_steps = sorted({0, OUTER_STEPS // 2, OUTER_STEPS})
    runs = []

    for steps in [160]:
        per_unroll_key = data_key
        problem = GradientBasedHPO(
            model=model,
            train_data=(train_images, train_targets),
            val_data=(val_images, val_targets),
            num_steps=steps,
            batch_size=args.inner_batch,
            key=per_unroll_key,
            weight_scale=args.weight_scale,
            loss_interval=args.loss_interval,
        )

        outer_params = encode_hyperparams(INIT_LR)
        snapshot_payloads: list[dict] = []
        variance_state = {"key": variance_key}
        seen_steps: set[int] = set()

        def snapshot_callback(step: int, params: jax.Array) -> None:
            if step not in snapshot_steps or step in seen_steps:
                return
            seen_steps.add(step)
            variance_state["key"], sample_key = jr.split(variance_state["key"])
            variances = compute_grad_variances(
                problem,
                params,
                init_params_batched,
                key=sample_key,
                sigma=args.variance_sigma,
                num_samples=args.variance_samples,
                data_key=per_unroll_key,
            )
            lr_value = float(decode_hyperparams(params)[0])
            snapshot_payloads.append(
                {
                    "step": int(step),
                    "lr": lr_value,
                    "variances": jnp.asarray(variances).tolist(),
                }
            )

        run_outer_loop(
            problem,
            outer_params,
            outer_opt,
            data_key=per_unroll_key,
            windowing=NO_WINDOWING,
            method="raw",
            outer_steps=OUTER_STEPS,
            normalize_window_grads=args.normalize_window_grads,
            different_init_each_outer_step=args.different_init_each_outer_step,
            step_callback=snapshot_callback,
            on_exists=on_exists,
            run_label="variance",
        )

        runs.append(
            {
                "unroll_length": int(steps),
                "snapshots": snapshot_payloads,
            }
        )

    payload = {
        "type": "variance_snapshots",
        "timestamp": int(time.time()),
        "run_id": run_id,
        "method": "ours",
        "snapshot_steps": snapshot_steps,
        "outer_steps": OUTER_STEPS,
        "unroll_lengths": unroll_lengths,
        "variance_sigma": args.variance_sigma,
        "variance_samples": args.variance_samples,
        "num_inits": args.num_inits,
        "weight_scale": args.weight_scale,
        "loss_interval": args.loss_interval,
        "inner_batch": args.inner_batch,
        "different_init_each_outer_step": bool(args.different_init_each_outer_step),
        "runs": runs,
    }
    path = save_json(payload, filename, on_exists=on_exists)
    if path:
        print(f"Saved variance snapshot results to {path}")


def run_train_experiment(args: argparse.Namespace) -> None:
    unroll_lengths = parse_int_list(args.train_unroll_lengths)
    windowings = parse_int_list(args.windowings)
    run_config = {
        "type": "train_hpo",
        "unroll_lengths": unroll_lengths,
        "windowings": windowings,
        "outer_steps": OUTER_STEPS,
        "outer_lr": OUTER_LR,
        "outer_optimizer": OUTER_OPTIMIZER,
        "init_lr": INIT_LR,
        "inner_batch": args.inner_batch,
        "inner_steps": INNER_STEPS,
        "weight_scale": args.weight_scale,
        "loss_interval": args.loss_interval,
        "different_init_each_outer_step": bool(args.different_init_each_outer_step),
    }
    run_id = stable_json_hash(run_config)
    on_exists = results_on_exists_policy(args)
    key = jr.PRNGKey(SEED)
    data_key, model_key = jr.split(key, 2)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        NUM_TRAIN, NUM_VAL
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1
    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)
    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]
    model = SimpleMLP(
        height, width, num_classes, key=model_key, weight_scale=args.weight_scale
    )

    if OUTER_OPTIMIZER == "adam":
        outer_opt = optax.adam(OUTER_LR)
    else:
        outer_opt = optax.sgd(OUTER_LR)

    for steps in unroll_lengths:
        valid_windows = [w for w in windowings if 0 < w <= steps]
        per_unroll_key = data_key
        problem = GradientBasedHPO(
            model=model,
            train_data=(train_images, train_targets),
            val_data=(val_images, val_targets),
            num_steps=steps,
            batch_size=args.inner_batch,
            key=per_unroll_key,
            weight_scale=args.weight_scale,
            loss_interval=args.loss_interval,
        )

        filename = f"train_{run_id}_ours_unroll{steps}.json"
        outer_params = encode_hyperparams(INIT_LR)

        run_outer_loop(
            problem,
            outer_params,
            outer_opt,
            data_key=per_unroll_key,
            windowing=NO_WINDOWING,
            method="ours",
            outer_steps=OUTER_STEPS,
            normalize_window_grads=args.normalize_window_grads,
            different_init_each_outer_step=args.different_init_each_outer_step,
            save_filename=filename,
            save_payload_fn=make_train_payload_fn(
                run_id=run_id,
                method="ours",
                windowing=None,
                unroll_length=steps,
                weight_scale=args.weight_scale,
                loss_interval=args.loss_interval,
                inner_batch=args.inner_batch,
                different_init_each_outer_step=args.different_init_each_outer_step,
                dummy=True,
            ),
            on_exists=on_exists,
            run_label="train",
        )

        for window in valid_windows:
            filename = f"train_{run_id}_window{window}_unroll{steps}.json"
            outer_params = encode_hyperparams(INIT_LR)
            run_outer_loop(
                problem,
                outer_params,
                outer_opt,
                data_key=per_unroll_key,
                windowing=window,
                method="windowing",
                outer_steps=OUTER_STEPS,
                normalize_window_grads=args.normalize_window_grads,
                different_init_each_outer_step=args.different_init_each_outer_step,
                save_filename=filename,
                save_payload_fn=make_train_payload_fn(
                    run_id=run_id,
                    method="windowing",
                    windowing=window,
                    unroll_length=steps,
                    weight_scale=args.weight_scale,
                    loss_interval=args.loss_interval,
                    inner_batch=args.inner_batch,
                    different_init_each_outer_step=args.different_init_each_outer_step,
                ),
                on_exists=on_exists,
                run_label="train",
            )


def run_hpo(args: argparse.Namespace) -> None:
    key = jr.PRNGKey(SEED)

    data_key, model_key = jr.split(key, 2)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        NUM_TRAIN, NUM_VAL
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1

    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)

    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]

    model = SimpleMLP(
        height, width, num_classes, key=model_key, weight_scale=args.weight_scale
    )

    problem = GradientBasedHPO(
        model=model,
        train_data=(train_images, train_targets),
        val_data=(val_images, val_targets),
        num_steps=INNER_STEPS,
        batch_size=INNER_BATCH,
        key=data_key,
        weight_scale=args.weight_scale,
        loss_interval=args.loss_interval,
    )

    outer_params = encode_hyperparams(INIT_LR)

    if OUTER_OPTIMIZER == "adam":
        outer_opt = optax.adam(OUTER_LR)
    else:
        outer_opt = optax.sgd(OUTER_LR)

    outer_opt_state = outer_opt.init(outer_params)

    for step in range(OUTER_STEPS):
        if args.different_init_each_outer_step:
            data_key, step_key, init_key = jr.split(data_key, 3)
            init_params = problem.sample_init_params(init_key)
            init_args = (init_params,)
        else:
            data_key, step_key = jr.split(data_key)
            init_args = None
        problem_step = problem.new(key=step_key)
        val_loss, grads = problem_step.grad(windowing=args.windowing, expanded=True)(
            outer_params, init_args=init_args
        )

        grads_for_upd = grads.sum(axis=0)
        grad_norm = float(jnp.linalg.norm(grads_for_upd))
        updates, outer_opt_state = outer_opt.update(grads_for_upd, outer_opt_state)
        decoded_params = decode_hyperparams(outer_params)
        print(
            "\nHyperparameters: "
            f"lr={float(decoded_params[0]):.4f}\n"
            f"Had Loss Of: {val_loss:.5f}\n"
            f"Grad Norm: {grad_norm:.5f}"
        )

        outer_params = jax.tree.map(lambda p, u: p + u, outer_params, updates)
        outer_params = project_hyperparams(outer_params)

    decoded_params = decode_hyperparams(outer_params)
    print(f"\nFinal hyperparameters: lr={float(decoded_params[0]):.5f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gradient-based hyperparameter optimization baseline that differentiates "
            "through MLP training dynamics using Equinox."
        )
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="scale initial network weights to trigger gradient explosion (e.g., 5.0 or 10.0)",
    )
    parser.add_argument(
        "--windowing",
        type=int,
        default=NO_WINDOWING,
        help="truncate gradients every N steps (use a huge number to avoid truncation)",
    )
    parser.add_argument(
        "--different-init-each-outer-step",
        action="store_true",
        help="re-initialize the inner model parameters on every outer step",
    )
    parser.add_argument(
        "--loss-interval",
        type=int,
        default=LOSS_INTERVAL,
        help=f"compute validation loss every N steps (default: {LOSS_INTERVAL})",
    )
    parser.add_argument(
        "--run-variance-experiment",
        action="store_true",
        help="compute mean gradient variance across unroll lengths and save results",
    )
    parser.add_argument(
        "--run-variance-snapshots-experiment",
        action="store_true",
        help="track gradient variance snapshots during HPO and save results for plotting",
    )
    parser.add_argument(
        "--run-train-experiment",
        action="store_true",
        help="run HPO training sweeps across unrolls/methods and save results",
    )
    parser.add_argument(
        "--unroll-lengths",
        type=str,
        default="10,15,20,30,40,60,80,120,160,240,320",
        help="comma-separated list of unroll lengths for variance experiment",
    )
    parser.add_argument(
        "--train-unroll-lengths",
        type=str,
        default="20,40,80,160,320,640",
        help="comma-separated list of unroll lengths for training experiment",
    )
    parser.add_argument(
        "--inner-batch",
        type=int,
        default=INNER_BATCH,
        help=f"batch size for inner training loop (default: {INNER_BATCH})",
    )
    parser.add_argument(
        "--windowings",
        type=str,
        default="10,20,40,80,160,320,640",
        help="comma-separated list of window sizes for truncated gradients",
    )
    parser.add_argument(
        "--variance-samples",
        type=int,
        default=32,
        help="number of Gaussian samples for gradient variance estimate",
    )
    parser.add_argument(
        "--variance-sigma",
        type=float,
        default=0.01,
        help="stddev for Gaussian smoothing of outer parameters",
    )
    parser.add_argument(
        "--normalize-window-grads",
        action="store_true",
        help="scale windowed gradients by unroll/window size when divisible",
    )
    results_group = parser.add_mutually_exclusive_group()
    results_group.add_argument(
        "--overwrite-results",
        action="store_true",
        help="overwrite existing result files instead of failing",
    )
    results_group.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="skip writing results when a file already exists",
    )
    parser.add_argument(
        "--num-inits",
        type=int,
        default=50,
        help="number of random model initializations to plot",
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_parser()
    args = arg_parser.parse_args()

    if args.run_variance_experiment:
        run_variance_experiment(args)
    if args.run_variance_snapshots_experiment:
        run_variance_snapshots_experiment(args)
    if args.run_train_experiment:
        run_train_experiment(args)
