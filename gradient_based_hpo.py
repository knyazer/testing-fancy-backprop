import argparse
import hashlib
import json
import os
import time
from typing import Any, Callable, Tuple

import equinox as eqx
import equinox.nn as nn
import optax
import seaborn as sns
import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt

# Hyperparameter optimization constants
RESULTS_DIR = os.path.join("results", "gradient_based_hpo")
NUM_TRAIN = 4_000
NUM_VAL = 100
INNER_BATCH = 32
LOSS_INTERVAL = 1
INNER_STEPS = 640
OUTER_STEPS = 200
OUTER_BATCH = 4
OUTER_LR = 0.01
OUTER_OPTIMIZER = "adam"
INIT_LR = 0.01
SEED = 0


class SimpleMLP(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(
        self,
        height: int,
        width: int,
        num_classes: int,
        *,
        key: jax.Array,
        weight_scale: float = 1.0,
    ):
        k1, k2, k3 = jr.split(key, 3)
        in_dim = height * width
        hidden = 32
        self.linear1 = nn.Linear(in_dim, hidden, key=k1)
        self.linear2 = nn.Linear(hidden, hidden, key=k2)
        self.linear3 = nn.Linear(hidden, num_classes, key=k3)
        self.height = height
        self.width = width

        # Scale weights to trigger gradient explosion
        if weight_scale != 1.0:
            self.linear1 = eqx.tree_at(
                lambda m: m.weight, self.linear1, self.linear1.weight * weight_scale
            )
            self.linear2 = eqx.tree_at(
                lambda m: m.weight, self.linear2, self.linear2.weight * weight_scale
            )
            self.linear3 = eqx.tree_at(
                lambda m: m.weight, self.linear3, self.linear3.weight * weight_scale
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1)
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)


def load_mnist_arrays(num_train: int, num_val: int):
    import tensorflow_datasets as tfds  # type: ignore

    train_ds = tfds.load(
        "mnist", split=f"train[:{num_train}]", as_supervised=True, batch_size=-1
    )
    test_ds = tfds.load(
        "mnist", split=f"test[:{num_val}]", as_supervised=True, batch_size=-1
    )
    train_images, train_labels = tfds.as_numpy(train_ds)
    val_images, val_labels = tfds.as_numpy(test_ds)

    return (
        jnp.asarray(train_images, dtype=jnp.float32),
        jnp.asarray(train_labels, dtype=jnp.int32),
        jnp.asarray(val_images, dtype=jnp.float32),
        jnp.asarray(val_labels, dtype=jnp.int32),
    )


def preprocess_images(images: jax.Array) -> jax.Array:
    images = jnp.asarray(images, dtype=jnp.float32)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = jnp.squeeze(images, axis=-1)
    max_val = jnp.max(images)
    scale = jnp.where(max_val > 1.0, 255.0, 1.0)
    return images / scale


def decode_hyperparams(values: jax.Array) -> Tuple[jax.Array]:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 1 - values.shape[0])))[:1]
    return (values[0],)


def encode_hyperparams(lr: float) -> jax.Array:
    return jnp.array([lr], dtype=jnp.float32)


def project_hyperparams(values: jax.Array) -> jax.Array:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 1 - values.shape[0])))[:1]
    return jnp.array(
        [
            jnp.clip(values[0], 1e-8, 1.0),  # lr
        ]
    )


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


def make_train_payload_fn(
    *,
    run_id: str,
    method: str,
    windowing: int | None,
    unroll_length: int,
    weight_scale: float,
    loss_interval: int,
    fixed_batch: bool,
    inner_batch: int,
    dummy: bool = False,
):
    def build(losses, lrs, grad_norms, final_params):
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
            "fixed_batch": fixed_batch,
            "inner_batch": inner_batch,
        }
        if dummy:
            payload["dummy"] = True
        return payload

    return build


def list_result_files() -> list[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(
        os.path.join(RESULTS_DIR, name)
        for name in os.listdir(RESULTS_DIR)
        if name.endswith(".json")
    )


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def results_on_exists_policy(args) -> str:
    if args.overwrite_results:
        return "overwrite"
    if args.skip_existing_results:
        return "skip"
    return "error"


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
    problem: "GradientBasedHPO",
    base_params: jax.Array,
    sigma: float,
    num_samples: int,
    *,
    key: jax.Array,
    init_params,
) -> jax.Array:
    grad_fn = problem.grad(expanded=False)
    _, grad0 = grad_fn(base_params, init_args=(init_params,))
    grad0 = grad0.reshape(-1)

    def body(carry, key_i):
        mean, m2, count = carry
        noise = jr.normal(key_i, base_params.shape) * sigma
        _, grads = grad_fn(base_params + noise, init_args=(init_params,))
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


def run_outer_loop(
    problem: "GradientBasedHPO",
    outer_params: jax.Array,
    outer_opt: optax.GradientTransformation,
    *,
    data_key: jax.Array,
    windowing: int,
    method: str,
    outer_steps: int,
    normalize_window_grads: bool = False,
    save_filename: str | None = None,
    save_payload_fn: Callable[[list[float], list[float], list[float], jax.Array], dict]
    | None = None,
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
        data_key, step_key = jr.split(data_key)
        val_loss, grads = jit_grad_fn(outer_params, init_args=None, data_key=step_key)
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

    if save_filename and save_payload_fn:
        payload = save_payload_fn(losses, lrs, grad_norms, outer_params)
        path = save_json(payload, save_filename, on_exists=on_exists)
        if path:
            print(f"Saved {run_label} results to {path}")
            print_train_summary(losses, lrs)

    return losses, lrs, grad_norms, outer_params


def run_variance_experiment(args):
    unroll_lengths = parse_int_list(args.unroll_lengths)
    config = {
        "type": "variance",
        "unroll_lengths": unroll_lengths,
        "num_inits": args.num_inits,
        "variance_sigma": args.variance_sigma,
        "variance_samples": args.variance_samples,
        "weight_scale": args.weight_scale,
        "loss_interval": args.loss_interval,
        "fixed_batch": bool(args.fixed_batch),
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

    init_keys = jr.split(model_key, args.num_inits)
    init_params_list = []
    for init_key in init_keys:
        init_model = SimpleMLP(
            height, width, num_classes, key=init_key, weight_scale=args.weight_scale
        )
        init_params, _ = eqx.partition(init_model, eqx.is_inexact_array)
        init_params_list.append(init_params)

    init_params_batched = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *init_params_list
    )

    for steps in unroll_lengths:
        per_unroll_key = data_key if args.fixed_batch else jr.fold_in(data_key, steps)
        problem = GradientBasedHPO(
            model=model,
            train_data=(train_images, train_targets),
            val_data=(val_images, val_targets),
            num_steps=steps,
            batch_size=INNER_BATCH,
            key=per_unroll_key,
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
        max_step = jnp.max(variances_step)
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
        "fixed_batch": config["fixed_batch"],
    }
    path = save_json(payload, filename, on_exists=results_on_exists_policy(args))
    if path:
        print(f"Saved variance results to {path}")
    else:
        print(
            f"Skipped existing variance results at {os.path.join(RESULTS_DIR, filename)}"
        )


def run_train_experiment(args):
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
        "fixed_batch": bool(args.fixed_batch),
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
        per_unroll_key = data_key if args.fixed_batch else jr.fold_in(data_key, steps)
        problem = GradientBasedHPO(
            model=model,
            train_data=(train_images, train_targets),
            val_data=(val_images, val_targets),
            num_steps=steps,
            batch_size=args.inner_batch,
            key=per_unroll_key,
            loss_interval=args.loss_interval,
        )

        filename = f"train_{run_id}_ours_unroll{steps}.json"
        outer_params = encode_hyperparams(INIT_LR)
        run_outer_loop(
            problem,
            outer_params,
            outer_opt,
            data_key=per_unroll_key,
            windowing=-1,
            method="ours",
            outer_steps=OUTER_STEPS,
            normalize_window_grads=args.normalize_window_grads,
            save_filename=filename,
            save_payload_fn=make_train_payload_fn(
                run_id=run_id,
                method="ours",
                windowing=None,
                unroll_length=steps,
                weight_scale=args.weight_scale,
                loss_interval=args.loss_interval,
                fixed_batch=bool(args.fixed_batch),
                inner_batch=args.inner_batch,
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
                save_filename=filename,
                save_payload_fn=make_train_payload_fn(
                    run_id=run_id,
                    method="windowing",
                    windowing=window,
                    unroll_length=steps,
                    weight_scale=args.weight_scale,
                    loss_interval=args.loss_interval,
                    fixed_batch=bool(args.fixed_batch),
                    inner_batch=args.inner_batch,
                ),
                on_exists=on_exists,
                run_label="train",
            )


def plot_results(args):
    files = list_result_files()
    if not files:
        print("No results found to plot.")
        return

    os.makedirs(args.plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    variance_runs = []
    train_runs = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") == "variance":
            variance_runs.append(payload)
        elif payload.get("type") == "train_hpo":
            train_runs.append(payload)

    for payload in variance_runs:
        unroll_lengths = payload["unroll_lengths"]
        variances = jnp.array(payload["variances"])
        if variances.size == 0:
            continue
        plt.figure()
        for idx in range(variances.shape[0]):
            plt.plot(unroll_lengths, variances[idx], color="orange", alpha=0.1)
        median_curve = jnp.median(variances, axis=0)
        plt.plot(unroll_lengths, median_curve, color="black", linewidth=2.0)
        plt.yscale("log")
        median_valid = median_curve[~jnp.isnan(median_curve)]
        if median_valid.size > 0:
            median_min = float(jnp.min(median_valid))
            median_max = float(jnp.max(median_valid))
            plt.ylim(median_min / 10.0, median_max * 1000.0)
        plt.xlabel("unroll length")
        plt.ylabel("gradient variance at random init")
        ts = payload.get("timestamp", "unknown")
        out_path = os.path.join(args.plot_dir, f"variance_{ts}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Wrote plot {out_path}")

    def exp_moving_average(values: jax.Array, alpha: float = 0.2) -> jax.Array:
        if values.size == 0:
            return values
        init = values[0]

        def body(avg, x):
            avg = alpha * x + (1.0 - alpha) * avg
            return avg, avg

        _, smoothed = jax.lax.scan(body, init, values[1:])
        return jnp.concatenate([jnp.array([init]), smoothed])

    if train_runs:
        by_unroll: dict[int, list[dict]] = {}
        for run in train_runs:
            by_unroll.setdefault(run["unroll_length"], []).append(run)
        for unroll, runs in sorted(by_unroll.items()):
            plt.figure()
            for run in runs:
                label = run["method"]
                if run["method"] == "windowing":
                    label = f"window={run['windowing']}"
                plt.plot(run["losses"], label=label)
            plt.xlabel("outer step")
            plt.ylabel("meta loss")
            plt.legend()
            out_path = os.path.join(args.plot_dir, f"train_unroll_{unroll}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Wrote plot {out_path}")

            plt.figure()
            grad_by_label: dict[str, list[jax.Array]] = {}
            for run in runs:
                label = run["method"]
                if run["method"] == "windowing":
                    label = f"window={run['windowing']}"
                grad_norms = jnp.array(run.get("grad_norms", []), dtype=jnp.float32)
                if grad_norms.size == 0:
                    continue
                smoothed = exp_moving_average(grad_norms)
                grad_by_label.setdefault(label, []).append(smoothed)
            for label, curves in grad_by_label.items():
                min_len = min(curve.size for curve in curves)
                if min_len == 0:
                    continue
                stacked = jnp.stack([curve[:min_len] for curve in curves])
                mean_curve = jnp.mean(stacked, axis=0)
                plt.plot(mean_curve, label=label, linewidth=2.0)
            plt.xlabel("outer step")
            plt.ylabel("grad norm")
            plt.yscale("log")
            plt.legend()
            out_path = os.path.join(
                args.plot_dir, f"train_unroll_{unroll}_gradnorms.png"
            )
            plt.savefig(out_path)
            plt.close()
            print(f"Wrote plot {out_path}")


def run_hpo(args):
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
        loss_interval=args.loss_interval,
    )

    outer_params = encode_hyperparams(INIT_LR)

    if OUTER_OPTIMIZER == "adam":
        outer_opt = optax.adam(OUTER_LR)
    else:
        outer_opt = optax.sgd(OUTER_LR)

    outer_opt_state = outer_opt.init(outer_params)

    for step in range(OUTER_STEPS):
        problem = problem.new()
        val_loss, grads = problem.grad(windowing=args.windowing, expanded=True)(
            outer_params, init_args=None
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


def build_parser():
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
        default=-1,
        help="truncate gradients every N steps (-1 for no truncation)",
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
        "--run-train-experiment",
        action="store_true",
        help="run HPO training sweeps across unrolls/methods and save results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="regenerate plots from stored results without running training",
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
        default="10,20,40,80,160",
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
        "--fixed-batch",
        action="store_true",
        help="use a fixed RNG key so batch sequence is identical across unrolls",
    )
    parser.add_argument(
        "--normalize-window-grads",
        action="store_true",
        help="scale windowed gradients by unroll/window size when divisible",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "plots"),
        help="directory to save plots when --plot is used",
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


class Problem(eqx.Module):
    max_steps: int = 1000  # the maximum number of steps

    def __init__(self, *args, **kwargs):
        # store constant hypers
        raise NotImplementedError

    # defines common methods to interact with the thing in question
    def step(self, state, params, stepwise_aux):
        # returns new_state, step_aux - with aux being some random data
        raise NotImplementedError

    def loss(self, params, init_args=None):
        # convenience wrapper to get the loss directly
        if init_args is None:
            init_state = self.initial_state()
        else:
            init_state = self.initial_state(*init_args)

        def body(state, xs_i):
            stepwise_aux, step_idx = xs_i
            new_state, step_aux = self.step(state, params, stepwise_aux)
            loss = self.single_step_loss(new_state, step_idx=step_idx)
            return new_state, (state, aux, loss)

        xs = (self.stepwise_data(), jnp.arange(self.max_steps))
        last_state, (past_states, aux, losses) = jax.lax.scan(body, init_state, xs)

        total_loss = jnp.sum(losses)

        return total_loss

    def initial_state(self, args=None):
        # generates the initial state based off whatever you want
        raise NotImplementedError

    def single_step_loss(self, state, step_idx=None, step_aux=None):
        raise NotImplementedError

    def stepwise_data(self):
        # generates input data for the forward/backward pass, per-step
        # this could be e.g. batches for meta-optimization, character at i^th index
        # for training sentence copying, etc
        raise NotImplementedError

    def grad(
        self,
        windowing: int = -1,
        expanded: bool = False,
        filter=eqx.is_inexact_array,
        ours_simple: bool = False,
        ours_lambda: float = 0.95,
    ):
        def fn(
            params: eqx.Module, init_args: Any = None, data_key: jax.Array | None = None
        ):
            problem = (
                self
                if data_key is None
                else eqx.tree_at(lambda p: p.data_key, self, data_key)
            )
            # standard use for init_args: store the random key there
            params_dyn, params_st = eqx.partition(params, filter)
            if init_args is None:
                init_state = problem.initial_state()
            else:
                init_state = problem.initial_state(*init_args)
            state_dyn, state_st = eqx.partition(init_state, eqx.is_array_like)

            xs = (problem.stepwise_data(), jnp.arange(problem.max_steps))
            if not expanded:
                # Avoid materializing per-step outputs for the non-expanded path.
                def truncate_state(state, step_idx: int):
                    if ours_simple:
                        # print(
                        #    f"Using our-simple method, no_grad * (1.0 - {ours_lambda}) + grad * {ours_lambda}"
                        # )

                        def discount(v: jnp.array):
                            return (
                                jax.lax.stop_gradient(v) * (1.0 - ours_lambda)
                                + v * ours_lambda
                            )

                        return jax.tree.map(
                            lambda v: discount(v) if eqx.is_inexact_array(v) else v,
                            state,
                            is_leaf=eqx.is_inexact_array,
                        )
                    return jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda s: jax.lax.stop_gradient(s),
                        lambda s: s,
                        state,
                    )

                def loss_body(carry, xs_i):
                    state, loss_acc = carry
                    stepwise_aux, step_idx = xs_i
                    state = truncate_state(state, step_idx=step_idx)
                    new_state, _ = problem.step(
                        eqx.combine(state, state_st), params, stepwise_aux
                    )
                    loss = problem.single_step_loss(new_state, step_idx=step_idx)
                    return (
                        eqx.filter(new_state, eqx.is_array_like),
                        loss_acc + loss,
                    ), None

                loss_body = jax.checkpoint(loss_body)
                init_state_f = eqx.filter(init_state, eqx.is_array_like)
                (_, total_loss), _ = jax.lax.scan(
                    loss_body, (init_state_f, jnp.array(0.0)), xs
                )
                return total_loss

            # the loop for scan: a single node in the dependency graph
            def body(state, xs_i):
                stepwise_aux, step_idx = xs_i
                if windowing != -1:
                    state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda: jax.lax.stop_gradient(state),
                        lambda: state,
                    )
                new_state, step_aux = problem.step(
                    eqx.combine(state, state_st), params, stepwise_aux
                )
                loss = problem.single_step_loss(new_state, step_idx=step_idx)
                return eqx.filter(new_state, eqx.is_array_like), (state, step_aux, loss)

            body = body if expanded else jax.checkpoint(body)
            last_state, (past_states, aux, losses) = jax.lax.scan(body, init_state, xs)

            # the final loss is the sum of reported losses
            total_loss = jnp.sum(losses)

            # if we are expanded -> use hand-crafted backprop
            def backward_body(grad_carry, scan_inputs):
                state, stepwise_aux, step_idx = scan_inputs

                new_state, step_aux = problem.step(
                    eqx.combine(state, state_st),
                    eqx.combine(params_dyn, params_st),
                    stepwise_aux,
                )

                grad_from_loss = eqx.filter_grad(
                    lambda s: problem.single_step_loss(s, step_idx=step_idx)
                )(new_state)
                grad_total = jax.tree.map(
                    lambda g1, g2: g1 + g2, grad_carry, grad_from_loss
                )
                vjp_fn = eqx.filter_vjp(
                    lambda p, s: problem.step(
                        eqx.combine(s, state_st),
                        eqx.combine(p, params_st),
                        stepwise_aux,
                    )[0],
                    params_dyn,
                    state,
                )[1]
                grad_for_params, grad_for_state = vjp_fn(grad_total)

                # Apply windowing: zero out grad_for_state AFTER vjp to truncate gradient flow
                # This prevents gradients from flowing to earlier timesteps at window boundaries
                if windowing != -1:
                    grad_for_state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda gs: jax.tree.map(jnp.zeros_like, gs),
                        lambda gs: gs,
                        grad_for_state,
                    )

                return grad_for_state, grad_for_params

            # initialize the grad with zeroes -> \frac{d_L}{d_s} = 0 (gradient w.r.t. state)
            grad_init = jax.tree.map(
                lambda t: jnp.zeros_like(t),
                eqx.filter(
                    init_state, eqx.is_inexact_array
                ),  # that's what's returned by eqx.filter_grad - inexact arrays
            )
            _, stepwise_grads_pytree = jax.lax.scan(
                backward_body, grad_init, (past_states, *xs), reverse=True
            )

            # invert such that the first value in the vector is 1-long gradient, k-th is k-long gradient
            stepwise_grads_pytree = jax.tree.map(
                lambda x: x[::-1], stepwise_grads_pytree
            )

            return total_loss, stepwise_grads_pytree

        if expanded:
            return eqx.filter_jit(fn)
        else:
            return eqx.filter_value_and_grad(fn)


class TrainState(eqx.Module):
    params: Any
    step: jax.Array


class GradientBasedHPO(Problem):
    train_inputs: jax.Array
    train_targets: jax.Array
    val_inputs: jax.Array
    val_targets: jax.Array
    model_static: Any = eqx.field(static=True)
    initial_params: Any
    data_key: jax.Array
    max_steps: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    loss_interval: int = eqx.field(static=True)

    def __init__(
        self,
        model: SimpleMLP,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array],
        num_steps: int,
        batch_size: int,
        key: jax.Array,
        loss_interval: int = LOSS_INTERVAL,
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        self.initial_params = params
        self.model_static = static
        self.train_inputs, self.train_targets = train_data
        self.val_inputs, self.val_targets = val_data
        self.data_key = key
        self.max_steps = num_steps
        self.batch_size = batch_size
        self.loss_interval = loss_interval

    def new(self, key: jax.Array | None = None) -> "GradientBasedHPO":
        if key is None:
            key, _ = jr.split(self.data_key)
        return eqx.tree_at(lambda p: p.data_key, self, key)

    def _train_loss(self, params, batch_inputs, batch_targets):
        model = eqx.combine(params, self.model_static)
        preds = jax.vmap(model)(batch_inputs)
        loss = optax.softmax_cross_entropy(preds, batch_targets)
        return jnp.mean(loss)

    def initial_state(self, init_params=None):
        if init_params is None:
            init_params = self.initial_params
        return TrainState(
            params=init_params,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def single_step_loss(self, state: TrainState, step_idx=None, step_aux=None):
        def fn():
            model = eqx.combine(state.params, self.model_static)
            preds = eqx.filter_vmap(model)(self.val_inputs)
            loss = optax.softmax_cross_entropy(preds, self.val_targets)
            return jnp.mean(loss) / self.max_steps

        return jax.lax.cond(
            step_idx % self.loss_interval == self.loss_interval - 1, fn, lambda: 0.0
        )

    def stepwise_data(self):
        indices = jr.randint(
            self.data_key, (self.max_steps, self.batch_size), 0, len(self.train_inputs)
        )
        return self.train_inputs[indices], self.train_targets[indices]

    def step(self, state, params, stepwise_aux):
        batch_inputs, batch_targets = stepwise_aux
        (lr,) = decode_hyperparams(params)

        _step_loss, grads = eqx.filter_value_and_grad(
            lambda p: self._train_loss(p, batch_inputs, batch_targets),
        )(state.params)

        new_params = jax.tree.map(
            lambda p, g: p - lr * g,
            state.params,
            grads,
        )
        new_step = state.step + 1

        return TrainState(params=new_params, step=new_step), None


if __name__ == "__main__":
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    if args.plot:
        plot_results(args)
    else:
        if args.run_variance_experiment:
            run_variance_experiment(args)
        if args.run_train_experiment:
            run_train_experiment(args)
        if not args.run_variance_experiment and not args.run_train_experiment:
            run_hpo(args)
