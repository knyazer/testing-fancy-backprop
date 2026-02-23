import argparse
import hashlib
import json
import os
import time
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from problems import problems, ProblemSpec
from problems.base import NO_WINDOWING, Problem

RESULTS_DIR = "results"
OUTER_STEPS = 200
SEED = 0

METHODS: list[tuple[str, dict]] = [
    ("raw", dict(windowing=NO_WINDOWING, ours_simple=False)),
    ("ours_0.99", dict(windowing=NO_WINDOWING, ours_simple=True, ours_lambda=0.99)),
    ("ours_0.95", dict(windowing=NO_WINDOWING, ours_simple=True, ours_lambda=0.95)),
    ("ours_0.90", dict(windowing=NO_WINDOWING, ours_simple=True, ours_lambda=0.90)),
    ("window_10", dict(windowing=10, ours_simple=False)),
    ("window_20", dict(windowing=20, ours_simple=False)),
    ("window_40", dict(windowing=40, ours_simple=False)),
    ("window_80", dict(windowing=80, ours_simple=False)),
]


def _results_dir(spec: ProblemSpec) -> str:
    return os.path.join(RESULTS_DIR, spec.name)


def stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_json(
    payload: dict, path: str, *, on_exists: str = "error"
) -> str | None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        if on_exists == "skip":
            return None
        if on_exists == "error":
            raise FileExistsError(f"{path} already exists.")
        if on_exists != "overwrite":
            raise ValueError(f"Unknown on_exists policy: {on_exists}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def should_skip_file(path: str, *, on_exists: str) -> bool:
    if not os.path.exists(path):
        return False
    if on_exists == "overwrite":
        return False
    if on_exists == "error":
        raise FileExistsError(f"{path} already exists.")
    if on_exists == "skip":
        print(f"Skipping existing {path}")
        return True
    raise ValueError(f"Unknown on_exists policy: {on_exists}")


def list_result_files() -> list[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    result = []
    for dirpath, _dirnames, filenames in os.walk(RESULTS_DIR):
        for name in filenames:
            if name.endswith(".json"):
                result.append(os.path.join(dirpath, name))
    return sorted(result)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def results_on_exists_policy(args: argparse.Namespace) -> str:
    if args.overwrite_results:
        return "overwrite"
    if args.skip_existing_results:
        return "skip"
    return "error"


def find_spec(name: str) -> ProblemSpec:
    for spec in problems:
        if spec.name == name:
            return spec
    available = ", ".join(s.name for s in problems)
    raise ValueError(f"Unknown problem {name!r}. Available: {available}")


# ---------------------------------------------------------------------------
# Eval: gradient norm vs unroll length
# ---------------------------------------------------------------------------


def run_eval_grad_norms(spec: ProblemSpec, args: argparse.Namespace) -> None:
    unroll_lengths = parse_int_list(args.eval_unroll_lengths)
    on_exists = results_on_exists_policy(args)
    config = {
        "type": "eval_grad_norms",
        "problem": spec.name,
        "unroll_lengths": unroll_lengths,
        "num_inits": args.eval_num_inits,
    }
    run_id = stable_json_hash(config)
    out_dir = _results_dir(spec)
    filename = os.path.join(out_dir, f"eval_grad_norms_{run_id}.json")
    if should_skip_file(filename, on_exists=on_exists):
        return

    key = jr.key(SEED)
    data_key, params_key, init_key = jr.split(key, 3)

    outer_params = spec.default_outer_params(key=params_key)

    results: dict[str, dict[str, list[float]]] = {
        name: {"mean_norms": [], "std_norms": [], "max_norms": [], "nan_frac": []}
        for name, _ in METHODS
    }

    for steps in unroll_lengths:
        problem = spec.build(num_steps=steps, key=data_key)

        for method_name, method_kwargs in METHODS:
            grad_fn = problem.grad(expanded=False, **method_kwargs)
            jit_fn = eqx.filter_jit(grad_fn)

            norms: list[float] = []
            for i in range(args.eval_num_inits):
                sample_key = jr.fold_in(init_key, i)
                init_params = spec.sample_init_params(key=sample_key)
                init_args = (init_params,) if init_params is not None else None
                _, grads = jit_fn(outer_params, init_args=init_args)
                flat_grads = jax.tree.leaves(grads)
                norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads)))
                norms.append(norm)

            norms_arr = jnp.array(norms)
            nan_count = int(jnp.sum(jnp.isnan(norms_arr)))
            finite = norms_arr[~jnp.isnan(norms_arr)]
            mean_norm = float(jnp.mean(finite)) if finite.size > 0 else float("nan")
            std_norm = float(jnp.std(finite)) if finite.size > 0 else float("nan")
            max_norm = float(jnp.max(finite)) if finite.size > 0 else float("nan")

            results[method_name]["mean_norms"].append(mean_norm)
            results[method_name]["std_norms"].append(std_norm)
            results[method_name]["max_norms"].append(max_norm)
            results[method_name]["nan_frac"].append(nan_count / len(norms))

            print(
                f"[{spec.name}] steps={steps:4d} {method_name:12s} "
                f"mean={mean_norm:.4e} std={std_norm:.4e} max={max_norm:.4e} "
                f"nan={nan_count}/{len(norms)}"
            )
        print()

    payload = {
        "type": "eval_grad_norms",
        "timestamp": int(time.time()),
        "run_id": run_id,
        "problem": spec.name,
        "unroll_lengths": unroll_lengths,
        "num_inits": args.eval_num_inits,
        "methods": {name: results[name] for name, _ in METHODS},
    }
    path = save_json(payload, filename, on_exists=on_exists)
    if path:
        print(f"Saved eval_grad_norms to {path}")


# ---------------------------------------------------------------------------
# Eval: training comparison across methods
# ---------------------------------------------------------------------------


def run_eval_training(spec: ProblemSpec, args: argparse.Namespace) -> None:
    unroll_lengths = parse_int_list(args.eval_unroll_lengths)
    on_exists = results_on_exists_policy(args)

    key = jr.key(SEED)
    data_key, params_key = jr.split(key)

    outer_opt = spec.outer_optimizer()

    for steps in unroll_lengths:
        problem = spec.build(num_steps=steps, key=data_key)

        for method_name, method_kwargs in METHODS:
            config = {
                "type": "eval_training",
                "problem": spec.name,
                "unroll_length": steps,
                "method": method_name,
                "outer_steps": OUTER_STEPS,
                "outer_optimizer": str(spec.outer_optimizer()),
            }
            run_id = stable_json_hash(config)
            out_dir = _results_dir(spec)
            filepath = os.path.join(out_dir, f"eval_training_{run_id}.json")
            if should_skip_file(filepath, on_exists=on_exists):
                continue

            outer_params = spec.default_outer_params(key=params_key)
            opt_state = outer_opt.init(eqx.filter(outer_params, eqx.is_inexact_array))
            grad_fn = problem.grad(expanded=False, **method_kwargs)
            jit_fn = eqx.filter_jit(grad_fn)

            losses: list[float] = []
            param_descriptions: list[str] = []
            grad_norms: list[float] = []

            run_data_key = data_key
            for step in range(OUTER_STEPS):
                run_data_key, step_key, init_key = jr.split(run_data_key, 3)
                init_params = spec.sample_init_params(key=init_key)
                init_args = (init_params,) if init_params is not None else None
                loss, grads = jit_fn(
                    outer_params, init_args=init_args, data_key=step_key
                )

                flat_grads = jax.tree.leaves(grads)
                g_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads)))

                updates, opt_state = outer_opt.update(grads, opt_state)
                outer_params = eqx.apply_updates(outer_params, updates)
                outer_params = spec.project_outer_params(outer_params)

                losses.append(float(loss))
                param_descriptions.append(spec.describe_outer_params(outer_params))
                grad_norms.append(g_norm)

            losses_arr = jnp.array(losses)
            best_idx = int(jnp.argmin(losses_arr))
            print(
                f"[{spec.name}] steps={steps:4d} {method_name:12s} "
                f"best_loss={losses[best_idx]:.4f} final={losses[-1]:.4f} "
                f"{param_descriptions[-1]}"
            )

            payload = {
                "type": "eval_training",
                "timestamp": int(time.time()),
                "run_id": run_id,
                "problem": spec.name,
                "method": method_name,
                "unroll_length": steps,
                "outer_steps": OUTER_STEPS,
                "outer_optimizer": str(spec.outer_optimizer()),
                "losses": losses,
                "param_descriptions": param_descriptions,
                "grad_norms": grad_norms,
            }
            path = save_json(payload, filepath, on_exists=on_exists)
            if path:
                print(f"  -> saved to {path}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gradient-based optimization experiments."
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="hpo",
        help="problem name (available: " + ", ".join(s.name for s in problems) + ")",
    )
    parser.add_argument(
        "--run-eval-grad-norms",
        action="store_true",
        help="run eval: gradient norm vs unroll length for each method",
    )
    parser.add_argument(
        "--run-eval-training",
        action="store_true",
        help="run eval: training comparison across methods",
    )
    parser.add_argument(
        "--eval-unroll-lengths",
        type=str,
        default="100,200,400,640,1000",
        help="comma-separated unroll lengths for eval experiments",
    )
    parser.add_argument(
        "--eval-num-inits",
        type=int,
        default=16,
        help="number of model initializations for eval gradient norm sweep",
    )
    results_group = parser.add_mutually_exclusive_group()
    results_group.add_argument(
        "--overwrite-results",
        action="store_true",
        help="overwrite existing result files",
    )
    results_group.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="skip runs whose results already exist",
    )
    return parser


if __name__ == "__main__":
    arg_parser = build_parser()
    args = arg_parser.parse_args()
    spec = find_spec(args.problem)

    if args.run_eval_grad_norms:
        run_eval_grad_norms(spec, args)
    if args.run_eval_training:
        run_eval_training(spec, args)
