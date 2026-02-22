import argparse
import json
import os

import jax
import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt

from run import RESULTS_DIR, list_result_files


def exp_moving_average(values: jax.Array, alpha: float = 0.2) -> jax.Array:
    if values.size == 0:
        return values
    init = values[0]

    def body(avg, x):
        avg = alpha * x + (1.0 - alpha) * avg
        return avg, avg

    _, smoothed = jax.lax.scan(body, init, values[1:])
    return jnp.concatenate([jnp.array([init]), smoothed])


def plot_variance_snapshot(
    variances: jax.Array, *, out_path: str, title: str | None = None
) -> None:
    if variances.size == 0:
        return
    xs = jnp.arange(variances.shape[0])
    plt.figure()
    plt.scatter(xs, variances, color="orange", alpha=0.6)
    median = jnp.median(variances)
    plt.axhline(median, color="black", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("init index")
    plt.ylabel("gradient variance")
    if title:
        plt.title(title)
    plt.savefig(out_path)
    plt.close()


def plot_results(plot_dir: str) -> None:
    files = list_result_files()
    if not files:
        print("No results found to plot.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    variance_runs: list[dict] = []
    variance_snapshot_runs: list[dict] = []
    train_runs: list[dict] = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") == "variance":
            variance_runs.append(payload)
        elif payload.get("type") == "variance_snapshots":
            variance_snapshot_runs.append(payload)
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
        out_path = os.path.join(plot_dir, f"variance_{ts}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Wrote plot {out_path}")

    for payload in variance_snapshot_runs:
        for run in payload.get("runs", []):
            unroll_length = run["unroll_length"]
            for snapshot in run.get("snapshots", []):
                step = snapshot["step"]
                lr_value = snapshot.get("lr")
                variances = jnp.array(
                    snapshot.get("variances", []), dtype=jnp.float32
                )
                title = (
                    f"{payload.get('method', 'ours')} unroll={unroll_length} "
                    f"step={step} lr={lr_value:.6f}"
                    if lr_value is not None
                    else None
                )
                out_path = os.path.join(
                    plot_dir,
                    f"variance_snapshot_{payload.get('run_id')}_unroll{unroll_length}_step{step}.png",
                )
                plot_variance_snapshot(variances, out_path=out_path, title=title)
                print(f"Wrote plot {out_path}")

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
            out_path = os.path.join(plot_dir, f"train_unroll_{unroll}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Wrote plot {out_path}")

            plt.figure()
            grad_by_label: dict[str, list[jax.Array]] = {}
            for run in runs:
                label = run["method"]
                if run["method"] == "windowing":
                    label = f"window={run['windowing']}"
                grad_norms = jnp.array(
                    run.get("grad_norms", []), dtype=jnp.float32
                )
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
                plot_dir, f"train_unroll_{unroll}_gradnorms.png"
            )
            plt.savefig(out_path)
            plt.close()
            print(f"Wrote plot {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "plots"),
        help="directory to save plots",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_results(args.plot_dir)
