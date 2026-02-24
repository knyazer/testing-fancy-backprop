import argparse
import json
import os
from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from run import RESULTS_DIR, list_result_files

OURS_METHODS = ["ours_0.70", "ours_0.80", "ours_0.90", "ours_0.95", "ours_0.98", "ours_0.99"]
WINDOW_METHODS = ["window_3", "window_5", "window_10", "window_20", "window_50", "window_100"]

OURS_CMAP = plt.cm.Reds
WINDOW_CMAP = plt.cm.Blues


def _method_color(name: str) -> str:
    if name == "raw":
        return "black"
    ours_list = OURS_METHODS
    window_list = WINDOW_METHODS
    if name in ours_list:
        idx = ours_list.index(name)
        return OURS_CMAP(0.3 + 0.6 * idx / max(len(ours_list) - 1, 1))
    if name in window_list:
        idx = window_list.index(name)
        return WINDOW_CMAP(0.3 + 0.6 * idx / max(len(window_list) - 1, 1))
    return "gray"


def _method_style(name: str) -> str:
    if name.startswith("ours_"):
        return "-"
    if name.startswith("window_"):
        return "--"
    return "-."


def _load_results_by_type() -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for path in list_result_files():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rtype = payload.get("type", "unknown")
        by_type[rtype].append(payload)
    return by_type


def _latest_by_problem(runs: list[dict]) -> dict[str, dict]:
    by_problem: dict[str, dict] = {}
    for run in runs:
        problem = run.get("problem")
        if problem is None:
            continue
        if problem not in by_problem or run.get("timestamp", 0) > by_problem[problem].get("timestamp", 0):
            by_problem[problem] = run
    return by_problem


def plot_grad_norms(plot_dir: str, grad_norm_runs: list[dict]) -> None:
    by_problem = _latest_by_problem(grad_norm_runs)

    for problem, payload in sorted(by_problem.items()):
        unroll_lengths = payload["unroll_lengths"]
        methods = payload["methods"]

        fig, ax = plt.subplots(figsize=(8, 5))
        for method_name, data in sorted(methods.items()):
            mean_norms = np.array(data["mean_norms"])
            valid = ~np.isnan(mean_norms)
            if not np.any(valid):
                continue
            ax.plot(
                np.array(unroll_lengths)[valid],
                mean_norms[valid],
                label=method_name,
                color=_method_color(method_name),
                linestyle=_method_style(method_name),
                linewidth=2,
                marker="o",
                markersize=4,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Unroll length")
        ax.set_ylabel("Mean gradient norm")
        ax.set_title(f"Gradient Norms — {problem}")
        ax.legend(fontsize=7, ncol=2)
        out_path = os.path.join(plot_dir, f"grad_norms_{problem}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def _loss_series(run: dict) -> list[float]:
    """Return eval_losses if available, otherwise training losses."""
    eval_losses = run.get("eval_losses", [])
    return eval_losses if eval_losses else run["losses"]


def _untrained_loss(training_runs: list[dict], problem: str) -> float | None:
    """Extract the untrained model loss (step-0 loss) for a given problem."""
    for run in training_runs:
        if run.get("problem") == problem and run["method"] == "raw" and run.get("losses"):
            series = _loss_series(run)
            return series[0]
    return None


def _aggregate_best_losses(
    training_runs: list[dict],
) -> dict[str, dict[str, dict[int, float]]]:
    current_runs = [r for r in training_runs if r.get("outer_optimizer", "?") != "?"]
    by_problem: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for run in current_runs:
        problem = run["problem"]
        method = run["method"]
        unroll = run["unroll_length"]
        best = min(_loss_series(run))
        existing = by_problem[problem][method].get(unroll)
        if existing is None or best < existing:
            by_problem[problem][method][unroll] = best
    return by_problem


def _plot_best_loss_family(
    ax: plt.Axes,
    methods_data: dict[str, dict[int, float]],
    family_methods: list[str],
    problem: str,
    untrained: float | None,
) -> None:
    # Always include raw as baseline
    for method_name in ["raw"] + family_methods:
        unroll_losses = methods_data.get(method_name)
        if unroll_losses is None:
            continue
        unrolls = sorted(unroll_losses.keys())
        losses = [unroll_losses[u] for u in unrolls]
        valid = [(u, l) for u, l in zip(unrolls, losses) if not np.isnan(l)]
        if not valid:
            continue
        us, ls = zip(*valid)
        ax.plot(
            us, ls,
            label=method_name,
            color=_method_color(method_name),
            linestyle=_method_style(method_name),
            linewidth=2,
            marker="o",
            markersize=4,
        )

    if untrained is not None and not np.isnan(untrained):
        ax.axhline(untrained, color="gray", linestyle=":", linewidth=1.5,
                    label=f"untrained ({untrained:.2f})")

    if problem == "linear_rnn_copy":
        ax.set_yscale("log")

    ax.set_xlabel("Unroll length")
    ax.set_ylabel("Best loss achieved")


def plot_training_best_loss(plot_dir: str, training_runs: list[dict]) -> None:
    by_problem = _aggregate_best_losses(training_runs)

    for problem, methods_data in sorted(by_problem.items()):
        untrained = _untrained_loss(training_runs, problem)

        for family_name, family_methods in [("ours", OURS_METHODS), ("window", WINDOW_METHODS)]:
            fig, ax = plt.subplots(figsize=(8, 5))
            _plot_best_loss_family(ax, methods_data, family_methods, problem, untrained)
            ax.set_title(f"Training Performance — {problem} ({family_name})")
            ax.legend(fontsize=7, ncol=2)
            out_path = os.path.join(plot_dir, f"training_best_loss_{problem}_{family_name}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_path}")


def plot_training_curves(plot_dir: str, training_runs: list[dict]) -> None:
    current_runs = [r for r in training_runs if r.get("outer_optimizer", "?") != "?"]

    by_problem_unroll: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in current_runs:
        by_problem_unroll[(run["problem"], run["unroll_length"])].append(run)

    for (problem, unroll), runs in sorted(by_problem_unroll.items()):
        for family_name, family_methods in [("ours", OURS_METHODS), ("window", WINDOW_METHODS)]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for run in sorted(runs, key=lambda r: r["method"]):
                method = run["method"]
                if method != "raw" and method not in family_methods:
                    continue
                losses = np.array(_loss_series(run))
                ax.plot(
                    losses,
                    label=method,
                    color=_method_color(method),
                    linestyle=_method_style(method),
                    linewidth=1.5,
                    alpha=0.8,
                )

            ax.set_xlabel("Outer step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Curves — {problem}, unroll={unroll} ({family_name})")
            ax.legend(fontsize=6, ncol=2)
            out_path = os.path.join(plot_dir, f"training_curves_{problem}_unroll{unroll}_{family_name}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_path}")


def plot_gradient_profile(plot_dir: str, profile_runs: list[dict]) -> None:
    for payload in profile_runs:
        problem = payload["problem"]
        unroll = payload["unroll_length"]
        methods = payload["methods"]

        fig, ax = plt.subplots(figsize=(8, 5))
        for method_name, data in sorted(methods.items()):
            mean_norms = np.array(data["per_step_mean_norms"])
            std_norms = np.array(data["per_step_std_norms"])
            steps = np.arange(len(mean_norms))

            ax.plot(
                steps, mean_norms,
                label=method_name,
                color=_method_color(method_name),
                linestyle=_method_style(method_name),
                linewidth=1.5,
            )
            ax.fill_between(
                steps,
                np.maximum(mean_norms - std_norms, 1e-10),
                mean_norms + std_norms,
                color=_method_color(method_name),
                alpha=0.1,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Per-step gradient norm")
        ax.set_title(f"Gradient Profile — {problem}, unroll={unroll}")
        ax.legend(fontsize=7)
        out_path = os.path.join(plot_dir, f"gradient_profile_{problem}_unroll{unroll}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_all(plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    by_type = _load_results_by_type()

    if "eval_grad_norms" in by_type:
        plot_grad_norms(plot_dir, by_type["eval_grad_norms"])

    if "eval_training" in by_type:
        plot_training_best_loss(plot_dir, by_type["eval_training"])
        plot_training_curves(plot_dir, by_type["eval_training"])

    if "eval_gradient_profile" in by_type:
        plot_gradient_profile(plot_dir, by_type["eval_gradient_profile"])


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
    plot_all(args.plot_dir)
