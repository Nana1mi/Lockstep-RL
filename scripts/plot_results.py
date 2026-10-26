"""Plot saved measurement files only; never synthesizes an experiment result."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=["consistency", "training"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.kind == "consistency":
        data = json.loads(args.input.read_text(encoding="utf-8"))
        values = data["delta_logp"]
        if not values:
            parser.error("no measured logprob differences in input")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(50, max(1, len(values))))
        ax.set(xlabel="Reference logprob - rollout logprob", ylabel="Token count",
               title="Known-error test fixture" if data.get("known_error_injection") else "Measured fixed-trajectory differences")
    else:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            parser.error("empty metrics input")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        x = [row["iteration"] for row in rows]
        axes[0].plot(x, [row["mean_reward"] for row in rows], marker="o", label="Training reward")
        validation = [row for row in rows if "validation" in row]
        if validation:
            axes[0].plot([row["iteration"] for row in validation], [row["validation"]["accuracy"] for row in validation], marker="o", label="Validation accuracy")
        axes[0].legend()
        for metric in ("ratio_outside_fraction", "clip_active_fraction"):
            points = [(row["iteration"], row["steps"][-1][metric]) for row in rows if row["steps"]]
            if points:
                axes[1].plot([p[0] for p in points], [p[1] for p in points], label=metric)
        axes[1].legend()
        for stage in ("rollout", "reward", "training", "sync"):
            axes[2].plot(x, [row["timings"][stage] for row in rows], label=stage)
        axes[2].legend()
        axes[2].set_ylabel("Seconds")
        for ax in axes:
            ax.set_xlabel("Iteration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
