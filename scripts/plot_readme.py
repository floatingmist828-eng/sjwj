import argparse
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
INK, BLUE, ORANGE, GRAY = "#202832", "#21618C", "#B95724", "#6B7580"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
    "svg.fonttype": "none", "svg.hashsalt": "strata",
    "pdf.fonttype": 42, "figure.facecolor": "white",
})


def clustering(source):
    data = pd.read_csv(source / "cluster/assignments.csv")
    fig, ax = plt.subplots(figsize=(5.1, 3.15), layout="constrained")
    colors = [BLUE, ORANGE, "#17836D", "#A05279", "#6557A5", "#73664B"]
    for (name, cluster), color, marker in zip(
        data.groupby("pca_kmeans_matched"), colors, ("o", "s", "^", "D", "P", "X")
    ):
        ax.scatter(cluster["pc1"], cluster["pc2"], s=11, alpha=0.68, linewidths=0,
                   color=color, marker=marker, label=name)
    ax.set(xlabel="Principal component 1", ylabel="Principal component 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    return fig


def weather(source):
    data = pd.read_csv(source / "weather/test_predictions.csv", parse_dates=["date"])
    shown = data[data["date"] < data["date"].iloc[0] + pd.Timedelta(days=2)]
    # Break the lines at missing observations.
    shown = shown.set_index("date").asfreq("10min")
    metrics = pd.read_csv(source / "weather/metrics.csv")
    selected = metrics.loc[metrics["split"].eq("test") & metrics["selected"]].iloc[0]
    fig, ax = plt.subplots(figsize=(6.8, 2.65), layout="constrained")
    ax.plot(shown.index, shown["actual"], color=INK, lw=1.05, label="Observed")
    ax.plot(shown.index, shown[selected["model"]], color=ORANGE, lw=0.95, ls="--",
            label="Forecast")
    ax.xaxis.set_major_locator(HourLocator(byhour=(0, 12)))
    ax.xaxis.set_major_formatter(DateFormatter("%b %d, %H:%M"))
    ax.set(xlabel="Time", ylabel="CO₂ concentration (ppm)")
    ax.legend(loc="upper right", ncols=2)
    return fig


def zipper_pr(source):
    folder = source / "image/zipper"
    fig, ax = plt.subplots(figsize=(6.1, 3.0), layout="constrained")
    for name, label, color, linestyle in [
        ("patch_memory", "Patch memory", GRAY, "-."),
        ("density_memory", "Density correction", BLUE, "-"),
        ("global_svm", "Global SVM", "#17836D", ":"),
        ("fusion", "Fusion", ORANGE, "--"),
    ]:
        frame = pd.read_csv(folder / f"{name}.csv")
        precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
        ap = average_precision_score(frame["label"], frame["score"])
        ax.step(recall[::-1], precision[::-1], where="pre", color=color,
                linestyle=linestyle, lw=1.25, label=f"{label} ({ap:.3f})")
    prevalence = frame["label"].mean()
    ax.axhline(prevalence, color=GRAY, ls=(0, (2, 4)), lw=0.8,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.015), ylim=(0, 1.045))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
              title="Average precision", title_fontsize=7.5)
    return fig


def cross_modal(source):
    data = pd.read_csv(source / "cross-modal/domain_metrics.csv")
    scores = data.pivot(index="domain", columns="model", values="average_precision")
    scores = scores.loc[["hazelnut", "zipper", "thyroid"]]
    scores.loc["macro"] = scores.mean()
    original, adjusted = scores["shared_memory"], scores["density_memory"]
    rows = np.arange(len(scores))[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 2.5), layout="constrained")
    ax.hlines(rows, original, adjusted, color="#B8C0CA", lw=1.15)
    ax.scatter(original, rows, s=34, marker="o", edgecolors=GRAY, facecolors="white",
               linewidths=1.1, label="Normal memory", zorder=3)
    ax.scatter(adjusted, rows, s=19, marker="D", color=BLUE, linewidths=0,
               label="Density correction", zorder=4)
    ax.set_yticks(rows, ["Hazelnut", "Zipper", "Thyroid", "Macro average"])
    ax.set(xlabel="Average precision", xlim=(0.72, 1.025), ylim=(-0.55, 3.55))
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Render README figures from experiment outputs")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs")
    parser.add_argument("--qa", type=Path, help="Optional PDF and PNG export directory")
    args = parser.parse_args()
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    if args.qa:
        args.qa.mkdir(parents=True, exist_ok=True)
    for name, plot in [
        ("clustering", clustering), ("co2-forecast", weather),
        ("zipper-pr", zipper_pr), ("cross-modal-ap", cross_modal),
    ]:
        fig = plot(args.results)
        svg = StringIO()
        fig.savefig(svg, format="svg", metadata={"Date": None})
        lines = svg.getvalue().splitlines()
        (assets / f"{name}.svg").write_text(
            "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8", newline="\n")
        if args.qa:
            fig.savefig(args.qa / f"{name}.pdf")
            fig.savefig(args.qa / f"{name}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
