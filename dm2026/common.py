from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)


def read_labels(archive, name):
    with ZipFile(archive) as source:
        return json.loads(source.read(f"{archive.stem}/{name}.json"))


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def binary_metrics(y, score, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(
        n=len(y), positives=int(np.sum(y)), auroc=roc_auc_score(y, score),
        average_precision=average_precision_score(y, score),
        precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0),
        f1=f1_score(y, prediction, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, prediction),
        fpr=float(fp / (fp + tn)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def binary_outputs(folder, frames, title):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, frame in frames.items():
        y, score = frame["label"], frame["score"]
        rows.append(dict(model=name, **binary_metrics(y, score, frame["prediction"])))
        frame.to_csv(folder / f"{name}.csv", index=False)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f'{name} ({rows[-1]["auroc"]:.3f})')
        axes[1].plot(recall, precision, label=f'{name} ({rows[-1]["average_precision"]:.3f})')
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[1].axhline(float(y.mean()), linestyle=":", color="gray", label="prevalence")
    for ax, x, y_label in zip(axes, ["False positive rate", "Recall"], ["Recall", "Precision"]):
        ax.set(xlabel=x, ylabel=y_label, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(folder / "curves.png", dpi=150)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "metrics.csv", index=False)
    return summary
