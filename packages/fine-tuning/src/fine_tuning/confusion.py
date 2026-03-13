from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.special import xlog1py
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

LABELS = [
    "O",
    "B-Document",
    "I-Document",
    "B-Class",
    "I-Class",
    "B-Abstraction",
    "I-Abstraction",
    "B-Act",
    "I-Act",
    "B-Organization",
    "I-Organization",
    "B-Person",
    "I-Person",
]

DISPLAY_LABELS = [
    "O",
    "B-Doc",
    "I-Doc",
    "B-Cls",
    "I-Cls",
    "B-Abs",
    "I-Abs",
    "B-Act",
    "I-Act",
    "B-Org",
    "I-Org",
    "B-Per",
    "I-Per",
]


def confusion(predictions: Path, references: Path, output_path: Path) -> None:
    with (
        open(predictions, "r", encoding="utf-8") as preds,
        open(references, "r", encoding="utf-8") as labels,
    ):
        y_true: list[str] = []
        y_pred: list[str] = []
        for line in zip(labels, preds):
            if line and line[0] != "\n":
                _, label = line[0].split(" ")
                label = label.strip()
                _, pred = line[1].split(" ")
                pred = pred.strip()
                y_true.append(label)
                y_pred.append(pred)

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_perc = (cm.astype("float") / row_sums) * 100

        plt.figure(figsize=(8, 8))

        sns.heatmap(
            cm_perc,
            annot=True,  # We put the REAL percentages in the text labels
            fmt=".1f",  # Format labels as percentages
            cmap="viridis",  # 'viridis' or 'magma' work great with log scales
            vmin=0,
            vmax=100,
            xticklabels=DISPLAY_LABELS,
            yticklabels=DISPLAY_LABELS,
            annot_kws={"size": 9},  # SHRINK this if numbers don't fit
            cbar_kws={"label": "Percentage (%)"},
        )

        plt.rcParams.update({"font.size": 8})
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
