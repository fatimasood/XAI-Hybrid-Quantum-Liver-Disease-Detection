#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from utils.config import PLOTS_DIR


class EvaluationPlotter:

    def __init__(
        self,
        y_true,
        y_pred,
        y_prob,
        class_names=["No Disease", "Disease"]
    ):

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.class_names = class_names

    def plot_confusion_matrix(self):

        cm = confusion_matrix(
            self.y_true,
            self.y_pred
        )

        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14,6)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        ax1.set_title(
            "Classical Baseline Confusion Matrix",
            fontsize=13,
            fontweight="bold"
        )

        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            ax=ax2,
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        ax2.set_title(
            "Normalized Confusion Matrix",
            fontsize=13,
            fontweight="bold"
        )

        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_confusion_matrix.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    def plot_roc_curve(self, ci_bounds=None):

        fpr, tpr, _ = roc_curve(
            self.y_true,
            self.y_prob
        )

        plt.figure(figsize=(8,8))

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label="Classical Baseline"
        )

        plt.plot(
            [0,1],
            [0,1],
            "--",
            color="gray",
            label="Random"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title(
            "ROC Curve",
            fontsize=14,
            fontweight="bold"
        )

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_roc_curve.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    def plot_pr_curve(self):

        precision, recall, _ = precision_recall_curve(
            self.y_true,
            self.y_prob
        )

        plt.figure(figsize=(8,8))

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label="Classical Baseline"
        )

        plt.axhline(
            y=self.y_true.mean(),
            linestyle="--",
            color="gray",
            label="Baseline"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title(
            "Precision Recall Curve",
            fontsize=14,
            fontweight="bold"
        )

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_pr_curve.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    def plot_prediction_distribution(self):

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14,5)
        )

        ax1.hist(
            self.y_prob[self.y_true==0],
            bins=30,
            alpha=0.7,
            label="No Disease"
        )

        ax1.hist(
            self.y_prob[self.y_true==1],
            bins=30,
            alpha=0.7,
            label="Disease"
        )

        ax1.set_title(
            "Prediction Distribution",
            fontweight="bold"
        )

        ax1.set_xlabel("Predicted Probability")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        ax2.boxplot([
            self.y_prob[self.y_true==0],
            self.y_prob[self.y_true==1]
        ])

        ax2.set_xticklabels([
            "No Disease",
            "Disease"
        ])

        ax2.set_ylabel("Predicted Probability")

        ax2.set_title(
            "Prediction Boxplot",
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_prediction_distribution.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    def plot_calibration_curve(self):

        prob_true, prob_pred = calibration_curve(
            self.y_true,
            self.y_prob,
            n_bins=10
        )

        plt.figure(figsize=(8,8))

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label="Classical Baseline"
        )

        plt.plot(
            [0,1],
            [0,1],
            "--",
            color="gray",
            label="Perfect Calibration"
        )

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Frequency")

        plt.title(
            "Calibration Curve",
            fontsize=14,
            fontweight="bold"
        )

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_calibration_curve.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    def plot_metrics_radar(self, metrics_dict):

        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        N = len(categories)

        angles = [
            n / float(N) * 2 * np.pi
            for n in range(N)
        ]

        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(8,8),
            subplot_kw=dict(projection="polar")
        )

        ax.plot(
            angles,
            values,
            linewidth=2
        )

        ax.fill(
            angles,
            values,
            alpha=0.25
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        ax.set_ylim(0,1)

        ax.set_title(
            "Classical Baseline Performance",
            fontsize=14,
            fontweight="bold",
            pad=20
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "classical_metrics_radar.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()