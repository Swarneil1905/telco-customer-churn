from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

COLORS = ["#176b87", "#ed7d31", "#4477aa", "#228833", "#aa3377"]
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def save_figure(figure, folder, name):
    figure.savefig(Path(folder) / f"{name}.png", dpi=180, bbox_inches="tight")
    return figure


def comparison_plot(metrics, folder):
    values = metrics.sort_values("AP")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(values.index, values.AP, color=COLORS[0])
    ax.set(xlabel="Test average precision (higher is better)", xlim=(0, 1))
    ax.set_title("How well do the models rank churners?")
    for i, value in enumerate(values.AP):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")
    return save_figure(fig, folder, "model_comparison")


def diagnostic_plots(y, probabilities, winner, threshold, folder):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in dict.fromkeys(
        [winner, "Logistic regression", "Prior baseline"]
    ):
        PrecisionRecallDisplay.from_predictions(
            y, probabilities[name], name=name, ax=axes[0]
        )
        RocCurveDisplay.from_predictions(
            y, probabilities[name], name=name, ax=axes[1]
        )
    axes[0].axhline(np.mean(y), color="grey", linestyle=":")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("Precision and recall on unseen customers")
    axes[1].set_title("ROC curve on unseen customers")
    fig.tight_layout()
    save_figure(fig, folder, "ranking_curves")
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, value in zip(axes, [0.5, threshold]):
        ConfusionMatrixDisplay.from_predictions(
            y,
            probabilities[winner] >= value,
            display_labels=["Stayed", "Churned"],
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )
        ax.set_title(f"Frozen threshold: {value:.2f}")
    fig2.tight_layout()
    save_figure(fig2, folder, "confusion_matrices")
    fig3, ax = plt.subplots(figsize=(6, 5))
    actual, predicted = calibration_curve(
        y, probabilities[winner], n_bins=10, strategy="quantile"
    )
    ax.plot(predicted, actual, "o-", color=COLORS[0], label=winner)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Ideal calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate")
    ax.set_title("Are the risk probabilities well calibrated?")
    ax.legend()
    save_figure(fig3, folder, "calibration")
    return fig, fig2, fig3


def threshold_plot(table, threshold, folder):
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, color in zip(["precision", "recall", "F1"], COLORS):
        ax.plot(table.threshold, table[col], label=col, color=color)
    ax.axvline(
        threshold, color="black", linestyle="--", label="Chosen on validation"
    )
    ax.set(xlabel="Decision threshold", ylabel="Validation score", ylim=(0, 1))
    ax.set_title("The operating threshold changes who gets flagged")
    ax.legend(loc="lower left")
    return save_figure(fig, folder, "threshold_tradeoff")


def light_scene(fig, title, x, y, z):
    axis = dict(
        showbackground=True,
        backgroundcolor="#f1f5f9",
        gridcolor="#cbd5e1",
        color="#182b49",
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=660,
        paper_bgcolor="white",
        font=dict(color="#182b49"),
        margin=dict(l=5, r=5, b=25, t=65),
        scene=dict(
            xaxis=dict(title=x, **axis),
            yaxis=dict(title=y, **axis),
            zaxis=dict(title=z, **axis),
            camera=dict(eye=dict(x=1.65, y=-1.8, z=1.25)),
        ),
    )
    return fig


def cohort_plot(train, folder):
    frame = train.copy()
    frame["tenure_band"] = pd.cut(
        frame.tenure,
        [-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    frame["charge_band"] = pd.cut(
        frame.MonthlyCharges,
        [0, 35, 55, 75, 95, 130],
        labels=["0-35", "35-55", "55-75", "75-95", "95-130"],
    )
    frame["churn"] = frame.Churn.eq("Yes").astype(int)
    cohort = (
        frame.groupby(["tenure_band", "charge_band"], observed=True)
        .churn.agg(["mean", "size", "sum"])
        .reset_index()
    )
    n, rate, z = cohort["size"], cohort["mean"], 1.96
    center = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
    half /= 1 + z**2 / n
    cohort["lower"], cohort["upper"] = center - half, center + half
    cohort.to_csv(Path(folder).parent / "training_cohorts.csv", index=False)
    supported = cohort[cohort["size"] >= 20].copy()
    x = supported.tenure_band.cat.codes.to_numpy()
    y = supported.charge_band.cat.codes.to_numpy()
    rate = supported["mean"].to_numpy()
    custom = np.column_stack(
        [
            supported.tenure_band.astype(str),
            supported.charge_band.astype(str),
            supported["size"],
            supported.lower,
            supported.upper,
        ]
    )
    fig = go.Figure(
        go.Scatter3d(
            x=x,
            y=y,
            z=rate,
            mode="markers",
            marker=dict(
                size=np.sqrt(supported["size"]) / 2 + 5,
                color=rate,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Churn rate"),
                opacity=0.95,
            ),
            error_z=dict(
                type="data",
                symmetric=False,
                array=supported.upper - rate,
                arrayminus=rate - supported.lower,
                color="#475569",
                thickness=2,
            ),
            customdata=custom,
            hovertemplate=(
                "Tenure: %{customdata[0]} months<br>"
                "Monthly charge: %{customdata[1]}<br>"
                "Customers: %{customdata[2]}<br>"
                "Churn rate: %{z:.1%}<extra></extra>"
            ),
        )
    )
    light_scene(
        fig,
        "Observed churn across customer cohorts",
        "Tenure band (months)",
        "Monthly charge band",
        "Observed churn rate",
    )
    fig.update_scenes(
        xaxis=dict(
            tickvals=list(range(5)),
            ticktext=["0-6", "7-12", "13-24", "25-48", "49-72"],
        ),
        yaxis=dict(
            tickvals=list(range(5)),
            ticktext=["0-35", "35-55", "55-75", "75-95", "95-130"],
        ),
        zaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    fig.write_json(Path(folder).parent / "cohort_3d.json")
    fig.write_html(
        Path(folder).parent / "cohort_3d.html", include_plotlyjs=True
    )
    static = plt.figure(figsize=(10, 7))
    ax = static.add_subplot(projection="3d")
    scatter = ax.scatter(
        x, y, rate, c=rate, cmap="viridis", vmin=0, vmax=1, s=70
    )
    for a, b, low, high in zip(x, y, supported.lower, supported.upper):
        ax.plot([a, a], [b, b], [low, high], color="#64748b")
    ax.set_xticks(range(5), ["0-6", "7-12", "13-24", "25-48", "49-72"])
    ax.set_yticks(range(5), ["0-35", "35-55", "55-75", "75-95", "95-130"])
    ax.set(
        xlabel="Tenure (months)",
        ylabel="Monthly charges",
        zlabel="Churn rate",
        zlim=(0, 1),
    )
    ax.set_title("Training cohorts: observed rates and 95% Wilson intervals")
    static.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    save_figure(static, folder, "cohort_3d")
    return fig


def cost_plot(table, sample_count, folder):
    ratios = np.linspace(1, 10, 40)
    thresholds = table.threshold.to_numpy()
    x, y = np.meshgrid(thresholds, ratios)
    cost = table.FP.to_numpy()[None, :] + y * table.FN.to_numpy()[None, :]
    cost = cost * 100 / sample_count
    fig = go.Figure(
        go.Surface(
            x=x,
            y=y,
            z=cost,
            colorscale="Viridis",
            colorbar=dict(title="Cost units"),
            lighting=dict(ambient=0.85, diffuse=0.7),
        )
    )
    light_scene(
        fig,
        "Threshold and error-cost trade-off (validation)",
        "Decision threshold",
        "Missed churn / false alarm cost",
        "Cost units per 100 customers",
    )
    fig.write_json(Path(folder).parent / "cost_3d.json")
    fig.write_html(Path(folder).parent / "cost_3d.html", include_plotlyjs=True)
    static = plt.figure(figsize=(10, 7))
    ax = static.add_subplot(projection="3d")
    surface = ax.plot_surface(x, y, cost, cmap="viridis", linewidth=0)
    ax.set(
        xlabel="Threshold", ylabel="Error cost ratio", zlabel="Cost per 100"
    )
    ax.set_title("Hypothetical classification costs, not retention ROI")
    static.colorbar(surface, ax=ax, shrink=0.55, pad=0.12)
    save_figure(static, folder, "cost_3d")
    return fig


def importance_plot(importance, folder):
    values = importance.sort_values("mean").tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        values.feature, values["mean"], xerr=values["std"], color=COLORS[0]
    )
    ax.set(xlabel="Validation AP drop after feature shuffling")
    ax.set_title("Predictive contribution, not causal effect")
    return save_figure(fig, folder, "permutation_importance")
