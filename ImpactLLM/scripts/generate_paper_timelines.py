#!/usr/bin/env python3
"""Generate the paper timeline figures from the current market model catalog."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "market_models.csv"
FIGURES_DIR = ROOT.parent / "ImpactLLM-paper" / "figures"

PROVIDER_COLORS = {
    "openai": "#0f766e",
    "anthropic": "#b45309",
    "google": "#2563eb",
    "xai": "#7c3aed",
    "meta": "#dc2626",
    "mistral": "#ea580c",
    "alibaba": "#0284c7",
    "deepseek": "#1d4ed8",
}


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("release_date"):
                continue
            try:
                release_date = datetime.strptime(row["release_date"], "%Y-%m-%d")
            except ValueError:
                continue
            rows.append(
                {
                    "display_name": row["display_name"],
                    "provider": row["provider"],
                    "release_date": release_date,
                    "inference_carbon": float(row["screening_per_hour_carbon_gco2e_central"] or 0.0),
                    "training_carbon": float(row["training_carbon_tco2e_central"] or 0.0),
                }
            )
    return sorted(rows, key=lambda item: item["release_date"])


def plot_timeline(rows, metric_key, output_name, title, ylabel):
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.set_facecolor("#fcfcfd")

    for provider in sorted({row["provider"] for row in rows}):
        provider_rows = [row for row in rows if row["provider"] == provider]
        color = PROVIDER_COLORS.get(provider, "#475569")
        ax.scatter(
            [row["release_date"] for row in provider_rows],
            [row[metric_key] for row in provider_rows],
            s=42,
            alpha=0.9,
            color=color,
            label=provider.title(),
        )
        ax.plot(
            [row["release_date"] for row in provider_rows],
            [row[metric_key] for row in provider_rows],
            color=color,
            alpha=0.28,
            linewidth=1.2,
        )

    top_rows = sorted(rows, key=lambda item: item[metric_key], reverse=True)[:12]
    labeled = {row["display_name"] for row in top_rows}
    labeled.update({"GPT-3.5 Turbo", "GPT-4", "Claude 3.5 Sonnet", "Grok 4", "Gemini 2.5 Pro"})
    for row in rows:
        if row["display_name"] not in labeled:
            continue
        ax.annotate(
            row["display_name"],
            (row["release_date"], row[metric_key]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#0f172a",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model release date")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")

    output_path = FIGURES_DIR / output_name
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main():
    rows = load_rows()
    plot_timeline(
        rows,
        metric_key="inference_carbon",
        output_name="inference_release_timeline.png",
        title="Inference carbon by model release date",
        ylabel="Inference carbon for one standardized hour (gCO2e, log scale)",
    )
    plot_timeline(
        rows,
        metric_key="training_carbon",
        output_name="training_release_timeline.png",
        title="Training carbon by model release date",
        ylabel="Training carbon estimate (tCO2e, log scale)",
    )


if __name__ == "__main__":
    main()
