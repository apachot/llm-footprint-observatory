#!/usr/bin/env python3
"""Generate academic-style discussion figures for doubling views in the paper."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.estimator import get_country_mix, to_float, wh_to_gco2e

DATA_PATH = ROOT / "data" / "market_models.csv"
FIGURES_DIR = ROOT.parent / "ImpactLLM-paper" / "figures"

KEEP_NAMES = [
    "GPT-3.5 Turbo",
    "GPT-4",
    "GPT-5.2",
    "GPT-5.4",
    "GPT-5.5",
    "Claude 2",
    "Claude 3.5 Sonnet",
    "Claude Sonnet 4",
    "Claude Sonnet 4.6",
    "Claude Opus 4.1",
    "Claude Opus 4.7",
    "Claude Mythos Preview",
    "Grok 1",
    "Grok 2",
    "Grok 4",
    "Grok 4.1 Fast",
    "Grok 4.20",
    "Grok 4.3",
]

FAMILY_COLORS = {
    "gpt": "#1f4e79",
    "claude": "#8c6d46",
    "grok": "#9c2f2f",
}


def family_of(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("gpt"):
        return "gpt"
    if lowered.startswith("claude"):
        return "claude"
    return "grok"


def load_rows(metric_key: str) -> list[dict[str, object]]:
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["display_name"].strip()
            if name not in KEEP_NAMES:
                continue
            if metric_key == "training_carbon_tco2e_central":
                energy_wh = to_float(row["training_energy_wh_central"], default=0.0)
                country_mix = get_country_mix(row.get("estimation_country_code"))
                retained_grid_carbon_intensity = to_float(
                    (country_mix or {}).get("grid_carbon_intensity_gco2_per_kwh"),
                    default=None,
                )
                value = (
                    wh_to_gco2e(energy_wh, retained_grid_carbon_intensity) / 1_000_000.0
                    if retained_grid_carbon_intensity is not None
                    else float(row[metric_key])
                )
            else:
                value = float(row[metric_key])
            rows.append(
                {
                    "name": name,
                    "family": family_of(name),
                    "date": datetime.strptime(row["release_date"], "%Y-%m-%d"),
                    "value": value,
                }
            )
    return sorted(rows, key=lambda item: item["date"])


def compute_doubling_months(rows: list[dict[str, object]]) -> float:
    base_date = rows[0]["date"]
    xs = [((row["date"] - base_date).days / 30.4375) for row in rows]
    ys = [math.log(float(row["value"]), 2) for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den
    return 1.0 / slope


def draw_figure(rows: list[dict[str, object]], title: str, ylabel: str, output_name: str) -> None:
    doubling_months = compute_doubling_months(rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)

    for family, family_rows in grouped.items():
        color = FAMILY_COLORS[family]
        dates = [row["date"] for row in family_rows]
        values = [row["value"] for row in family_rows]
        ax.plot(dates, values, color=color, linewidth=1.8, alpha=0.9, label=family.upper() if family == "gpt" else family.capitalize())
        ax.scatter(dates, values, color=color, s=28, zorder=3)

    label_offsets = {
        "GPT-3.5 Turbo": (4, 8),
        "GPT-4": (4, -12),
        "GPT-5.2": (4, 8),
        "GPT-5.4": (4, -12),
        "GPT-5.5": (4, 8),
        "Claude 2": (4, 8),
        "Claude 3.5 Sonnet": (4, -12),
        "Claude Sonnet 4": (4, -12),
        "Claude Sonnet 4.6": (4, 8),
        "Claude Opus 4.1": (4, 8),
        "Claude Opus 4.7": (4, -12),
        "Claude Mythos Preview": (4, 8),
        "Grok 1": (4, 8),
        "Grok 2": (4, -12),
        "Grok 4": (4, 8),
        "Grok 4.1 Fast": (4, -12),
        "Grok 4.20": (4, 8),
        "Grok 4.3": (4, -12),
    }
    for row in rows:
        dx, dy = label_offsets.get(row["name"], (4, 8))
        ax.annotate(
            row["name"],
            (row["date"], row["value"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.8,
            color="#1f1f1f",
        )

    ax.set_title(title, fontsize=11.5, loc="left")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xlabel("Model release date", fontsize=9.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.4, alpha=0.2)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.995,
        0.04,
        f"Estimated doubling time: {doubling_months:.1f} months",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4d4d4d",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw_figure(
        load_rows("training_carbon_tco2e_central"),
        title="Training-screening trajectory of flagship GPT, Claude, and Grok models",
        ylabel="Central training CO2 estimate (tCO2e, log scale)",
        output_name="paper_training_co2_doubling.png",
    )
    draw_figure(
        load_rows("screening_per_request_carbon_gco2e_central"),
        title="Inference-screening trajectory of flagship GPT, Claude, and Grok models",
        ylabel="Central inference CO2 estimate (gCO2e/request, log scale)",
        output_name="paper_inference_co2_doubling.png",
    )


if __name__ == "__main__":
    main()
