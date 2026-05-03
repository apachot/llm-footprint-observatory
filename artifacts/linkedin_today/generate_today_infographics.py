from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ImpactLLM" / "data" / "market_models.csv"
OUT_DIR = Path(__file__).resolve().parent

BG = "#f6f1e8"
PANEL = "#fbf8f2"
INK = "#1f2430"
MUTED = "#5f6773"
GRID = "#d9d1c4"

FAMILY_COLORS = {
    "gpt": "#243b63",
    "claude": "#8c7a5b",
    "grok": "#bc4b51",
}

PROVIDER_COLORS = {
    "openai": "#243b63",
    "anthropic": "#8c7a5b",
    "google": "#6a8f5b",
    "xai": "#bc4b51",
    "meta": "#4b7f9f",
    "mistral": "#d17a22",
    "deepseek": "#7b5ea7",
    "microsoft": "#3f6b8c",
    "alibaba": "#5a9367",
    "deepmind": "#7b6d5d",
    "ai21": "#a85d5d",
}

PUBLICATION_DATE_LABEL = "3 mai 2026"

KEEP_DOUBLING = {
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
}

USAGE_POINTS = [
    {"date": datetime(2024, 4, 1), "weekly_users_m": 100, "label": "100M", "source": "OpenAI, Start using ChatGPT instantly"},
    {"date": datetime(2024, 10, 2), "weekly_users_m": 250, "label": "250M", "source": "OpenAI, New funding to scale the benefits of AI"},
    {"date": datetime(2025, 2, 4), "weekly_users_m": 300, "label": "300M", "source": "OpenAI, CSU system"},
    {"date": datetime(2025, 3, 31), "weekly_users_m": 500, "label": "500M", "source": "OpenAI, March funding updates"},
    {"date": datetime(2025, 9, 29), "weekly_users_m": 700, "label": "700M", "source": "OpenAI, Buy it in ChatGPT"},
    {"date": datetime(2025, 12, 8), "weekly_users_m": 800, "label": "800M", "source": "OpenAI, The state of enterprise AI"},
]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_PATH.open() as handle:
        for row in csv.DictReader(handle):
            if not row.get("display_name"):
                continue
            rows.append(
                {
                    "model_id": row["model_id"],
                    "name": row["display_name"],
                    "provider": row["provider"],
                    "date": datetime.strptime(row["release_date"], "%Y-%m-%d") if row["release_date"] else None,
                    "active_b": float(row["active_parameters_billion"] or 0),
                    "inference_g_h": float(row["screening_per_hour_carbon_gco2e_central"] or 0),
                    "training_t": float(row["training_carbon_tco2e_central"] or 0),
                }
            )
    return rows


def family_of(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("gpt"):
        return "gpt"
    if lowered.startswith("claude"):
        return "claude"
    return "grok"


def style_axes(ax, y_grid: bool = True, x_grid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=10)
    if x_grid:
        ax.grid(axis="x", color=GRID, alpha=0.7, linewidth=0.8)
    if y_grid:
        ax.grid(axis="y", color=GRID, alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.125, 0.025, text, fontsize=9, color=MUTED)


def compute_doubling_months(rows: list[dict[str, object]], key: str) -> float:
    base_date = rows[0]["date"]
    xs = [((row["date"] - base_date).days / 30.4375) for row in rows]
    ys = [math.log(float(row[key]), 2) for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return 1.0 / (numerator / denominator)


def compute_log2_trend(rows: list[dict[str, object]], key: str) -> tuple[datetime, float, float]:
    base_date = rows[0]["date"]
    xs = [((row["date"] - base_date).days / 30.4375) for row in rows]
    ys = [math.log(float(row[key]), 2) for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    intercept = mean_y - slope * mean_x
    return base_date, intercept, slope


def chart_doubling(rows: list[dict[str, object]], key: str, title: str, subtitle: str, ylabel: str, filename: str) -> None:
    selected = [row for row in rows if row["name"] in KEEP_DOUBLING and row["date"]]
    selected.sort(key=lambda row: row["date"])
    doubling = compute_doubling_months(selected, key)
    base_date, intercept, slope = compute_log2_trend(selected, key)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        groups[family_of(str(row["name"]))].append(row)

    for family, family_rows in groups.items():
        color = FAMILY_COLORS[family]
        dates = [row["date"] for row in family_rows]
        values = [row[key] for row in family_rows]
        ax.plot(dates, values, color=color, linewidth=2.4, alpha=0.9, zorder=2)
        ax.scatter(dates, values, s=95, color=color, edgecolor="#ffffff", linewidth=1.2, zorder=3)

    for row in selected:
        ax.annotate(
            row["name"],
            (row["date"], row[key]),
            xytext=(6, 8 if row["name"] not in {"GPT-4", "Claude 3.5 Sonnet", "Claude Sonnet 4", "Grok 2"} else -14),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color=INK,
        )

    trend_dates = [row["date"] for row in selected]
    trend_values = []
    for current_date in trend_dates:
        months = (current_date - base_date).days / 30.4375
        trend_values.append(2 ** (intercept + slope * months))
    ax.plot(
        trend_dates,
        trend_values,
        color="#2f4858",
        linewidth=2.2,
        linestyle=(0, (5, 4)),
        alpha=0.9,
        zorder=1,
    )

    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", which="major", color="#ece4d8", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title(
        title.format(months=round(doubling)),
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=28,
    )
    fig.text(0.125, 0.89, subtitle, fontsize=12.5, color="#4f5d73")
    ax.set_ylabel(ylabel, fontsize=12, color="#243b63")
    ax.set_xlabel("Date de sortie", fontsize=12, color="#243b63")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS["gpt"], marker="o", lw=2.4, label="GPT"),
        plt.Line2D([0], [0], color=FAMILY_COLORS["claude"], marker="o", lw=2.4, label="Claude"),
        plt.Line2D([0], [0], color=FAMILY_COLORS["grok"], marker="o", lw=2.4, label="Grok"),
        plt.Line2D([0], [0], color="#2f4858", lw=2.2, linestyle=(0, (5, 4)), label="Tendance exponentielle"),
    ]
    legend = ax.legend(handles=legend_handles, ncol=4, fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    for text in legend.get_texts():
        text.set_color("#495057")
    ax.text(
        0.985,
        0.03,
        "En échelle logarithmique,\nune trajectoire quasi droite signale\nune croissance exponentielle.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2ebdf", edgecolor="#e1d8ca"),
    )
    add_footer(
        fig,
        f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Estimations centrales de screening, pas des mesures directes fournisseur. Echelle logarithmique.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.86))
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def chart_doubling_inference_emphasized(rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["name"] in KEEP_DOUBLING and row["date"]]
    selected.sort(key=lambda row: row["date"])
    doubling = compute_doubling_months(selected, "inference_g_h")
    base_date, intercept, slope = compute_log2_trend(selected, "inference_g_h")

    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        groups[family_of(str(row["name"]))].append(row)

    for family, family_rows in groups.items():
        color = FAMILY_COLORS[family]
        dates = [row["date"] for row in family_rows]
        values = [row["inference_g_h"] for row in family_rows]
        ax.plot(dates, values, color=color, linewidth=2.2, alpha=0.72, zorder=2)
        ax.scatter(dates, values, s=88, color=color, edgecolor="#ffffff", linewidth=1.2, zorder=3)

    months_span = [0]
    current = 0.0
    last_month = (selected[-1]["date"] - base_date).days / 30.4375
    while current + doubling <= last_month + 1:
        current += doubling
        months_span.append(current)
    trend_dates = []
    trend_values = []
    doubling_dates = []
    doubling_values = []
    for months in np.linspace(0, last_month, 120):
        current_date = base_date + (selected[-1]["date"] - base_date) * (months / last_month if last_month else 0)
        trend_dates.append(current_date)
        trend_values.append(2 ** (intercept + slope * months))
    for months in months_span:
        current_date = base_date + (selected[-1]["date"] - base_date) * (months / last_month if last_month else 0)
        value = 2 ** (intercept + slope * months)
        doubling_dates.append(current_date)
        doubling_values.append(value)
    ax.plot(trend_dates, trend_values, color="#111827", linewidth=2.8, linestyle=(0, (6, 4)), zorder=1, label="Tendance exponentielle")
    ax.scatter(doubling_dates, doubling_values, s=46, color="#111827", zorder=4)
    for idx, (d, v) in enumerate(zip(doubling_dates[1:], doubling_values[1:]), start=1):
        ax.annotate(
            f"x{2**idx:.0f}",
            (d, v),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#111827",
            fontweight="bold",
        )

    highlight_names = {"GPT-3.5 Turbo", "GPT-5.5", "Claude Opus 4.7", "Claude Mythos Preview", "Grok 4.3", "GPT-4"}
    for row in selected:
        if row["name"] not in highlight_names:
            continue
        ax.annotate(
            row["name"],
            (row["date"], row["inference_g_h"]),
            xytext=(6, 8 if row["name"] not in {"GPT-4", "Claude Sonnet 4"} else -14),
            textcoords="offset points",
            fontsize=10,
            fontweight="semibold",
            color=INK,
        )

    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.9, alpha=0.9)
    ax.grid(axis="x", which="major", color="#ece4d8", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Le CO2 d'inférence suit une trajectoire de doublement annuel",
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=28,
    )
    fig.text(
        0.125,
        0.89,
        f"Sur la sélection GPT / Claude / Grok, l'estimation centrale suggère un doublement tous les ~{round(doubling):d} mois.",
        fontsize=12.5,
        color="#4f5d73",
    )
    ax.set_ylabel("CO2 d'inférence central (gCO2e par heure, échelle logarithmique)", fontsize=12, color="#243b63")
    ax.set_xlabel("Date de sortie", fontsize=12, color="#243b63")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS["gpt"], marker="o", lw=2.2, label="GPT"),
        plt.Line2D([0], [0], color=FAMILY_COLORS["claude"], marker="o", lw=2.2, label="Claude"),
        plt.Line2D([0], [0], color=FAMILY_COLORS["grok"], marker="o", lw=2.2, label="Grok"),
        plt.Line2D([0], [0], color="#111827", lw=2.8, linestyle=(0, (6, 4)), label="Rythme de doublement"),
    ]
    legend = ax.legend(handles=legend_handles, ncol=4, fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    for text in legend.get_texts():
        text.set_color("#495057")
    ax.text(
        0.985,
        0.03,
        "Lecture : en échelle log,\nles repères x2, x4, x8 rendent\nvisible la dynamique exponentielle.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2ebdf", edgecolor="#e1d8ca"),
    )
    add_footer(
        fig,
        f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Estimations centrales de screening, pas des mesures directes fournisseur. Echelle logarithmique.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.86))
    fig.savefig(OUT_DIR / "01b_doubling_inference_exponential_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_inference_index_non_log(rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["name"] in KEEP_DOUBLING and row["date"]]
    selected.sort(key=lambda row: row["date"])
    base_value = float(selected[0]["inference_g_h"])
    doubling = compute_doubling_months(selected, "inference_g_h")

    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    index_values = [100 * float(row["inference_g_h"]) / base_value for row in selected]
    dates = [row["date"] for row in selected]

    ax.plot(dates, index_values, color="#243b63", linewidth=3.2, zorder=2)
    ax.fill_between(dates, index_values, [100] * len(index_values), color="#9db4d3", alpha=0.25, zorder=1)
    ax.scatter(dates, index_values, s=96, color="#243b63", edgecolor="#ffffff", linewidth=1.2, zorder=3)

    label_names = {"GPT-3.5 Turbo", "GPT-5.5", "Claude Opus 4.7", "Claude Mythos Preview", "Grok 4.3", "GPT-4"}
    growth_factor = index_values[-1] / 100 if index_values else 1.0
    for row, idx_value in zip(selected, index_values):
        if row["name"] not in label_names:
            continue
        ax.annotate(
            row["name"],
            (row["date"], idx_value),
            xytext=(6, 8 if row["name"] not in {"GPT-4", "Claude Sonnet 4"} else -14),
            textcoords="offset points",
            fontsize=10,
            fontweight="semibold",
            color=INK,
        )

    for level, label in [(100, "base 100"), (200, "x2"), (400, "x4"), (800, "x8")]:
        ax.axhline(level, color="#cfc5b6", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
        ax.text(dates[0], level * 1.015, label, fontsize=9.5, color=MUTED, va="bottom")

    ax.set_title(
        f"Le CO2 d'inference a ete multiplie par environ x{growth_factor:.1f} depuis GPT-3.5",
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=28,
    )
    fig.text(
        0.125,
        0.89,
        f"Indice base 100 = GPT-3.5 Turbo. Lecture non logarithmique. Le rythme moyen implicite reste d'environ un doublement tous les ~{round(doubling):d} mois.",
        fontsize=12.5,
        color="#4f5d73",
    )
    ax.set_ylabel("Indice de CO2 d'inférence (base 100 = GPT-3.5 Turbo)", fontsize=12, color="#243b63")
    ax.set_xlabel("Date de sortie", fontsize=12, color="#243b63")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axes(ax, y_grid=False, x_grid=True)
    ax.set_ylim(0, max(index_values) * 1.18)
    ax.text(
        0.985,
        0.03,
        "Cette version n'utilise pas d'échelle logarithmique.\nElle montre la hausse relative par rapport\nau point de départ.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2ebdf", edgecolor="#e1d8ca"),
    )
    add_footer(
        fig,
        f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Indice construit a partir des estimations centrales de screening. Base 100 = GPT-3.5 Turbo.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.86))
    fig.savefig(OUT_DIR / "01c_doubling_inference_index_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_gpt_inference_index_non_log(rows: list[dict[str, object]]) -> None:
    keep = {"GPT-3.5 Turbo", "GPT-4", "GPT-5.2", "GPT-5.4", "GPT-5.5"}
    selected = [row for row in rows if row["name"] in keep and row["date"]]
    selected.sort(key=lambda row: row["date"])
    base_value = float(selected[0]["inference_g_h"])
    doubling = compute_doubling_months(selected, "inference_g_h")

    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    dates = [row["date"] for row in selected]
    index_values = [100 * float(row["inference_g_h"]) / base_value for row in selected]
    growth_factor = index_values[-1] / 100 if index_values else 1.0

    ax.plot(dates, index_values, color=FAMILY_COLORS["gpt"], linewidth=3.4, zorder=2)
    ax.fill_between(dates, index_values, [100] * len(index_values), color="#9db4d3", alpha=0.28, zorder=1)
    ax.scatter(dates, index_values, s=110, color=FAMILY_COLORS["gpt"], edgecolor="#ffffff", linewidth=1.2, zorder=3)

    for row, idx_value in zip(selected, index_values):
        ax.annotate(
            row["name"],
            (row["date"], idx_value),
            xytext=(6, 8 if row["name"] != "GPT-4" else -14),
            textcoords="offset points",
            fontsize=11,
            fontweight="semibold",
            color=INK,
        )

    for level, label in [(100, "base 100"), (200, "x2"), (400, "x4"), (800, "x8")]:
        ax.axhline(level, color="#cfc5b6", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
        ax.text(dates[0], level * 1.015, label, fontsize=10, color=MUTED, va="bottom")

    ax.set_title(
        f"Chez GPT, le CO2 d'inference a ete multiplie par environ x{growth_factor:.1f}",
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=28,
    )
    fig.text(
        0.125,
        0.89,
        f"Indice base 100 = GPT-3.5 Turbo. Lecture non logarithmique. La trajectoire implicite correspond à un doublement environ tous les ~{round(doubling):d} mois.",
        fontsize=12.5,
        color="#4f5d73",
    )
    ax.set_ylabel("Indice de CO2 d'inférence (base 100 = GPT-3.5 Turbo)", fontsize=12, color="#243b63")
    ax.set_xlabel("Date de sortie", fontsize=12, color="#243b63")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axes(ax, y_grid=False, x_grid=True)
    ax.set_ylim(0, max(index_values) * 1.22)
    ax.text(
        0.985,
        0.03,
        "Version centrée sur la famille GPT.\nElle rend la trajectoire plus lisible\npour un post LinkedIn.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2ebdf", edgecolor="#e1d8ca"),
    )
    add_footer(
        fig,
        f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Indice construit a partir des estimations centrales de screening. Base 100 = GPT-3.5 Turbo.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.86))
    fig.savefig(OUT_DIR / "01d_gpt_inference_index_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_usage_vs_gpt_intensity(rows: list[dict[str, object]]) -> None:
    keep = {"GPT-3.5 Turbo", "GPT-4", "GPT-5.2", "GPT-5.4", "GPT-5.5"}
    gpt_rows = [row for row in rows if row["name"] in keep and row["date"]]
    gpt_rows.sort(key=lambda row: row["date"])
    base_gpt = float(gpt_rows[0]["inference_g_h"])
    gpt_dates = [row["date"] for row in gpt_rows]
    gpt_index = [100 * float(row["inference_g_h"]) / base_gpt for row in gpt_rows]
    gpt_multiplier = gpt_index[-1] / 100 if gpt_index else 1.0

    usage_dates = [row["date"] for row in USAGE_POINTS]
    usage_index = [100 * row["weekly_users_m"] / USAGE_POINTS[0]["weekly_users_m"] for row in USAGE_POINTS]
    usage_multiplier = usage_index[-1] / 100 if usage_index else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=220, sharex=True, gridspec_kw={"hspace": 0.15})
    fig.patch.set_facecolor(BG)

    ax1, ax2 = axes
    style_axes(ax1, y_grid=True, x_grid=True)
    style_axes(ax2, y_grid=True, x_grid=True)

    ax1.plot(usage_dates, usage_index, color="#2f6b8c", linewidth=3.2)
    ax1.scatter(usage_dates, usage_index, s=92, color="#2f6b8c", edgecolor="#ffffff", linewidth=1.2, zorder=3)
    for row, idx_value in zip(USAGE_POINTS, usage_index):
        ax1.annotate(row["label"], (row["date"], idx_value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax1.set_ylabel("Indice usage\n(base 100 = avr. 2024)", color=INK, fontsize=11)
    ax1.set_title(
        "L'usage explose pendant que l'intensité d'inférence reste élevée",
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=20,
    )
    fig.text(
        0.125,
        0.90,
        "Proxy d'usage : utilisateurs hebdomadaires de ChatGPT (sources OpenAI). Intensité : CO2 d'inférence de la famille GPT dans l'observatoire.",
        fontsize=12.5,
        color="#4f5d73",
    )
    ax1.text(0.985, 0.90, f"x{usage_multiplier:.1f}", transform=ax1.transAxes, ha="right", va="top", fontsize=28, fontweight="bold", color="#2f6b8c")

    ax2.plot(gpt_dates, gpt_index, color=FAMILY_COLORS["gpt"], linewidth=3.2)
    ax2.scatter(gpt_dates, gpt_index, s=110, color=FAMILY_COLORS["gpt"], edgecolor="#ffffff", linewidth=1.2, zorder=3)
    for row, idx_value in zip(gpt_rows, gpt_index):
        ax2.annotate(row["name"], (row["date"], idx_value), xytext=(6, 8 if row["name"] != "GPT-4" else -14), textcoords="offset points", fontsize=10.5, fontweight="semibold", color=INK)
    ax2.set_ylabel("Indice CO2\n(base 100 = GPT-3.5)", color=INK, fontsize=11)
    ax2.set_xlabel("Date", color=INK, fontsize=11)
    ax2.text(0.985, 0.90, f"x{gpt_multiplier:.1f}", transform=ax2.transAxes, ha="right", va="top", fontsize=28, fontweight="bold", color=FAMILY_COLORS["gpt"])

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    add_footer(
        fig,
        "Usage = ChatGPT weekly users from official OpenAI posts (2024-04-01, 2024-10-02, 2025-02-04, 2025-03-31, 2025-09-29, 2025-12-08). Intensité = estimations centrales ImpactLLM.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.88))
    fig.savefig(OUT_DIR / "05_usage_vs_gpt_inference_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_usage_vs_frontier_average(rows: list[dict[str, object]]) -> None:
    frontier = [row for row in rows if row["name"] in KEEP_DOUBLING and row["date"]]
    frontier.sort(key=lambda row: row["date"])

    cumulative_dates = [row["date"] for row in USAGE_POINTS]
    cumulative_means = []
    for point in USAGE_POINTS:
        values = [float(row["inference_g_h"]) for row in frontier if row["date"] <= point["date"]]
        cumulative_means.append(sum(values) / len(values))

    base_mean = cumulative_means[0]
    mean_index = [100 * value / base_mean for value in cumulative_means]
    usage_index = [100 * row["weekly_users_m"] / USAGE_POINTS[0]["weekly_users_m"] for row in USAGE_POINTS]
    usage_dates = [row["date"] for row in USAGE_POINTS]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    style_axes(ax, y_grid=True, x_grid=True)

    ax.plot(usage_dates, usage_index, color="#2f6b8c", linewidth=3.1, marker="o", markersize=7.5, label="Usage ChatGPT (indice)")
    ax.plot(usage_dates, mean_index, color="#8c7a5b", linewidth=3.1, marker="o", markersize=7.5, label="CO2 d'inférence moyen frontier (indice)")

    ax.set_title(
        "L'usage augmente plus vite que l'intensité moyenne, mais les deux montent",
        loc="left",
        fontsize=24,
        fontweight="bold",
        color="#243b63",
        pad=20,
    )
    fig.text(
        0.125,
        0.90,
        "Même en lissant sur une moyenne des modèles frontier déjà sortis, le signal d'inférence reste orienté à la hausse.",
        fontsize=12.5,
        color="#4f5d73",
    )
    ax.set_ylabel("Indice base 100 (avril 2024)", color=INK, fontsize=11)
    ax.set_xlabel("Date", color=INK, fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend = ax.legend(frameon=False, loc="upper left", fontsize=11)
    for text in legend.get_texts():
        text.set_color("#495057")
    ax.text(usage_dates[-1], usage_index[-1] + 12, "x8", color="#2f6b8c", fontsize=18, fontweight="bold", ha="right")
    ax.text(usage_dates[-1], mean_index[-1] + 8, f"x{mean_index[-1]/100:.1f}", color="#8c7a5b", fontsize=18, fontweight="bold", ha="right")
    add_footer(
        fig,
        "Usage = ChatGPT weekly users from official OpenAI posts. Intensité = moyenne cumulative du CO2 d'inférence central sur GPT, Claude et Grok dans ImpactLLM.",
    )
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.88))
    fig.savefig(OUT_DIR / "06_usage_vs_frontier_average_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_usage_vs_gpt_intensity_academic(rows: list[dict[str, object]]) -> None:
    keep = {"GPT-3.5 Turbo", "GPT-4", "GPT-5.2", "GPT-5.4", "GPT-5.5"}
    gpt_rows = [row for row in rows if row["name"] in keep and row["date"]]
    gpt_rows.sort(key=lambda row: row["date"])
    base_gpt = float(gpt_rows[0]["inference_g_h"])
    gpt_dates = [row["date"] for row in gpt_rows]
    gpt_index = [100 * float(row["inference_g_h"]) / base_gpt for row in gpt_rows]
    gpt_multiplier = gpt_index[-1] / 100 if gpt_index else 1.0

    usage_dates = [row["date"] for row in USAGE_POINTS]
    usage_index = [100 * row["weekly_users_m"] / USAGE_POINTS[0]["weekly_users_m"] for row in USAGE_POINTS]
    usage_multiplier = usage_index[-1] / 100 if usage_index else 1.0

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.4, 8.6),
        dpi=240,
        sharex=True,
        gridspec_kw={"hspace": 0.10, "height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("#ffffff")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(colors="#333333", labelsize=10)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.7)
        ax.grid(axis="x", color="#efefef", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    usage_color = "#334e68"
    gpt_color = "#7a1f1f"

    ax1.plot(usage_dates, usage_index, color=usage_color, linewidth=2.2)
    ax1.scatter(usage_dates, usage_index, s=34, color=usage_color, zorder=3)
    for row, idx_value in zip(USAGE_POINTS, usage_index):
        ax1.annotate(row["label"], (row["date"], idx_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=usage_color)
    ax1.set_ylabel("Indice d'usage hebdomadaire\n(base 100)", fontsize=10.5, color="#222222")
    ax1.text(0.99, 0.86, f"x{usage_multiplier:.1f}", transform=ax1.transAxes, ha="right", va="top", fontsize=19, color=usage_color, fontweight="semibold")

    ax2.plot(gpt_dates, gpt_index, color=gpt_color, linewidth=2.2)
    ax2.scatter(gpt_dates, gpt_index, s=40, color=gpt_color, zorder=3)
    for row, idx_value in zip(gpt_rows, gpt_index):
        ax2.annotate(row["name"], (row["date"], idx_value), xytext=(6, 8 if row["name"] != "GPT-4" else -14), textcoords="offset points", fontsize=9.5, color=gpt_color)
    ax2.set_ylabel("Indice de CO2 d'inférence\n(base 100)", fontsize=10.5, color="#222222")
    ax2.set_xlabel("Chronologie de sortie", fontsize=10.5, color="#222222")
    ax2.text(0.99, 0.86, f"x{gpt_multiplier:.1f}", transform=ax2.transAxes, ha="right", va="top", fontsize=19, color=gpt_color, fontweight="semibold")

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "L'usage des LLM et l'intensité d'inférence augmentent ensemble",
        x=0.125,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color="#111111",
    )
    fig.text(
        0.125,
        0.03,
        f"Sources : publications officielles d'OpenAI pour les jalons d'usage (2024-2025) ; Observatoire ImpactLLM, {PUBLICATION_DATE_LABEL}, pour les estimations de CO2 d'inference. Les indices servent ici a la comparabilite.",
        fontsize=8.7,
        color="#666666",
    )

    fig.tight_layout(rect=(0.07, 0.07, 0.98, 0.90))
    fig.savefig(OUT_DIR / "05b_usage_vs_gpt_inference_academic_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_inference_concentration(rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["inference_g_h"]), reverse=True)[:12]
    values = [float(row["inference_g_h"]) for row in ordered]
    names = [str(row["name"]) for row in ordered]
    colors = [PROVIDER_COLORS.get(str(row["provider"]), "#7a7a7a") for row in ordered]
    top_three_share = sum(values[:3]) / sum(values)
    top_three_names = ", ".join(names[:3])
    total_models = len(rows)

    fig, ax = plt.subplots(figsize=(13, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    style_axes(ax, y_grid=False, x_grid=True)
    ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.72)
    ax.set_xlabel("CO2 d'inférence central (gCO2e par heure standardisée)", color=INK, fontsize=11)
    ax.set_title(
        f"3 modeles concentrent deja {top_three_share * 100:.0f}% du CO2 d'inference du top 12",
        loc="left",
        fontsize=23,
        color=INK,
        fontweight="bold",
        pad=20,
    )
    fig.text(
        0.125,
        0.90,
        f"Dans le top 12 actuel de l'observatoire, {top_three_names} captent l'essentiel du signal.",
        fontsize=12,
        color=MUTED,
    )
    for idx, value in enumerate(values[::-1]):
        ax.text(value + max(values) * 0.012, idx, f"{value:.1f}", va="center", fontsize=10, color=INK)
    fig.text(0.125, 0.12, f"Soit {top_three_share * 100:.1f}% du top 12 et {3 / total_models * 100:.1f}% des {total_models} modeles du referentiel.", fontsize=11, color=INK)
    add_footer(fig, f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Heure d'usage standardisee = scenario commun de comparaison.")
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.88))
    fig.savefig(OUT_DIR / "03_concentration_inference_top12_fr.png", bbox_inches="tight")
    plt.close(fig)


def chart_provider_dispersion(rows: list[dict[str, object]]) -> None:
    grouped: list[dict[str, object]] = []
    providers = defaultdict(list)
    for row in rows:
        providers[str(row["provider"])].append(float(row["inference_g_h"]))
    for provider, values in providers.items():
        if len(values) < 3:
            continue
        values = sorted(values)
        grouped.append(
            {
                "provider": provider,
                "n": len(values),
                "min": min(values),
                "max": max(values),
                "median": float(np.median(values)),
                "spread": max(values) / max(min(values), 1e-9),
            }
        )
    grouped.sort(key=lambda row: row["spread"], reverse=True)
    top_group = grouped[0]

    fig, ax = plt.subplots(figsize=(13, 8), dpi=220)
    fig.patch.set_facecolor(BG)
    style_axes(ax, y_grid=False, x_grid=True)
    labels = [f"{row['provider']} (n={row['n']})" for row in grouped][::-1]
    ypos = np.arange(len(grouped))
    for idx, row in enumerate(grouped[::-1]):
        color = PROVIDER_COLORS.get(str(row["provider"]), "#7a7a7a")
        ax.hlines(idx, row["min"], row["max"], color=color, linewidth=3, alpha=0.85)
        ax.scatter(row["median"], idx, s=85, color=color, edgecolor="#fff", linewidth=1.1, zorder=3)
        ax.text(row["max"] * 1.05, idx, f"x{row['spread']:.1f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("CO2 d'inférence central (gCO2e par heure standardisée, échelle log)", color=INK, fontsize=11)
    ax.set_title(
        f"Chez {str(top_group['provider']).capitalize()}, l'ecart entre modeles va jusqu'a x{top_group['spread']:.0f}",
        loc="left",
        fontsize=23,
        color=INK,
        fontweight="bold",
        pad=20,
    )
    fig.text(
        0.125,
        0.90,
        "Le nom du fournisseur ne suffit pas : la dispersion interne des catalogues est parfois énorme.",
        fontsize=12,
        color=MUTED,
    )
    add_footer(fig, f"ImpactLLM Observatory, {PUBLICATION_DATE_LABEL}. Trait = min-max, point = mediane. Echelle logarithmique.")
    fig.tight_layout(rect=(0.07, 0.08, 0.98, 0.88))
    fig.savefig(OUT_DIR / "04_dispersion_fournisseurs_fr.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    chart_doubling(
        rows,
        key="inference_g_h",
        title="Le CO2 d'inférence double tous les ~{months} mois",
        subtitle="Estimation centrale du CO2 d'inférence par date de sortie. GPT, Claude et Grok uniquement.",
        ylabel="CO2 d'inférence central (gCO2e par heure, échelle logarithmique)",
        filename="01_doubling_inference_fr.png",
    )
    chart_doubling_inference_emphasized(rows)
    chart_inference_index_non_log(rows)
    chart_gpt_inference_index_non_log(rows)
    chart_usage_vs_gpt_intensity(rows)
    chart_usage_vs_gpt_intensity_academic(rows)
    chart_usage_vs_frontier_average(rows)
    chart_doubling(
        rows,
        key="training_t",
        title="Le CO2 d'entraînement double tous les ~{months} mois",
        subtitle="Estimation centrale du CO2 d'entraînement par date de sortie. GPT, Claude et Grok uniquement.",
        ylabel="CO2 d'entraînement central (tCO2e, échelle logarithmique)",
        filename="02_doubling_training_fr.png",
    )
    chart_inference_concentration(rows)
    chart_provider_dispersion(rows)


if __name__ == "__main__":
    main()
