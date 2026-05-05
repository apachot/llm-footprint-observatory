from __future__ import annotations

import csv
from datetime import date
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ImpactLLM" / "data" / "market_models.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "11_request_growth_frontier_only_linear_fr.png"

BG = "#F6F1E8"
PANEL = "#FBF8F2"
INK = "#1F2430"
MUTED = "#5F6773"
GRID = "#D9D1C4"
ACCENT = "#8B5E3C"
FILL = "#B39A74"
EDGE = "#FFFFFF"

YEARS = ["2022", "2023", "2024", "2025", "2026"]
CORE_PROVIDERS = {"openai", "anthropic", "google", "xai", "deepseek", "mistral", "meta", "alibaba"}


def to_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def bubble_size(active_b: float) -> float:
    return 180 + 34 * math.sqrt(max(active_b, 1.0))


def parse_release_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def load_frontier_rows() -> list[dict[str, object]]:
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, object]] = []
    for year in YEARS:
        snapshot_date = date(int(year), 12, 31)
        candidates = []
        for row in rows:
            if (row.get("provider") or "").strip().lower() not in CORE_PROVIDERS:
                continue
            if (row.get("market_status") or "").strip().lower() == "research":
                continue
            release_date = parse_release_date(row.get("release_date"))
            if release_date is None or release_date > snapshot_date:
                continue
            active_b = to_float(row.get("active_parameters_billion"))
            request_wh = to_float(row.get("screening_per_request_energy_wh_central"))
            if active_b <= 0 or request_wh <= 0:
                continue
            candidates.append(
                {
                    "year": year,
                    "model_id": row["model_id"],
                    "name": row["display_name"],
                    "active_b": active_b,
                    "request_wh": request_wh,
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda item: (float(item["active_b"]), float(item["request_wh"])), reverse=True)
        selected.append(candidates[0])
    return selected


def add_label(
    ax: plt.Axes,
    x_value: int,
    y_value: float,
    label: str,
    ha: str,
    va: str,
    dx: float,
    dy: float,
) -> None:
    ax.annotate(
        label,
        (x_value, y_value),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=11.5,
        color=INK,
        fontweight="bold",
        linespacing=1.25,
    )


def build_chart() -> None:
    rows = load_frontier_rows()
    if len(rows) != 5:
        raise SystemExit("Expected one frontier row per year from 2022 to 2026.")

    x_values = [int(row["year"]) for row in rows]
    y_values = [float(row["request_wh"]) for row in rows]
    sizes = [bubble_size(float(row["active_b"])) for row in rows]
    y_max = max(4.4, max(y_values) + 0.7)

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(14, 8.6), dpi=220, facecolor=BG)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.58], facecolor=PANEL)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(2021.7, 2026.3)
    ax.set_ylim(0, y_max)
    ax.set_xticks([2022, 2023, 2024, 2025, 2026])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(axis="y", color=GRID, alpha=0.85, linewidth=0.9)
    ax.grid(axis="x", color=GRID, alpha=0.35, linewidth=0.9)
    ax.tick_params(axis="x", labelsize=13, colors=INK, length=0)
    ax.tick_params(axis="y", labelsize=12, colors=MUTED)
    ax.set_axisbelow(True)

    ax.plot(x_values, y_values, color=ACCENT, linewidth=3.2, zorder=2)
    ax.fill_between(x_values, y_values, 0, color="#E7D9C5", alpha=0.45, zorder=1)
    ax.scatter(
        x_values,
        y_values,
        s=sizes,
        color=FILL,
        edgecolor=EDGE,
        linewidth=2.0,
        zorder=3,
    )

    label_specs = {
        2022: ("left", "bottom", 12, 10),
        2023: ("right", "bottom", -12, 10),
        2024: ("left", "top", 14, -12),
        2025: ("center", "bottom", 0, 12),
        2026: ("center", "bottom", 0, 10),
    }
    previous_model_id = None
    for row in rows:
        year = int(row["year"])
        if row["model_id"] == previous_model_id:
            label = (
                f"{row['name']}\n"
                f"{float(row['active_b']):.0f}B actifs\n"
                f"reste en tete\n"
                f"{float(row['request_wh']):.2f} Wh / requete"
            )
        else:
            label = (
                f"{row['name']}\n"
                f"{float(row['active_b']):.0f}B actifs\n"
                f"{float(row['request_wh']):.2f} Wh / requete"
            )
        ha, va, dx, dy = label_specs[year]
        add_label(ax, year, float(row["request_wh"]), label, ha=ha, va=va, dx=dx, dy=dy)
        previous_model_id = row["model_id"]

    ax.set_ylabel("Wh par requete standardisee", fontsize=13, color=MUTED, labelpad=12)

    fig.text(0.08, 0.925, "ImpactLLM", fontsize=24, fontweight="bold", color=INK)
    fig.text(
        0.08,
        0.885,
        "2022-2026 | un seul point par an : le plus gros modele de marche disponible",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.845,
        "Lecture immediate : plus le point monte, plus la requete standardisee consomme. "
        "Plus la bulle grossit, plus le modele prend de l'embonpoint. "
        "Taille des bulles = parametres actifs retenus.",
        fontsize=11.5,
        color=MUTED,
    )

    growth_ratio = y_values[-1] / y_values[0] if y_values[0] else 0.0
    fig.text(
        0.08,
        0.812,
        f"Le plus gros modele disponible passe d'environ {y_values[0]:.2f} Wh/requete en 2022 a {y_values[-1]:.2f} Wh/requete en 2026, soit x{growth_ratio:.1f}.",
        fontsize=12,
        color=ACCENT,
        fontweight="bold",
    )

    fig.text(
        0.08,
        0.055,
        "Champ = grands fournisseurs de marche, hors lignes research. Selection annuelle = maximum des parametres actifs retenus parmi les modeles deja sortis au 31 decembre ; faute de date de retrait normalisee, un modele est suppose encore disponible ensuite.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.030,
        "Source : ImpactLLM market_models.csv | Requete standardisee = 1000 tokens in + 550 out | Valeurs centrales de screening",
        fontsize=10,
        color=MUTED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    build_chart()


if __name__ == "__main__":
    main()
