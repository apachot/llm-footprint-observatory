from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ImpactLLM" / "data" / "market_models.csv"
OUTPUT_PATH = ROOT / "artifacts" / "linkedin_impactllm_results_latest_models.png"

SELECTED_MODELS = [
    "Claude Mythos Preview",
    "Claude Opus 4.7",
    "GPT-5.5",
    "Grok 4.3",
    "Claude Sonnet 4.6",
    "Gemini 3.1 Pro",
    "DeepSeek V4 Pro",
    "Llama 4 Scout",
]


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {row["display_name"]: row for row in rows}
    return [by_name[name] for name in SELECTED_MODELS]


def format_value(value: float) -> str:
    return f"{value:.1f}"


def add_value_labels(ax: plt.Axes, bars, values: list[float], color: str) -> None:
    max_value = max(values)
    for bar, value in zip(bars, values):
        ax.text(
            value + max_value * 0.012,
            bar.get_y() + bar.get_height() / 2,
            format_value(value),
            va="center",
            ha="left",
            fontsize=12,
            color=color,
            fontweight="bold",
        )


def build_visual(rows: list[dict[str, str]], catalog_count: int) -> None:
    names = [row["display_name"] for row in rows]
    energy = [float(row["screening_per_hour_energy_wh_central"]) for row in rows]
    carbon = [float(row["screening_per_hour_carbon_gco2e_central"]) for row in rows]

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="#F7F2EA")
    gs = fig.add_gridspec(
        14,
        14,
        left=0.05,
        right=0.97,
        top=0.92,
        bottom=0.08,
        hspace=1.0,
        wspace=0.8,
    )

    fig.text(0.05, 0.94, "ImpactLLM", fontsize=26, fontweight="bold", color="#18212F")
    fig.text(
        0.05,
        0.905,
        "Quelques ordres de grandeur sur les LLMs recents du referentiel",
        fontsize=20,
        fontweight="bold",
        color="#18212F",
    )
    fig.text(
        0.05,
        0.874,
        "Scenario standardise : 1 heure d'usage actif, 34,6 interactions/h, "
        "1000 tokens en entree, 550 en sortie.",
        fontsize=12,
        color="#5E6472",
    )
    fig.text(
        0.05,
        0.848,
        "Valeurs centrales de screening issues du referentiel public ImpactLLM. "
        "Objectif : comparer des ordres de grandeur, pas revendiquer une mesure instrumentee.",
        fontsize=11.5,
        color="#5E6472",
    )

    ax_energy = fig.add_subplot(gs[3:8, 0:9], facecolor="#FFF9F0")
    ax_carbon = fig.add_subplot(gs[9:14, 0:9], facecolor="#FFF9F0")
    ax_note = fig.add_subplot(gs[3:14, 9:14])
    ax_note.axis("off")

    y_positions = list(range(len(names)))

    energy_bars = ax_energy.barh(y_positions, energy, color="#255C4A", edgecolor="#255C4A", height=0.58)
    ax_energy.set_yticks(y_positions, labels=names)
    ax_energy.invert_yaxis()
    ax_energy.set_title("Energie d'inference estimee", loc="left", fontsize=17, pad=10, color="#18212F")
    ax_energy.set_xlabel("Wh pour 1 heure d'usage standardise", fontsize=11, color="#5E6472", labelpad=8)
    ax_energy.grid(axis="x", color="#D9CFBF", alpha=0.7)
    ax_energy.tick_params(axis="y", labelsize=11, length=0, colors="#2E3440")
    ax_energy.tick_params(axis="x", labelsize=10, colors="#6B6F76")
    for spine in ax_energy.spines.values():
        spine.set_visible(False)
    add_value_labels(ax_energy, energy_bars, energy, "#364152")

    carbon_bars = ax_carbon.barh(y_positions, carbon, color="#A3523A", edgecolor="#A3523A", height=0.58)
    ax_carbon.set_yticks(y_positions, labels=names)
    ax_carbon.invert_yaxis()
    ax_carbon.set_title("Carbone d'inference estime", loc="left", fontsize=17, pad=10, color="#18212F")
    ax_carbon.set_xlabel("gCO2e pour 1 heure d'usage standardise", fontsize=11, color="#5E6472", labelpad=8)
    ax_carbon.grid(axis="x", color="#D9CFBF", alpha=0.7)
    ax_carbon.tick_params(axis="y", labelsize=11, length=0, colors="#2E3440")
    ax_carbon.tick_params(axis="x", labelsize=10, colors="#6B6F76")
    for spine in ax_carbon.spines.values():
        spine.set_visible(False)
    add_value_labels(ax_carbon, carbon_bars, carbon, "#364152")

    card = FancyBboxPatch(
        (0.02, 0.52),
        0.96,
        0.42,
        boxstyle="round,pad=0.018,rounding_size=18",
        linewidth=0,
        facecolor="#E7D9C5",
        transform=ax_note.transAxes,
    )
    ax_note.add_patch(card)
    ax_note.text(0.08, 0.88, "Ce que montre ImpactLLM", transform=ax_note.transAxes, fontsize=16, fontweight="bold", color="#18212F")
    ax_note.text(
        0.08,
        0.80,
        f"- {catalog_count} modeles LLM compares dans le referentiel\n"
        "- Hypotheses, sources et bornes explicites\n"
        "- Meme cadre pour des modeles ouverts et fermes",
        transform=ax_note.transAxes,
        fontsize=12.5,
        color="#364152",
        va="top",
        linespacing=1.6,
    )

    card2 = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.34,
        boxstyle="round,pad=0.018,rounding_size=18",
        linewidth=1.2,
        edgecolor="#D6C6AE",
        facecolor="#FFF4E3",
        transform=ax_note.transAxes,
    )
    ax_note.add_patch(card2)
    ax_note.text(0.08, 0.34, "Et aussi", transform=ax_note.transAxes, fontsize=16, fontweight="bold", color="#18212F")
    ax_note.text(
        0.08,
        0.27,
        "Decrivez un usage ou votre site en langage naturel,\n"
        "ImpactLLM calcule ensuite un ordre de grandeur coherent\n"
        "avec le referentiel public.",
        transform=ax_note.transAxes,
        fontsize=12.5,
        color="#364152",
        va="top",
        linespacing=1.55,
    )

    fig.text(
        0.05,
        0.045,
        "Source : dev.emotia.com/impact-llm    |    Code et donnees : github.com/apachot/ImpactLLM",
        fontsize=11,
        color="#907A59",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    build_visual(selected_rows(rows), len(rows))


if __name__ == "__main__":
    main()
