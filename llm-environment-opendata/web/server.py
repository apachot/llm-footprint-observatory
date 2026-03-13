#!/usr/bin/env python3
from datetime import datetime
import json
import re
import sys
from email.utils import formatdate
from functools import lru_cache
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.estimator import (
    build_energy_inference_anchors,
    build_market_model_predictions,
    build_training_market_predictions,
    compute_token_ratio,
    estimate_feature_externalities,
    get_record,
    load_country_energy_mix,
    load_market_models,
    load_models,
    load_records,
    wh_to_gco2e,
    wh_to_liters,
)
from core.openai_parser import (
    OpenAIModerationError,
    OpenAIParserError,
    moderate_application_description_with_openai,
    parse_application_description_with_openai,
)


PROJECT_NAME = "LLM Environment Open Data"
BIB_PATH = ROOT.parent / "llm-environment-opendata-paper" / "references_llm_environment_opendata.bib"
ANALYSIS_LOG_PATH = ROOT / "data" / "analysis_runs.json"
PAPER_TEX_PATH = ROOT.parent / "llm-environment-opendata-paper" / "llm_environment_opendata_paper.tex"
PAPER_PDF_PATH = ROOT.parent / "llm-environment-opendata-paper" / "llm_environment_opendata_paper.pdf"
REFERENCE_PAGE_TOKENS = 750.0
DEFAULT_PROMPT_TOKENS = 1550.0
PROJECT_PAPER_BIBTEX = """@misc{llm_environment_open_data_2026,
  title = {An Open Tool for Exploring and Estimating the Environmental Footprint of Large Language Models: Dataset, Observatory, API, MCP Server, and Web Interface},
  author = {Pachot, Arnault and Petit, Thierry},
  year = {2026},
  month = mar,
  note = {Working paper},
  url = {https://github.com/apachot/Publications-scientifiques}
}"""
SITE_CONTEXT_REFERENCES = [
    {
        "key": "brysbaert2019",
        "category": "Méthodologie d'usage",
        "citation": "Brysbaert, M. (2019). How many words do we read per minute? A review and meta-analysis of reading rate. Journal of Memory and Language, 109, 104047. https://www.sciencedirect.com/science/article/pii/S0749596X19300786",
        "url": "https://www.sciencedirect.com/science/article/pii/S0749596X19300786",
    },
    {
        "key": "icct2025cars",
        "category": "Repère de comparaison",
        "citation": "International Council on Clean Transportation. (2025). Life-cycle greenhouse gas emissions from passenger cars in the European Union: A 2025 update and key factors to consider. https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/",
        "url": "https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/",
    },
    {
        "key": "gosslingklower2026aviation",
        "category": "Repère de comparaison",
        "citation": "Gössling, S., Klöwer, M., Leitão, J. C., Hirsch, S., Brockhagen, D., & Humpe, A. (2026). Large carbon dioxide emissions avoidance potential in improved commercial air transport efficiency. Communications Earth & Environment, 7, 13. https://www.nature.com/articles/s43247-025-03069-4",
        "url": "https://www.nature.com/articles/s43247-025-03069-4",
    },
    {
        "key": "purdue_extension_energy",
        "category": "Repère de comparaison",
        "citation": "Carroll, N. J., & Kruse, J. (n.d.). Energy investigators 2: Facilitator’s guide. Purdue Extension. https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
        "url": "https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
    },
    {
        "key": "inc2019serviceeau",
        "category": "Repère de comparaison",
        "citation": "Institut national de la consommation. (2019, October 31). Le service de l'eau. https://www.inc-conso.fr/content/le-service-de-leau",
        "url": "https://www.inc-conso.fr/content/le-service-de-leau",
    },
    {
        "key": "inc2016salledebains",
        "category": "Repère de comparaison",
        "citation": "Institut national de la consommation. (2016, March 21). L'eau du robinet dans la salle de bains. https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
        "url": "https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
    },
    {
        "key": "anses2020eau",
        "category": "Repère de comparaison",
        "citation": "Anses. (2020, September 18). Eau en bouteille ou eau du robinet : bonnes pratiques de consommation. https://www.anses.fr/fr/content/eau-de-boisson-bonnes-pratiques-de-consommation",
        "url": "https://www.anses.fr/fr/content/eau-de-boisson-bonnes-pratiques-de-consommation",
    },
    {
        "key": "ademe2026eau",
        "category": "Repère de comparaison",
        "citation": "ADEME. (n.d.). 10 conseils pour faire des économies d’eau à la maison. Agir pour la transition écologique. https://agirpourlatransition.ademe.fr/particuliers/maison/economies-denergie-deau/comment-reduire-consommation-facture-deau",
        "url": "https://agirpourlatransition.ademe.fr/particuliers/maison/economies-denergie-deau/comment-reduire-consommation-facture-deau",
    },
    {
        "key": "rte2022bilan",
        "category": "Repère de comparaison",
        "citation": "RTE. (2022, February 25). Bilan électrique 2021 - Une production d’électricité assurée à plus de 92% par des sources n’émettant pas de gaz à effet de serre. https://www.rte-france.com/actualites/bilan-electrique-2021",
        "url": "https://www.rte-france.com/actualites/bilan-electrique-2021",
    },
    {
        "key": "macknick2011water",
        "category": "Méthodologie eau-électricité",
        "citation": "Macknick, J., Newmark, R., Heath, G., & Hallett, K. C. (2011). A review of operational water consumption and withdrawal factors for electricity generating technologies. Environmental Research Letters, 7(4), 045802. https://iopscience.iop.org/article/10.1088/1748-9326/7/4/045802",
        "url": "https://iopscience.iop.org/article/10.1088/1748-9326/7/4/045802",
    },
    {
        "key": "meldrum2013water",
        "category": "Méthodologie eau-électricité",
        "citation": "Meldrum, J., Nettles-Anderson, S., Heath, G., & Macknick, J. (2013). Life cycle water use for electricity generation: A review and harmonization of literature estimates. Environmental Research Letters, 8(1), 015031. https://iopscience.iop.org/article/10.1088/1748-9326/8/1/015031",
        "url": "https://iopscience.iop.org/article/10.1088/1748-9326/8/1/015031",
    },
]

REAL_WORLD_INDICATOR_ROWS = [
    {
        "domain": "Energy",
        "indicator": "Fluorescent lamp for 1 hour",
        "value": "9,3 Wh",
        "citation": "Carroll, N. J., & Kruse, J. (n.d.). Energy investigators 2: Facilitator’s guide. Purdue Extension. https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
        "locator": "Purdue Extension guide used for household-use examples; extraction still to be normalized more precisely.",
        "url": "https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
    },
    {
        "domain": "Energy",
        "indicator": "Laptop for 1 hour",
        "value": "32 Wh",
        "citation": "Carroll, N. J., & Kruse, J. (n.d.). Energy investigators 2: Facilitator’s guide. Purdue Extension. https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
        "locator": "Purdue Extension guide used for household-use examples; extraction still to be normalized more precisely.",
        "url": "https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf",
    },
    {
        "domain": "Energy",
        "indicator": "Electric space heater for 1 hour",
        "value": "1.5 kWh",
        "citation": "Project calculation convention.",
        "locator": "Nominal power assumption fixed at 1,500 W, i.e. 1.5 kWh for 1 hour.",
        "url": "",
    },
    {
        "domain": "Energy",
        "indicator": "10,000 French households over one year of domestic use",
        "value": "25 GWh",
        "citation": "RTE. (2022, February 25). Bilan électrique 2021 - Une production d’électricité assurée à plus de 92% par des sources n’émettant pas de gaz à effet de serre. https://www.rte-france.com/actualites/bilan-electrique-2021",
        "locator": "Project convention based on an average consumption of 2,500 kWh/year per household; comparison value made explicit in the training-chart note.",
        "url": "https://www.rte-france.com/actualites/bilan-electrique-2021",
    },
    {
        "domain": "Carbon",
        "indicator": "Average internal-combustion car",
        "value": "235 gCO2e/km",
        "citation": "International Council on Clean Transportation. (2025). Life-cycle greenhouse gas emissions from passenger cars in the European Union: A 2025 update and key factors to consider. https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/",
        "locator": "Key findings ; Figure 1 ; gasoline ICEV running on the average blend of fossil gasoline and ethanol estimated at 235 gCO2e/km.",
        "url": "https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/",
    },
    {
        "domain": "Carbon",
        "indicator": "Average full commercial flight (derived value)",
        "value": "≈ 21.1 tCO2 per flight",
        "citation": "Gössling, S., Klöwer, M., Leitão, J. C., Hirsch, S., Brockhagen, D., & Humpe, A. (2026). Large carbon dioxide emissions avoidance potential in improved commercial air transport efficiency. Communications Earth & Environment, 7, 13. https://www.nature.com/articles/s43247-025-03069-4",
        "locator": "Results, “Emissions and efficiency”: 27,451,887 flights in 2023 causing 577,968,750 tCO2 emissions; the site then derives an average per flight.",
        "url": "https://www.nature.com/articles/s43247-025-03069-4",
    },
    {
        "domain": "Water",
        "indicator": "Shower",
        "value": "9 L/min",
        "citation": "ADEME. (n.d.). La Clef Verte. Agir pour la transition écologique. https://agirpourlatransition.ademe.fr/particuliers/mieux-consommer/mieux-choisir/labels-environnementaux/hebergement/gite-chambre-hotes-la-clef-verte",
        "locator": "Mandatory criteria: “100% of showers: 9 liters/minute”.",
        "url": "https://agirpourlatransition.ademe.fr/particuliers/mieux-consommer/mieux-choisir/labels-environnementaux/hebergement/gite-chambre-hotes-la-clef-verte",
    },
    {
        "domain": "Water",
        "indicator": "4 to 5 minute shower",
        "value": "30 to 80 L",
        "citation": "Institut national de la consommation. (2016, March 21). L'eau du robinet dans la salle de bains. https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
        "locator": "Section “1 - Prefer alternating showers and baths”: “A 4 to 5 minute shower consumes 30 to 80 liters of water”.",
        "url": "https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
    },
    {
        "domain": "Water",
        "indicator": "Bath",
        "value": "150 to 200 L",
        "citation": "Institut national de la consommation. (2016, March 21). L'eau du robinet dans la salle de bains. https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
        "locator": "Section “1 - Prefer alternating showers and baths”: “a bath 150 to 200 liters”.",
        "url": "https://www.inc-conso.fr/content/leau-du-robinet-dans-la-salle-de-bains",
    },
    {
        "domain": "Water",
        "indicator": "Average daily water consumption per person in France",
        "value": "148 L/day",
        "citation": "ADEME. (n.d.). Nos conseils pour économiser l'eau à la maison. Agir pour la transition écologique. https://agirpourlatransition.ademe.fr/particuliers/maison/economies-denergie-deau/comment-reduire-consommation-facture-deau",
        "locator": "Section “Did you know?”: “On average, in France, we consume 148 liters of water per person each day.”",
        "url": "https://agirpourlatransition.ademe.fr/particuliers/maison/economies-denergie-deau/comment-reduire-consommation-facture-deau",
    },
]


def normalize_model_label(value):
    if not value:
        return ""
    lowered = str(value).lower()
    for char in (" ", "-", "_", ".", ",", ":", ";", "/", "(", ")"):
        lowered = lowered.replace(char, "")
    return lowered


def format_apa_hover(row):
    apa_citation = format_apa_citation(row)
    locator = str((row or {}).get("source_locator", "")).strip()
    metric_name = str((row or {}).get("metric_name", "")).strip()
    if not apa_citation:
        return locator or metric_name
    extras = [part for part in (metric_name, locator) if part]
    if extras:
        return f"{apa_citation}. {' | '.join(extras)}"
    return apa_citation


def format_apa_citation(row):
    study_key = str((row or {}).get("study_key", "")).strip()
    if study_key:
        bib_entry = load_bibliography_index().get(study_key)
        if bib_entry:
            return format_bib_entry_apa(bib_entry)

    citation = str((row or {}).get("citation", "")).strip()
    if not citation:
        return ""
    return citation


@lru_cache(maxsize=1)
def load_bibliography_index():
    if not BIB_PATH.exists():
        return {}
    content = BIB_PATH.read_text(encoding="utf-8")
    entries = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", content, re.DOTALL):
        entry_type = match.group(1).strip().lower()
        key = match.group(2).strip()
        body = match.group(3)
        fields = {}
        for field_match in re.finditer(r"(\w+)\s*=\s*\{(.*?)\},?", body, re.DOTALL):
            field_name = field_match.group(1).strip().lower()
            field_value = " ".join(field_match.group(2).strip().split())
            fields[field_name] = field_value
        fields["entry_type"] = entry_type
        entries[key] = fields
    return entries


def format_bib_author_list(author_field):
    if not author_field:
        return ""
    authors = [part.strip().strip("{}") for part in author_field.split(" and ") if part.strip()]
    formatted = []
    for author in authors:
        if "," in author:
            last, first = [part.strip() for part in author.split(",", 1)]
            initials = " ".join(f"{chunk[0]}." for chunk in re.split(r"[\s-]+", first) if chunk)
            formatted.append(f"{last}, {initials}".strip())
        else:
            parts = author.split()
            if len(parts) == 1:
                formatted.append(parts[0])
            else:
                last = parts[-1]
                first = " ".join(parts[:-1])
                initials = " ".join(f"{chunk[0]}." for chunk in re.split(r"[\s-]+", first) if chunk)
                formatted.append(f"{last}, {initials}".strip())
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def format_bib_entry_apa(entry):
    authors = format_bib_author_list(entry.get("author", ""))
    year = entry.get("year", "n.d.")
    title = entry.get("title", "").replace("{", "").replace("}", "")
    journal = entry.get("journal", "") or entry.get("booktitle", "") or entry.get("institution", "")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")
    url = entry.get("url", "")

    parts = []
    if authors:
        parts.append(f"{authors} ({year}).")
    else:
        parts.append(f"({year}).")
    if title:
        parts.append(f"{title}.")
    if journal:
        container = journal
        if volume:
            container += f", {volume}"
            if number:
                container += f"({number})"
        elif number:
            container += f", ({number})"
        if pages:
            container += f", {pages}"
        container += "."
        parts.append(container)
    if url:
        parts.append(url)
    return " ".join(part for part in parts if part)


def reference_anchor_id(row):
    record_id = str((row or {}).get("record_id", "")).strip()
    if not record_id:
        return ""
    return f"ref-{record_id}"


def format_reference_parameters(value):
    raw = str(value or "").strip()
    if not raw:
        return "n.d."
    if "est" in raw.lower():
        return f"{raw}*"
    return raw


def is_estimated_parameter_status(value):
    return str(value or "").strip().lower() == "estimated"


def format_parameter_billions(value, estimated=False):
    raw = str(value or "").strip()
    if not raw:
        return "n.d."
    suffix = "*" if estimated else ""
    return f"{raw}B{suffix}"


def obfuscate_email(value):
    text = str(value or "").strip()
    return "".join(f"&#{ord(char)};" for char in text)


def html_id_attr(value):
    if not value:
        return ""
    return f' id="{escape(value, quote=True)}"'


@lru_cache(maxsize=1)
def build_reference_number_map():
    mapping = {}
    for index, row in enumerate(build_literature_catalog_rows(), start=1):
        record_id = str(row.get("record_id", "")).strip()
        if record_id:
            mapping[record_id] = index
    return mapping


def classify_evidence_level(parsed_payload, factor_rows):
    model_id = parsed_payload.get("model_id", "")
    normalized_model = normalize_model_label(model_id)
    if not normalized_model or normalized_model in {"unknown", "generic"}:
        return {
            "level": "proxy_scientifique",
            "label": "Scientific proxy",
            "description": "The estimate relies on literature factors applicable to a family of uses, without a measurement attributable to one specific target model.",
        }

    direct_match = False
    family_match = False
    provider = normalize_model_label(parsed_payload.get("provider", ""))
    prefixes = tuple(part for part in normalized_model.split() if part)

    for row in factor_rows:
        haystack = normalize_model_label(
            " ".join(
                [
                    row.get("metric_name", ""),
                    row.get("citation", ""),
                    row.get("source_locator", ""),
                ]
            )
        )
        if normalized_model and normalized_model in haystack:
            direct_match = True
            break
        if provider and provider in haystack:
            family_match = True
        if normalized_model.startswith(("gpt", "gemini", "claude", "llama", "mistral", "qwen", "deepseek")):
            family = "".join(ch for ch in normalized_model if not ch.isdigit())
            if family and family in haystack:
                family_match = True

    if direct_match:
        return {
            "level": "mesure_directe",
            "label": "Direct measurement",
            "description": "At least one selected factor explicitly matches the model mentioned in the request.",
        }
    if family_match:
        return {
            "level": "proxy_scientifique",
            "label": "Scientific proxy",
            "description": "The retained factors are close to the service family or provider mentioned, but do not constitute a direct measurement of the target model.",
        }
    return {
        "level": "extrapolation",
        "label": "Extrapolation",
        "description": "No direct measurement of the target model is available in the corpus used; the estimate is derived from reference factors and contextual adjustments.",
    }


def format_scaled_value(value, unit_kind):
    value = 0.0 if value is None else float(value)
    abs_value = abs(value)

    if unit_kind == "energy":
        if abs_value >= 1000:
            return f"{value / 1000.0:.1f}", "kWh"
        if abs_value >= 1:
            return f"{value:.1f}", "Wh"
        if abs_value >= 0.1:
            return f"{value:.2f}", "Wh"
        if abs_value >= 0.01:
            return f"{value:.4f}", "Wh"
        if abs_value > 0:
            return f"{value:.5f}", "Wh"
        return f"{value:.1f}", "Wh"

    if unit_kind == "carbon":
        if abs_value >= 1000:
            return f"{value / 1000.0:.2f}", "kgCO2e"
        if abs_value >= 1:
            return f"{value:.1f}", "gCO2e"
        if abs_value >= 0.1:
            return f"{value:.2f}", "gCO2e"
        if abs_value > 0:
            return f"{value:.4f}", "gCO2e"
        return f"{value:.1f}", "gCO2e"

    if unit_kind == "water":
        if abs_value >= 1000:
            return f"{value / 1000.0:.1f}", "L"
        if abs_value >= 1:
            return f"{value:.1f}", "mL"
        if abs_value >= 0.1:
            return f"{value:.2f}", "mL"
        if abs_value > 0:
            return f"{value:.4f}", "mL"
        return f"{value:.1f}", "mL"

    return f"{value:.1f}", ""


def format_range_display(range_obj, unit_kind):
    low_value, unit = format_scaled_value(range_obj["low"], unit_kind)
    high_value, _ = format_scaled_value(range_obj["high"], unit_kind)
    return f"{low_value} - {high_value} {unit}"


def format_result_card_display(range_obj, unit_kind):
    if not range_obj:
        return "n.d."
    low = float(range_obj.get("low", 0.0) or 0.0)
    high = float(range_obj.get("high", 0.0) or 0.0)
    if abs(low - high) < 1e-12:
        return format_value_display(range_obj.get("central", low), unit_kind)
    return format_range_display(range_obj, unit_kind)


def format_central_display(range_obj, unit_kind):
    if not range_obj:
        return "n.d."
    return format_value_display(range_obj.get("central", 0.0), unit_kind)


def format_dispersion_ratio(range_obj):
    if not range_obj:
        return "n.d."
    low = float(range_obj.get("low", 0.0) or 0.0)
    high = float(range_obj.get("high", 0.0) or 0.0)
    central = float(range_obj.get("central", 0.0) or 0.0)
    if central <= 0 or low <= 0 or high <= 0:
        return "n.d."
    spread = high / low if low > 0 else None
    if spread is None:
        return "n.d."
    return f"×{spread:.1f}"


def format_value_display(value, unit_kind):
    formatted_value, unit = format_scaled_value(value, unit_kind)
    return f"{formatted_value} {unit}".strip()


def format_count(value):
    return f"{int(round(float(value))):,}".replace(",", " ")


def humanize_assumption(text):
    value = str(text)
    replacements = {
        "Parametric extrapolation enabled for target model": "Parametric extrapolation applied to the target model",
        "Reference inference scaling derived from Ren et al. 2024 with page-level measurements at": "Scaling derived from Ren et al. (2024), using page-level measurements at",
        "Token scaling applied relative to": "Token scaling applied relative to",
        "Energy is treated as the primary quantity; carbon and water are derived from the electricity mix of the retained country": "Energy is the primary quantity, and carbon is contextualized with the retained electricity mix.",
        "Carbon contextualized using country electricity carbon intensity": "Carbon emissions are adjusted using the retained country's electricity carbon intensity.",
        "Carbon adjusted using provided electricity carbon intensity": "Carbon emissions are recalculated using the provided electricity carbon intensity.",
        "Water contextualized using country electricity water intensity": "Electricity water intensity remains part of the method context but is not shown in the personalized result.",
        "Water adjusted using provided electricity water intensity": "Provided electricity water intensity remains part of the method context but is not shown in the personalized result.",
        "Request type classified as": "Request type classified as",
        "Country mix fallback applied for": "Default electricity mix applied for",
        "Carbon and water recalculated with the publisher-country mix for": "Carbon is recalculated using the publisher-country electricity mix for",
        "because the model is treated as a proprietary hosted service": "because the model is treated as a hosted proprietary service.",
        "Carbon and water recalculated with the project country mix for": "Carbon is recalculated using the project-country electricity mix for",
        "because the model is treated as open-weight or self-hosted": "because the model is treated as open-weight or self-hosted.",
        "Page-based method anchored on the nearest available source model in parameter count:": "Page-based method anchored on the nearest available source model by parameter count:",
        "Final central result uses the prompt-based method only because no empirical prompt-to-page conversion is available and the target model is treated as a hosted proprietary service": "The final central result uses only the prompt-based method because no empirical prompt-to-page conversion is available and the target model is treated as a hosted proprietary service.",
        "Final central result uses the page-based method only because no empirical prompt-to-page conversion is available and the target model is treated as open-weight or self-hosted": "The final central result uses only the page-based method because no empirical prompt-to-page conversion is available and the target model is treated as open-weight or self-hosted.",
        "Final central result averages the available inference methods": "The final central result averages the available inference methods.",
        "Unified inference model calibrated from the nearest literature energy anchor after harmonizing the observed value to Wh per request": "Unified inference model calibrated from the nearest literature energy anchor after harmonizing the observed value to Wh per request.",
        "Nearest calibration anchor:": "Nearest calibration anchor:",
        "LLM request(s) per feature use": "LLM request(s) per feature use",
        "feature uses per year": "feature uses per year",
        "A page calibration is computed from the mean Wh per parameter observed in page-generation inference records": "A page calibration is computed from the mean Wh per parameter observed in page-generation inference records.",
        "Page-family annualization uses generated page equivalents, with": "Page-family annualization uses generated page equivalents, with",
        "tokens per reference page when no explicit page count is provided": "tokens per reference page when no explicit page count is provided.",
        "Page-family method marked as not applicable for this scenario by the parser": "The page-family method was marked as not applicable for this scenario by the parser.",
    }
    for source, target in replacements.items():
        if source in value:
            value = value.replace(source, target)
    value = value.replace("750 tokens", "750 tokens")
    value = value.replace("12.0", "12")
    value = value.replace("1.0", "1")
    return value


def matching_factor_rows(factor_rows, keywords):
    matches = []
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for row in factor_rows or []:
        haystack = " ".join(
            [
                str(row.get("metric_name", "")),
                str(row.get("metric_unit", "")),
                str(row.get("citation", "")),
                str(row.get("source_locator", "")),
            ]
        ).lower()
        if any(keyword in haystack for keyword in lowered_keywords):
            matches.append(row)
    return matches


def render_source_refs(rows):
    number_map = build_reference_number_map()
    refs = []
    for row in rows:
        title = escape(format_apa_hover(row))
        href = f"#{reference_anchor_id(row)}" if reference_anchor_id(row) else "#"
        ref_number = number_map.get(str(row.get("record_id", "")).strip())
        if not ref_number:
            continue
        refs.append(
            f'<a class="inline-ref" href="{href}" title="{title}">[{ref_number}]</a>'
        )
    return " ".join(refs)


def render_single_source_ref(row):
    if not row:
        return ""
    number_map = build_reference_number_map()
    ref_number = number_map.get(str(row.get("record_id", "")).strip())
    if not ref_number:
        return ""
    title = escape(format_apa_hover(row))
    href = f"#{reference_anchor_id(row)}" if reference_anchor_id(row) else "#"
    return f'<a class="inline-ref" href="{href}" title="{title}">[{ref_number}]</a>'


def render_sourced_value(value_text, rows):
    if not rows:
        return f"<code>{escape(value_text)}</code>"
    title = " ; ".join(format_apa_hover(row) for row in rows)
    return (
        f'<span class="sourced-value" title="{escape(title)}">'
        f'<code>{escape(value_text)}</code>'
        f'{render_source_refs(rows)}'
        f"</span>"
    )


def render_extrapolation_details(result, metric_label, source_rows):
    if result.get("method") != "parametric_extrapolation":
        return ""
    unit_key = {
        "energy": "energy_wh",
        "carbon": "carbon_gco2e",
        "water": "water_ml",
    }.get(metric_label)
    detail = ((result.get("per_request_llm") or {}) and (result.get("extrapolation_details") or {}).get(unit_key)) or {}
    row_lookup = {row.get("record_id"): row for row in source_rows or []}
    detail_lines = []
    for anchor in detail.get("anchors", []):
        source_value = anchor.get("source_value")
        source_unit = anchor.get("source_unit", "")
        factor_value = anchor.get("factor_central")
        extrapolated_value = anchor.get("extrapolated_value_central")
        if source_value is None or factor_value is None or extrapolated_value is None:
            continue
        row = row_lookup.get(anchor.get("record_id"))
        citation_link = ""
        if row:
            title = escape(format_apa_hover(row))
            href = f"#{reference_anchor_id(row)}" if reference_anchor_id(row) else "#"
            citation_link = (
                f' <a class="inline-ref" href="{href}" title="{title}">'
                f'{escape(row.get("citation", "source"))}</a>'
            )
        formatted_source = f"{source_value} {source_unit}".strip()
        formatted_extrapolated = format_value_display(extrapolated_value, metric_label if metric_label != "carbon" else "carbon")
        detail_lines.append(
            f"<li><strong>{escape(anchor.get('source_model', 'source model'))}</strong>: source value "
            f"<code>{escape(formatted_source)}</code>{citation_link} x applied factor <code>{factor_value:.3f}</code> "
            f"= extrapolated value <code>{escape(formatted_extrapolated)}</code></li>"
        )

    if not detail_lines:
        return ""
    return f'<ul class="extrapolation-list">{"".join(detail_lines)}</ul>'


def render_metric_detail(result, factor_rows, metric_label, title):
    annual_llm = result["annual_llm"]
    scope = result["feature_scope"]
    annual_requests = float(scope["annual_llm_requests"])
    per_request = result["per_request_llm"]
    per_feature = result["per_feature_llm"]
    source_rows = matching_factor_rows(
        factor_rows,
        {
            "energy": ["energy", "wh"],
            "carbon": ["carbon", "emission", "gco2"],
            "water": ["water", "ml", "liter", "litre"],
        }[metric_label],
    )
    unit_key = {
        "energy": "energy_wh",
        "carbon": "carbon_gco2e",
        "water": "water_ml",
    }[metric_label]
    return f"""
    <details class="metric-detail">
      <summary class="metric-detail-toggle">
        <span class="metric-detail-icon" aria-hidden="true">+</span>
        <span>See calculation details</span>
      </summary>
      <div class="metric-detail-body">
        <p class="math-detail">
          <strong>{escape(title)}</strong> :
          <code>{escape(format_range_display(annual_llm[unit_key], metric_label))}</code>
        </p>
        {render_extrapolation_details(result, metric_label, source_rows)}
        <p class="math-detail">
          Estimated impact per LLM request:
          <code>{escape(format_range_display(per_request[unit_key], metric_label))}</code>
        </p>
        <p class="math-detail">
          Estimated impact per feature use:
          <code>{escape(format_range_display(per_feature[unit_key], metric_label))}</code>
        </p>
        <p class="math-detail">
          Projected across <code>{format_count(annual_requests)}</code> calls per year,
          this gives <code>{escape(format_range_display(annual_llm[unit_key], metric_label))}</code>.
        </p>
      </div>
    </details>
    """


def render_assumptions_summary(result):
    assumptions = result.get("assumptions", [])
    if not assumptions:
        return ""
    return f"""
    <div class="assumptions-box assumptions-box-compact">
      <span class="math-label">Retained assumptions</span>
      <ul class="assumptions-list">
        {''.join(f'<li>{escape(humanize_assumption(item))}</li>' for item in assumptions)}
      </ul>
    </div>
    """


def translate_method_text(value):
    text = str(value or "")
    replacements = {
        "Moyenne Wh/prompt|requête": "Average Wh/prompt|request",
        "Moyenne Wh/page": "Average Wh/page",
        "Moyenne des intensités énergétiques Wh/paramètre calibrées sur les ancrages prompt/requête.": "Average Wh/parameter energy intensities calibrated on prompt/request anchors.",
        "Moyenne des intensités énergétiques Wh/paramètre calibrées sur les ancrages page.": "Average Wh/parameter energy intensities calibrated on page anchors.",
        "Wh/prompt|requête": "Wh/prompt|request",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def format_scalar(value, decimals=3):
    if value is None:
        return "n.d."
    text = f"{float(value):.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_raw_metric(value, unit):
    return f"{format_scalar(value)} {unit}".strip()


def format_literature_metric(value, unit):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{value} {unit}".strip()

    if numeric == 0:
        return f"0 {unit}".strip()
    if abs(numeric) < 0.001:
        text = f"{numeric:.6f}".rstrip("0").rstrip(".")
    elif abs(numeric) < 0.01:
        text = f"{numeric:.5f}".rstrip("0").rstrip(".")
    else:
        text = f"{numeric:.3f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


def infer_source_intensity(energy_record, metric_record, metric_kind):
    if not energy_record or not metric_record:
        return None
    try:
        energy_value = float(energy_record["metric_value"])
        metric_value = float(metric_record["metric_value"])
    except (TypeError, ValueError):
        return None
    energy_unit = str(energy_record.get("metric_unit", "")).lower()
    metric_unit = str(metric_record.get("metric_unit", "")).lower()

    if "/page" in energy_unit:
        energy_kwh = energy_value
    elif "/prompt" in energy_unit or "/query" in energy_unit:
        energy_kwh = energy_value / 1000.0
    else:
        return None

    if energy_kwh <= 0:
        return None

    if metric_kind == "carbon":
        if "gco2" not in metric_unit:
            return None
        return metric_value / energy_kwh

    if metric_kind == "water":
        if metric_unit.startswith("l/"):
            water_l = metric_value
        elif metric_unit.startswith("ml/") or "ml/" in metric_unit:
            water_l = metric_value / 1000.0
        else:
            return None
        return water_l / energy_kwh

    return None


def build_method_modal_body(method):
    annual_requests = float(method.get("annual_requests", 0.0) or 0.0)
    annual_feature_uses = float(method.get("annual_feature_uses", 0.0) or 0.0)
    requests_per_feature = float(method.get("requests_per_feature", 0.0) or 0.0)
    months_per_year = float(method.get("months_per_year", 0.0) or 0.0)
    feature_uses_per_month = float(method.get("feature_uses_per_month", 0.0) or 0.0)
    token_ratio = method.get("token_ratio")
    page_ratio = method.get("page_ratio")
    target_mix = method.get("target_mix") or {}
    target_country = target_mix.get("country_name") or method.get("target_country") or "not specified"
    target_carbon = method.get("target_grid_carbon_intensity")
    detail = method.get("detail", {})
    factor_rows = method.get("factor_rows") or []
    row_by_id = {row.get("record_id"): row for row in factor_rows}
    sections = []

    if detail.get("kind") == "wh_parameter_model":
        standard_request = detail.get("standard_request") or {}
        family = detail.get("family")
        pages_per_request_equivalent = detail.get("pages_per_request_equivalent")
        annual_page_equivalents = detail.get("annual_page_equivalents")
        reference_page_tokens = detail.get("reference_page_tokens")
        token_source_note = detail.get("token_source_note")
        annual_multiplier = detail.get("annual_multiplier", annual_requests)
        annualization_sentence = (
            f"<p>The annual call volume is calculated as:</p>"
            f"<p>\\["
            f"N_{{appels/an}} = {format_count(feature_uses_per_month)} \\times {format_scalar(months_per_year, 0)} \\times {format_scalar(requests_per_feature, 0)} = {format_count(annual_requests)}"
            f"\\]</p>"
        )
        if family == "page":
            source_note = "derived from the user description" if token_source_note in {"user_tokens", "parser_page_equivalent", "output_tokens"} else "project default value"
            annualization_sentence += (
                f"<p>For the <code>Wh/page</code> family, the engine then converts outputs into page equivalents:</p>"
                f"<p>\\["
                f"P_{{eq/appel}} = \\frac{{{format_scalar(standard_request.get('output_tokens', 0), 0)}}}{{{format_scalar(reference_page_tokens, 0)}}} = {format_scalar(pages_per_request_equivalent, 3)}"
                f"\\]</p>"
                f"<p>This conversion is based on {source_note}.</p>"
                f"<p>\\["
                f"P_{{eq/an}} = {format_count(annual_requests)} \\times {format_scalar(pages_per_request_equivalent, 3)} = {format_count(annual_page_equivalents)}"
                f"\\]</p>"
            )
        else:
            annualization_sentence += (
                "<p>In the <code>Wh/prompt|request</code> family, one LLM request directly corresponds to one inference unit. "
                "Annualization therefore relies on the number of LLM calls per year.</p>"
            )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">1. Scenario input data</div>
              <p>The interpreted scenario uses <code>{format_scalar(standard_request.get('input_tokens', 0), 0)}</code> input tokens and <code>{format_scalar(standard_request.get('output_tokens', 0), 0)}</code> output tokens per call.</p>
              {annualization_sentence}
            </div>
            """
        )
        anchor_lines = []
        for anchor in detail.get("anchors", []):
            row = row_by_id.get(anchor.get("record_id"))
            ref = render_single_source_ref(row)
            anchor_lines.append(
                f"""
                <li>
                  <p><strong>{escape(anchor.get('source_model', 'source'))}</strong> {ref}</p>
                  <p>Observed literature value: <code>{escape(anchor.get('source_energy', 'n.d.'))}</code> {ref}</p>
                  <p>Source parameter count: <code>{format_scalar(anchor.get('source_params'))}B</code>. Target parameter count: <code>{format_scalar(anchor.get('target_params'))}B</code>.</p>
                  <p>Applied parameter factor:</p>
                  <p>\\[
                  r_P = \\frac{{P_t}}{{P_s}} = \\frac{{{format_scalar(anchor.get('target_params'))}}}{{{format_scalar(anchor.get('source_params'))}}} = {format_scalar(anchor.get('parameter_factor'), 4)}
                  \\]</p>
                  <p>Extrapolated energy for one inference unit:</p>
                  <p>\\[
                  E_t = E_s \\times r_P = {escape(anchor.get('source_energy', 'n.d.'))} \\times {format_scalar(anchor.get('parameter_factor'), 4)} = {escape(format_range_display(anchor.get('per_request_energy', {'low':0,'high':0}), 'energy'))}
                  \\]</p>
                </li>
                """
            )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">2. Literature anchors and extrapolation</div>
              <p>The method starts from literature energy values published for the <code>{escape(detail.get('unit_basis', 'Wh'))}</code> family, then applies scaling by parameter count.</p>
              <ul class="extrapolation-list">{''.join(anchor_lines) or '<li>n.d.</li>'}</ul>
              <p>When several anchors exist in the same family, the engine computes an average energy intensity per billion parameters to obtain the central value shown in the result block.</p>
            </div>
            """
        )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">3. Carbon derivation from the country mix</div>
              <p>Carbon is not reused directly from the literature. It is derived from extrapolated energy using the retained country electricity mix, here <strong>{escape(target_country)}</strong>.</p>
              <p>\\[
              CO2_{{unitaire}} = \\frac{{E_{{unitaire}}}}{{1000}} \\times CI_c
              \\]</p>
              <p>Avec \\(CI_c = {format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\).</p>
              <p>The unit result retained for this method then leads to the following annualized values: energy <code>{escape(method['energy'])}</code> and carbon <code>{escape(method['carbon'])}</code>.</p>
            </div>
            <div class="method-modal-section">
              <div class="math-label">4. Final annualization</div>
              <p>The final annual projection is based on <code>{format_count(annual_multiplier)}</code> inference unit(s) per year.</p>
              <p>\\[
              Impact_{{annuel}} = Impact_{{unitaire}} \\times N_{{annuel}}
              \\]</p>
            </div>
            """
        )
        return "".join(sections)

    sections.append(
        f"""
        <div class="method-modal-section">
          <div class="math-label">1. Call annualization</div>
          <p><code>{format_count(feature_uses_per_month)}</code> uses/month × <code>{format_scalar(months_per_year, 0)}</code> months = <code>{format_count(annual_feature_uses)}</code> uses/year</p>
          <p><code>{format_count(annual_feature_uses)}</code> uses/year × <code>{format_scalar(requests_per_feature, 0)}</code> LLM call(s)/use = <code>{format_count(annual_requests)}</code> LLM calls/year</p>
        </div>
        """
    )

    anchors = detail.get("anchors", [])
    unit_basis = detail.get("unit_basis") or "facteur"
    ratio_value = detail.get("ratio")
    anchor_lines = []
    for anchor in anchors:
        carbon_note = ""
        if anchor.get("source_carbon_intensity") is not None and target_carbon is not None:
            carbon_note = (
                f" source intensity ≈ \\({format_scalar(anchor['source_carbon_intensity'])}\\ \\text{{gCO2e/kWh}}\\) "
                f"replaced by \\({format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\)"
            )
        anchor_lines.append(
            f"""
            <li>
              <strong>{escape(anchor.get('source_model', 'source'))}</strong> ({escape(anchor.get('source_country', 'n.d.'))}) :
              published energy <code>{escape(anchor.get('source_energy', 'n.d.'))}</code>,
              applied ratio <code>{format_scalar(ratio_value)}</code>,
              energy per request <code>{escape(format_range_display(anchor.get('per_request_energy', {'low':0,'high':0}), 'energy'))}</code>,
              carbon per request <code>{escape(format_range_display(anchor.get('per_request_carbon', {'low':0,'high':0}), 'carbon'))}</code> ({carbon_note.strip() or 'target mix applied'}).
            </li>
            """
        )
    sections.append(
        f"""
        <div class="method-modal-section">
          <div class="math-label">2. Multiples method on {escape(unit_basis)} indicators</div>
          <p>Each scientific indicator available in this unit family is recalculated for the target scenario. When several anchors exist, the method first retains the closest model by parameter count, then builds the central value from that selected anchor.</p>
          <ul class="extrapolation-list">{''.join(anchor_lines) or '<li>n.d.</li>'}</ul>
        </div>
        """
    )
    sections.append(
        f"""
        <div class="method-modal-section">
          <div class="math-label">3. Averaging and annualization</div>
          <p>Retained annual energy: average of recalculated indicators × <code>{format_count(annual_requests)}</code> calls/year = <code>{escape(method['energy'])}</code>.</p>
          <p>Retained annual carbon: average of recalculated indicators for <strong>{escape(target_country)}</strong> = <code>{escape(method['carbon'])}</code>.</p>
        </div>
        """
    )

    return "".join(sections)


def build_method_comparisons(records, parsed_payload, result):
    methods = []
    scope = result.get("feature_scope", {})
    annual_requests = float(scope.get("annual_llm_requests", 0.0) or 0.0)
    annual_feature_uses = float(scope.get("annual_feature_uses", 0.0) or 0.0)
    requests_per_feature = float(scope.get("requests_per_feature", 0.0) or 0.0)
    feature_uses_per_month = float(scope.get("feature_uses_per_month", 0.0) or 0.0)
    months_per_year = float(scope.get("months_per_year", 0.0) or 0.0)
    target_mix = result.get("country_energy_mix") or {}
    target_country = target_mix.get("country_name") or parsed_payload.get("country")
    target_carbon = target_mix.get("grid_carbon_intensity_gco2_per_kwh") or parsed_payload.get("grid_carbon_intensity_gco2_per_kwh")
    for method in result.get("method_results", []):
        rows = factor_details(records, method.get("record_ids", []))
        detail = dict(method.get("detail", {}))
        methods.append(
            {
                "label": translate_method_text(method.get("label", "Method")),
                "basis": translate_method_text(method.get("basis", "")),
                "energy": format_result_card_display(method["annual_energy_wh"], "energy"),
                "carbon": format_result_card_display(method["annual_carbon_gco2e"], "carbon"),
                "refs": render_source_refs(rows),
                "factor_rows": rows,
                "annual_requests": annual_requests,
                "annual_feature_uses": annual_feature_uses,
                "requests_per_feature": requests_per_feature,
                "feature_uses_per_month": feature_uses_per_month,
                "months_per_year": months_per_year,
                "token_ratio": (method.get("detail") or {}).get("token_ratio"),
                "page_ratio": (method.get("detail") or {}).get("page_ratio"),
                "target_country": target_country,
                "target_grid_carbon_intensity": float(target_carbon) if target_carbon not in (None, "") else None,
                "target_mix": target_mix,
                "detail": detail,
            }
        )
    return methods


def render_method_comparisons(methods):
    if not methods:
        return ""
    cards = "".join(
        f"""
        <article class="result-method-card">
          <div class="result-method-head">
            <div>
              <div class="result-method-kicker">Result</div>
              <h4>{escape(method['label'])}</h4>
            </div>
            <div class="result-method-refs">{method['refs']}</div>
          </div>
          <p class="result-method-basis">{escape(method['basis'])}</p>
          <div class="result-method-metrics">
            <div class="result-method-metric">
              <span class="result-method-label">Annual energy</span>
              <strong>{escape(method['energy'])}</strong>
            </div>
            <div class="result-method-metric">
              <span class="result-method-label">Annual carbon</span>
              <strong>{escape(method['carbon'])}</strong>
            </div>
          </div>
        </article>
        """
        for method in methods
    )
    return f"""
    <div class="method-panel result-panel">
      <div class="result-method-grid">
        {cards}
      </div>
    </div>
    """


def render_method_calculation_details(methods):
    if not methods:
        return ""
    blocks = "".join(
        f"""
        <article class="result-method-detail-card">
          <div class="summary-header">
            <div>
              <div class="summary-kicker">Calculation details</div>
              <h3>{escape(method['label'])}</h3>
            </div>
            <div class="result-method-refs">{method['refs']}</div>
          </div>
          <p class="summary-intro">{escape(method['basis'])}</p>
          {build_method_modal_body(method)}
        </article>
        """
        for method in methods
    )
    return f"""
    <div class="method-panel">
      {blocks}
    </div>
    """


def render_summary_html(summary_text, factor_rows):
    text = escape(summary_text or "")
    source_map = {}
    number_map = build_reference_number_map()
    for row in factor_rows or []:
        ref_number = number_map.get(str(row.get("record_id", "")).strip())
        if not ref_number:
            continue
        source_map[str(ref_number)] = row
        source_map[f"SRC{ref_number}"] = row

    def replace_source_tag(match):
        tag = match.group(1)
        row = source_map.get(tag)
        if not row:
            return f"[{escape(tag)}]"
        title = escape(format_apa_hover(row))
        href = f"#{reference_anchor_id(row)}" if reference_anchor_id(row) else "#"
        display_tag = re.sub(r"^SRC", "", tag)
        return (
            f'<a class="source-tag" href="{href}" '
            f'title="{title}">[{escape(display_tag)}]</a>'
        )

    html = re.sub(r"\[(SRC\d+|\d+)\]", replace_source_tag, text)
    return html.replace("\n", "<br>")


def describe_record_type_fr(record):
    phase = str(record.get("phase", "")).strip()
    metric_name = str(record.get("metric_name", "")).strip()
    model_or_scope = str(record.get("model_or_scope", "")).strip()

    labels = {
        ("training", "training_emissions"): "Greenhouse gas emissions from training",
        ("training", "compute_time"): "Compute time used for training",
        ("training", "training_tokens"): "Training token volume",
        ("lifecycle", "creation_lifecycle_emissions"): "Emissions across the model creation lifecycle",
        ("lifecycle", "creation_lifecycle_water"): "Water consumption across the model creation lifecycle",
        ("lifecycle", "development_share"): "Impact share attributed to development",
        ("lifecycle", "power_utilization_range"): "Infrastructure power utilization range",
        ("lifecycle", "training_water_total"): "Total water consumption associated with training",
        ("lifecycle", "training_water_onsite"): "On-site water consumption associated with training",
        ("infrastructure", "ai_share_of_datacenter_electricity"): "AI share of data center electricity demand",
        ("infrastructure", "annual_electricity"): "Annual data center electricity consumption",
        ("infrastructure", "electricity_share"): "Share of electricity demand attributed to data centers",
        ("infrastructure", "annual_growth_rate"): "Annual electricity consumption growth rate",
        ("infrastructure", "ai_power_demand"): "Power demand from AI systems",
        ("infrastructure", "annualized_energy"): "Annualized energy consumption of AI systems",
        ("inference", "query_energy"): "Energy consumption per query",
        ("inference", "prompt_energy"): "Energy consumption per prompt",
        ("inference", "prompt_emissions"): "Emissions per prompt",
        ("inference", "prompt_water"): "Water consumption per prompt",
        ("inference", "efficiency_gain"): "Reported efficiency gain between inference configurations",
        ("inference", "energy_component"): "Inference energy consumption by component",
        ("inference", "page_generation_energy"): "Energy consumption to generate one page",
        ("inference", "page_generation_emissions"): "Emissions to generate one page",
        ("inference", "page_generation_water"): "Water consumption to generate one page",
        ("inference", "response_water_equivalent"): "Water consumption for a small set of responses",
    }

    label = labels.get((phase, metric_name))
    if not label:
        phase_label = {
            "training": "Training",
            "inference": "Inference",
            "infrastructure": "Infrastructure",
            "lifecycle": "Lifecycle",
        }.get(phase, phase.capitalize() if phase else "Indicator")
        metric_label = metric_name.replace("_", " ") if metric_name else "not specified"
        label = f"{phase_label} : {metric_label}"

    if model_or_scope:
        return f"{label} ({model_or_scope})"
    return label


def build_literature_catalog_rows():
    rows = []
    excluded_study_keys = {"lbl2025", "devriesgao2025joule", "iea2025", "epri2024"}
    for record in load_records():
        source_url = str(record.get("source_url", ""))
        if record.get("study_key") in excluded_study_keys:
            continue
        if record.get("record_id") in {
            "morrison2025_dev_share",
            "morrison2025_power_variation",
            "li2025_chatbot_water",
            "strubell2019_co2_tuning_pipeline",
            "strubell2019_co2_nas",
            "elsworth2025_efficiency_energy",
            "elsworth2025_efficiency_carbon",
            "elsworth2025_accelerator_energy",
            "elsworth2025_cpu_dram_energy",
            "elsworth2025_idle_energy",
            "elsworth2025_datacenter_overhead",
        }:
            continue
        if "iea.org" in source_url or "publicpower.org" in source_url:
            continue
        rows.append(
            {
                "record_id": record.get("record_id", ""),
                "study_key": record.get("study_key", ""),
                "phase": record.get("phase", ""),
                "data_type": describe_record_type_fr(record),
                "model_or_scope": record.get("llm_normalized", "") or "n.d.",
                "model_parameters": record.get("model_parameters_normalized", "") or "n.d.",
                "geography": record.get("country_normalized", "") or "n.d.",
                "metric_value": f"{record.get('metric_value', '')} {record.get('metric_unit', '')}".strip(),
                "citation": record.get("citation", ""),
                "source_locator": record.get("source_locator", ""),
                "source_url": record.get("source_url", "#"),
            }
        )
    return rows


def render_reference_catalog_sections():
    rows = build_literature_catalog_rows()
    if not rows:
        return {"training": "", "inference": "", "counts": {"total": 0, "training": 0, "inference": 0}}
    number_map = build_reference_number_map()
    def render_reference_table(table_rows, title):
        if not table_rows:
            return ""
        title_html = f"<h4>{escape(title)}</h4>" if title else ""
        return f"""
        <div class="reference-subtable">
          {title_html}
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Ref.</th>
                  <th>Data type</th>
                  <th>LLM model</th>
                  <th>Parameters</th>
                  <th>Country</th>
                  <th>Value</th>
                  <th>Citation</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </div>
        """

    grouped = {"training": [], "inference": [], "other": []}
    for row in rows:
        locator_html = ""
        if row["source_locator"]:
            locator_html = f'<div class="reference-locator">{escape(row["source_locator"])}</div>'
        row_id = reference_anchor_id(row)
        ref_number = number_map.get(str(row.get("record_id", "")).strip(), "")
        rendered = (
            f"<tr{html_id_attr(row_id)}>"
            f"<td class=\"reference-number\">[{escape(str(ref_number))}]</td>"
            f"<td>{escape(row['data_type'])}</td>"
            f"<td>{escape(row.get('model_or_scope', '') or 'n.d.')}</td>"
            f"<td>{escape(format_reference_parameters(row.get('model_parameters', '') or 'n.d.'))}</td>"
            f"<td>{escape(row.get('geography', '') or 'n.d.')}</td>"
            f"<td>{escape(row['metric_value'])}</td>"
            f"<td>"
            f"<a href=\"{escape(row['source_url'], quote=True)}\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"{escape(format_apa_hover(row))}\">{escape(format_apa_citation(row))}</a>"
            f"{locator_html}"
            f"</td>"
            f"</tr>"
        )
        phase = str(row.get("phase", "")).strip().lower()
        if phase in {"training", "lifecycle"}:
            grouped["training"].append(rendered)
        elif phase == "inference":
            grouped["inference"].append(rendered)
        else:
            grouped["other"].append(rendered)

    return {
        "training": render_reference_table("".join(grouped["training"]), ""),
        "inference": render_reference_table("".join(grouped["inference"]), ""),
        "counts": {
            "total": len(rows),
            "training": len(grouped["training"]),
            "inference": len(grouped["inference"]),
        },
    }


def build_site_bibliography_entries():
    entries = []
    seen = set()

    for row in build_literature_catalog_rows():
        citation = format_apa_citation(row)
        if not citation:
            continue
        key = ("literature", str(row.get("study_key") or citation).strip())
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "category": "LLM literature",
                "citation": citation,
                "url": str(row.get("source_url", "")).strip(),
            }
        )

    bibliography_index = load_bibliography_index()
    for key in ("brysbaert2019",):
        entry = bibliography_index.get(key)
        if not entry:
            continue
        citation = format_bib_entry_apa(entry)
        dedupe_key = ("context", key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "category": "Usage methodology",
                "citation": citation,
                "url": str(entry.get("url", "")).strip(),
            }
        )

    for entry in SITE_CONTEXT_REFERENCES:
        key = ("context", str(entry.get("key", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "category": str(entry.get("category", "Comparison benchmark")),
                "citation": str(entry.get("citation", "")).strip(),
                "url": str(entry.get("url", "")).strip(),
            }
        )

    entries.sort(key=lambda item: (item["category"], item["citation"]))
    return entries


def render_bibliography_tab():
    reference_sections = render_reference_catalog_sections()
    country_rows = load_country_energy_mix()
    market_model_rows = load_market_models()
    if not reference_sections:
        return ""
    filtered_market_model_rows = []
    for row in market_model_rows:
        parameter_status = str(row.get("parameter_value_status", "")).strip().lower()
        market_status = str(row.get("market_status", "")).strip().lower()
        serving_mode = str(row.get("serving_mode", "")).strip().lower()
        if parameter_status != "estimated":
            continue
        if market_status != "api" and serving_mode != "closed":
            continue
        filtered_market_model_rows.append(row)
    filtered_market_model_rows.sort(
        key=lambda row: (
            str(row.get("provider", "") or "").lower(),
            str(row.get("display_name", "") or row.get("model_id", "") or "").lower(),
        )
    )
    proprietary_parameter_rows = []
    for row in filtered_market_model_rows:
        source_url = str(row.get("parameter_source_url", "") or "").strip()
        source_label = escape(str(row.get("parameter_source", "") or "Source not specified"))
        if source_url:
            source_html = f'<a href="{escape(source_url, quote=True)}" target="_blank" rel="noopener noreferrer">{source_label}</a>'
        else:
            source_html = source_label
        proprietary_parameter_rows.append(
            f"""
        <tr>
          <td>{escape(row.get('display_name', '') or row.get('model_id', '') or 'n.d.')}</td>
          <td>{escape(row.get('provider', 'n.d.') or 'n.d.')}</td>
          <td>{escape(format_market_parameter_display(row))}</td>
          <td>{escape(row.get('parameter_confidence', 'n.d.') or 'n.d.')}</td>
          <td>{source_html}</td>
          <td>{escape(row.get('notes', 'n.d.') or 'n.d.')}</td>
        </tr>
            """
        )
    visible_real_world_rows = [entry for entry in REAL_WORLD_INDICATOR_ROWS if entry.get("domain") != "Water"]
    real_world_rows = "".join(
        f"""
        <tr>
          <td>{escape(str(index))}</td>
          <td>{escape(entry['domain'])}</td>
          <td>{escape(entry['indicator'])}</td>
          <td>{escape(entry['value'])}</td>
          <td><a href="{escape(entry['url'] or '#', quote=True)}" target="_blank" rel="noopener noreferrer">{escape(entry['citation'])}</a><div class="reference-locator">{escape(entry['locator'])}</div></td>
        </tr>
        """
        for index, entry in enumerate(visible_real_world_rows, start=1)
    )
    country_mix_rows = "".join(
        f"""
        <tr>
          <td>{escape(str(index))}</td>
          <td>{escape(row.get('country_name', 'n.d.'))}</td>
          <td>{escape(str(row.get('year', 'n.d.')))}</td>
          <td>{escape(str(row.get('grid_carbon_intensity_gco2_per_kwh', 'n.d.')))} gCO2e/kWh</td>
          <td><a href="{escape(str(row.get('source_url', '') or '#'), quote=True)}" target="_blank" rel="noopener noreferrer">{escape(row.get('source_citation', 'n.d.'))}</a><div class="reference-locator">{escape(row.get('notes', 'Location not documented.'))}</div></td>
        </tr>
        """
        for index, row in enumerate(country_rows, start=1)
    )

    return f"""
    <section class="tab-panel" id="tab-bibliography-panel" data-tab-panel="bibliography">
      <section class="panel reference-panel">
        <div class="summary-header">
            <div>
            <div class="summary-kicker">References</div>
            <h3>Source annex used in the site</h3>
          </div>
        </div>
        <p class="summary-intro">This annex brings together the quantified reference material used in the interface, along with everyday comparison benchmarks and country factors used for carbon and water recalculation.</p>

        <div class="reference-subtable">
          <h4>Sources used to estimate proprietary LLM parameter counts</h4>
          <p class="summary-intro">This table lists the third-party sources used when a provider does not publish the parameter count of a closed model. Estimated values are marked with `*` throughout the interface.</p>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Estimated parameters</th>
                  <th>Confidence</th>
                  <th>Estimation source</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {"".join(proprietary_parameter_rows)}
              </tbody>
            </table>
          </div>
        </div>

        <div class="reference-subtable">
          <h4>Inference reference set</h4>
          {reference_sections['inference']}
        </div>

        <div class="reference-subtable">
          <h4>Training reference set</h4>
          {reference_sections['training']}
        </div>

        <div class="reference-subtable">
          <h4>Real-world comparison benchmarks</h4>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>No.</th>
                  <th>Domain</th>
                  <th>Indicator</th>
                  <th>Value</th>
                  <th>Reference</th>
                </tr>
              </thead>
              <tbody>
                {real_world_rows}
              </tbody>
            </table>
          </div>
        </div>

        <div class="reference-subtable">
          <h4>Country factors for carbon and water recalculation</h4>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>No.</th>
                  <th>Country</th>
                  <th>Year</th>
                  <th>Carbon intensity</th>
                  <th>Reference</th>
                </tr>
              </thead>
              <tbody>
                {country_mix_rows}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </section>
    """


def render_model_reference_table():
    rows = load_models()
    if not rows:
        return ""
    body = "".join(
        f"""
        <tr>
          <td>{escape(row.get('model_id', ''))}</td>
          <td>{escape(row.get('provider', ''))}</td>
          <td>{escape(format_parameter_billions(row.get('active_parameters_billion'), is_estimated_parameter_status(row.get('parameter_value_status'))))}</td>
          <td>{escape(format_parameter_billions(row.get('total_parameters_billion'), is_estimated_parameter_status(row.get('parameter_value_status'))))}</td>
          <td>{escape(row.get('parameter_value_status', 'n.d.'))}</td>
          <td>{escape(row.get('parameter_source', 'n.d.'))}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Models</div>
          <h3>Model reference table</h3>
        </div>
      </div>
      <p class="summary-intro">This table lists the reference models used by the project, their parameter counts, and the source of the observed or estimated values.</p>
      <div class="reference-table-wrap">
        <table class="reference-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Provider</th>
              <th>Active parameters</th>
              <th>Total parameters</th>
              <th>Status</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
      <p class="summary-intro">`*` indicates an estimated parameter count rather than a provider-published value.</p>
    </section>
    """


def render_country_mix_table():
    rows = load_country_energy_mix()
    if not rows:
        return ""
    body = "".join(
        f"""
        <tr>
          <td>{escape(row.get('country_code', ''))}</td>
          <td>{escape(row.get('country_name', ''))}</td>
          <td>{escape(row.get('year', ''))}</td>
          <td>{escape(row.get('grid_carbon_intensity_gco2_per_kwh', ''))} gCO2e/kWh</td>
          <td>{escape(row.get('source_citation', 'n.d.'))}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Countries</div>
          <h3>Electricity mix reference table</h3>
        </div>
      </div>
      <p class="summary-intro">This table lists the country factors used to contextualize carbon, with the source associated with each value.</p>
      <div class="reference-table-wrap">
        <table class="reference-table">
          <thead>
            <tr>
              <th>Code</th>
              <th>Country</th>
              <th>Year</th>
              <th>Carbon intensity</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </section>
    """


def render_documentation_tab():
    return """
    <section class="tab-panel" id="tab-documentation-panel" data-tab-panel="documentation">
      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Documentation</div>
            <h3>Install and run the project</h3>
          </div>
        </div>
        <p class="summary-intro">The project combines a local dataset, an HTTP API, an MCP server, and a web calculation interface. The main scripts rely on Python 3 and the repository files.</p>
        <div class="summary-body">
          <p><strong>Prerequisites.</strong> Python 3 is required. An OpenAI key is only needed for natural-language parsing in the web interface.</p>
          <p><strong>Move into the project.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>cd "llm-environment-opendata"</code></pre>
          <p><strong>Run the HTTP API.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>python3 api/server.py</code></pre>
          <p>Default address: <code>http://127.0.0.1:8000</code></p>
          <p><strong>Run the MCP server.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>python3 mcp/server.py</code></pre>
          <p><strong>Run the web interface.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>python3 web/server.py</code></pre>
          <p>Default address: <code>http://127.0.0.1:8080</code></p>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Configuration</div>
            <h3>OpenAI key and .env files</h3>
          </div>
        </div>
        <p class="summary-intro">The web frontend moderates and structures descriptions through OpenAI. Without API configuration, the web interface cannot interpret free-form descriptions.</p>
        <div class="summary-body">
          <p>Supported locations:</p>
          <pre style="margin:0; white-space:pre-wrap;"><code>llm-environment-opendata/.env
llm-environment-opendata/web/.env</code></pre>
          <p>Minimum configuration:</p>
          <pre style="margin:0; white-space:pre-wrap;"><code>OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini</code></pre>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">API HTTP</div>
            <h3>Available endpoints</h3>
          </div>
        </div>
        <p class="summary-intro">The local API returns JSON and provides access to the corpus, model profiles, country factors, and estimators.</p>
        <div class="summary-body">
          <p><strong>Reading and exploration.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>GET /health
GET /records
GET /records/&lt;record_id&gt;
GET /sources
GET /models
GET /models/&lt;model_id&gt;
GET /market-models
GET /energy-mix
GET /energy-mix/&lt;country_code&gt;
GET /extrapolation-rules
GET /stats</code></pre>
          <p><strong>Calculations.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>POST /estimate
POST /estimate_feature
POST /predict_inference</code></pre>
          <p><strong><code>/records</code> filters.</strong> <code>phase</code>, <code>impact_category</code>, <code>study_key</code>, <code>geography</code>.</p>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Exemples</div>
            <h3>curl examples</h3>
          </div>
        </div>
        <div class="summary-body">
          <p><strong>Check service status.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>curl http://127.0.0.1:8000/health</code></pre>
          <p><strong>List inference records.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>curl "http://127.0.0.1:8000/records?phase=inference"</code></pre>
          <p><strong>Read one specific record.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>curl http://127.0.0.1:8000/records/elsworth2025_prompt_energy</code></pre>
          <p><strong>Estimate a software feature.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>curl -X POST http://127.0.0.1:8000/estimate_feature \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "demo",
    "provider": "OpenAI",
    "model_id": "gpt-4o-mini",
    "request_type": "chat",
    "input_tokens": 1200,
    "output_tokens": 350,
    "requests_per_feature": 1,
    "feature_uses_per_month": 10000,
    "months_per_year": 12,
    "country": "FR"
  }'</code></pre>
          <p><strong>Project a normalized inference scenario.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>curl -X POST http://127.0.0.1:8000/predict_inference \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "catalog-demo",
    "provider": "OpenAI",
    "model_id": "gpt-4o-mini",
    "request_type": "chat",
    "input_tokens": 1000,
    "output_tokens": 550,
    "requests_per_feature": 1,
    "feature_uses_per_month": 1,
    "months_per_year": 12,
    "country": "FR"
  }'</code></pre>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">MCP</div>
            <h3>Tools exposed to agents</h3>
          </div>
        </div>
        <p class="summary-intro">The MCP server reads from standard input and returns serialized JSON responses. It exposes corpus lookup tools and reusable estimators.</p>
        <div class="summary-body">
          <pre style="margin:0; white-space:pre-wrap;"><code>list_records
get_record
aggregate_by_phase
list_sources
list_models
list_market_models
get_model_profile
list_country_energy_mix
get_country_energy_mix
list_extrapolation_rules
estimate_externalities
estimate_feature_externalities
predict_inference_externalities</code></pre>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Data</div>
            <h3>Dataset, validation, and exports</h3>
          </div>
        </div>
        <p class="summary-intro">The CSV remains the canonical dataset source. The JSON file is a convenience export for local or software reuse.</p>
        <div class="summary-body">
          <p><strong>Main files.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>data/records.csv
data/records.json
schema/llm_environment_record.schema.json
data/models.csv
data/market_models.csv
data/country_energy_mix.csv
data/extrapolation_rules.csv</code></pre>
          <p><strong>Validation and export.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>python3 scripts/validate_dataset.py
python3 scripts/export_json.py</code></pre>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Publication</div>
            <h3>LaTeX compilation</h3>
          </div>
        </div>
        <p class="summary-intro">The associated scientific paper is maintained in the neighboring <code>llm-environment-opendata-paper</code> folder. PDF generation requires a LaTeX installation providing <code>pdflatex</code>.</p>
        <div class="summary-body">
          <pre style="margin:0; white-space:pre-wrap;"><code>cd "../llm-environment-opendata-paper"
pdflatex -interaction=nonstopmode -halt-on-error llm_environment_opendata_paper.tex</code></pre>
        </div>
      </section>

      <section class="panel summary-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">License</div>
            <h3>Reuse conditions</h3>
          </div>
        </div>
        <div class="summary-body">
          <p>The software is distributed under the GNU GPL. Any redistribution or modification must remain compatible with the obligations of that free-software license.</p>
          <p>The dataset, comparability metadata, and scientific texts keep their own documentary constraints, especially regarding citation of the underlying sources.</p>
          <p>For a public release of the repository, it is recommended to keep the full license text and author notices explicitly in the project.</p>
        </div>
      </section>
    </section>
    """.replace("__PROJECT_PAPER_BIBTEX__", escape(PROJECT_PAPER_BIBTEX))


def format_market_country_status(value):
    labels = {
        "multi_region": "Documented multi-region",
        "documented_multi_region": "Documented multi-region",
        "self_hosted_variable": "Varies by hosting provider",
        "provider_country_proxy": "Provider-country proxy",
        "screening_proxy": "Screening proxy",
        "comparative_reference": "Comparative reference country",
        "documented_region_proxy": "Documented region, country retained as reference",
        "non_specified": "Not specified",
    }
    return labels.get(value, value or "n.d.")


def format_market_parameter_display(row):
    active = str(row.get("active_parameters_billion", "") or "").strip()
    total = str(row.get("total_parameters_billion", "") or "").strip()
    estimated = is_estimated_parameter_status(row.get("parameter_value_status"))
    if active and total and active != total:
        active_text = format_parameter_billions(active, estimated)
        total_text = format_parameter_billions(total, estimated)
        return f"{active_text} active / {total_text} total"
    if active:
        return format_parameter_billions(active, estimated)
    if total:
        return format_parameter_billions(total, estimated)
    return "n.d."


def market_parameter_sort_value(row):
    for key in ("active_parameters_billion", "total_parameters_billion"):
        raw = str(row.get(key, "") or "").strip()
        if not raw:
            continue
        try:
            return str(float(raw))
        except ValueError:
            continue
    return ""


def render_market_models_table(records):
    rows = build_market_model_predictions(records)
    if not rows:
        return ""
    standard_scenario = rows[0].get("standard_scenario", {}) if rows else {}
    requests_per_hour = standard_scenario.get("requests_per_hour", 0)
    reading_wpm = standard_scenario.get("reading_words_per_minute", 0)
    words_per_token = standard_scenario.get("words_per_token", 0)
    chart_rows = []
    body = []
    for row in rows:
        parameter_title = escape(str(row.get("parameter_source", "") or "Source not specified"))
        server_title = escape(str(row.get("server_country_source", "") or "Source not specified"))
        estimation_title = escape(str(row.get("estimation_country_source", "") or "Source not specified"))
        method_map = row.get("method_results_by_id") or {}
        prompt_method = method_map.get("prompt_query_average") or {}
        page_method = method_map.get("page_average") or {}
        prompt_energy = format_central_display(prompt_method.get("annual_energy_wh", {}), "energy") if prompt_method else "n.d."
        page_energy = format_central_display(page_method.get("annual_energy_wh", {}), "energy") if page_method else "n.d."
        prompt_carbon = format_central_display(prompt_method.get("annual_carbon_gco2e", {}), "carbon") if prompt_method else "n.d."
        page_carbon = format_central_display(page_method.get("annual_carbon_gco2e", {}), "carbon") if page_method else "n.d."
        chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": row.get("provider", ""),
                "kind": "model",
                "prompt_energy_wh": float((prompt_method.get("annual_energy_wh") or {}).get("central", 0.0) or 0.0),
                "page_energy_wh": float((page_method.get("annual_energy_wh") or {}).get("central", 0.0) or 0.0),
                "prompt_carbon_gco2e": float((prompt_method.get("annual_carbon_gco2e") or {}).get("central", 0.0) or 0.0),
                "page_carbon_gco2e": float((page_method.get("annual_carbon_gco2e") or {}).get("central", 0.0) or 0.0),
            }
        )
        body.append(
            f"""
            <tr>
              <td><strong>{escape(row.get('display_name', row.get('model_id', '')))}</strong><div class="method-basis">{escape(row.get('provider', ''))}</div></td>
              <td data-sort-value="{escape(market_parameter_sort_value(row), quote=True)}">{escape(format_market_parameter_display(row))}</td>
              <td>{escape(row.get('server_country', 'n.d.') or 'n.d.')}<div class="reference-locator">{escape(format_market_country_status(row.get('server_country_status')))}</div></td>
              <td>{escape(row.get('estimation_country_code', 'n.d.') or 'n.d.')}<div class="reference-locator">{escape(format_market_country_status(row.get('estimation_country_status')))}</div></td>
              <td>{escape(prompt_energy)}</td>
              <td>{escape(page_energy)}</td>
              <td>{escape(prompt_carbon)}</td>
              <td>{escape(page_carbon)}</td>
            </tr>
            """
        )

    chart_rows.extend(
        [
            {
                "label": "Lampe fluorescente 1 h",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 9.3,
                "page_energy_wh": 9.3,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
            },
            {
                "label": "Ordinateur portable 1 h",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 32.0,
                "page_energy_wh": 32.0,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
            },
            {
                "label": "Electric heater 11 min",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 275.0,
                "page_energy_wh": 275.0,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
            },
            {
                "label": "Electric heater 10 min (US mix)",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 0.0,
                "page_energy_wh": 0.0,
                "prompt_carbon_gco2e": 96.0,
                "page_carbon_gco2e": 96.0,
            },
        ]
    )

    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Comparative environmental impact of models</h3>
        </div>
      </div>
      <p class="summary-intro">The chart below shows the estimated central values for all catalog models under a standardized inference scenario corresponding to <strong>1 hour of active use</strong>: <strong>{requests_per_hour} interactions/hour</strong>, <strong>1000 input tokens</strong>, <strong>550 output tokens</strong>, and one LLM request per use. The hourly pace is derived from an average reading speed of <strong>{reading_wpm} words/min</strong> (Brysbaert, 2019) and a project convention of <strong>1 token ≈ {words_per_token} word</strong>.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Inference chart indicator">
        <button type="button" class="chart-tab-button is-active" data-model-chart-control="metric-tab" data-metric-value="energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-model-chart-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="models-impact-chart" class="models-impact-chart" data-chart-rows='{escape(json.dumps(chart_rows, ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Benchmarks integrated into the chart, all expressed over one hour or rescaled to a comparable order of magnitude: household electricity from Purdue Extension measurements (fluorescent lamp ≈ 9.3 Wh over 1 h; laptop ≈ 32 Wh over 1 h) and a 1500 W electric space heater rescaled here to 11 minutes to obtain ≈ 275 Wh; for carbon, a 10-minute electric-heater benchmark recalculated with the US electricity mix retained by the project (0.25 kWh × 384 gCO2e/kWh ≈ 96 gCO2e), closer to the order of magnitude of the highest models in the inference scenario.</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Models</div>
          <h3>{len(rows)} current models tracked by the project</h3>
        </div>
      </div>
      <p class="summary-intro">The table below compares the models tracked by the project under the same inference scenario. For each model, the application shows separately the estimate derived from the <strong>Wh/prompt|request</strong> average and the estimate derived from the <strong>Wh/page</strong> average, along with their carbon counterparts.</p>
      <div class="table-toolbar">
        <label class="table-search-label" for="market-model-search">Search for a model</label>
        <input id="market-model-search" class="table-search-input" type="search" placeholder="Example: GPT, Claude, Mistral, US, 70B" data-table-search="market-models-table">
      </div>
      <div class="reference-table-wrap">
        <table class="reference-table sortable-table" id="market-models-table">
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="0" data-sort-type="text">Model</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="1" data-sort-type="number">Parameters</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="2" data-sort-type="text">Server country</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="3" data-sort-type="text">Retained country</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="4" data-sort-type="number">Energy / h prompt-req</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="5" data-sort-type="number">Energy / h page</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="6" data-sort-type="number">Carbon / h prompt-req</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="7" data-sort-type="number">Carbon / h page</button></th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
      <p class="summary-intro">`Server country` describes the published information about where the service is hosted or, for open-weight models, the fact that hosting depends on deployment. `Retained country` is the country actually used to recalculate CO2 via the electricity mix. When the exact country is not published, the project uses an explicit screening proxy rather than presenting a location as certain.</p>
      <p class="summary-intro">`*` indicates an estimated parameter count rather than a provider-published value.</p>
    </section>
    """


def format_training_estimate(value, unit):
    if value in (None, ""):
        return "n.d."
    value = float(value)
    if unit == "Wh":
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.1f} GWh".replace(",", " ")
        if value >= 1_000_000:
            return f"{value / 1_000_000:,.1f} MWh".replace(",", " ")
        if value >= 1000:
            return f"{value / 1000:,.1f} kWh".replace(",", " ")
        return f"{value:,.0f} Wh".replace(",", " ")
    if unit == "tCO2e":
        if value >= 1000:
            return f"{value:,.0f} tCO2e".replace(",", " ")
        if value >= 100:
            return f"{value:,.1f} tCO2e".replace(",", " ")
        return f"{value:,.2f} tCO2e".replace(",", " ")
    if unit == "kL":
        if value >= 1000:
            return f"{value:,.0f} kL".replace(",", " ")
        if value >= 100:
            return f"{value:,.1f} kL".replace(",", " ")
        return f"{value:,.2f} kL".replace(",", " ")
    return f"{value:,.2f} {unit}".replace(",", " ")


def render_training_models_table(records):
    rows = build_training_market_predictions(records)
    if not rows:
        return ""
    chart_rows = []
    body = []
    for row in rows:
        results = row.get("training_results_by_id") or {}
        direct_energy = results.get("direct_training_energy") or {}
        direct_carbon = results.get("direct_training_carbon") or {}
        lifecycle_water = results.get("creation_lifecycle_water") or {}
        chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": row.get("provider", ""),
                "kind": "model",
                "direct_training_energy_wh": float(direct_energy.get("value", 0.0) or 0.0),
                "direct_training_carbon_tco2e": float(direct_carbon.get("value", 0.0) or 0.0),
            }
        )
        body.append(
            f"""
            <tr>
              <td><strong>{escape(row.get('display_name', row.get('model_id', '')))}</strong><div class="method-basis">{escape(row.get('provider', ''))}</div></td>
              <td data-sort-value="{escape(market_parameter_sort_value(row), quote=True)}">{escape(format_market_parameter_display(row))}</td>
              <td>{escape(format_training_estimate(direct_energy.get('value'), direct_energy.get('unit')))}</td>
              <td>{escape(format_training_estimate(direct_carbon.get('value'), direct_carbon.get('unit')))}</td>
            </tr>
            """
        )

    chart_rows.extend(
        [
            {
                "label": "10,000 households (annual domestic use)",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "direct_training_energy_wh": 25000000000.0,
                "direct_training_carbon_tco2e": 235.0,
            },
            {
                "label": "4,955 full commercial flights",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 104560.5,
            },
        ]
    )

    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Comparative training impacts of models</h3>
        </div>
      </div>
      <p class="summary-intro">The chart below shows the extrapolated central values for all catalog models across two training indicator families: training energy and direct training CO2e. Everyday benchmarks are inserted directly into the list to situate the orders of magnitude.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Training chart indicator">
        <button type="button" class="chart-tab-button is-active" data-training-chart-control="metric-tab" data-metric-value="direct_training_energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-training-chart-control="metric-tab" data-metric-value="direct_training_carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="training-impact-chart" class="models-impact-chart" data-training-chart-rows='{escape(json.dumps(chart_rows, ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Benchmarks integrated into the chart: household electricity for 10,000 households over one year of domestic use, i.e. ≈ 25 GWh based on an average consumption of 2,500 kWh per household (RTE, 2021 estimate), road transport from the ICCT (2025, 235 gCO2e/km for an average gasoline car), and full-flight aviation derived from Klöwer et al. (2025) from 577.97 MtCO2 and 27.45 million commercial flights observed in 2023, i.e. ≈ 104,560.5 tCO2 for 4,955 full flights.</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Models</div>
          <h3>{len(rows)} current models with estimated training impacts</h3>
        </div>
      </div>
      <p class="summary-intro">This table projects the training orders of magnitude of current models from the indicator families actually available in the literature: <strong>training energy</strong> derived from emissions when the source country is documented in the electricity-mix table, and <strong>direct training CO2e</strong>. Values are extrapolated by parameter count. Training energy therefore remains a more fragile screening reconstruction than direct carbon.</p>
      <div class="table-toolbar">
        <label class="table-search-label" for="training-model-search">Search for a model</label>
        <input id="training-model-search" class="table-search-input" type="search" placeholder="Example: GPT, Claude, 70B, Meta" data-table-search="training-models-table">
      </div>
      <div class="reference-table-wrap">
        <table class="reference-table sortable-table" id="training-models-table">
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="0" data-sort-type="text">Model</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="1" data-sort-type="number">Parameters</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="2" data-sort-type="number">Training energy</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="3" data-sort-type="number">Direct training CO2e</button></th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
      <p class="summary-intro">`*` indicates an estimated parameter count rather than a provider-published value.</p>
    </section>
    """


def factor_details(records, factor_ids):
    rows = []
    for factor_id in factor_ids:
        record = get_record(records, factor_id)
        if not record:
            continue
        rows.append(
            {
                "record_id": record["record_id"],
                "study_key": record.get("study_key", ""),
                "metric_name": record["metric_name"],
                "metric_value": record["metric_value"],
                "metric_unit": record["metric_unit"],
                "citation": record["citation"],
                "source_locator": record["source_locator"],
                "source_url": record["source_url"],
                "country_normalized": record.get("country_normalized", ""),
                "geography": record.get("geography", ""),
                "llm_normalized": record.get("llm_normalized", ""),
                "model_parameters_normalized": record.get("model_parameters_normalized", ""),
            }
        )
    return rows


def process_description(form):
    description = form.get("description", [""])[0]
    moderation = moderate_application_description_with_openai(description)
    if moderation["decision"] != "allow":
        guidance = (
            "Describe an application, feature, or workflow using an LLM, "
            "with its usage pattern, scale, or technical context."
        )
        raise OpenAIModerationError(
            f"This description does not clearly correspond to software or an LLM usage scenario that the platform can process. "
            f"{moderation['reason']} {guidance}"
        )
    parsed_payload, parser_notes, parser_meta = parse_application_description_with_openai(description)
    parser_meta["moderation"] = moderation
    parsed_payload["software_components"] = []
    parser_notes.append("The estimate was restricted to LLM consumption; non-LLM software components were excluded from the calculation.")
    apply_overrides(parsed_payload, form)
    records = load_records()
    result = estimate_feature_externalities(records, parsed_payload)
    rows = factor_details(records, result["selected_factors"])
    parser_meta["evidence"] = classify_evidence_level(parsed_payload, rows)
    return description, parsed_payload, parser_notes, parser_meta, result, rows


def persist_analysis_run(description, parsed_payload, parser_notes, parser_meta, result, factor_rows):
    entry = {
        "analysis_date": datetime.now().astimezone().isoformat(),
        "description": description,
        "parsed_payload": parsed_payload,
        "parser_notes": parser_notes or [],
        "parser_meta": parser_meta or {},
        "result": result,
        "factor_rows": factor_rows or [],
    }

    ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ANALYSIS_LOG_PATH.exists():
        try:
            current = json.loads(ANALYSIS_LOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(current, list):
                current = []
        except json.JSONDecodeError:
            current = []
    else:
        current = []

    current.append(entry)
    ANALYSIS_LOG_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_page(result=None, description="", parsed_payload=None, parser_notes=None, parser_meta=None, factor_rows=None, error_message=None):
    error_block = ""
    if error_message:
        error_block = f"""
        <section class="panel alert-panel error">
          <h2>Description cannot be processed</h2>
          <p class="lead">{escape(error_message)}</p>
          <p class="lead">The application expects a description of an application, feature, or workflow using one or more LLMs.</p>
        </section>
        """

    all_records = load_records()
    market_models_block = render_market_models_table(all_records)
    training_models_block = render_training_models_table(all_records)
    bibliography_tab = render_bibliography_tab()
    documentation_tab = render_documentation_tab()
    contact_tab = f"""
    <section class="tab-panel" id="tab-contact-panel" data-tab-panel="contact">
      <section class="panel reference-panel">
        <div class="summary-body">
          <p><strong>Arnault Pachot</strong></p>
          <p><a href="mailto:{obfuscate_email('apachot@pollitics.com')}">{obfuscate_email('apachot@pollitics.com')}</a></p>
          <p><strong>Thierry Petit</strong></p>
          <p><a href="mailto:{obfuscate_email('tpetit@pollitics.com')}">{obfuscate_email('tpetit@pollitics.com')}</a></p>
          <p class="summary-intro">Email addresses are obfuscated in the page source to reduce basic scraping.</p>
        </div>
      </section>
    </section>
    """
    cite_tab = f"""
    <section class="tab-panel" id="tab-cite-panel" data-tab-panel="cite">
      <section class="panel reference-panel">
        <div class="summary-body">
          <p><strong>Short reference.</strong></p>
          <p>Pachot, A., &amp; Petit, T. (2026, March 12). <em>An open tool for exploring and estimating the environmental footprint of large language models: Dataset, observatory, API, MCP server, and web interface.</em></p>
          <p><strong>BibTeX.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>{escape(PROJECT_PAPER_BIBTEX)}</code></pre>
          <p><a href="/downloads/llm_environment_opendata_paper.pdf">Download PDF</a> | <a href="/downloads/llm_environment_opendata_paper.bib">Download BibTeX entry</a></p>
        </div>
      </section>
    </section>
    """
    result_block = ""
    result_methods_block = ""
    if result:
        annual = result["annual_llm"]
        method_comparisons = build_method_comparisons(load_records(), parsed_payload, result)
        result_methods_block = render_method_comparisons(method_comparisons)
        evidence = (parser_meta or {}).get("evidence", {})
        method_label = {
            "parametric_extrapolation": "Parametric extrapolation",
            "literature_proxy": "Literature proxy",
            "literature_multiples": "Multi-indicator inference aggregation",
            "wh_parameter_model": "Unified Wh -> parameters model",
        }.get(result.get("method"), "Unqualified method")
        model_profile = result.get("model_profile") or {}
        country_mix = result.get("country_energy_mix") or {}
        country_resolution_label = {
            "publisher_country": "publisher country",
            "project_country": "project country",
            "fallback_reference_country": "reference country",
            "explicit_country": "explicit country",
        }.get(result.get("country_resolution"), "retained country")
        result_block = f"""
        <section class="panel result hero-card">
          <div class="summary-header">
            <div>
              <div class="summary-kicker">Calculation</div>
              <h3>Calculation details</h3>
            </div>
          </div>
          <p class="lead">Inference estimate based on source-linked scientific indicators and a traceable calculation.</p>
          <p class="scope-note">Retained scope: only LLM inference externalities are included. Model training, software-system consumption, and ancillary infrastructure are excluded from the displayed estimate.</p>
          <p class="meta-inline">Evidence level: <strong>{escape(evidence.get('label', 'Unqualified'))}</strong></p>
          <p class="meta-inline">Method: <strong>{escape(method_label)}</strong></p>
          <p class="meta-inline">Reference model: <strong>{escape(model_profile.get('model_id', parsed_payload.get('model_id', 'not specified')))}</strong>{' | Approx. active parameters: <strong>' + escape(format_parameter_billions(model_profile.get('active_parameters_billion'), is_estimated_parameter_status(model_profile.get('parameter_value_status')))) + '</strong>' if model_profile.get('active_parameters_billion') else ''}</p>
          <p class="meta-inline">Electricity mix: <strong>{escape(country_mix.get('country_code', parsed_payload.get('country', 'not specified')))}</strong> <span class="method-basis">({escape(country_resolution_label)})</span>{' | ' + escape(country_mix.get('grid_carbon_intensity_gco2_per_kwh', '')) + ' gCO2e/kWh' if country_mix.get('grid_carbon_intensity_gco2_per_kwh') else ''}</p>
          {render_assumptions_summary(result)}
          {render_method_calculation_details(method_comparisons)}
        </section>
        """

    home_tab = f"""
    <section class="tab-panel is-active" id="tab-home-panel" data-tab-panel="home">
      <header class="hero">
        <h1>An Open Tool for Exploring and Estimating the Environmental Footprint of Large Language Models</h1>
        <p class="meta-inline"><a href="/downloads/llm_environment_opendata_paper.pdf">Download the associated scientific publication PDF</a> | <a href="/downloads/llm_environment_opendata_paper.bib">Download the BibTeX entry</a></p>
      </header>
      <form class="panel" method="post" action="/" id="estimate-form">
        <label for="description">Describe your application in natural language to obtain an inference estimate, its assumptions, and its source-linked calculation details.</label>
        <textarea id="description" name="description" placeholder="Example: We have a RAG assistant on GPT-4 via API, used 4,000 times per month in France. Each request sends 2,200 input tokens and receives 500 output tokens. There is a vector database, embeddings, and logging.">{escape(description)}</textarea>
        <button type="submit" id="submit-button">
          <span class="spinner" aria-hidden="true"></span>
          <span class="default-text">Estimate application</span>
          <span class="loading-text">Estimating...</span>
        </button>
      </form>
      {result_methods_block}
      {error_block}
      {result_block}
    </section>
    """
    observatory_tab = f"""
    <section class="tab-panel" id="tab-observatory-panel" data-tab-panel="observatory">
      <nav class="subtabs" aria-label="Observatory">
        <button type="button" class="subtab-button is-active" data-subtab-target="observatory-inference">Inference</button>
        <button type="button" class="subtab-button" data-subtab-target="observatory-training">Training</button>
      </nav>

      <section class="subtab-panel is-active" id="subtab-observatory-inference-panel" data-subtab-panel="observatory-inference">
        {market_models_block}
      </section>

      <section class="subtab-panel" id="subtab-observatory-training-panel" data-subtab-panel="observatory-training">
        {training_models_block}
      </section>
    </section>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PROJECT_NAME}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']]
      }},
      chtml: {{
        displayAlign: 'left',
        displayIndent: '0'
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root {{
      --bg: #ffffff;
      --paper: #ffffff;
      --ink: #1f2430;
      --muted: #5f6773;
      --line: #d8dde6;
      --accent: #243b63;
      --accent-soft: #ffffff;
      --error: #9f2f3f;
      --shadow: none;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, "URW Palladio L", serif;
      color: var(--ink);
      background: var(--bg);
      scroll-behavior: smooth;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; padding: 32px 24px 56px; }}
    .hero {{ margin-bottom: 28px; }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 0;
      border-radius: 0;
      background: transparent;
      color: var(--accent);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(1.55rem, 3vw, 1.95rem); line-height: 1.24; font-weight: 700; color: var(--ink); }}
    h2 {{ margin: 0 0 0.7rem; font-size: 1.22rem; line-height: 1.3; font-weight: 700; color: var(--ink); }}
    h3 {{ margin: 0 0 0.7rem; font-size: 1.1rem; line-height: 1.32; font-weight: 700; color: var(--ink); }}
    h4 {{ line-height: 1.35; }}
    .subtitle {{ max-width: 720px; color: var(--muted); font-size: 1rem; line-height: 1.68; margin: 0; }}
    .panel {{
      background: var(--paper);
      border: 0;
      border-radius: 0.2rem;
      padding: 1.15rem 0;
      box-shadow: var(--shadow);
      margin-bottom: 1.5rem;
    }}
    .hero-card {{
      background: var(--paper);
    }}
    .alert-panel.error {{
      border: 0;
      background: #fff;
    }}
    textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      padding: 0.95rem 1rem;
      font: inherit;
      background: #fff;
    }}
    #description {{
      background: rgba(140, 122, 91, 0.08);
    }}
    textarea:focus, input:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: none;
    }}
    textarea {{ min-height: 200px; resize: vertical; }}
    label {{ display: block; font-size: 0.95rem; line-height: 1.45; font-weight: 600; color: var(--ink); margin-bottom: 0.55rem; }}
    button {{
      margin-top: 1rem;
      border: 1px solid var(--accent);
      border-radius: 0.2rem;
      background: var(--paper);
      color: var(--accent);
      padding: 0.78rem 1.15rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.65rem;
    }}
    button:hover {{ background: #fff; }}
    button:disabled {{
      background: #fff;
      color: #8c94a0;
      border-color: #c7ccd5;
      cursor: wait;
    }}
    .spinner {{
      width: 1rem;
      height: 1rem;
      border: 2px solid rgba(36,59,99,0.18);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }}
    .is-loading .spinner {{
      display: inline-block;
    }}
    .loading-text {{
      display: none;
    }}
    .is-loading .loading-text {{
      display: inline;
    }}
    .is-loading .default-text {{
      display: none;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .metric {{
      margin-top: 1rem;
      padding: 0.2rem 0;
      border-radius: 0.2rem;
      background: #fff;
      border: 0;
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: none;
      letter-spacing: 0.01em;
      margin-bottom: 0.35rem;
    }}
    .metric strong {{
      font-size: 1.08rem;
      line-height: 1.45;
    }}
    .submetrics-block {{
      margin-top: 1.15rem;
      padding-top: 1rem;
      border-top: 1px solid var(--line);
    }}
    .submetrics-title {{
      margin: 0 0 0.8rem;
      color: var(--ink);
      font-size: 0.96rem;
      font-weight: 700;
    }}
    .submetrics-title-secondary {{
      margin-top: 1rem;
    }}
    .submetrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.9rem;
    }}
    .submetric {{
      padding: 0.15rem 0;
      border: 0;
      border-radius: 0.2rem;
      background: #fff;
    }}
    .submetric-label {{
      display: block;
      margin-bottom: 0.35rem;
      color: var(--muted);
      font-size: 0.8rem;
      line-height: 1.35;
    }}
    .submetric strong {{
      font-size: 0.98rem;
      line-height: 1.35;
    }}
    .submetrics-native {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .submetric-native {{
      background: #fff;
    }}
    .method-panel {{
      margin-top: 1.2rem;
      padding-top: 1.05rem;
      border-top: 1px solid var(--line);
    }}
    .result-panel {{
      margin-bottom: 2rem;
      padding-top: 0;
      border-top: 0;
    }}
    .result-method-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.15rem;
    }}
    .result-method-card {{
      border: 0;
      border-radius: 0.2rem;
      background: var(--paper);
      padding: 0.2rem 0;
      box-shadow: none;
    }}
    .result-method-head {{
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: flex-start;
      margin-bottom: 0.7rem;
    }}
    .result-method-head h4 {{
      margin: 0;
      font-size: 1.02rem;
      line-height: 1.4;
    }}
    .result-method-kicker {{
      display: none;
    }}
    .result-method-refs {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 0.3rem;
      min-width: 3rem;
    }}
    .result-method-basis {{
      margin: 0 0 0.95rem;
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.62;
    }}
    .result-method-metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1rem;
    }}
    .result-method-metric {{
      padding: 0.15rem 0;
      border: 0;
      border-radius: 0.2rem;
      background: #fff;
    }}
    .result-method-label {{
      display: block;
      margin-bottom: 0.3rem;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: none;
      letter-spacing: 0.01em;
    }}
    .result-method-metric strong {{
      font-size: 1rem;
      line-height: 1.45;
    }}
    .result-method-actions {{
      margin-top: 0.9rem;
    }}
    .result-method-detail {{
      margin-top: 1.1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--line);
    }}
    .result-method-detail-card {{
      border: 0;
      border-radius: 0.2rem;
      background: var(--paper);
      padding: 0.15rem 0;
      box-shadow: none;
    }}
    .result-method-detail-card + .result-method-detail-card {{
      margin-top: 0.9rem;
    }}
    .result-method-detail .method-modal-section + .method-modal-section {{
      margin-top: 0.95rem;
      padding-top: 0.95rem;
      border-top: 1px solid var(--line);
    }}
    .method-basis {{
      margin-top: 0.25rem;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.55;
    }}
    .method-detail-button,
    .modal-close-button {{
      appearance: none;
      border: 1px solid transparent;
      background: transparent;
      color: var(--accent);
      border-radius: 0.2rem;
      padding: 0.35rem 0.1rem;
      font: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      margin: 0;
    }}
    .method-detail-button:hover,
    .modal-close-button:hover {{
      background: #fff;
    }}
    .method-modal {{
      border: 0;
      padding: 0;
      background: transparent;
      max-width: min(920px, calc(100vw - 2rem));
      width: 100%;
    }}
    .method-modal::backdrop {{
      background: rgba(255,255,255,0.96);
    }}
    .method-modal-card {{
      background: var(--paper);
      border: 0;
      border-radius: 0.9rem;
      box-shadow: var(--shadow);
      padding: 0.25rem 0;
    }}
    .method-modal-section + .method-modal-section {{
      margin-top: 1.05rem;
      padding-top: 1.05rem;
      border-top: 1px solid var(--line);
    }}
    .metric-detail {{
      margin-top: 1rem;
      border-top: 1px solid var(--line);
      padding-top: 0.85rem;
    }}
    .metric-detail-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      cursor: pointer;
      color: var(--accent);
      font-size: 0.95rem;
      font-weight: 600;
      list-style: none;
    }}
    .metric-detail-toggle::-webkit-details-marker {{
      display: none;
    }}
    .metric-detail-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 1.25rem;
      height: 1.25rem;
      border-radius: 0.2rem;
      background: transparent;
      border: 1px solid var(--line);
      color: var(--accent);
      font-weight: 700;
      line-height: 1;
    }}
    .metric-detail[open] .metric-detail-icon {{
      transform: rotate(45deg);
    }}
    .metric-detail-body {{
      margin-top: 0.7rem;
      padding: 0.15rem 0;
      border: 0;
      border-radius: 0.2rem;
      background: #fff;
    }}
    .lead {{ color: var(--muted); line-height: 1.68; margin: 0; max-width: 720px; }}
    .scope-note {{
      margin: 0.7rem 0 0;
      padding: 0.15rem 0 0.15rem 0.9rem;
      border-left: 2px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-size: 0.95rem;
      line-height: 1.65;
    }}
    .meta-inline {{
      margin: 0.5rem 0 0;
      color: var(--muted);
      font-size: 0.94rem;
    }}
    .summary-panel {{
      background: var(--paper);
    }}
    .tabs {{
      display: flex;
      gap: 1rem;
      margin-bottom: 1.5rem;
      flex-wrap: wrap;
    }}
    .tab-button {{
      appearance: none;
      border: 0;
      border-bottom: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      border-radius: 0;
      padding: 0.3rem 0 0.55rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      margin: 0 1.2rem 0 0;
    }}
    .tab-button:hover {{
      border-color: rgba(36,59,99,0.35);
      background: transparent;
    }}
    .tab-button.is-active {{
      background: transparent;
      color: var(--accent);
      border-color: rgb(140, 122, 91);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.is-active {{
      display: block;
    }}
    .subtabs {{
      display: flex;
      gap: 1rem;
      margin: 0 0 1.5rem;
      flex-wrap: wrap;
    }}
    .subtab-button {{
      appearance: none;
      border: 0;
      border-bottom: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      padding: 0.25rem 0 0.45rem;
      font: inherit;
      font-size: 0.98rem;
      font-weight: 600;
      cursor: pointer;
      margin: 0 1.1rem 0 0;
    }}
    .subtab-button:hover {{
      border-color: rgba(36,59,99,0.35);
    }}
    .subtab-button.is-active {{
      color: var(--accent);
      border-color: rgb(140, 122, 91);
    }}
    .subtab-panel {{
      display: none;
    }}
    .subtab-panel.is-active {{
      display: block;
    }}
    .summary-header {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.7rem;
    }}
    .summary-kicker {{
      display: none;
    }}
    .summary-header h3 {{
      color: var(--ink);
    }}
    .result-method-head h4 {{
      color: var(--ink);
    }}
    .math-label,
    .submetrics-title,
    .table-search-label {{
      color: var(--ink);
    }}
    .summary-intro {{
      margin: 0 0 1rem;
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.68;
      font-size: 0.96rem;
    }}
    .summary-body {{
      border: 0;
      padding: 0.15rem 0;
      background: #fff;
      border-radius: 0.35rem;
      line-height: 1.78;
      font-size: 0.98rem;
    }}
    .reference-panel {{
    }}
    .table-toolbar {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin: 0 0 1rem;
    }}
    .table-search-label {{
      margin: 0;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--ink);
    }}
    .table-search-input {{
      max-width: 420px;
      padding: 0.8rem 0.9rem;
    }}
    .models-chart-panel {{
      margin-bottom: 1.5rem;
      padding: 0.2rem 0;
      border: 0;
      border-radius: 0.2rem;
      background: var(--paper);
    }}
    .models-chart-toolbar {{
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }}
    .models-chart-field {{
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      min-width: 200px;
    }}
    .models-chart-field label {{
      margin: 0;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--ink);
    }}
    .models-chart-field select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      padding: 0.7rem 0.8rem;
      font: inherit;
      background: #fff;
    }}
    .chart-tabbar {{
      display: flex;
      gap: 0.65rem;
      flex-wrap: wrap;
      margin-bottom: 1.15rem;
    }}
    .chart-tab-button {{
      margin-top: 0;
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      background: transparent;
      color: var(--muted);
      padding: 0.38rem 0.72rem;
      font-size: 0.9rem;
      font-weight: 600;
    }}
    .chart-tab-button:hover {{
      background: #fff;
      color: var(--accent);
      border-color: rgba(36,59,99,0.24);
    }}
    .chart-tab-button.is-active {{
      background: #fff;
      color: var(--accent);
      border-color: rgba(36,59,99,0.24);
    }}
    .models-impact-chart {{
      border: 0;
      border-radius: 0.2rem;
      background: rgba(140, 122, 91, 0.08);
      padding: 0.45rem 0.5rem;
      min-height: 480px;
    }}
    .models-benchmark-note {{
      margin-top: 1rem;
      font-size: 0.92rem;
      line-height: 1.65;
      color: var(--muted);
    }}
    .models-impact-chart svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .reference-table-wrap {{
      overflow-x: auto;
      border: 0;
      border-radius: 0.2rem;
      background: var(--paper);
    }}
    .reference-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.96rem;
    }}
    .reference-table th,
    .reference-table td {{
      padding: 0.9rem 0.9rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .reference-table th {{
      background: #fff;
      font-size: 0.8rem;
      text-transform: none;
      letter-spacing: 0.01em;
      color: var(--muted);
    }}
    .sort-button {{
      margin: 0;
      padding: 0;
      border: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      font-weight: 700;
      text-transform: inherit;
      letter-spacing: inherit;
      cursor: pointer;
    }}
    .sort-button:hover {{
      color: var(--accent);
      background: transparent;
    }}
    .reference-table tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .reference-table tbody tr:target td {{
      background: #fff;
    }}
    .reference-locator {{
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.55;
    }}
    .source-tag {{
      display: inline-block;
      margin-left: 0.15rem;
      padding: 0.05rem 0.35rem;
      border-radius: 0.25rem;
      background: #fff;
      border: 0;
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      text-decoration: none;
      vertical-align: baseline;
    }}
    .source-tag:hover {{
      background: #fff;
      text-decoration: none;
    }}
    .math-panel {{
      background: #fff;
    }}
    .assumptions-box {{
      margin-bottom: 1rem;
      padding: 0.15rem 0;
      border: 0;
      border-radius: 0.2rem;
      background: #fff;
    }}
    .assumptions-box-compact {{
      margin-top: 0.9rem;
      margin-bottom: 0;
    }}
    #tab-documentation-panel pre {{
      margin: 0;
      padding: 0.85rem 0.95rem;
      border-radius: 0.2rem;
      background: rgba(140, 122, 91, 0.08);
      overflow-x: auto;
    }}
    #tab-documentation-panel pre code {{
      display: block;
      background: transparent;
      border: 0;
      padding: 0;
      white-space: pre-wrap;
    }}
    #tab-cite-panel pre {{
      margin: 0;
      padding: 0.85rem 0.95rem;
      border-radius: 0.2rem;
      background: rgba(140, 122, 91, 0.08);
      overflow-x: auto;
    }}
    #tab-cite-panel pre code {{
      display: block;
      background: transparent;
      border: 0;
      padding: 0;
      white-space: pre-wrap;
    }}
    .assumptions-list {{
      margin: 0;
      padding-left: 1.15rem;
      color: var(--ink);
      line-height: 1.7;
    }}
    .assumptions-list li + li {{
      margin-top: 0.35rem;
    }}
    .math-label {{
      display: block;
      margin-bottom: 0.6rem;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: none;
      letter-spacing: 0.01em;
      color: var(--ink);
    }}
    .math-formula {{
      margin-bottom: 0.85rem;
      font-size: 1.02rem;
    }}
    mjx-container[jax="CHTML"][display="true"] {{
      text-align: left !important;
      margin: 0.45rem 0 0.85rem 0 !important;
    }}
    .math-formula code,
    .metric-detail-body code,
    .math-panel code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #fff;
      border: 0;
      border-radius: 0.45rem;
      padding: 0.2rem 0.4rem;
    }}
    .sourced-value {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      flex-wrap: wrap;
    }}
    .inline-ref {{
      display: inline-block;
      padding: 0.1rem 0.38rem;
      border-radius: 0.25rem;
      background: #fff;
      border: 0;
      color: var(--accent);
      font-size: 0.74rem;
      font-weight: 700;
      text-decoration: none;
    }}
    .inline-ref:hover {{
      background: #fff;
      text-decoration: none;
    }}
    .metric-detail-body p,
    .math-panel p {{
      margin: 0;
      color: var(--ink);
      line-height: 1.75;
    }}
    .extrapolation-list {{
      margin: 0 0 0.75rem;
      padding-left: 1.2rem;
      color: var(--ink);
      line-height: 1.68;
      font-size: 0.94rem;
    }}
    .extrapolation-list li + li {{
      margin-top: 0.35rem;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 900px) {{
      .wrap {{ padding: 24px 18px 40px; }}
      .submetrics {{ grid-template-columns: 1fr; }}
      .summary-header {{ flex-direction: column; align-items: flex-start; }}
      .result-method-metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
  <body>
  <main class="wrap">
    <nav class="tabs" aria-label="Main navigation">
      <button type="button" class="tab-button is-active" data-tab-target="home">Home</button>
      <button type="button" class="tab-button" data-tab-target="observatory">Observatory</button>
      <button type="button" class="tab-button" data-tab-target="documentation">Documentation</button>
      <button type="button" class="tab-button" data-tab-target="cite">Cite</button>
      <button type="button" class="tab-button" data-tab-target="contact">Contact</button>
      <button type="button" class="tab-button" data-tab-target="bibliography">References</button>
    </nav>

    {home_tab}
    {observatory_tab}
    {documentation_tab}
    {cite_tab}
    {contact_tab}
    {bibliography_tab}
  </main>
  <script>
    const estimateForm = document.getElementById('estimate-form');
    const submitButton = document.getElementById('submit-button');
    const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
    const tabPanels = Array.from(document.querySelectorAll('[data-tab-panel]'));
    const subtabButtons = Array.from(document.querySelectorAll('[data-subtab-target]'));
    const subtabPanels = Array.from(document.querySelectorAll('[data-subtab-panel]'));
    const searchInputs = Array.from(document.querySelectorAll('[data-table-search]'));
    const sortButtons = Array.from(document.querySelectorAll('[data-sort-table]'));
    const modelsChart = document.getElementById('models-impact-chart');
    const chartControls = Array.from(document.querySelectorAll('[data-model-chart-control="metric-tab"]'));
    const trainingChart = document.getElementById('training-impact-chart');
    const trainingChartControls = Array.from(document.querySelectorAll('[data-training-chart-control="metric-tab"]'));
    if (estimateForm && submitButton) {{
      estimateForm.addEventListener('submit', function () {{
        submitButton.disabled = true;
        submitButton.classList.add('is-loading');
      }});
    }}
    tabButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        const target = button.getAttribute('data-tab-target');
        tabButtons.forEach((item) => item.classList.toggle('is-active', item === button));
        tabPanels.forEach((panel) => {{
          panel.classList.toggle('is-active', panel.getAttribute('data-tab-panel') === target);
        }});
      }});
    }});
    subtabButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        const target = button.getAttribute('data-subtab-target');
        subtabButtons.forEach((item) => item.classList.toggle('is-active', item === button));
        subtabPanels.forEach((panel) => {{
          panel.classList.toggle('is-active', panel.getAttribute('data-subtab-panel') === target);
        }});
      }});
    }});
    searchInputs.forEach((input) => {{
      input.addEventListener('input', function () {{
        const tableId = input.getAttribute('data-table-search');
        const table = document.getElementById(tableId);
        if (!table) return;
        const query = input.value.trim().toLowerCase();
        const rows = Array.from(table.querySelectorAll('tbody tr'));
        rows.forEach((row) => {{
          const haystack = row.textContent.toLowerCase();
          row.style.display = (!query || haystack.includes(query)) ? '' : 'none';
        }});
      }});
    }});
    sortButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        const tableId = button.getAttribute('data-sort-table');
        const table = document.getElementById(tableId);
        if (!table) return;
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        const index = Number(button.getAttribute('data-sort-index'));
        const sortType = button.getAttribute('data-sort-type') || 'text';
        const currentDirection = button.getAttribute('data-sort-direction') || 'desc';
        const nextDirection = currentDirection === 'asc' ? 'desc' : 'asc';
        sortButtons
          .filter((item) => item.getAttribute('data-sort-table') === tableId)
          .forEach((item) => item.setAttribute('data-sort-direction', ''));
        button.setAttribute('data-sort-direction', nextDirection);
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const numericValue = (text) => {{
          const normalized = text.replace(/[^0-9.,-]/g, '').replace(',', '.');
          const match = normalized.match(/-?[0-9]+(?:\\.[0-9]+)?/);
          return match ? Number(match[0]) : Number.NEGATIVE_INFINITY;
        }};
        rows.sort((rowA, rowB) => {{
          const cellA = rowA.children[index];
          const cellB = rowB.children[index];
          const textA = cellA ? (cellA.getAttribute('data-sort-value') || cellA.textContent.trim()) : '';
          const textB = cellB ? (cellB.getAttribute('data-sort-value') || cellB.textContent.trim()) : '';
          let comparison = 0;
          if (sortType === 'number') {{
            comparison = numericValue(textA) - numericValue(textB);
          }} else {{
            comparison = textA.localeCompare(textB, 'fr', {{ sensitivity: 'base' }});
          }}
          return nextDirection === 'asc' ? comparison : -comparison;
        }});
        rows.forEach((row) => tbody.appendChild(row));
      }});
    }});
    const formatChartValue = (value, metric) => {{
      if (metric === 'energy') {{
        if (value >= 1000) return `${{(value / 1000).toFixed(1)}} kWh`;
        if (value >= 1) return `${{value.toFixed(1)}} Wh`;
        return `${{value.toFixed(4)}} Wh`;
      }}
      if (metric === 'carbon') {{
        if (value >= 1000) return `${{(value / 1000).toFixed(2)}} kgCO2e`;
        return `${{value.toFixed(1)}} gCO2e`;
      }}
      if (metric === 'water') {{
        if (value >= 1000) return `${{(value / 1000).toFixed(1)}} L`;
        return `${{value.toFixed(1)}} mL`;
      }}
      return String(value);
    }};
    const buildChartMarkup = (rows, metric, family) => {{
      if (!rows.length) {{
        return '<p class="lead">No data available for this selection.</p>';
      }}
      const key = `${{family}}_${{metric}}_${{metric === 'energy' ? 'wh' : metric === 'carbon' ? 'gco2e' : 'ml'}}`;
      const sorted = rows
        .map((row) => ({{
          label: row.label,
          provider: row.provider,
          value: Number(row[key] || 0),
          kind: row.kind || 'model',
        }}))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value);
      if (!sorted.length) {{
        return '<p class="lead">No usable value available for this selection.</p>';
      }}
      const maxValue = sorted[0].value || 1;
      const barHeight = 26;
      const rowGap = 18;
      const chartWidth = 980;
      const labelWidth = 320;
      const valueWidth = 150;
      const barStart = labelWidth + 12;
      const barMaxWidth = chartWidth - labelWidth - valueWidth - 40;
      const chartHeight = sorted.length * (barHeight + rowGap) + 24;
      const bars = sorted.map((row, index) => {{
        const y = 12 + index * (barHeight + rowGap);
        const width = Math.max(2, (row.value / maxValue) * barMaxWidth);
        const valueText = formatChartValue(row.value, metric);
        const fill = row.kind === 'reference' ? '#8c7a5b' : '#3f5a49';
        return `
          <text x="0" y="${{y + 17}}" font-size="13" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 31}}" font-size="11" fill="#6c757d">${{row.provider}}</text>
          <rect x="${{barStart}}" y="${{y}}" width="${{width}}" height="${{barHeight}}" rx="4" fill="${{fill}}"></rect>
          <text x="${{barStart + width + 10}}" y="${{y + 17}}" font-size="12" fill="#212529">${{valueText}}</text>
        `;
      }}).join('');
      const titleMetric = metric === 'energy' ? 'Energy' : metric === 'carbon' ? 'Carbon' : 'Water';
      const titleFamily = family === 'prompt' ? 'prompt|request' : 'page';
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">Comparison of estimated central values for the <strong>${{titleMetric}}</strong> indicator, using the <strong>${{titleFamily}}</strong> family.</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="Comparative chart of models">${{bars}}</svg>
      `;
    }};
    const renderModelsChart = () => {{
      if (!modelsChart) return;
      const metricControl = document.querySelector('[data-model-chart-control="metric-tab"].is-active');
      const metric = metricControl ? metricControl.getAttribute('data-metric-value') : 'energy';
      const family = 'prompt';
      let rows = [];
      try {{
        rows = JSON.parse(modelsChart.getAttribute('data-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      modelsChart.innerHTML = buildChartMarkup(rows, metric, family);
    }};
    chartControls.forEach((control) => {{
      control.addEventListener('click', () => {{
        chartControls.forEach((item) => {{
          const isActive = item === control;
          item.classList.toggle('is-active', isActive);
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        renderModelsChart();
      }});
    }});
    renderModelsChart();
    const formatTrainingChartValue = (value, metric) => {{
      if (metric === 'direct_training_energy') {{
        if (value >= 1_000_000_000) return `${{(value / 1_000_000_000).toFixed(1)}} GWh`;
        if (value >= 1_000_000) return `${{(value / 1_000_000).toFixed(1)}} MWh`;
        if (value >= 1000) return `${{(value / 1000).toFixed(1)}} kWh`;
        return `${{value.toFixed(0)}} Wh`;
      }}
      if (value >= 1000) return `${{value.toFixed(0)}} tCO2e`;
      if (value >= 100) return `${{value.toFixed(1)}} tCO2e`;
      return `${{value.toFixed(2)}} tCO2e`;
    }};
    const buildTrainingChartMarkup = (rows, metric) => {{
      if (!rows.length) {{
        return '<p class="lead">No data available for this selection.</p>';
      }}
      const key = metric === 'direct_training_energy'
        ? 'direct_training_energy_wh'
        : 'direct_training_carbon_tco2e';
      const sorted = rows
        .map((row) => ({{
          label: row.label,
          provider: row.provider,
          value: Number(row[key] || 0),
          kind: row.kind || 'model',
        }}))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value);
      if (!sorted.length) {{
        return '<p class="lead">No usable value available for this selection.</p>';
      }}
      const maxValue = sorted[0].value || 1;
      const barHeight = 26;
      const rowGap = 18;
      const chartWidth = 980;
      const labelWidth = 320;
      const valueWidth = 150;
      const barStart = labelWidth + 12;
      const barMaxWidth = chartWidth - labelWidth - valueWidth - 40;
      const chartHeight = sorted.length * (barHeight + rowGap) + 24;
      const bars = sorted.map((row, index) => {{
        const y = 12 + index * (barHeight + rowGap);
        const width = Math.max(2, (row.value / maxValue) * barMaxWidth);
        const valueText = formatTrainingChartValue(row.value, metric);
        const fill = row.kind === 'reference' ? '#8c7a5b' : '#3f5a49';
        return `
          <text x="0" y="${{y + 17}}" font-size="13" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 31}}" font-size="11" fill="#6c757d">${{row.provider}}</text>
          <rect x="${{barStart}}" y="${{y}}" width="${{width}}" height="${{barHeight}}" rx="4" fill="${{fill}}"></rect>
          <text x="${{barStart + width + 10}}" y="${{y + 17}}" font-size="12" fill="#212529">${{valueText}}</text>
        `;
      }}).join('');
      const titleMetric = metric === 'direct_training_energy'
        ? "Training energy"
        : "Direct training CO2e";
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">Comparison of extrapolated central values for the <strong>${{titleMetric}}</strong> indicator.</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="Comparative chart of training impacts">${{bars}}</svg>
      `;
    }};
    const renderTrainingChart = () => {{
      if (!trainingChart) return;
      const metricControl = document.querySelector('[data-training-chart-control="metric-tab"].is-active');
      const metric = metricControl ? metricControl.getAttribute('data-metric-value') : 'direct_training_energy';
      let rows = [];
      try {{
        rows = JSON.parse(trainingChart.getAttribute('data-training-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      trainingChart.innerHTML = buildTrainingChartMarkup(rows, metric);
    }};
    trainingChartControls.forEach((control) => {{
      control.addEventListener('click', () => {{
        trainingChartControls.forEach((item) => {{
          const isActive = item === control;
          item.classList.toggle('is-active', isActive);
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        renderTrainingChart();
      }});
    }});
    renderTrainingChart();
  </script>
</body>
</html>
"""
class Handler(BaseHTTPRequestHandler):
    def _write_bytes(self, body, content_type, filename=None, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Last-Modified", formatdate(usegmt=True))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/downloads/llm_environment_opendata_paper.pdf":
            if PAPER_PDF_PATH.exists():
                self._write_bytes(
                    PAPER_PDF_PATH.read_bytes(),
                    "application/pdf",
                    filename="llm_environment_opendata_paper.pdf",
                )
                return
            self._write_html(render_page(error_message="Publication PDF not found."), status=404)
            return
        if self.path == "/downloads/llm_environment_opendata_paper.bib":
            self._write_bytes(
                PROJECT_PAPER_BIBTEX.encode("utf-8"),
                "application/x-bibtex; charset=utf-8",
                filename="llm_environment_opendata_paper.bib",
            )
            return
        self._write_html(render_page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw)
        description = form.get("description", [""])[0]

        try:
            description, parsed_payload, parser_notes, parser_meta, result, rows = process_description(form)
        except (OpenAIModerationError, OpenAIParserError) as exc:
            self._write_html(render_page(description=description, error_message=str(exc)), status=502)
            return

        persist_analysis_run(description, parsed_payload, parser_notes, parser_meta, result, rows)

        self._write_html(
            render_page(
                result=result,
                description=description,
                parsed_payload=parsed_payload,
                parser_notes=parser_notes,
                parser_meta=parser_meta,
                factor_rows=rows,
            )
        )


def apply_overrides(payload, form):
    return payload


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8080), Handler)
    print(f"{PROJECT_NAME} web app running on http://127.0.0.1:8080")
    server.serve_forever()
