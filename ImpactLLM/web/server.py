#!/usr/bin/env python3
from datetime import datetime
import json
import os
import re
import sys
from email.utils import formatdate
from functools import lru_cache
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
    market_architecture_factor,
    market_context_factor,
    market_modality_factor,
    market_serving_factor,
    parse_market_bool,
    training_parameter_count_billion,
    training_tokens_estimate_trillion,
    to_float,
    wh_to_gco2e,
    wh_to_liters,
)
from core.openai_parser import (
    OpenAIModerationError,
    OpenAIParserError,
    moderate_application_description_with_openai,
    parse_application_description_with_openai,
)


PROJECT_NAME = "ImpactLLM"
ANALYSIS_LOG_PATH = ROOT / "data" / "analysis_runs.json"
LOGO_PATH = ROOT / "web" / "impactllm-logo.svg"
LOGO_MARK_PATH = ROOT / "web" / "impactllm-mark.svg"
REFERENCE_PAGE_TOKENS = 750.0
DEFAULT_PROMPT_TOKENS = 1550.0
PROJECT_PAPER_BIBTEX = """@misc{impactllm_screening_2026,
  title = {Transparent Screening for LLM Inference and Training Impacts},
  author = {Pachot, Arnault and Petit, Thierry},
  year = {2026},
  month = mar,
  note = {Working paper},
  url = {https://dev.emotia.com/impact-llm/downloads/ImpactLLM_paper.pdf}
}"""


def normalize_url_prefix(prefix):
    prefix = (prefix or "").strip()
    if not prefix:
        return ""
    prefix = "/" + prefix.strip("/")
    return "" if prefix == "/" else prefix


URL_PREFIX = normalize_url_prefix(os.environ.get("LLM_WEB_PREFIX", ""))


def first_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


BIB_PATH = first_existing_path(
    ROOT.parent / "ImpactLLM-paper" / "references_ImpactLLM.bib",
    ROOT.parent / "llm-environment-opendata-paper" / "references_llm_environment_opendata.bib",
)
PAPER_TEX_PATH = first_existing_path(
    ROOT.parent / "ImpactLLM-paper" / "ImpactLLM_paper.tex",
    ROOT.parent / "llm-environment-opendata-paper" / "llm_environment_opendata_paper.tex",
)
PAPER_PDF_PATH = first_existing_path(
    ROOT.parent / "ImpactLLM-paper" / "ImpactLLM_paper.pdf",
    ROOT.parent / "llm-environment-opendata-paper" / "llm_environment_opendata_paper.pdf",
)
PAPER_PREVIEW_PATH = first_existing_path(
    ROOT / "web" / "ImpactLLM_paper_preview.png",
    ROOT / "web" / "llm_environment_opendata_paper_preview.png",
)
TRAINING_DOUBLING_FIGURE_PATH = ROOT.parent / "ImpactLLM-paper" / "figures" / "linkedin_training_co2_doubling_en.png"
INFERENCE_DOUBLING_FIGURE_PATH = ROOT.parent / "ImpactLLM-paper" / "figures" / "linkedin_inference_co2_doubling_en.png"


def app_url(path="/"):
    if not path.startswith("/"):
        path = f"/{path}"
    if not URL_PREFIX:
        return path
    if path == "/":
        return f"{URL_PREFIX}/"
    return f"{URL_PREFIX}{path}"


@lru_cache(maxsize=1)
def render_logo_markup():
    if not LOGO_PATH.exists():
        return '<div class="hero-logo-fallback" aria-hidden="true">I&gt;</div>'
    return LOGO_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def render_logo_mark_markup():
    if not LOGO_MARK_PATH.exists():
        return '<span class="nav-logo-fallback" aria-hidden="true">I&gt;</span>'
    return LOGO_MARK_PATH.read_text(encoding="utf-8")
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
        "indicator": "Average gasoline car for 0.17 km",
        "value": "40.0 gCO2e",
        "citation": "International Council on Clean Transportation. (2025). Life-cycle greenhouse gas emissions from passenger cars in the European Union: A 2025 update and key factors to consider. https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/",
        "locator": "Project derivation from the ICCT benchmark of 235 gCO2e/km for an average gasoline car, rescaled here to 0.17 km to align with the order of magnitude of Claude Opus 4.1 in the inference comparison.",
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


def build_analysis_bibliography_entries(factor_rows, result):
    entries = []
    seen = set()

    for row in factor_rows or []:
        citation = format_apa_citation(row)
        url = str((row or {}).get("source_url", "") or "").strip()
        locator = str((row or {}).get("source_locator", "") or "").strip()
        entry_key = str((row or {}).get("record_id", "") or "").strip()
        dedupe_key = ("literature", entry_key, citation, url, locator)
        if not citation or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "key": entry_key or f"literature-{len(entries) + 1}",
                "label": "Observed literature value",
                "citation": citation,
                "url": url,
                "locator": locator,
            }
        )

    market_profile = (
        (result or {}).get("market_model_profile")
        or (result or {}).get("model_profile")
        or {}
    )
    characteristic_sources = [
        (
            "parameters",
            "Active-parameter characteristic",
            market_profile.get("parameter_source"),
            market_profile.get("parameter_source_url"),
            market_profile.get("notes"),
        ),
        (
            "context_profile",
            "Context-window characteristic",
            market_profile.get("context_source"),
            market_profile.get("context_source_url"),
            market_profile.get("context_window_tokens"),
        ),
        (
            "serving_profile",
            "Serving-mode characteristic",
            market_profile.get("architecture_source") or market_profile.get("context_source") or market_profile.get("modalities_source"),
            market_profile.get("architecture_source_url") or market_profile.get("context_source_url") or market_profile.get("modalities_source_url"),
            market_profile.get("serving_mode"),
        ),
        (
            "vision_profile",
            "Vision characteristic",
            market_profile.get("modalities_source"),
            market_profile.get("modalities_source_url"),
            market_profile.get("vision_support"),
        ),
    ]
    for entry_key, label, citation, url, locator in characteristic_sources:
        source_citation = str(citation or "").strip()
        source_url = str(url or "").strip()
        source_locator = str(locator or "").strip()
        dedupe_key = ("market-characteristic", entry_key, source_citation, source_url, source_locator)
        if not source_citation or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "key": entry_key,
                "label": label,
                "citation": source_citation,
                "url": source_url,
                "locator": source_locator,
            }
        )

    country_mix = (result or {}).get("country_energy_mix") or {}
    mix_citation = str(country_mix.get("source_citation", "") or "").strip()
    mix_url = str(country_mix.get("source_url", "") or "").strip()
    mix_note = str(country_mix.get("notes", "") or "").strip()
    mix_dedupe_key = ("mix", mix_citation, mix_url, mix_note)
    if mix_citation and mix_dedupe_key not in seen:
        seen.add(mix_dedupe_key)
        entries.append(
            {
                "key": "mix",
                "label": "Electricity mix",
                "citation": mix_citation,
                "url": mix_url,
                "locator": mix_note,
            }
        )

    for index, entry in enumerate(entries, start=1):
        entry["number"] = index
        entry["anchor_id"] = f"analysis-ref-{index}"
        entry["title"] = f"{entry['label']}. {entry['citation']}"
        if entry["locator"]:
            entry["title"] += f". {entry['locator']}"

    return entries


def build_analysis_bibliography_map(entries):
    return {
        str(entry.get("key", "")).strip(): entry
        for entry in entries or []
        if str(entry.get("key", "")).strip()
    }


def render_analysis_entry_ref(entry_key, entries_by_key):
    if not entry_key:
        return ""
    entry = (entries_by_key or {}).get(str(entry_key).strip())
    if not entry:
        return ""
    return (
        f'<a class="inline-ref" href="#{escape(entry["anchor_id"], quote=True)}" '
        f'title="{escape(entry["title"])}">[{entry["number"]}]</a>'
    )


def render_analysis_bibliography(entries):
    if not entries:
        return ""

    items = []
    for entry in entries or []:
        citation_html = escape(entry["citation"])
        if entry["url"]:
            citation_html = f'<a href="{escape(entry["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{citation_html}</a>'
        locator_html = f'<div class="reference-locator">{escape(entry["locator"])}</div>' if entry["locator"] else ""
        items.append(
            f"""
            <li class="analysis-bibliography-item" id="{escape(entry['anchor_id'], quote=True)}">
              <span class="analysis-bibliography-number">[{entry['number']}]</span>
              <span class="analysis-bibliography-label">{escape(entry['label'])}.</span>
              {citation_html}
              {locator_html}
            </li>
            """
        )

    return f"""
    <section class="analysis-bibliography">
      <div class="math-label">Bibliography for this analysis</div>
      <ul class="analysis-bibliography-list">
        {''.join(items)}
      </ul>
    </section>
    """


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
        "Energy is treated as the primary quantity; carbon is derived from the electricity mix of the retained country": "Energy is the primary quantity, and carbon is derived from the electricity mix of the retained country.",
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
        "A prompt/query proxy is calibrated from": "A prompt/query proxy is calibrated from",
        "prompt/query anchor(s), then adjusted by a token ratio relative to": "prompt/query anchor(s), then adjusted by a token ratio relative to",
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
        "Prompt-energy estimate calibrated on Elsworth et al. (2025) at 0.24 Wh/prompt for Gemini Apps": "Prompt-energy estimate calibrated on Elsworth et al. (2025) at 0.24 Wh/prompt for Gemini Apps.",
        "Weighted prompt compute uses input tokens + 1.8 x output tokens relative to the project reference scenario": "Weighted prompt compute uses input tokens + 1.8 x output tokens relative to the project reference scenario.",
        "Effective active parameters adjust the raw model size with context window, serving mode, modality support, and architecture overhead": "Effective active parameters adjust the raw model size with context window, serving mode, modality support, and architecture overhead.",
        "Exact market-model profile retained for": "Exact market-model profile retained for",
        "Exact market-model profile unavailable; synthetic multifactor fallback built from the target parameter estimate": "Exact market-model profile unavailable; synthetic multifactor fallback built from the target parameter estimate.",
        "Carbon recalculated with the publisher-country mix for": "Carbon is recalculated using the publisher-country electricity mix for",
        "Carbon recalculated with the project country mix for": "Carbon is recalculated using the project-country electricity mix for",
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
        "Proxy Wh/prompt|requête": "Wh/prompt|request proxy",
        "Proxy Wh/page": "Wh/page proxy",
        "Proxy prompt multi-facteurs": "Multi-factor prompt proxy",
        "Multi-factor prompt proxy": "Multi-factor prompt proxy",
        "Proxy paramétrique Wh/prompt|requête calibré sur les ancrages de la littérature, avec ajustement simple au volume de tokens.": "Parametric Wh/prompt|request proxy calibrated on literature anchors, with a simple token-volume adjustment.",
        "Proxy paramétrique Wh/page calibré sur les ancrages de génération de pages de la littérature.": "Parametric Wh/page proxy calibrated on literature page-generation anchors.",
        "Proxy de screening en énergie par prompt calibré sur Elsworth et al. (2025), puis ajusté par les paramètres actifs effectifs, les hypothèses de service, l’overhead d’architecture et un volume de tokens pondéré.": "Prompt-energy screening proxy calibrated on Elsworth et al. (2025), then adjusted by effective active parameters, serving assumptions, architecture overhead, and weighted token volume.",
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


def build_method_modal_body(method, analysis_refs=None):
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
    parameter_ref = render_analysis_entry_ref("parameters", analysis_refs)
    mix_ref = render_analysis_entry_ref("mix", analysis_refs)
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
                "<p>In the <code>Wh/prompt|request</code> family, one LLM request remains the base inference unit, then the proxy is adjusted with a simple token ratio relative to the project reference prompt. "
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
            literature_ref = render_analysis_entry_ref(anchor.get("record_id"), analysis_refs)
            token_factor = float(anchor.get("token_factor", 1.0) or 1.0)
            token_factor_block = ""
            energy_formula_suffix = ""
            energy_multiplier_suffix = ""
            if token_factor != 1.0:
                token_factor_block = (
                    "<p>Applied token factor:</p>"
                    f"<p>\\[r_T = {format_scalar(token_factor, 4)}\\]</p>"
                )
                energy_formula_suffix = " \\times r_T"
                energy_multiplier_suffix = f" \\times {format_scalar(token_factor, 4)}"
            anchor_lines.append(
                f"""
                <li>
                  <p><strong>{escape(anchor.get('source_model', 'source'))}</strong></p>
                  <p>Observed literature value: <code>{escape(anchor.get('source_energy', 'n.d.'))}</code> {literature_ref}</p>
                  <p>Source parameter count: <code>{format_scalar(anchor.get('source_params'))}B</code>. Target parameter count: <code>{format_scalar(anchor.get('target_params'))}B</code>. {parameter_ref}</p>
                  <p>Applied parameter factor:</p>
                  <p>\\[
                  r_P = \\frac{{P_t}}{{P_s}} = \\frac{{{format_scalar(anchor.get('target_params'))}}}{{{format_scalar(anchor.get('source_params'))}}} = {format_scalar(anchor.get('parameter_factor'), 4)}
                  \\]</p>
                  {token_factor_block}
                  <p>Extrapolated energy for one inference unit:</p>
                  <p>\\[
                  E_t = E_s \\times r_P{energy_formula_suffix} = {escape(anchor.get('source_energy', 'n.d.'))} \\times {format_scalar(anchor.get('parameter_factor'), 4)}{energy_multiplier_suffix} = {escape(format_range_display(anchor.get('per_request_energy', {'low':0,'high':0}), 'energy'))}
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
              <p>{'This family currently relies on a single literature anchor, so the displayed central value is a calibrated proxy rather than a cross-study average.' if int(detail.get('family_anchor_count', 0) or 0) == 1 else 'When several anchors exist in the same family, the engine computes an average energy intensity per billion parameters to obtain the central value shown in the result block.'}</p>
            </div>
            """
        )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">3. Carbon derivation from the country mix</div>
              <p>Carbon is not reused directly from the literature. It is derived from extrapolated energy using the retained country electricity mix, here <strong>{escape(target_country)}</strong> {mix_ref}.</p>
              <p>\\[
              CO2_{{unitaire}} = \\frac{{E_{{unitaire}}}}{{1000}} \\times CI_c
              \\]</p>
              <p>Avec \\(CI_c = {format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\) {mix_ref}.</p>
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

    if detail.get("kind") == "market_multifactor_prompt_proxy":
        standard_request = detail.get("standard_request") or {}
        effective_params = detail.get("effective_active_parameters_billion") or {}
        token_factor = detail.get("token_factor") or {}
        context_factor = detail.get("context_factor") or {}
        serving_factor = detail.get("serving_factor") or {}
        modality_factor = detail.get("modality_factor") or {}
        architecture_factor = detail.get("architecture_factor") or {}
        scaling_exponent = detail.get("scaling_exponent") or {}
        annual_multiplier = detail.get("annual_multiplier", annual_requests)
        literature_ref = render_analysis_entry_ref("elsworth2025_prompt_energy", analysis_refs)
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">1. Scenario input data</div>
              <p>The interpreted scenario uses <code>{format_scalar(standard_request.get('input_tokens', 0), 0)}</code> input tokens and <code>{format_scalar(standard_request.get('output_tokens', 0), 0)}</code> output tokens per call.</p>
              <p>The annual call volume is calculated as:</p>
              <p>\\[
              N_{{appels/an}} = {format_count(feature_uses_per_month)} \\times {format_scalar(months_per_year, 0)} \\times {format_scalar(requests_per_feature, 0)} = {format_count(annual_requests)}
              \\]</p>
              <p>In this method, one LLM request remains the base inference unit, then the prompt-energy anchor is adjusted with weighted token volume and multi-factor effective parameters.</p>
            </div>
            """
        )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">2. Prompt anchor and multi-factor extrapolation</div>
              <p>Observed literature anchor: <code>{escape(detail.get('reference_anchor', '0.24 Wh/prompt'))}</code> {literature_ref}</p>
              <p>Target raw parameter count: <code>{format_scalar(detail.get('target_params'))}B</code> {parameter_ref}</p>
              <p>Central effective active parameters:</p>
              <p>\\[
              P^{{eff}}_c = P_t \\times F_{{ctx}} \\times F_{{srv}} \\times F_{{mod}} \\times F_{{arch}} = {format_scalar(detail.get('target_params'))} \\times {format_scalar(context_factor.get('central'))} \\times {format_scalar(serving_factor.get('central'))} \\times {format_scalar(modality_factor.get('central'))} \\times {format_scalar(architecture_factor.get('central'))} = {format_scalar(effective_params.get('central'))}
              \\]</p>
              <p class="notranslate">Where \\(P^{{eff}}_c\\) is the central effective active-parameter proxy, \\(P_t\\) the retained raw active-parameter count, \\(F_{{ctx}}\\) the context-window factor, \\(F_{{srv}}\\) the serving-mode factor, \\(F_{{mod}}\\) the modality factor, and \\(F_{{arch}}\\) the architecture-overhead factor.</p>
              <p>See the table <a href="{app_url('/#tab-bibliography')}" target="_blank" rel="noopener noreferrer">Central screening factors retained for market models</a> in Sources for the retained screening values by model.</p>
              <p>Central token factor:</p>
              <p>\\[
              F_{{tok,c}} = {format_scalar(token_factor.get('central'), 4)}
              \\]</p>
              <p>Central per-request energy:</p>
              <p>\\[
              E_c = 0.24 \\times \\left(\\frac{{{format_scalar(effective_params.get('central'))}}}{{180}}\\right)^{{{format_scalar(scaling_exponent.get('central'), 2)}}} \\times {format_scalar(token_factor.get('central'), 4)} = {escape(format_central_display(method.get('per_request_energy_wh', {'low': 0, 'central': 0, 'high': 0}), 'energy'))}
              \\]</p>
              <p>The displayed range widens the scaling exponent and the contextual overhead factors between low, central, and high scenarios.</p>
            </div>
            """
        )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">3. Carbon derivation from the country mix</div>
              <p>Carbon is not reused directly from the literature. It is derived from extrapolated energy using the retained country electricity mix, here <strong>{escape(target_country)}</strong> {mix_ref}.</p>
              <p>\\[
              CO2_{{unitaire}} = \\frac{{E_{{unitaire}}}}{{1000}} \\times CI_c
              \\]</p>
              <p>Avec \\(CI_c = {format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\) {mix_ref}.</p>
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
        literature_ref = render_analysis_entry_ref(anchor.get("record_id"), analysis_refs)
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
              published energy <code>{escape(anchor.get('source_energy', 'n.d.'))}</code> {literature_ref},
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
          <p>Retained annual carbon: average of recalculated indicators for <strong>{escape(target_country)}</strong> {mix_ref} = <code>{escape(method['carbon'])}</code>.</p>
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
    method_source = result.get("primary_method_results") or result.get("method_results") or []
    for method in method_source:
        rows = factor_details(records, method.get("record_ids", []))
        detail = dict(method.get("detail", {}))
        methods.append(
            {
                "label": translate_method_text(method.get("label", "Method")),
                "basis": translate_method_text(method.get("basis", "")),
                "energy": format_result_card_display(method["annual_energy_wh"], "energy"),
                "carbon": format_result_card_display(method["annual_carbon_gco2e"], "carbon"),
                "per_request_energy_wh": method.get("per_request_energy_wh"),
                "per_request_carbon_gco2e": method.get("per_request_carbon_gco2e"),
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
                "model_profile": result.get("model_profile") or {},
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
          </div>
          <p class="result-method-basis">{escape(method['basis'])}</p>
          <div class="result-method-metrics">
            <div class="result-method-metric result-method-metric-energy">
              <span class="result-method-label">Annual energy</span>
              <strong>{escape(method['energy'])}</strong>
            </div>
            <div class="result-method-metric result-method-metric-carbon">
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


def render_method_calculation_details(methods, analysis_refs=None):
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
          </div>
          <p class="summary-intro">{escape(method['basis'])}</p>
          {build_method_modal_body(method, analysis_refs)}
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
                "impact_category": record.get("impact_category", ""),
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
        if str(row.get("impact_category", "")).strip().lower() == "water":
            continue
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
    factor_source_entries = []
    factor_source_index = {}

    def register_factor_source(category, citation, url):
        source_citation = str(citation or "").strip()
        source_url = str(url or "").strip()
        if source_citation and not source_url and (
            source_citation.startswith("Project screening prior")
            or source_citation.startswith("ImpactLLM method note")
            or source_citation.startswith("Project screening default")
        ):
            source_url = app_url("/downloads/ImpactLLM_paper.pdf")
        if not source_citation:
            return ""
        key = (category, source_citation, source_url)
        if key not in factor_source_index:
            number = len(factor_source_entries) + 1
            anchor_id = f"factor-source-{number}"
            factor_source_index[key] = {
                "number": number,
                "anchor_id": anchor_id,
            }
            factor_source_entries.append(
                {
                    "number": number,
                    "anchor_id": anchor_id,
                    "category": category,
                    "citation": source_citation,
                    "url": source_url,
                }
            )
        entry = factor_source_index[key]
        return f' <a class="inline-ref" href="#{escape(entry["anchor_id"], quote=True)}">[{entry["number"]}]</a>'

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
    factor_table_rows = []
    training_factor_table_rows = []
    for row in sorted(
        market_model_rows,
        key=lambda item: (
            str(item.get("provider", "") or "").lower(),
            str(item.get("display_name", "") or item.get("model_id", "") or "").lower(),
        ),
    ):
        active = to_float(row.get("active_parameters_billion"), default=None)
        if active in (None, 0):
            continue
        f_ctx = market_context_factor(row.get("context_window_tokens"), scenario="central")
        f_srv = market_serving_factor(row.get("serving_mode"), scenario="central")
        f_mod = market_modality_factor(row, scenario="central")
        f_arch = market_architecture_factor(row, scenario="central")
        p_eff = active * f_ctx * f_srv * f_mod * f_arch

        parameter_ref = register_factor_source("Active parameters", row.get("parameter_source"), row.get("parameter_source_url"))
        context_ref = register_factor_source("Context window", row.get("context_source"), row.get("context_source_url"))
        modalities_ref = register_factor_source("Vision", row.get("modalities_source"), row.get("modalities_source_url"))
        serving_ref = register_factor_source(
            "Serving mode",
            row.get("context_source") or row.get("modalities_source") or row.get("architecture_source"),
            row.get("context_source_url") or row.get("modalities_source_url") or row.get("architecture_source_url"),
        )

        factor_table_rows.append(
            f"""
        <tr>
          <td>{render_model_detail_trigger(row)}</td>
          <td>{escape(row.get('provider', 'n.d.') or 'n.d.')}</td>
          <td>{escape(format_market_parameter_display(row))}{parameter_ref}</td>
          <td>{to_float(row.get('context_window_tokens'), default=0):,.0f}{context_ref}</td>
          <td>{escape(row.get('serving_mode', 'n.d.') or 'n.d.')}{serving_ref}</td>
          <td>{'yes' if parse_market_bool(row.get('vision_support')) else 'no'}{modalities_ref}</td>
          <td>{f_ctx:.3f}</td>
          <td>{f_srv:.3f}</td>
          <td>{f_mod:.3f}</td>
          <td>{f_arch:.3f}</td>
          <td>{p_eff:.3f}B</td>
        </tr>
            """
        )
        training_params = training_parameter_count_billion(row)
        training_tokens = to_float(row.get("training_tokens_estimate_trillion"), default=training_tokens_estimate_trillion(row) or 0.0)
        training_regime = str(row.get("training_regime", "n.d.") or "n.d.")
        training_hardware = str(row.get("training_hardware_class_proxy", "n.d.") or "n.d.")
        training_multimodal = "yes" if parse_market_bool(row.get("training_multimodal")) else "no"
        training_param_ref = register_factor_source("Training parameters", row.get("parameter_source"), row.get("parameter_source_url"))
        training_tokens_ref = register_factor_source("Training tokens", row.get("training_tokens_source"), row.get("training_tokens_source_url"))
        training_regime_ref = register_factor_source("Training regime", row.get("training_regime_source"), row.get("training_regime_source_url"))
        training_modality_ref = register_factor_source("Training modality", row.get("training_multimodal_source"), row.get("training_multimodal_source_url"))
        training_hardware_ref = register_factor_source("Training hardware", row.get("training_hardware_source"), row.get("training_hardware_source_url"))
        training_factor_table_rows.append(
            f"""
        <tr>
          <td>{render_model_detail_trigger(row)}</td>
          <td>{escape(row.get('provider', 'n.d.') or 'n.d.')}</td>
          <td>{escape(format_scalar(training_params))}B{training_param_ref}</td>
          <td>{training_tokens:,.2f}T{training_tokens_ref}</td>
          <td>{escape(training_regime)}{training_regime_ref}</td>
          <td>{training_multimodal}{training_modality_ref}</td>
          <td>{escape(training_hardware)}{training_hardware_ref}</td>
          <td>{escape(str(row.get('training_f_regime_central', 'n.d.') or 'n.d.'))}</td>
          <td>{escape(str(row.get('training_f_arch_central', 'n.d.') or 'n.d.'))}</td>
          <td>{escape(str(row.get('training_f_hardware_central', 'n.d.') or 'n.d.'))}</td>
        </tr>
            """
        )
    factor_source_rows = "".join(
        f"""
        <li id="{escape(entry['anchor_id'], quote=True)}">
          <span class="analysis-bibliography-number">[{entry['number']}]</span>
          <span class="analysis-bibliography-label">{escape(entry['category'])}.</span>
          {'<a href="' + escape(entry['url'], quote=True) + '" target="_blank" rel="noopener noreferrer">' + escape(entry['citation']) + '</a>' if entry['url'] else escape(entry['citation'])}
        </li>
        """
        for entry in factor_source_entries
    )

    return f"""
    <section class="tab-panel" id="tab-bibliography-panel" data-tab-panel="bibliography">
      <section class="panel reference-panel">
        <div class="summary-header">
            <div>
            <div class="summary-kicker">Sources</div>
          </div>
        </div>
        <div class="reference-copy-block">
          <p class="summary-intro">This annex brings together the quantified reference material used in the interface, along with everyday comparison benchmarks and country factors used for carbon and water recalculation.</p>
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

        <div class="reference-subtable" id="market-screening-factors">
          <h4>Central screening factors retained for market models</h4>
          <div class="reference-copy-block">
            <p class="summary-intro">This table documents the central values retained by the project for the multi-factor prompt proxy of each catalog model: raw active parameters, context window, serving mode, modality support, the resulting central factors <code>F_ctx</code>, <code>F_srv</code>, <code>F_mod</code>, <code>F_arch</code>, and the resulting central effective active-parameter proxy <code>P_eff,c</code>. These are project screening factors, not provider-published measurements.</p>
          </div>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Active parameters</th>
                  <th>Context window</th>
                  <th>Serving mode</th>
                  <th>Vision</th>
                  <th>F_ctx</th>
                  <th>F_srv</th>
                  <th>F_mod</th>
                  <th>F_arch</th>
                  <th>P_eff,c</th>
                </tr>
              </thead>
              <tbody>
                {"".join(factor_table_rows)}
              </tbody>
            </table>
          </div>
        </div>

        <div class="reference-subtable" id="market-training-screening-factors">
          <h4>Central training screening factors retained for market models</h4>
          <div class="reference-copy-block">
            <p class="summary-intro">This table documents the central values retained by the project for the multi-factor training proxy of each catalog model: retained training parameter count, training-token prior, training regime, multimodal training flag, hardware-class proxy, and the resulting central factors <code>F_reg</code>, <code>F_arch-tr</code>, and <code>F_hw</code>. These are project screening factors, not provider-published measurements.</p>
          </div>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Retained parameters</th>
                  <th>Training tokens</th>
                  <th>Training regime</th>
                  <th>Multimodal</th>
                  <th>Hardware class</th>
                  <th>F_reg</th>
                  <th>F_arch</th>
                  <th>F_hw</th>
                </tr>
              </thead>
              <tbody>
                {"".join(training_factor_table_rows)}
              </tbody>
            </table>
          </div>
        </div>

        <div class="reference-subtable">
          <h4>Numbered source list for retained screening characteristics</h4>
          <div class="reference-copy-block">
            <p class="summary-intro">The numbered references used in the retained inference and training screening-characteristic tables are listed below.</p>
          </div>
          <ol class="analysis-bibliography-list">
            {factor_source_rows}
          </ol>
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
          <p><strong>GitHub repository.</strong> <a href="https://github.com/apachot/ImpactLLM" target="_blank" rel="noreferrer">ImpactLLM on GitHub</a></p>
          <p><strong>Prerequisites.</strong> Python 3 is required. An OpenAI key is only needed for natural-language parsing in the web interface.</p>
          <p><strong>Move into the project.</strong></p>
          <pre style="margin:0; white-space:pre-wrap;"><code>cd "ImpactLLM"</code></pre>
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
          <pre style="margin:0; white-space:pre-wrap;"><code>ImpactLLM/.env
ImpactLLM/web/.env</code></pre>
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
    "model_id": "gpt-4",
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
    "model_id": "gpt-4",
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
        <p class="summary-intro">The associated scientific paper is maintained in the neighboring <code>ImpactLLM-paper</code> folder. PDF generation requires a LaTeX installation providing <code>pdflatex</code>.</p>
        <div class="summary-body">
          <pre style="margin:0; white-space:pre-wrap;"><code>cd "../ImpactLLM-paper"
pdflatex -interaction=nonstopmode -halt-on-error ImpactLLM_paper.tex</code></pre>
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
          <p>This program is free software: you can redistribute it and/or modify it under the terms of the <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">GNU General Public License</a> as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.</p>
          <p>This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.</p>
          <p>You should have received a copy of the GNU General Public License along with this program. If not, see <a href="https://www.gnu.org/licenses/" target="_blank" rel="noopener noreferrer">https://www.gnu.org/licenses/</a>.</p>
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


def format_model_field_status(value):
    labels = {
        "observed": "Observed",
        "documented": "Documented",
        "estimated": "Estimated",
        "screening_prior": "Screening prior",
        "screening_proxy": "Screening proxy",
        "provider_country_proxy": "Provider-country proxy",
        "multi_region": "Documented multi-region",
        "documented_multi_region": "Documented multi-region",
        "self_hosted_variable": "Varies by hosting provider",
        "comparative_reference": "Comparative reference",
        "documented_region_proxy": "Documented region proxy",
        "non_specified": "Not specified",
        "api": "API",
        "open": "Open",
        "closed": "Closed",
        "hybrid": "Hybrid",
        "pretraining": "Pretraining",
        "continued_pretraining": "Continued pretraining",
        "instruction_tuning": "Instruction tuning",
        "alignment_or_rl": "Alignment / RL",
        "modern_hyperscale_gpu": "Modern hyperscale GPU",
        "mixed_gpu_cluster": "Mixed GPU cluster",
        "standard_gpu_cluster": "Standard GPU cluster",
        "older_or_unknown_cluster": "Older or unknown cluster",
    }
    return labels.get(str(value or "").strip(), str(value or "n.d.").replace("_", " ").title())


def render_model_detail_trigger(row):
    model_id = str(row.get("model_id", "") or "").strip()
    label = str(row.get("display_name", "") or row.get("model_id", "") or "n.d.")
    provider = str(row.get("provider", "") or "")
    return (
        f'<button type="button" class="model-detail-trigger" data-model-detail-key="{escape(model_id, quote=True)}">'
        f"<strong>{escape(label)}</strong>"
        f'<div class="method-basis">{escape(provider)}</div>'
        f"</button>"
    )


def render_model_detail_inline_trigger(model_id, label):
    model_key = str(model_id or "").strip()
    if not model_key:
        return f"<strong>{escape(label or 'n.d.')}</strong>"
    return (
        f'<button type="button" class="model-detail-inline-trigger" data-model-detail-key="{escape(model_key, quote=True)}">'
        f"<strong>{escape(label or model_key)}</strong>"
        f"</button>"
    )


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


def build_market_models_view(records):
    rows = build_market_model_predictions(records)
    standard_scenario = rows[0].get("standard_scenario", {}) if rows else {}
    requests_per_hour = standard_scenario.get("requests_per_hour", 0)
    reading_wpm = standard_scenario.get("reading_words_per_minute", 0)
    words_per_token = standard_scenario.get("words_per_token", 0)
    chart_rows = []
    params_chart_rows = []
    scatter_chart_rows = []
    factor_heatmap_rows = []
    uncertainty_chart_rows = []
    cross_impact_chart_rows = []
    country_mix_chart_rows = []
    release_timeline_rows = []
    body = []
    for row in rows:
        architecture_notes = str(row.get("architecture_notes", "") or "").lower()
        serving_mode = str(row.get("serving_mode", "") or "").strip().lower()
        screening_method = row.get("screening_method_id", "") or "n.d."
        provider = row.get("provider", "")
        raw_active_parameters = to_float(row.get("active_parameters_billion"), default=0.0)
        f_ctx = market_context_factor(row.get("context_window_tokens"), scenario="central")
        f_srv = market_serving_factor(row.get("serving_mode"), scenario="central")
        f_mod = market_modality_factor(row, scenario="central")
        f_arch = market_architecture_factor(row, scenario="central")
        hour_energy_range = {
            "low": to_float(row.get("screening_per_hour_energy_wh_low"), default=0.0),
            "central": to_float(row.get("screening_per_hour_energy_wh_central"), default=0.0),
            "high": to_float(row.get("screening_per_hour_energy_wh_high"), default=0.0),
        }
        hour_carbon_range = {
            "low": to_float(row.get("screening_per_hour_carbon_gco2e_low"), default=0.0),
            "central": to_float(row.get("screening_per_hour_carbon_gco2e_central"), default=0.0),
            "high": to_float(row.get("screening_per_hour_carbon_gco2e_high"), default=0.0),
        }
        request_energy_range = {
            "low": to_float(row.get("screening_per_request_energy_wh_low"), default=0.0),
            "central": to_float(row.get("screening_per_request_energy_wh_central"), default=0.0),
            "high": to_float(row.get("screening_per_request_energy_wh_high"), default=0.0),
        }
        request_carbon_range = {
            "low": to_float(row.get("screening_per_request_carbon_gco2e_low"), default=0.0),
            "central": to_float(row.get("screening_per_request_carbon_gco2e_central"), default=0.0),
            "high": to_float(row.get("screening_per_request_carbon_gco2e_high"), default=0.0),
        }
        hour_energy = format_central_display(hour_energy_range, "energy")
        hour_carbon = format_central_display(hour_carbon_range, "carbon")
        request_energy = format_central_display(request_energy_range, "energy")
        request_carbon = format_central_display(request_carbon_range, "carbon")
        chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": row.get("provider", ""),
                "kind": "model",
                "prompt_energy_wh": hour_energy_range["central"],
                "page_energy_wh": hour_energy_range["central"],
                "prompt_carbon_gco2e": hour_carbon_range["central"],
                "page_carbon_gco2e": hour_carbon_range["central"],
            }
        )
        effective_params = to_float(row.get("screening_effective_active_parameters_billion_central"), default=0.0)
        params_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": row.get("provider", ""),
                "kind": "model",
                "effective_active_parameters_billion": effective_params,
            }
        )
        factor_heatmap_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "f_ctx": f_ctx,
                "f_srv": f_srv,
                "f_mod": f_mod,
                "f_arch": f_arch,
                "p_eff_ratio": (effective_params / raw_active_parameters) if raw_active_parameters else 0.0,
            }
        )
        uncertainty_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "energy_low": hour_energy_range["low"],
                "energy_central": hour_energy_range["central"],
                "energy_high": hour_energy_range["high"],
                "carbon_low": hour_carbon_range["low"],
                "carbon_central": hour_carbon_range["central"],
                "carbon_high": hour_carbon_range["high"],
            }
        )
        scatter_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "active_parameters_billion": raw_active_parameters,
                "effective_active_parameters_billion": effective_params,
                "context_window_tokens": to_float(row.get("context_window_tokens"), default=0.0),
                "serving_mode_score": 1.0 if serving_mode == "closed" else 0.5 if serving_mode == "hybrid" else 0.0,
                "vision_support_score": 1.0 if parse_market_bool(row.get("vision_support")) else 0.0,
                "moe_score": 1.0 if "moe" in architecture_notes else 0.0,
                "reasoning_score": 1.0 if "reason" in architecture_notes else 0.0,
                "hour_energy_wh": hour_energy_range["central"],
                "hour_carbon_gco2e": hour_carbon_range["central"],
                "request_energy_wh": request_energy_range["central"],
                "request_carbon_gco2e": request_carbon_range["central"],
            }
        )
        training_energy_central = to_float(row.get("training_energy_wh_central"), default=0.0)
        training_carbon_central = to_float(row.get("training_carbon_tco2e_central"), default=0.0)
        cross_impact_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "active_parameters_billion": raw_active_parameters,
                "hour_energy_wh": hour_energy_range["central"],
                "hour_carbon_gco2e": hour_carbon_range["central"],
                "direct_training_energy_wh": training_energy_central,
                "direct_training_carbon_tco2e": training_carbon_central,
            }
        )
        country_mix_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "country_code": row.get("estimation_country_code", "n.d.") or "n.d.",
                "hour_energy_wh": hour_energy_range["central"],
                "hour_carbon_gco2e": hour_carbon_range["central"],
            }
        )
        release_date = row.get("release_date", "")
        if release_date:
            release_timeline_rows.append(
                {
                    "label": row.get("display_name", row.get("model_id", "")),
                    "provider": provider,
                    "release_date": release_date,
                    "hour_carbon_gco2e": hour_carbon_range["central"],
                }
            )
        body.append(
            f"""
            <tr>
              <td>{render_model_detail_trigger(row)}</td>
              <td data-sort-value="{escape(market_parameter_sort_value(row), quote=True)}">{escape(format_market_parameter_display(row))}</td>
              <td>{escape(row.get('estimation_country_code', 'n.d.') or 'n.d.')}<div class="reference-locator">{escape(format_market_country_status(row.get('estimation_country_status')))}</div></td>
              <td>{escape(hour_energy)}</td>
              <td>{escape(hour_carbon)}</td>
              <td>{escape(request_energy)}</td>
              <td>{escape(request_carbon)}</td>
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
                "label": "Electric heater 4.1 min",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 103.5,
                "page_energy_wh": 103.5,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
            },
            {
                "label": "Average gasoline car for 0.17 km",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "prompt_energy_wh": 0.0,
                "page_energy_wh": 0.0,
                "prompt_carbon_gco2e": 40.0,
                "page_carbon_gco2e": 40.0,
            },
        ]
    )
    return {
        "rows": rows,
        "requests_per_hour": requests_per_hour,
        "reading_wpm": reading_wpm,
        "words_per_token": words_per_token,
        "chart_rows": chart_rows,
        "params_chart_rows": params_chart_rows,
        "scatter_chart_rows": scatter_chart_rows,
        "factor_heatmap_rows": factor_heatmap_rows,
        "uncertainty_chart_rows": uncertainty_chart_rows,
        "cross_impact_chart_rows": cross_impact_chart_rows,
        "country_mix_chart_rows": country_mix_chart_rows,
        "release_timeline_rows": release_timeline_rows,
        "table_body": "".join(body),
    }


def render_market_models_charts(records):
    view = build_market_models_view(records)
    rows = view["rows"]
    if not rows:
        return ""
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Comparative environmental impact of models</h3>
        </div>
      </div>
      <div class="chart-tabbar" role="tablist" aria-label="Inference chart indicator">
        <button type="button" class="chart-tab-button is-active" data-model-chart-control="metric-tab" data-metric-value="energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-model-chart-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="models-impact-chart" class="models-impact-chart" data-chart-rows='{escape(json.dumps(view["chart_rows"], ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-chart-note">The chart below shows the estimated central values for all catalog models under a standardized inference scenario corresponding to <strong>1 hour of active use</strong>: <strong>{view["requests_per_hour"]} interactions/hour</strong>, <strong>1000 input tokens</strong>, <strong>550 output tokens</strong>, and one LLM request per use. The hourly pace is derived from an average reading speed of <strong>{view["reading_wpm"]} words/min</strong> (<a href="https://www.sciencedirect.com/science/article/pii/S0749596X19300786" target="_blank" rel="noopener noreferrer">Brysbaert, 2019</a>) and a project convention of <strong>1 token ≈ {view["words_per_token"]} word</strong>.</p>
      <p class="summary-intro models-chart-note models-benchmark-note">Benchmarks integrated into the chart, all expressed over one hour or rescaled to a comparable order of magnitude: household electricity from <a href="https://www.extension.purdue.edu/extmedia/4H/4-H-1015-W.pdf" target="_blank" rel="noopener noreferrer">Purdue Extension</a> measurements (fluorescent lamp ≈ 9.3 Wh over 1 h; laptop ≈ 32 Wh over 1 h) and a 1500 W electric space heater rescaled here to <strong>4.1 minutes</strong> to obtain ≈ <strong>103.5 Wh</strong>, close to the order of magnitude of Claude Opus 4.1 in the inference scenario; for carbon, an average gasoline car benchmark derived from the <a href="https://theicct.org/publication/electric-cars-life-cycle-analysis-emissions-europe-jul25/" target="_blank" rel="noopener noreferrer">ICCT (2025)</a> factor retained by the project (235 gCO2e/km), here rescaled to <strong>0.17 km</strong> to obtain ≈ <strong>40.0 gCO2e</strong>, close to the order of magnitude of Claude Opus 4.1 in the inference scenario.</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Trade-off</div>
          <h3>Inference vs. training impact map</h3>
        </div>
      </div>
      <p class="summary-intro">This scatter plot compares each catalog model on two axes at once: standardized inference impact over one hour on the horizontal axis and retained training impact on the vertical axis. Point size follows the retained active parameter count, while colors distinguish providers.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Inference and training trade-off metric">
        <button type="button" class="chart-tab-button is-active" data-cross-impact-control="metric-tab" data-metric-value="energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-cross-impact-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="inference-training-tradeoff-chart" class="models-impact-chart" data-cross-impact-chart-rows='{escape(json.dumps(view["cross_impact_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Inference bubble chart</h3>
        </div>
      </div>
      <p class="summary-intro">This explicit bubble chart positions each model by effective active parameters and by its retained inference impact. Bubble size reflects the retained context window, while colors distinguish providers.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Inference bubble chart indicator">
        <button type="button" class="chart-tab-button is-active" data-inference-bubble-control="metric-tab" data-metric-value="energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-inference-bubble-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="inference-bubble-chart" class="models-impact-chart" data-scatter-chart-rows='{escape(json.dumps(view["scatter_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Uncertainty</div>
          <h3>Inference uncertainty span by model</h3>
        </div>
      </div>
      <p class="summary-intro">This chart makes the project’s retained inference range explicit for each model by showing the low, central, and high values under the standardized one-hour scenario.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Inference uncertainty indicator">
        <button type="button" class="chart-tab-button is-active" data-inference-uncertainty-control="metric-tab" data-metric-value="energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-inference-uncertainty-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="inference-uncertainty-chart" class="models-impact-chart" data-inference-uncertainty-rows='{escape(json.dumps(view["uncertainty_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Inference model landscape</h3>
        </div>
      </div>
      <p class="summary-intro">This landscape view clusters the catalog models from the characteristics retained by the project for inference screening: active and effective parameters, context window, serving mode, modality support, architecture notes, and central energy and carbon outputs. Nearby points indicate models with similar retained screening profiles, not a simple one-metric ranking.</p>
      <div id="carbon-params-scatter-chart" class="models-impact-chart" data-scatter-chart-rows='{escape(json.dumps(view["scatter_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Proxy</div>
          <h3>Inference screening factor heatmap</h3>
        </div>
      </div>
      <p class="summary-intro">This heatmap exposes the central screening factors retained for each market model. It shows the four multiplicative factors used by the project’s prompt proxy and the resulting ratio between effective and raw active parameters.</p>
      <div id="inference-factor-heatmap" class="models-impact-chart" data-factor-heatmap-rows='{escape(json.dumps(view["factor_heatmap_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Inference carbon vs. parameter count</h3>
        </div>
      </div>
      <p class="summary-intro">This complementary view places models by retained active parameter count on the horizontal axis and by central inference carbon over one hour on the vertical axis, using logarithmic scaling on both axes.</p>
      <div id="carbon-params-linear-chart" class="models-impact-chart" data-scatter-chart-rows='{escape(json.dumps(view["scatter_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Sensitivity</div>
          <h3>Country-mix sensitivity</h3>
        </div>
      </div>
      <p class="summary-intro">This view compares central inference energy and carbon over one hour, while coloring each model by the retained electricity-mix country used for carbon recalculation. It helps separate model-size effects from country-mix effects.</p>
      <div id="country-mix-sensitivity-chart" class="models-impact-chart" data-country-mix-chart-rows='{escape(json.dumps(view["country_mix_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Timeline</div>
          <h3>Inference carbon by model release date</h3>
        </div>
      </div>
      <p class="summary-intro">This timeline follows the evolution of the project’s central inference CO2e estimate over time for the OpenAI, Claude, Grok, and Mistral families, using the release month of each model as the horizontal axis.</p>
      <div id="inference-release-timeline-chart" class="models-impact-chart" data-release-timeline-rows='{escape(json.dumps(view["release_timeline_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Perspective</div>
          <h3>Inference CO2 doubling view</h3>
        </div>
      </div>
      <p class="summary-intro">This discussion-oriented chart summarizes the central inference screening values for flagship GPT, Claude, and Grok models as a simple doubling-time reading. It should be read as an interpretation of the retained observatory values, not as a provider-side measurement law.</p>
      <div id="inference-doubling-timeline-chart" class="models-impact-chart" data-release-timeline-rows='{escape(json.dumps(view["release_timeline_rows"], ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Under the current central screening profile, the flagship inference series suggests a slower increase than training, because standardized usage, active compute, and per-request serving assumptions damp part of the growth that appears in total model scale.</p>
    </section>
    """


def render_market_models_table(records):
    view = build_market_models_view(records)
    rows = view["rows"]
    if not rows:
        return ""
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Models</div>
          <h3>{len(rows)} current models tracked by the project</h3>
        </div>
      </div>
      <p class="summary-intro">The table below compares the models tracked by the project under the same inference scenario. For each model, the application shows the central values produced by the project’s multi-factor prompt proxy, both per hour of standardized use and per request.</p>
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
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="2" data-sort-type="text">Retained country</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="3" data-sort-type="number">Energy / h</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="4" data-sort-type="number">Carbon / h</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="5" data-sort-type="number">Energy / request</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="6" data-sort-type="number">Carbon / request</button></th>
            </tr>
          </thead>
          <tbody>{view["table_body"]}</tbody>
        </table>
      </div>
      <p class="summary-intro">`Retained country` is the country actually used to recalculate CO2 via the electricity mix. When the exact country is not published, the project uses an explicit screening proxy rather than presenting a location as certain.</p>
      <p class="summary-intro">`*` indicates an estimated parameter count rather than a provider-published value.</p>
      <p class="summary-intro">The market-model comparison now relies on <code>market_multifactor_prompt_proxy_v1</code>: a prompt-energy screening proxy whose main prompt-level calibration anchor comes from Elsworth et al. (2025), then adjusted by active parameters, context window, serving mode, modality support, architecture overhead, and standardized token volume, and interpreted alongside other inference references.</p>
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


def build_model_detail_index(records):
    market_rows = load_market_models()
    training_rows = build_training_market_predictions(records)
    training_by_model_id = {
        str(row.get("model_id", "") or "").strip(): row
        for row in training_rows
        if str(row.get("model_id", "") or "").strip()
    }
    details = {}
    for row in market_rows:
        model_id = str(row.get("model_id", "") or "").strip()
        if not model_id:
            continue
        training_row = training_by_model_id.get(model_id, {})
        training_results = training_row.get("training_results_by_id") or {}
        energy_result = training_results.get("direct_training_energy") or {}
        carbon_result = training_results.get("direct_training_carbon") or {}
        training_energy_range = energy_result.get("range") or {}
        training_carbon_range = carbon_result.get("range") or {}
        inference_energy = {
            "low": to_float(row.get("screening_per_hour_energy_wh_low"), default=0.0),
            "central": to_float(row.get("screening_per_hour_energy_wh_central"), default=0.0),
            "high": to_float(row.get("screening_per_hour_energy_wh_high"), default=0.0),
        }
        inference_carbon = {
            "low": to_float(row.get("screening_per_hour_carbon_gco2e_low"), default=0.0),
            "central": to_float(row.get("screening_per_hour_carbon_gco2e_central"), default=0.0),
            "high": to_float(row.get("screening_per_hour_carbon_gco2e_high"), default=0.0),
        }
        training_energy = {
            "low": to_float(training_energy_range.get("low"), default=to_float(row.get("training_energy_wh_low"), default=0.0)),
            "central": to_float(training_energy_range.get("central"), default=to_float(row.get("training_energy_wh_central"), default=0.0)),
            "high": to_float(training_energy_range.get("high"), default=to_float(row.get("training_energy_wh_high"), default=0.0)),
        }
        training_carbon = {
            "low": to_float(training_carbon_range.get("low"), default=to_float(row.get("training_carbon_tco2e_low"), default=0.0)),
            "central": to_float(training_carbon_range.get("central"), default=to_float(row.get("training_carbon_tco2e_central"), default=0.0)),
            "high": to_float(training_carbon_range.get("high"), default=to_float(row.get("training_carbon_tco2e_high"), default=0.0)),
        }

        source_entries = []
        seen_sources = set()

        def register_source(label, citation, url, status=""):
            citation_text = str(citation or "").strip()
            url_text = str(url or "").strip()
            if not citation_text and not url_text:
                return
            key = (label, citation_text, url_text, status)
            if key in seen_sources:
                return
            seen_sources.add(key)
            source_entries.append(
                {
                    "label": label,
                    "citation": citation_text or "n.d.",
                    "url": url_text,
                    "status": format_model_field_status(status),
                }
            )

        register_source("Parameters", row.get("parameter_source"), row.get("parameter_source_url"), row.get("parameter_value_status"))
        register_source("Release date", row.get("release_source"), row.get("release_source_url"), "documented")
        register_source("Context window", row.get("context_source"), row.get("context_source_url"), "documented")
        register_source("Modalities", row.get("modalities_source"), row.get("modalities_source_url"), "documented")
        register_source("Architecture", row.get("architecture_source"), row.get("architecture_source_url"), "documented")
        register_source("Retained country", row.get("estimation_country_source"), row.get("estimation_country_source_url"), row.get("estimation_country_status"))
        register_source("Server country", row.get("server_country_source"), row.get("server_country_source_url"), row.get("server_country_status"))
        register_source("Inference method anchor", row.get("screening_reference_anchor"), "", row.get("screening_method_id"))
        register_source("Training tokens", row.get("training_tokens_source"), row.get("training_tokens_source_url"), row.get("training_tokens_status"))
        register_source("Training regime", row.get("training_regime_source"), row.get("training_regime_source_url"), row.get("training_regime_status"))
        register_source("Training modality", row.get("training_multimodal_source"), row.get("training_multimodal_source_url"), "documented")
        register_source("Training hardware", row.get("training_hardware_source"), row.get("training_hardware_source_url"), "screening_proxy")
        register_source("Training method anchor", row.get("training_multifactor_anchor"), "", row.get("training_multifactor_method_id"))

        details[model_id] = {
            "key": model_id,
            "display_name": str(row.get("display_name", "") or model_id),
            "model_id": model_id,
            "provider": str(row.get("provider", "") or ""),
            "market_status": format_model_field_status(row.get("market_status")),
            "serving_mode": format_model_field_status(row.get("serving_mode")),
            "release_date": str(row.get("release_date", "") or "n.d."),
            "retained_country": str(row.get("estimation_country_code", "") or "n.d."),
            "retained_country_status": format_market_country_status(row.get("estimation_country_status")),
            "server_country": str(row.get("server_country", "") or "n.d."),
            "parameter_display": format_market_parameter_display(row),
            "parameter_status": format_model_field_status(row.get("parameter_value_status")),
            "effective_active_parameters_billion": format_scalar(to_float(row.get("screening_effective_active_parameters_billion_central"), default=0.0)),
            "context_window_tokens": format_count(to_float(row.get("context_window_tokens"), default=0.0)) if to_float(row.get("context_window_tokens"), default=0.0) else "n.d.",
            "vision_support": "Yes" if parse_market_bool(row.get("vision_support")) else "No",
            "input_modalities": str(row.get("input_modalities", "") or "n.d."),
            "output_modalities": str(row.get("output_modalities", "") or "n.d."),
            "architecture_notes": str(row.get("architecture_notes", "") or "n.d."),
            "notes": str(row.get("notes", "") or "n.d."),
            "screening_method_id": str(row.get("screening_method_id", "") or "n.d."),
            "screening_reference_anchor": str(row.get("screening_reference_anchor", "") or "n.d."),
            "training_method_id": str(row.get("training_multifactor_method_id", "") or "n.d."),
            "training_multifactor_anchor": str(row.get("training_multifactor_anchor", "") or "n.d."),
            "training_tokens_estimate_trillion": format_scalar(to_float(row.get("training_tokens_estimate_trillion"), default=0.0)) if to_float(row.get("training_tokens_estimate_trillion"), default=0.0) else "n.d.",
            "training_tokens_status": format_model_field_status(row.get("training_tokens_status")),
            "training_regime": format_model_field_status(row.get("training_regime")),
            "training_regime_status": format_model_field_status(row.get("training_regime_status")),
            "training_hardware": format_model_field_status(row.get("training_hardware_class_proxy")),
            "training_multimodal": "Yes" if parse_market_bool(row.get("training_multimodal")) else "No",
            "inference": {
                "energy": {
                    "low": format_value_display(inference_energy["low"], "energy"),
                    "central": format_value_display(inference_energy["central"], "energy"),
                    "high": format_value_display(inference_energy["high"], "energy"),
                },
                "carbon": {
                    "low": format_value_display(inference_carbon["low"], "carbon"),
                    "central": format_value_display(inference_carbon["central"], "carbon"),
                    "high": format_value_display(inference_carbon["high"], "carbon"),
                },
            },
            "training": {
                "energy": {
                    "low": format_training_estimate(training_energy["low"], "Wh"),
                    "central": format_training_estimate(training_energy["central"], "Wh"),
                    "high": format_training_estimate(training_energy["high"], "Wh"),
                },
                "carbon": {
                    "low": format_training_estimate(training_carbon["low"], "tCO2e"),
                    "central": format_training_estimate(training_carbon["central"], "tCO2e"),
                    "high": format_training_estimate(training_carbon["high"], "tCO2e"),
                },
            },
            "sources": source_entries,
        }
    return details


def build_training_models_view(records):
    rows = build_training_market_predictions(records)
    chart_rows = []
    scatter_chart_rows = []
    factor_heatmap_rows = []
    uncertainty_chart_rows = []
    release_timeline_rows = []
    body = []
    regime_scores = {
        "instruction_tuning": 0.2,
        "alignment_or_rl": 0.4,
        "continued_pretraining": 0.7,
        "pretraining": 1.0,
        "unknown": 0.5,
    }
    hardware_scores = {
        "modern_hyperscale_gpu": 1.0,
        "mixed_gpu_cluster": 0.75,
        "standard_gpu_cluster": 0.5,
        "older_or_unknown_cluster": 0.25,
        "unknown": 0.5,
    }
    for row in rows:
        architecture_notes = str(row.get("architecture_notes", "") or "").lower()
        training_profile = row.get("training_proxy_profile") or {}
        training_regime = str(training_profile.get("training_regime") or row.get("training_regime") or "unknown").strip().lower()
        hardware_class = str(training_profile.get("training_hardware_class_proxy") or row.get("training_hardware_class_proxy") or "unknown").strip().lower()
        provider = row.get("provider", "")
        retained_training_params = to_float(training_profile.get("target_params_billion"), default=training_parameter_count_billion(row) or 0.0)
        retained_training_tokens = to_float(training_profile.get("target_tokens_trillion"), default=training_tokens_estimate_trillion(row) or 0.0)
        training_results = row.get("training_results_by_id") or {}
        energy_result = training_results.get("direct_training_energy") or {}
        carbon_result = training_results.get("direct_training_carbon") or {}
        central_energy_value = energy_result.get("value")
        direct_energy_value = to_float(central_energy_value, default=to_float(row.get("training_energy_wh_central"), default=0.0))
        energy_range = energy_result.get("range") or {}
        direct_energy_range = {
            "low": to_float(energy_range.get("low"), default=direct_energy_value),
            "central": to_float(energy_range.get("central"), default=direct_energy_value),
            "high": to_float(energy_range.get("high"), default=direct_energy_value),
        }
        central_carbon_value = carbon_result.get("value")
        direct_carbon_value = to_float(central_carbon_value, default=to_float(row.get("training_carbon_tco2e_central"), default=0.0))
        carbon_range = carbon_result.get("range") or {}
        direct_carbon_range = {
            "low": to_float(carbon_range.get("low"), default=direct_carbon_value),
            "central": to_float(carbon_range.get("central"), default=direct_carbon_value),
            "high": to_float(carbon_range.get("high"), default=direct_carbon_value),
        }
        chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": row.get("provider", ""),
                "kind": "model",
                "direct_training_energy_wh": direct_energy_value,
                "direct_training_carbon_tco2e": direct_carbon_value,
            }
        )
        scatter_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "active_parameters_billion": retained_training_params,
                "training_tokens_estimate_trillion": retained_training_tokens,
                "training_regime_score": regime_scores.get(training_regime, regime_scores["unknown"]),
                "training_hardware_score": hardware_scores.get(hardware_class, hardware_scores["unknown"]),
                "vision_support_score": 1.0 if parse_market_bool(row.get("vision_support")) else 0.0,
                "moe_score": 1.0 if "moe" in architecture_notes else 0.0,
                "reasoning_score": 1.0 if "reason" in architecture_notes else 0.0,
                "direct_training_energy_wh": direct_energy_value,
                "direct_training_carbon_tco2e": direct_carbon_value,
            }
        )
        factor_heatmap_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "f_reg": to_float(row.get("training_f_regime_central"), default=0.0),
                "f_arch": to_float(row.get("training_f_arch_central"), default=0.0),
                "f_hw": to_float(row.get("training_f_hardware_central"), default=0.0),
                "token_ratio": (
                    retained_training_tokens / max(retained_training_params, 1e-9)
                    if retained_training_tokens > 0 and retained_training_params > 0
                    else 0.0
                ),
            }
        )
        uncertainty_chart_rows.append(
            {
                "label": row.get("display_name", row.get("model_id", "")),
                "provider": provider,
                "low": direct_carbon_range["low"],
                "central": direct_carbon_range["central"],
                "high": direct_carbon_range["high"],
            }
        )
        release_date = row.get("release_date", "")
        if release_date and direct_carbon_value > 0:
            release_timeline_rows.append(
                {
                    "label": row.get("display_name", row.get("model_id", "")),
                    "provider": provider,
                    "release_date": release_date,
                    "direct_training_carbon_tco2e": direct_carbon_value,
                }
            )
        body.append(
            f"""
            <tr>
              <td>{render_model_detail_trigger(row)}</td>
              <td data-sort-value="{escape(market_parameter_sort_value(row), quote=True)}">{escape(format_market_parameter_display(row))}</td>
              <td>{escape(format_training_estimate(direct_energy_value, 'Wh'))}</td>
              <td>{escape(format_training_estimate(direct_carbon_value, 'tCO2e'))}</td>
            </tr>
            """
        )

    chart_rows.extend(
        [
            {
                "label": "2,760,139 households (annual domestic use)",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "direct_training_energy_wh": 6900346573084.866,
                "direct_training_carbon_tco2e": 235.0,
            },
            {
                "label": "292,210 full commercial flights",
                "provider": "Everyday benchmark",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 6166824.8704,
            },
        ]
    )
    return {
        "rows": rows,
        "chart_rows": chart_rows,
        "scatter_chart_rows": scatter_chart_rows,
        "factor_heatmap_rows": factor_heatmap_rows,
        "uncertainty_chart_rows": uncertainty_chart_rows,
        "release_timeline_rows": release_timeline_rows,
        "table_body": "".join(body),
    }


def render_training_models_charts(records):
    view = build_training_models_view(records)
    rows = view["rows"]
    if not rows:
        return ""
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Comparative training impacts of models</h3>
        </div>
      </div>
      <p class="summary-intro">The chart below shows the central values retained for all catalog models across two training indicator families: training energy and direct training CO2e. The current screening method combines retained parameter count, a training-token prior, a training-regime prior, architecture features, and a hardware-class proxy. Under these central screening assumptions, frontier models can reach very large training orders of magnitude. Everyday benchmarks are inserted directly into the list to situate those scales, not to imply direct observed equivalence.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Training chart indicator">
        <button type="button" class="chart-tab-button is-active" data-training-chart-control="metric-tab" data-metric-value="direct_training_energy" aria-selected="true">Energy</button>
        <button type="button" class="chart-tab-button" data-training-chart-control="metric-tab" data-metric-value="direct_training_carbon" aria-selected="false">Carbon</button>
      </div>
      <div id="training-impact-chart" class="models-impact-chart" data-training-chart-rows='{escape(json.dumps(view["chart_rows"], ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Benchmarks integrated into the chart: household electricity for <strong>2,760,139 households</strong> over one year of domestic use, i.e. ≈ <strong>6.90 TWh</strong> based on an average consumption of 2,500 kWh per household (RTE, 2021 estimate), and full-flight aviation derived from Klöwer et al. (2025) from 577.97 MtCO2 and 27.45 million commercial flights observed in 2023, i.e. ≈ <strong>6,166,824.9 tCO2e</strong> for <strong>292,210 full flights</strong>. These comparison points are aligned with the current central screening order of magnitude of Claude Opus 4.1 in the training chart, not with a direct provider-side measurement.</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Training model landscape</h3>
        </div>
      </div>
      <p class="summary-intro">This landscape view clusters the catalog models from the characteristics retained by the project for training screening: retained parameter count, training-token prior, training regime, hardware-class proxy, modality support, architecture notes, and central training energy and carbon outputs. Nearby points indicate similar retained screening profiles rather than a direct ranking on one axis.</p>
      <div id="training-carbon-params-scatter-chart" class="models-impact-chart" data-training-scatter-chart-rows='{escape(json.dumps(view["scatter_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Training screening factor heatmap</h3>
        </div>
      </div>
      <p class="summary-intro">This heatmap exposes the central screening factors retained for each market model in the training proxy. It shows the regime, architecture, and hardware factors together with the retained training-token ratio per parameter.</p>
      <div id="training-factor-heatmap" class="models-impact-chart" data-training-factor-heatmap-rows='{escape(json.dumps(view["factor_heatmap_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Uncertainty</div>
          <h3>Training uncertainty span by model</h3>
        </div>
      </div>
      <p class="summary-intro">This view shows the low, central, and high direct training CO2e values retained by the project for each market model. It makes explicit how widely the training proxy can vary once the parameter and token exponents and contextual factors are widened.</p>
      <div id="training-uncertainty-chart" class="models-impact-chart" data-training-uncertainty-rows='{escape(json.dumps(view["uncertainty_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Positioning</div>
          <h3>Training carbon vs. parameter count</h3>
        </div>
      </div>
      <p class="summary-intro">This complementary view places models by retained parameter count on the horizontal axis and by direct training CO2e on the vertical axis, using logarithmic scaling on both axes.</p>
      <div id="training-carbon-params-log-chart" class="models-impact-chart" data-training-scatter-chart-rows='{escape(json.dumps(view["scatter_chart_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Timeline</div>
          <h3>Training carbon by model release date</h3>
        </div>
      </div>
      <p class="summary-intro">This timeline follows the evolution of the project’s retained direct training CO2e estimate over time for the OpenAI, Claude, Grok, and Mistral families, using the release month of each model as the horizontal axis.</p>
      <div id="training-release-timeline-chart" class="models-impact-chart" data-training-release-timeline-rows='{escape(json.dumps(view["release_timeline_rows"], ensure_ascii=False), quote=True)}'></div>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Perspective</div>
          <h3>Training CO2 doubling view</h3>
        </div>
      </div>
      <p class="summary-intro">This chart compresses the central training screening values of flagship GPT, Claude, and Grok models into a simple doubling-time interpretation. It is meant as a discussion support to make structural acceleration legible, not as a claim of direct industrial telemetry.</p>
      <div id="training-doubling-timeline-chart" class="models-impact-chart" data-training-release-timeline-rows='{escape(json.dumps(view["release_timeline_rows"], ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">The apparent acceleration is stronger for training because the current screening method compounds retained parameter count, token priors, architecture effects, and hardware assumptions. The resulting doubling pace is therefore a transparent scenario reading, not a universal empirical constant.</p>
    </section>
    """


def render_training_models_table(records):
    view = build_training_models_view(records)
    rows = view["rows"]
    if not rows:
        return ""
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Models</div>
          <h3>{len(rows)} current models with estimated training impacts</h3>
        </div>
      </div>
      <p class="summary-intro">This table projects the training orders of magnitude of current models from the indicator families actually available in the literature: <strong>training energy</strong> derived from emissions when the source country is documented in the electricity-mix table, and <strong>direct training CO2e</strong>. The current screening proxy combines retained parameter count, a training-token prior, a training-regime prior, architecture features, and a hardware-class proxy. Training energy therefore remains a more fragile screening reconstruction than direct carbon.</p>
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
          <tbody>{view["table_body"]}</tbody>
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


def normalize_description_cache_key(description):
    text = str(description or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def load_analysis_runs():
    if not ANALYSIS_LOG_PATH.exists():
        return []
    try:
        current = json.loads(ANALYSIS_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return current if isinstance(current, list) else []


def find_cached_analysis(description):
    cache_key = normalize_description_cache_key(description)
    if not cache_key:
        return None
    for entry in reversed(load_analysis_runs()):
        if entry.get("description_cache_key") == cache_key:
            return entry
        if normalize_description_cache_key(entry.get("description", "")) == cache_key:
            return entry
    return None


def process_description(form):
    description = form.get("description", [""])[0]
    cached_entry = find_cached_analysis(description)
    if cached_entry:
        parser_meta = dict(cached_entry.get("parser_meta") or {})
        parser_meta["cache"] = {"hit": True}
        return (
            cached_entry.get("description", description),
            cached_entry.get("parsed_payload") or {},
            cached_entry.get("parser_notes") or [],
            parser_meta,
            cached_entry.get("result") or {},
            cached_entry.get("factor_rows") or [],
            cached_entry.get("method_comparisons") or [],
        )

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
    apply_overrides(parsed_payload, form)
    records = load_records()
    result = estimate_feature_externalities(records, parsed_payload)
    rows = factor_details(records, result["selected_factors"])
    parser_meta["evidence"] = classify_evidence_level(parsed_payload, rows)
    parser_meta["cache"] = {"hit": False}
    method_comparisons = build_method_comparisons(records, parsed_payload, result)
    return description, parsed_payload, parser_notes, parser_meta, result, rows, method_comparisons


def persist_analysis_run(description, parsed_payload, parser_notes, parser_meta, result, factor_rows, method_comparisons):
    entry = {
        "analysis_date": datetime.now().astimezone().isoformat(),
        "description": description,
        "description_cache_key": normalize_description_cache_key(description),
        "parsed_payload": parsed_payload,
        "parser_notes": parser_notes or [],
        "parser_meta": parser_meta or {},
        "result": result,
        "factor_rows": factor_rows or [],
        "method_comparisons": method_comparisons or [],
    }

    ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = load_analysis_runs()
    current.append(entry)
    ANALYSIS_LOG_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_page(result=None, description="", parsed_payload=None, parser_notes=None, parser_meta=None, factor_rows=None, error_message=None, method_comparisons=None):
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
    model_detail_index = build_model_detail_index(all_records)
    market_models_chart_block = render_market_models_charts(all_records)
    market_models_table_block = render_market_models_table(all_records)
    training_models_chart_block = render_training_models_charts(all_records)
    training_models_table_block = render_training_models_table(all_records)
    bibliography_tab = render_bibliography_tab()
    method_tab = f"""
    <section class="tab-panel" id="tab-method-panel" data-tab-panel="method">
      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
          </div>
        </div>
        <div class="reference-copy-block">
          <p class="summary-intro">ImpactLLM is designed as a transparent screening tool, not as a black-box score. The current release starts from source-linked inference anchors, then exposes a bounded multi-factor proxy rather than a hidden single-number score.</p>
        </div>
        <div class="summary-body">
          <p><strong>1. Source-linked literature anchors.</strong></p>
          <p data-i18n-html="method-anchor-body">The application-level estimator starts from published inference indicators linked to an explicit source, model, geography, and system boundary. In the current market-model release, the predictive core uses Elsworth et al. (2025) as the main prompt-level calibration anchor, with a median prompt energy of <code>0.24 Wh/prompt</code> for Gemini Apps, and is interpreted alongside other inference references such as the <em>ML.ENERGY Benchmark</em>, Ren et al. (2024), and Li et al. (2025).</p>

          <p><strong>2. A multi-factor effective-parameter proxy.</strong></p>
          <p data-i18n-html="method-proxy-body">When direct telemetry is unavailable for a target model, ImpactLLM does not rely on a raw parameter multiple alone. It builds an effective active-parameter profile from the retained model characteristics: active parameters, context window, serving mode (<code>open</code>, <code>hybrid</code>, <code>closed</code>), modality support, and architecture notes such as MoE or reasoning-oriented overheads.</p>

          <p><strong>3. Token volume remains explicit.</strong></p>
          <p data-i18n-html="method-tokens-body">The current proxy adjusts the anchor with a weighted prompt-compute volume defined from input and output tokens. Output generation is weighted more heavily than input processing, so output-heavy scenarios and repeated LLM calls raise the estimate materially.</p>
          <p data-i18n-html="method-bound-body">The current prompt-level branch is a screening proxy, not an audited benchmark. For this reason, the application returns a bounded low-central-high result rather than one falsely precise deterministic value.</p>

          <p><strong>4. Carbon derived from context.</strong></p>
          <p data-i18n-html="method-carbon-body">Carbon is not copied mechanically from the source paper. It is recalculated from the retained energy estimate using the electricity mix associated with the selected country context.</p>

          <p><strong>5. A research-oriented estimator.</strong></p>
          <p data-i18n-html="method-research-body">The result is an auditable estimate intended for comparison, software design, and methodological discussion. It is useful precisely because the assumptions, factors, and retained sources remain visible and inspectable.</p>

          <p><strong>Scientific paper</strong></p>
          <div class="paper-preview-grid">
            <a class="paper-preview-card" href="{app_url('/downloads/ImpactLLM_paper.pdf')}" target="_blank" rel="noopener noreferrer" aria-label="Open the Transparent Screening paper PDF">
              <span class="paper-preview-frame">
                <img src="{app_url('/downloads/ImpactLLM_paper_preview.png')}" alt="Preview of the first page of Transparent Screening for LLM Inference and Training Impacts" loading="lazy">
              </span>
              <span class="paper-preview-caption">Transparent Screening paper PDF</span>
            </a>
          </div>
          <p class="paper-preview-reference notranslate">Pachot, A., &amp; Petit, T. (2026, March 14). <em>Transparent Screening for LLM Inference and Training Impacts.</em> <a href="{app_url('/downloads/ImpactLLM_paper.pdf')}">{app_url('/downloads/ImpactLLM_paper.pdf')}</a></p>
        </div>
      </section>
    </section>
    """
    contact_tab = f"""
    <section class="tab-panel" id="tab-contact-panel" data-tab-panel="contact">
      <section class="panel reference-panel">
        <div class="summary-body">
          <p>We work on responsible AI with a focus on methodological rigor, traceability, and real-world decision support. Our work combines scientific research, product design, and operational deployment to make AI systems more transparent, more accountable, and more useful in practice.</p>
          <p><strong>How to cite ImpactLLM</strong></p>
          <p class="notranslate">Pachot, A., &amp; Petit, T. (2026, March 14). <em><a href="{app_url('/downloads/ImpactLLM_paper.pdf')}">Transparent Screening for LLM Inference and Training Impacts.</a></em></p>
          <p><strong>BibTeX</strong></p>
          <pre class="citation-block"><code>{escape(PROJECT_PAPER_BIBTEX)}</code></pre>
          <p><a href="{app_url('/downloads/ImpactLLM_paper.pdf')}">Download paper PDF</a> | <a href="{app_url('/downloads/ImpactLLM_paper.bib')}">Download paper BibTeX</a></p>
          <p><strong>GitHub repository</strong></p>
          <p>The project repository is available on GitHub: <a href="https://github.com/apachot/ImpactLLM" target="_blank" rel="noreferrer">https://github.com/apachot/ImpactLLM</a>.</p>
          <p><strong>License</strong></p>
          <p>This program is free software: you can redistribute it and/or modify it under the terms of the <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">GNU General Public License</a> as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.</p>
          <p>This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.</p>
          <p>You should have received a copy of the GNU General Public License along with this program. If not, see <a href="https://www.gnu.org/licenses/" target="_blank" rel="noopener noreferrer">https://www.gnu.org/licenses/</a>.</p>
          <p><strong>Arnault Pachot</strong></p>
          <p>Arnault Pachot is a researcher and entrepreneur, founder of OpenStudio and now founder of Emotia. He works on responsible digital transformation, Green IT, and decision-oriented AI systems. He co-authored the Dunod book <em>Intelligence artificielle et environnement : alliance ou nuisance ?</em>, dedicated to practical pathways for environmentally responsible AI.</p>
          <p class="notranslate"><a class="profile-link" href="https://www.linkedin.com/in/arnaultpachot/" target="_blank" rel="noopener noreferrer"><span class="profile-link-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><path d="M6.94 8.5V19H3.5V8.5h3.44zM5.22 3A2 2 0 1 1 5.2 7a2 2 0 0 1 .02-4zM20.5 13.05V19h-3.43v-5.44c0-1.37-.5-2.3-1.71-2.3-.93 0-1.49.63-1.74 1.23-.09.22-.11.52-.11.82V19H10.1s.05-9.3 0-10.5h3.41v1.49l-.02.03h.02v-.03c.45-.69 1.25-1.68 3.05-1.68 2.23 0 3.9 1.46 3.9 4.74z"/></svg></span><span>LinkedIn: Arnault Pachot</span></a></p>
          <p class="notranslate"><a class="profile-link" href="https://scholar.google.com/citations?user=aFQ2gLMAAAAJ&hl=fr" target="_blank" rel="noopener noreferrer"><span class="profile-link-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><path d="M12 3 1 9l11 6 9-4.91V17h2V9L12 3zm-7.45 9L12 8.09 19.45 12 12 15.91 4.55 12z"/><path d="M7 14.96V18c0 1.9 2.69 3.5 5 3.5s5-1.6 5-3.5v-3.04l-5 2.73-5-2.73z"/></svg></span><span>Google Scholar: Arnault Pachot</span></a></p>
          <div class="paper-preview-grid">
            <a class="paper-preview-card book-preview-card" href="https://www.dunod.com/entreprise-et-economie/intelligence-artificielle-et-environnement-alliance-ou-nuisance-ia-face-aux" target="_blank" rel="noopener noreferrer" aria-label="Open Arnault Pachot's Dunod book page">
              <span class="paper-preview-frame">
                <img class="book-cover-preview" src="https://www.dunod.com/sites/default/files/styles/principal_desktop/public/thumbnails/image/9782100835683-001-X.jpeg" alt="Cover of the Dunod book Intelligence artificielle et environnement : alliance ou nuisance ?" loading="lazy" referrerpolicy="no-referrer">
              </span>
              <span class="paper-preview-caption">Arnault Pachot's Dunod book</span>
            </a>
          </div>
          <p><strong>Thierry Petit</strong></p>
          <p>Thierry Petit is a senior AI researcher and scientific leader with more than twenty years of academic and R&amp;D experience in Europe and the United States. His work spans trustworthy AI, simulation, optimization, and decision-grade platforms. At Emotia and Pollitics, he leads the scientific direction of systems designed to remain both operationally useful and methodologically robust.</p>
          <p class="notranslate"><a class="profile-link" href="https://www.linkedin.com/in/tpetit19/" target="_blank" rel="noopener noreferrer"><span class="profile-link-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><path d="M6.94 8.5V19H3.5V8.5h3.44zM5.22 3A2 2 0 1 1 5.2 7a2 2 0 0 1 .02-4zM20.5 13.05V19h-3.43v-5.44c0-1.37-.5-2.3-1.71-2.3-.93 0-1.49.63-1.74 1.23-.09.22-.11.52-.11.82V19H10.1s.05-9.3 0-10.5h3.41v1.49l-.02.03h.02v-.03c.45-.69 1.25-1.68 3.05-1.68 2.23 0 3.9 1.46 3.9 4.74z"/></svg></span><span>LinkedIn: Thierry Petit</span></a></p>
          <p class="notranslate"><a class="profile-link" href="https://scholar.google.com/citations?hl=fr&user=7-OxukEAAAAJ" target="_blank" rel="noopener noreferrer"><span class="profile-link-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><path d="M12 3 1 9l11 6 9-4.91V17h2V9L12 3zm-7.45 9L12 8.09 19.45 12 12 15.91 4.55 12z"/><path d="M7 14.96V18c0 1.9 2.69 3.5 5 3.5s5-1.6 5-3.5v-3.04l-5 2.73-5-2.73z"/></svg></span><span>Google Scholar: Thierry Petit</span></a></p>
          <p><strong>Selected references on AI and the environment</strong></p>
          <ul class="analysis-bibliography-list">
            <li class="analysis-bibliography-item">Pachot, A., Patissier, C., &amp; Open Studio. (2022). <em>Intelligence artificielle et environnement : alliance ou nuisance ? L'IA face aux défis écologiques d'aujourd'hui et de demain</em>. Dunod. <a href="https://www.dunod.com/entreprise-et-economie/intelligence-artificielle-et-environnement-alliance-ou-nuisance-ia-face-aux" target="_blank" rel="noopener noreferrer">https://www.dunod.com/entreprise-et-economie/intelligence-artificielle-et-environnement-alliance-ou-nuisance-ia-face-aux</a></li>
            <li class="analysis-bibliography-item">Pachot, A., &amp; Patissier, C. (2023). Toward Sustainable Artificial Intelligence: An Overview of Environmental Protection Uses and Issues. <em>Green and Low-Carbon Economy</em>, 3(2), 105-112. <a href="https://ojs.bonviewpress.com/index.php/GLCE/article/view/608" target="_blank" rel="noopener noreferrer">https://ojs.bonviewpress.com/index.php/GLCE/article/view/608</a></li>
          </ul>
        </div>
      </section>
    </section>
    """
    result_block = ""
    result_methods_block = ""
    if result:
        annual = result["annual_llm"]
        if method_comparisons is None:
            method_comparisons = build_method_comparisons(load_records(), parsed_payload, result)
        analysis_bibliography_entries = build_analysis_bibliography_entries(factor_rows, result)
        analysis_bibliography_map = build_analysis_bibliography_map(analysis_bibliography_entries)
        result_methods_block = render_method_comparisons(method_comparisons)
        evidence = (parser_meta or {}).get("evidence", {})
        method_label = {
            "parametric_extrapolation": "Parametric extrapolation",
            "literature_proxy": "Literature proxy",
            "literature_multiples": "Multi-indicator inference aggregation",
            "wh_parameter_model": "Unified Wh -> parameters model",
            "market_multifactor_prompt_proxy_v1": "Market multi-factor prompt proxy",
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
          <p class="lead">Inference-only estimate based on source-linked scientific indicators and a traceable screening calculation.</p>
          <p class="scope-note">Retained scope: only LLM inference externalities are included. Model training, software-system consumption, and ancillary infrastructure are excluded from the displayed estimate.</p>
          <p class="meta-inline">Evidence level: <strong>{escape(evidence.get('label', 'Unqualified'))}</strong></p>
          <p class="meta-inline">Method: <strong>{escape(method_label)}</strong></p>
          <p class="meta-inline">Reference model: {render_model_detail_inline_trigger(model_profile.get('model_id', parsed_payload.get('model_id', 'not specified')), model_profile.get('model_id', parsed_payload.get('model_id', 'not specified')))}{' | Approx. active parameters: <strong>' + escape(format_parameter_billions(model_profile.get('active_parameters_billion'), is_estimated_parameter_status(model_profile.get('parameter_value_status')))) + '</strong>' if model_profile.get('active_parameters_billion') else ''} {render_analysis_entry_ref('parameters', analysis_bibliography_map)}</p>
          <p class="meta-inline">Electricity mix: <strong>{escape(country_mix.get('country_code', parsed_payload.get('country', 'not specified')))}</strong> <span class="method-basis">({escape(country_resolution_label)})</span>{' | ' + escape(country_mix.get('grid_carbon_intensity_gco2_per_kwh', '')) + ' gCO2e/kWh' if country_mix.get('grid_carbon_intensity_gco2_per_kwh') else ''} {render_analysis_entry_ref('mix', analysis_bibliography_map)}</p>
          {render_assumptions_summary(result)}
          {render_method_calculation_details(method_comparisons, analysis_bibliography_map)}
          {render_analysis_bibliography(analysis_bibliography_entries)}
        </section>
        """

    home_tab = f"""
    <section class="tab-panel is-active" id="tab-home-panel" data-tab-panel="home">
      <header class="hero">
        <h1>An Open Tool for Estimating the Environmental Footprint of LLMs</h1>
      </header>
      <form class="panel" method="post" action="{app_url('/')}" id="estimate-form">
        <label for="description">Describe your application in natural language to obtain an inference estimate, its assumptions, and its source-linked calculation details.</label>
        <textarea id="description" name="description" placeholder="Describe your AI-enabled application in natural language...">{escape(description)}</textarea>
        <button type="submit" id="submit-button">
          <span class="spinner" aria-hidden="true"></span>
          <span class="default-text">Estimate application</span>
          <span class="loading-text">Estimating...</span>
        </button>
        <div class="example-prompts" aria-label="Application examples">
          <p class="example-prompts-label">Or click an example to test it</p>
          <button type="button" class="example-prompt" data-example-prompt="We have a customer-support assistant based on GPT-4, used about 4,000 times per month in France by our support team.">We have a customer-support assistant based on GPT-4, used about 4,000 times per month in France by our support team.</button>
          <button type="button" class="example-prompt" data-example-prompt="We use Claude 3.5 Sonnet in our app to summarize internal documents for around 120 consultants, with about 15,000 summaries generated per month.">We use Claude 3.5 Sonnet in our app to summarize internal documents for around 120 consultants, with about 15,000 summaries generated per month.</button>
          <button type="button" class="example-prompt" data-example-prompt="We have a RAG assistant based on Mistral Large, with a vector database and logging, used by about 800 employees and handling roughly 25,000 requests per month. If you know them, you can also add token volumes or request counts.">We have a RAG assistant based on Mistral Large, with a vector database and logging, used by about 800 employees and handling roughly 25,000 requests per month. If you know them, you can also add token volumes or request counts.</button>
        </div>
      </form>
      <div id="results-anchor">
        {result_methods_block}
        {error_block}
        {result_block}
      </div>
    </section>
    """
    observatory_tab = f"""
    <section class="tab-panel" id="tab-observatory-panel" data-tab-panel="observatory">
      <div class="observatory-layout">
        <aside class="observatory-sidebar">
          <nav class="subtabs" aria-label="Observatory">
            <button type="button" class="subtab-button is-active" data-subtab-target="observatory-inference">Inference</button>
            <button type="button" class="subtab-button" data-subtab-target="observatory-training">Training</button>
          </nav>
        </aside>

        <div class="observatory-content">
          <section class="subtab-panel is-active" id="subtab-observatory-inference-panel" data-subtab-panel="observatory-inference">
            {market_models_chart_block}
          </section>

          <section class="subtab-panel" id="subtab-observatory-training-panel" data-subtab-panel="observatory-training">
            {training_models_chart_block}
          </section>
        </div>
      </div>
    </section>
    """
    referential_tab = f"""
    <section class="tab-panel" id="tab-referential-panel" data-tab-panel="referential">
      <div class="observatory-layout">
        <aside class="observatory-sidebar">
          <nav class="subtabs" aria-label="Referential">
            <button type="button" class="subtab-button is-active" data-subtab-target="referential-inference">Inference</button>
            <button type="button" class="subtab-button" data-subtab-target="referential-training">Training</button>
          </nav>
        </aside>

        <div class="observatory-content">
          <section class="subtab-panel is-active" id="subtab-referential-inference-panel" data-subtab-panel="referential-inference">
            {market_models_table_block}
          </section>

          <section class="subtab-panel" id="subtab-referential-training-panel" data-subtab-panel="referential-training">
            {training_models_table_block}
          </section>
        </div>
      </div>
    </section>
    """
    model_detail_drawer = f"""
    <div id="model-detail-root" data-model-detail-index='{escape(json.dumps(model_detail_index, ensure_ascii=False), quote=True)}'>
      <div class="model-detail-overlay" id="model-detail-overlay" hidden></div>
      <aside class="model-detail-drawer" id="model-detail-drawer" aria-hidden="true" aria-labelledby="model-detail-title">
        <div class="model-detail-header">
          <div>
            <div class="summary-kicker">Model profile</div>
            <h3 id="model-detail-title">Model detail</h3>
          </div>
          <button type="button" class="model-detail-close" id="model-detail-close" aria-label="Close model detail">×</button>
        </div>
        <div class="model-detail-content" id="model-detail-content"></div>
      </aside>
    </div>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PROJECT_NAME}</title>
  <link rel="icon" type="image/svg+xml" href="{app_url('/favicon.svg')}">
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
      font-family: "Aptos", "Segoe UI", "Helvetica Neue", "Noto Sans", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      color: var(--ink);
      background: var(--bg);
      scroll-behavior: smooth;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; padding: 32px 24px 56px; }}
    .hero {{ margin-bottom: 28px; }}
    .hero-brand {{
      display: flex;
      align-items: center;
      gap: 1.1rem;
      margin-bottom: 0.9rem;
    }}
    .hero-logo {{
      flex: 0 0 auto;
      width: min(280px, 48vw);
      max-width: 100%;
    }}
    .hero-logo svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .hero-logo-fallback {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 72px;
      height: 72px;
      border-radius: 18px;
      background: #f7f8fa;
      color: var(--accent);
      font-size: 1.8rem;
      font-weight: 800;
    }}
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
    h1 {{ margin: 0 0 10px; font-size: clamp(1.18rem, 2.2vw, 1.45rem); line-height: 1.3; font-weight: 400; color: var(--ink); }}
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
      cursor: not-allowed;
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
      padding: 0.8rem 0.95rem;
      border: 1px solid var(--line);
      border-radius: 0.75rem;
      background: #fff;
    }}
    .result-method-metric-energy {{
      background: rgba(140, 122, 91, 0.08);
      border-color: rgba(140, 122, 91, 0.16);
    }}
    .result-method-metric-carbon {{
      background: rgba(140, 122, 91, 0.08);
      border-color: rgba(140, 122, 91, 0.16);
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
    .model-detail-trigger {{
      display: block;
      width: 100%;
      padding: 0;
      margin: 0;
      border: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    .model-detail-trigger strong {{
      color: var(--accent);
      text-decoration: underline;
      text-decoration-color: rgba(36, 59, 99, 0.22);
      text-underline-offset: 0.14em;
    }}
    .model-detail-inline-trigger {{
      display: inline;
      padding: 0;
      margin: 0;
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      font: inherit;
      vertical-align: baseline;
    }}
    .model-detail-inline-trigger strong {{
      color: var(--accent);
      text-decoration: underline;
      text-decoration-color: rgba(36, 59, 99, 0.22);
      text-underline-offset: 0.14em;
    }}
    .model-detail-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(17, 24, 39, 0.34);
      z-index: 39;
    }}
    .model-detail-drawer {{
      position: fixed;
      top: 0;
      right: 0;
      width: min(620px, 100vw);
      height: 100vh;
      background: #fff;
      border-left: 1px solid var(--line);
      box-shadow: -12px 0 36px rgba(17, 24, 39, 0.18);
      transform: translateX(100%);
      transition: transform 180ms ease;
      z-index: 40;
      display: flex;
      flex-direction: column;
    }}
    .model-detail-drawer.is-open {{
      transform: translateX(0);
    }}
    .model-detail-header {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: flex-start;
      padding: 1.1rem 1.2rem 0.9rem;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .model-detail-close {{
      width: 2.2rem;
      height: 2.2rem;
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--ink);
      font-size: 1.3rem;
      line-height: 1;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0;
    }}
    .model-detail-content {{
      overflow-y: auto;
      padding: 1rem 1.2rem 1.4rem;
    }}
    .model-detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.8rem;
      margin-bottom: 1rem;
    }}
    .model-detail-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 0.9rem 1rem;
      background: #fff;
    }}
    .model-detail-card h4,
    .model-detail-section h4 {{
      margin: 0 0 0.7rem;
      font-size: 1rem;
    }}
    .model-detail-list {{
      display: grid;
      gap: 0.48rem;
      margin: 0;
    }}
    .model-detail-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
    }}
    .model-detail-label {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .model-detail-value {{
      text-align: right;
      font-weight: 600;
      color: var(--ink);
    }}
    .model-detail-pillbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 0.7rem 0 0;
    }}
    .model-detail-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.34rem 0.65rem;
      border-radius: 999px;
      background: #f3f6fb;
      color: var(--accent);
      font-size: 0.84rem;
      font-weight: 600;
    }}
    .model-detail-section {{
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--line);
    }}
    .model-detail-copy {{
      margin: 0;
      color: var(--ink);
      line-height: 1.7;
    }}
    .model-detail-source-list {{
      display: grid;
      gap: 0.75rem;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .model-detail-source-list li {{
      padding: 0.8rem 0.9rem;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    .model-detail-source-meta {{
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      margin-bottom: 0.25rem;
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
      align-items: center;
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
      padding: 0.24rem 0 0.5rem;
      font: inherit;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      margin: 0 1rem 0 0;
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
    .tab-button.logo-tab {{
      padding: 0.28rem 0.6rem;
      line-height: 1;
    }}
    .nav-logo {{
      display: block;
      width: 128px;
      max-width: 22vw;
      min-width: 96px;
      height: auto;
    }}
    .nav-logo svg {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .nav-logo-fallback {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      color: var(--accent);
      font-size: 0.9rem;
      font-weight: 800;
    }}
    .language-control {{
      margin-left: auto;
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
    }}
    .language-label {{
      margin: 0;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--muted);
    }}
    .language-links {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.9rem;
    }}
    .language-link {{
      color: var(--muted);
      text-decoration: none;
      font-weight: 600;
      cursor: pointer;
    }}
    .language-link.is-active {{
      color: var(--accent);
    }}
    .language-separator {{
      color: var(--line);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.is-active {{
      display: block;
    }}
    .observatory-layout {{
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 2rem;
      align-items: start;
    }}
    .observatory-sidebar {{
      position: sticky;
      top: 1.25rem;
      align-self: start;
    }}
    .observatory-content {{
      min-width: 0;
    }}
    .subtabs {{
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      margin: 0;
      padding-right: 1rem;
      border-right: 1px solid var(--line);
    }}
    .subtab-button {{
      appearance: none;
      border: 1px solid transparent;
      border-radius: 0.4rem;
      background: transparent;
      color: var(--muted);
      padding: 0.72rem 0.85rem;
      font: inherit;
      font-size: 0.98rem;
      font-weight: 600;
      cursor: pointer;
      margin: 0;
      text-align: left;
      transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
    }}
    .subtab-button:hover {{
      border-color: rgba(36,59,99,0.18);
      background: rgba(36,59,99,0.04);
    }}
    .subtab-button.is-active {{
      color: var(--accent);
      border-color: rgba(140, 122, 91, 0.55);
      background: rgba(140, 122, 91, 0.08);
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
    .example-prompts {{
      display: grid;
      gap: 0.65rem;
      margin: 0.9rem 0 1rem;
    }}
    .example-prompts-label {{
      margin: 0;
      font-size: 0.84rem;
      font-weight: 700;
      letter-spacing: 0.01em;
      color: var(--ink);
    }}
    .example-prompt {{
      appearance: none;
      border: 1px solid rgba(140, 122, 91, 0.28);
      border-radius: 0.45rem;
      background: rgba(140, 122, 91, 0.08);
      color: var(--ink);
      padding: 0.85rem 0.95rem;
      font: inherit;
      font-size: 0.95rem;
      line-height: 1.6;
      font-style: italic;
      text-align: left;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease;
    }}
    .example-prompt::before {{
      content: '"';
    }}
    .example-prompt::after {{
      content: '"';
    }}
    .example-prompt:hover {{
      border-color: rgba(36,59,99,0.28);
      background: rgba(36,59,99,0.05);
    }}
    .analysis-bibliography {{
      margin-top: 1.5rem;
      padding-top: 1rem;
      border-top: 1px solid var(--line);
    }}
    .analysis-bibliography-list {{
      margin: 0.55rem 0 0;
      padding-left: 1.15rem;
      color: var(--ink);
    }}
    .analysis-bibliography-item + .analysis-bibliography-item {{
      margin-top: 0.65rem;
    }}
    .analysis-bibliography-number {{
      font-weight: 700;
      margin-right: 0.35rem;
      color: var(--ink);
    }}
    .analysis-bibliography-label {{
      font-weight: 700;
      color: var(--ink);
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
    .models-chart-note {{
      font-size: 0.88rem;
      line-height: 1.55;
    }}
    .models-benchmark-note {{
      margin-top: 1rem;
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
      background: transparent;
    }}
    .reference-copy-block {{
      width: 100%;
      box-sizing: border-box;
    }}
    .reference-copy-block .summary-intro {{
      max-width: none;
      width: 100%;
    }}
    .paper-preview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-top: 0.85rem;
      align-items: start;
    }}
    .paper-preview-card {{
      display: flex;
      flex-direction: column;
      gap: 0.55rem;
      margin: 0;
      text-decoration: none;
      color: inherit;
      max-width: 260px;
      width: 100%;
    }}
    .paper-preview-card.book-preview-card {{
      max-width: 182px;
    }}
    .paper-preview-frame {{
      display: block;
      width: 100%;
      aspect-ratio: 0.72;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: rgba(140, 122, 91, 0.08);
      box-shadow: 0 10px 24px rgba(31, 36, 48, 0.08);
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }}
    .paper-preview-frame img {{
      width: 100%;
      height: 100%;
      border: 0;
      object-fit: cover;
      background: #fff;
    }}
    .paper-preview-frame img.book-cover-preview {{
      object-fit: contain;
      padding: 0.35rem;
      box-sizing: border-box;
    }}
    .paper-preview-card:hover .paper-preview-frame {{
      transform: translateY(-2px);
      border-color: rgba(36,59,99,0.24);
      box-shadow: 0 14px 28px rgba(31, 36, 48, 0.12);
    }}
    .paper-preview-caption {{
      font-size: 0.86rem;
      font-weight: 600;
      color: var(--accent);
    }}
    .paper-preview-reference {{
      margin: 0.85rem auto 0;
      max-width: 620px;
      text-align: center;
      font-size: 0.9rem;
      line-height: 1.6;
      color: var(--muted);
    }}
    .profile-link {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .profile-link:hover {{
      text-decoration: underline;
    }}
    .profile-link-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 1rem;
      height: 1rem;
      color: currentColor;
      flex: 0 0 auto;
    }}
    .profile-link-icon svg {{
      display: block;
      width: 100%;
      height: 100%;
      fill: currentColor;
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
    .reference-table tbody tr:nth-child(even) td {{
      background: rgba(140, 122, 91, 0.08);
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
    .citation-block {{
      margin: 0;
      padding: 0.85rem 0.95rem;
      border-radius: 0.45rem;
      background: rgba(140, 122, 91, 0.08);
      border: 1px solid rgba(140, 122, 91, 0.18);
      overflow-x: auto;
      max-width: 100%;
      box-sizing: border-box;
    }}
    .citation-block code {{
      display: block;
      background: transparent;
      border: 0;
      padding: 0;
      font-size: 0.82rem;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
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
      font-size: 1rem;
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
      .nav-logo {{ width: 108px; min-width: 84px; }}
      .hero-brand {{ flex-direction: column; align-items: flex-start; gap: 0.6rem; }}
      .hero-logo {{ width: min(240px, 68vw); }}
      .tabs {{ align-items: flex-start; }}
      .language-control {{ margin-left: 0; }}
      .observatory-layout {{ grid-template-columns: 1fr; gap: 1.25rem; }}
      .observatory-sidebar {{ position: static; }}
      .subtabs {{
        flex-direction: row;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding-right: 0;
        padding-bottom: 0.8rem;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .subtab-button {{
        padding: 0.55rem 0.75rem;
      }}
      .submetrics {{ grid-template-columns: 1fr; }}
      .summary-header {{ flex-direction: column; align-items: flex-start; }}
      .result-method-metrics {{ grid-template-columns: 1fr; }}
      .model-detail-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
  <body>
  <main class="wrap">
    <nav class="tabs" aria-label="Main navigation">
      <button type="button" class="tab-button logo-tab is-active" data-tab-target="home" aria-label="Home">
        <span class="nav-logo" aria-hidden="true">{render_logo_markup()}</span>
      </button>
      <button type="button" class="tab-button" data-tab-target="observatory">Observatory</button>
      <button type="button" class="tab-button" data-tab-target="referential">Referential</button>
      <button type="button" class="tab-button" data-tab-target="method">Method</button>
      <button type="button" class="tab-button" data-tab-target="bibliography">Sources</button>
      <button type="button" class="tab-button" data-tab-target="contact">About</button>
      <div class="language-control">
        <span class="language-links" aria-label="Language selector">
          <a href="#" class="language-link is-active" data-language-option="en">EN</a>
          <span class="language-separator" aria-hidden="true">|</span>
          <a href="#" class="language-link" data-language-option="fr">FR</a>
        </span>
      </div>
    </nav>

    {home_tab}
    {observatory_tab}
    {referential_tab}
    {method_tab}
    {contact_tab}
    {bibliography_tab}
    {model_detail_drawer}
  </main>
  <script>
    const estimateForm = document.getElementById('estimate-form');
    const descriptionInput = document.getElementById('description');
    const resultsAnchor = document.getElementById('results-anchor');
    const submitButton = document.getElementById('submit-button');
    const examplePromptButtons = Array.from(document.querySelectorAll('[data-example-prompt]'));
    const languageLinks = Array.from(document.querySelectorAll('[data-language-option]'));
    const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
    const tabPanels = Array.from(document.querySelectorAll('[data-tab-panel]'));
    const subtabButtons = Array.from(document.querySelectorAll('[data-subtab-target]'));
    const subtabPanels = Array.from(document.querySelectorAll('[data-subtab-panel]'));
    const searchInputs = Array.from(document.querySelectorAll('[data-table-search]'));
    const sortButtons = Array.from(document.querySelectorAll('[data-sort-table]'));
    const modelsChart = document.getElementById('models-impact-chart');
    const paramsChart = document.getElementById('params-impact-chart');
    const scatterChart = document.getElementById('carbon-params-scatter-chart');
    const inferenceBubbleChart = document.getElementById('inference-bubble-chart');
    const factorHeatmap = document.getElementById('inference-factor-heatmap');
    const inferenceUncertaintyChart = document.getElementById('inference-uncertainty-chart');
    const inferenceTrainingTradeoffChart = document.getElementById('inference-training-tradeoff-chart');
    const scatterLinearChart = document.getElementById('carbon-params-linear-chart');
    const countryMixChart = document.getElementById('country-mix-sensitivity-chart');
    const inferenceReleaseTimelineChart = document.getElementById('inference-release-timeline-chart');
    const inferenceDoublingTimelineChart = document.getElementById('inference-doubling-timeline-chart');
    const chartControls = Array.from(document.querySelectorAll('[data-model-chart-control="metric-tab"]'));
    const inferenceBubbleControls = Array.from(document.querySelectorAll('[data-inference-bubble-control="metric-tab"]'));
    const inferenceUncertaintyControls = Array.from(document.querySelectorAll('[data-inference-uncertainty-control="metric-tab"]'));
    const crossImpactControls = Array.from(document.querySelectorAll('[data-cross-impact-control="metric-tab"]'));
    const trainingChart = document.getElementById('training-impact-chart');
    const trainingScatterChart = document.getElementById('training-carbon-params-scatter-chart');
    const trainingFactorHeatmap = document.getElementById('training-factor-heatmap');
    const trainingUncertaintyChart = document.getElementById('training-uncertainty-chart');
    const trainingScatterLogChart = document.getElementById('training-carbon-params-log-chart');
    const trainingReleaseTimelineChart = document.getElementById('training-release-timeline-chart');
    const trainingDoublingTimelineChart = document.getElementById('training-doubling-timeline-chart');
    const modelDetailRoot = document.getElementById('model-detail-root');
    const modelDetailOverlay = document.getElementById('model-detail-overlay');
    const modelDetailDrawer = document.getElementById('model-detail-drawer');
    const modelDetailContent = document.getElementById('model-detail-content');
    const modelDetailTitle = document.getElementById('model-detail-title');
    const modelDetailClose = document.getElementById('model-detail-close');
    const trainingChartControls = Array.from(document.querySelectorAll('[data-training-chart-control="metric-tab"]'));
    const LANGUAGE_STORAGE_KEY = 'llm-environment-language';
    const TAB_HASH_PREFIX = '#tab-';
    const MIN_DESCRIPTION_LENGTH = 50;
    const textNodeOriginals = new WeakMap();
    const currentLanguage = {{ value: 'en' }};
    let currentModelDetailKey = null;
    let modelDetailIndex = {{}};
    if (modelDetailRoot) {{
      try {{
        modelDetailIndex = JSON.parse(modelDetailRoot.getAttribute('data-model-detail-index') || '{{}}');
      }} catch (error) {{
        modelDetailIndex = {{}};
      }}
    }}
    const tabDefaultSubtabs = {{
      observatory: 'observatory-inference',
      referential: 'referential-inference',
    }};
    const uiText = {{
      en: {{
        title: 'ImpactLLM',
        navAriaLabel: 'Main navigation',
        homeAriaLabel: 'Home',
        languageLabel: 'Language',
        estimatePlaceholder: 'Describe your AI-enabled application in natural language...',
        examplePromptsLabel: 'Or click an example to test it',
        marketSearchPlaceholder: 'Example: GPT, Claude, Mistral, US, 70B',
        trainingSearchPlaceholder: 'Example: GPT, Claude, 70B, Meta',
        noData: 'No data available for this selection.',
        noUsableValue: 'No usable value available for this selection.',
        chartFamilyPrompt: 'prompt|request',
        chartFamilyPage: 'page',
        chartMetricEnergy: 'Energy',
        chartMetricCarbon: 'Carbon',
        trainingMetricCarbon: 'Direct training CO2e',
        comparisonEstimatedPrefix: 'Comparison of estimated central values over 1 hour of active use for the ',
        comparisonEstimatedMiddle: ' indicator, using the ',
        comparisonEstimatedSuffix: ' family.',
        comparisonTrainingPrefix: 'Comparison of extrapolated central values for the ',
        comparisonTrainingSuffix: ' indicator.',
        modelsAriaLabel: 'Comparative chart of models',
        trainingAriaLabel: 'Comparative chart of training impacts',
        methodAnchorBody: 'The application-level estimator starts from published inference indicators linked to an explicit source, model, geography, and system boundary. In the current market-model release, the predictive core uses Elsworth et al. (2025) as the main prompt-level calibration anchor, with a median prompt energy of <code>0.24 Wh/prompt</code> for Gemini Apps, and is interpreted alongside other inference references such as the <em>ML.ENERGY Benchmark</em>, Ren et al. (2024), and Li et al. (2025).',
        methodProxyBody: 'When direct telemetry is unavailable for a target model, ImpactLLM does not rely on a raw parameter multiple alone. It builds an effective active-parameter profile from the retained model characteristics: active parameters, context window, serving mode (<code>open</code>, <code>hybrid</code>, <code>closed</code>), modality support, and architecture notes such as MoE or reasoning-oriented overheads.',
        methodTokensBody: 'The current proxy adjusts the anchor with a weighted prompt-compute volume defined from input and output tokens. Output generation is weighted more heavily than input processing, so output-heavy scenarios and repeated LLM calls raise the estimate materially.',
        methodBoundBody: 'The current prompt-level branch is a screening proxy, not an audited benchmark. For this reason, the application returns a bounded low-central-high result rather than one falsely precise deterministic value.',
        methodCarbonBody: 'Carbon is not copied mechanically from the source paper. It is recalculated from the retained energy estimate using the electricity mix associated with the selected country context.',
        methodResearchBody: 'The result is an auditable estimate intended for comparison, software design, and methodological discussion. It is useful precisely because the assumptions, factors, and retained sources remain visible and inspectable.',
        examplePrompt1: 'We have a customer-support assistant based on GPT-4, used about 4,000 times per month in France by our support team.',
        examplePrompt2: 'We use Claude 3.5 Sonnet in our app to summarize internal documents for around 120 consultants, with about 15,000 summaries generated per month.',
        examplePrompt3: 'We have a RAG assistant based on Mistral Large, with a vector database and logging, used by about 800 employees and handling roughly 25,000 requests per month. If you know them, you can also add token volumes or request counts.',
      }},
      fr: {{
        title: 'Données ouvertes sur l’empreinte environnementale des LLM',
        navAriaLabel: 'Navigation principale',
        homeAriaLabel: 'Accueil',
        languageLabel: 'Langue',
        estimatePlaceholder: 'Décrivez votre application intégrant de l’IA en langage naturel...',
        examplePromptsLabel: 'Ou cliquez sur un exemple pour le tester',
        marketSearchPlaceholder: 'Exemple : GPT, Claude, Mistral, US, 70B',
        trainingSearchPlaceholder: 'Exemple : GPT, Claude, 70B, Meta',
        noData: 'Aucune donnée disponible pour cette sélection.',
        noUsableValue: 'Aucune valeur exploitable pour cette sélection.',
        chartFamilyPrompt: 'prompt|requête',
        chartFamilyPage: 'page',
        chartMetricEnergy: 'Énergie',
        chartMetricCarbon: 'Carbone',
        trainingMetricCarbon: 'CO2e direct d’entraînement',
        comparisonEstimatedPrefix: 'Comparaison des valeurs centrales estimées sur 1 heure d’utilisation active pour l’indicateur ',
        comparisonEstimatedMiddle: ', selon la famille ',
        comparisonEstimatedSuffix: '.',
        comparisonTrainingPrefix: 'Comparaison des valeurs centrales extrapolées pour l’indicateur ',
        comparisonTrainingSuffix: '.',
        modelsAriaLabel: 'Graphique comparatif des modèles',
        trainingAriaLabel: 'Graphique comparatif des impacts d’entraînement',
        methodAnchorBody: 'L’estimateur au niveau applicatif part d’indicateurs d’inférence publiés, reliés à une source, un modèle, une géographie et un périmètre système explicites. Dans la version actuelle pour les modèles du marché, le cœur prédictif utilise Elsworth et al. (2025) comme ancrage principal de calibration au niveau prompt, avec une énergie médiane de <code>0.24 Wh/prompt</code> pour Gemini Apps, et se lit aux côtés d’autres références d’inférence comme le <em>ML.ENERGY Benchmark</em>, Ren et al. (2024) et Li et al. (2025).',
        methodProxyBody: 'Quand aucune télémétrie directe n’est disponible pour un modèle cible, ImpactLLM ne repose pas uniquement sur un multiple brut de paramètres. Il construit un profil en paramètres actifs effectifs à partir des caractéristiques retenues du modèle : paramètres actifs, fenêtre de contexte, mode de service (<code>open</code>, <code>hybrid</code>, <code>closed</code>), support multimodal et notes d’architecture comme le MoE ou des surcoûts liés au raisonnement.',
        methodTokensBody: 'Le proxy actuel ajuste l’ancrage avec un volume de calcul par prompt pondéré, défini à partir des tokens d’entrée et de sortie. La génération de sortie est plus fortement pondérée que le traitement de l’entrée, de sorte que les scénarios riches en sortie et les appels LLM répétés augmentent sensiblement l’estimation.',
        methodBoundBody: 'La branche actuelle au niveau prompt est un proxy de screening, pas un benchmark audité. Pour cette raison, l’application renvoie un résultat borné bas-central-haut plutôt qu’une valeur déterministe faussement précise.',
        methodCarbonBody: 'Le carbone n’est pas repris mécaniquement depuis l’article source. Il est recalculé à partir de l’estimation énergétique retenue en utilisant le mix électrique associé au contexte pays sélectionné.',
        methodResearchBody: 'Le résultat est une estimation auditable destinée à la comparaison, à la conception logicielle et à la discussion méthodologique. Son intérêt vient précisément du fait que les hypothèses, les facteurs et les sources retenues restent visibles et inspectables.',
        examplePrompt1: 'Nous avons un assistant de support client basé sur GPT-4, utilisé environ 4 000 fois par mois en France par notre équipe support.',
        examplePrompt2: 'Nous utilisons Claude 3.5 Sonnet dans notre application pour résumer des documents internes pour environ 120 consultants, avec près de 15 000 résumés générés par mois.',
        examplePrompt3: 'Nous avons un assistant RAG basé sur Mistral Large, avec une base vectorielle et de la journalisation, utilisé par environ 800 collaborateurs et traitant près de 25 000 requêtes par mois.',
      }}
    }};
    const textReplacements = [
      ['Home', 'Accueil'],
      ['Observatory', 'Observatoire'],
      ['Referential', 'Référentiel'],
      ['Method', 'Méthode'],
      ['Documentation', 'Documentation'],
      ['About', 'À propos'],
      ['Contact', 'À propos'],
      ['About us', 'À propos'],
      ["Arnault Pachot's Dunod book", "Livre Dunod d’Arnault Pachot"],
      ['We work on responsible AI with a focus on methodological rigor, traceability, and real-world decision support. Our work combines scientific research, product design, and operational deployment to make AI systems more transparent, more accountable, and more useful in practice.', 'Nous travaillons sur l’IA responsable avec un accent mis sur la rigueur méthodologique, la traçabilité et l’aide à la décision dans des contextes réels. Notre travail combine recherche scientifique, conception produit et déploiement opérationnel pour rendre les systèmes d’IA plus transparents, plus responsables et plus utiles en pratique.'],
      ['Arnault Pachot is a researcher and entrepreneur, founder of OpenStudio and now founder of Emotia. He works on responsible digital transformation, Green IT, and decision-oriented AI systems. He co-authored the Dunod book ', 'Arnault Pachot est chercheur et entrepreneur, fondateur d’OpenStudio puis d’Emotia. Il travaille sur la transformation numérique responsable, le Green IT et les systèmes d’IA orientés décision. Il a coécrit chez Dunod l’ouvrage '],
      [' dedicated to practical pathways for environmentally responsible AI.', ', consacré à des trajectoires concrètes pour une IA écologiquement responsable.'],
      ['Thierry Petit is a senior AI researcher and scientific leader with more than twenty years of academic and R&D experience in Europe and the United States. His work spans trustworthy AI, simulation, optimization, and decision-grade platforms. At Emotia and Pollitics, he leads the scientific direction of systems designed to remain both operationally useful and methodologically robust.', 'Thierry Petit est chercheur senior en intelligence artificielle et directeur scientifique, avec plus de vingt ans d’expérience académique et de R&D en Europe et aux États-Unis. Ses travaux couvrent l’IA de confiance, la simulation, l’optimisation et les plateformes d’aide à la décision. Chez Emotia et Pollitics, il pilote la direction scientifique de systèmes conçus pour rester à la fois utiles opérationnellement et robustes méthodologiquement.'],
      ['References', 'Sources'],
      ['Biliography', 'Sources'],
      ['Bibliography', 'Sources'],
      ['Inference', 'Inférence'],
      ['Training', 'Entraînement'],
      ['Positioning', 'Positionnement'],
      ['Estimate application', 'Estimer l’application'],
      ['Estimating...', 'Estimation...'],
      ['How to cite ImpactLLM', 'Comment citer ImpactLLM'],
      ['GitHub repository', 'Dépôt GitHub'],
      ['The project repository is available on GitHub: ', 'Le dépôt du projet est disponible sur GitHub : '],
      ['License', 'Licence'],
      ['This program is free software: you can redistribute it and/or modify it under the terms of the <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">GNU General Public License</a> as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.', 'Ce programme est un logiciel libre : vous pouvez le redistribuer et/ou le modifier selon les termes de la <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">GNU General Public License</a> telle que publiée par la Free Software Foundation, soit la version 3 de la licence, soit, à votre choix, toute version ultérieure.'],
      ['This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.', 'Ce programme est distribué dans l’espoir qu’il sera utile, mais SANS AUCUNE GARANTIE, sans même la garantie implicite de QUALITÉ MARCHANDE ou D’ADÉQUATION À UN USAGE PARTICULIER. Consultez la GNU General Public License pour plus de détails.'],
      ['You should have received a copy of the GNU General Public License along with this program. If not, see <a href="https://www.gnu.org/licenses/" target="_blank" rel="noopener noreferrer">https://www.gnu.org/licenses/</a>.', 'Vous devriez avoir reçu une copie de la GNU General Public License avec ce programme. Sinon, voir <a href="https://www.gnu.org/licenses/" target="_blank" rel="noopener noreferrer">https://www.gnu.org/licenses/</a>.'],
      ['About us', 'À propos de nous'],
      ['Selected references on AI and the environment', 'Références choisies sur l’IA et l’environnement'],
      ['This annex brings together the quantified reference material used in the interface, along with everyday comparison benchmarks and country factors used for carbon and water recalculation.', 'Cette annexe rassemble les sources quantitatives mobilisées dans l’interface, ainsi que des repères de comparaison du quotidien et les facteurs pays utilisés pour le recalcul du carbone et de l’eau.'],
      ['Download PDF', 'Télécharger le PDF'],
      ['An Open Tool for Exploring and Estimating the Environmental Footprint of Large Language Models', 'Un outil libre pour estimer l’empreinte environnementale des LLMs'],
      ['An Open Tool for Estimating the Environmental Footprint of LLMs', 'Un outil libre pour estimer l’empreinte environnementale des LLMs'],
      ['Inference-only estimate based on source-linked scientific indicators and a traceable screening calculation.', 'Estimation limitée à l’inférence, fondée sur des indicateurs scientifiques reliés aux sources et un calcul de screening traçable.'],
      ['Inférence-only estimate based on source-linked scientific indicators and a traceable screening calculation.', 'Estimation limitée à l’inférence, fondée sur des indicateurs scientifiques reliés aux sources et un calcul de screening traçable.'],
      ['ImpactLLM is designed as a transparent screening tool, not as a black-box score. The current release starts from source-linked inference anchors, then exposes a bounded multi-factor proxy rather than a hidden single-number score.', 'ImpactLLM est conçu comme un outil transparent de screening, et non comme un score boîte noire. La version actuelle part d’ancrages d’inférence reliés aux sources, puis expose un proxy multi-facteurs borné plutôt qu’un score unique caché.'],
      ['1. Source-linked literature anchors.', '1. Ancrages bibliographiques reliés aux sources.'],
      ['The application-level estimator starts from published inference indicators linked to an explicit source, model, geography, and system boundary. In the current market-model release, the predictive core uses Elsworth et al. (2025) as the main prompt-level calibration anchor, with a median prompt energy of <code>0.24 Wh/prompt</code> for Gemini Apps, and is interpreted alongside other inference references such as the <em>ML.ENERGY Benchmark</em>, Ren et al. (2024), and Li et al. (2025).', 'L’estimateur au niveau applicatif part d’indicateurs d’inférence publiés, reliés à une source, un modèle, une géographie et un périmètre système explicites. Dans la version actuelle pour les modèles du marché, le cœur prédictif utilise Elsworth et al. (2025) comme ancrage principal de calibration au niveau prompt, avec une énergie médiane de <code>0.24 Wh/prompt</code> pour Gemini Apps, et se lit aux côtés d’autres références d’inférence comme le <em>ML.ENERGY Benchmark</em>, Ren et al. (2024) et Li et al. (2025).'],
      ['2. A multi-factor effective-parameter proxy.', '2. Un proxy multi-facteurs en paramètres actifs effectifs.'],
      ['When direct telemetry is unavailable for a target model, ImpactLLM does not rely on a raw parameter multiple alone. It builds an effective active-parameter profile from the retained model characteristics: active parameters, context window, serving mode (<code>open</code>, <code>hybrid</code>, <code>closed</code>), modality support, and architecture notes such as MoE or reasoning-oriented overheads.', 'Quand aucune télémétrie directe n’est disponible pour un modèle cible, ImpactLLM ne repose pas uniquement sur un multiple brut de paramètres. Il construit un profil en paramètres actifs effectifs à partir des caractéristiques retenues du modèle : paramètres actifs, fenêtre de contexte, mode de service (<code>open</code>, <code>hybrid</code>, <code>closed</code>), support multimodal et notes d’architecture comme le MoE ou des surcoûts liés au raisonnement.'],
      ['3. Token volume remains explicit.', '3. Le volume de tokens reste explicite.'],
      ['The current proxy adjusts the anchor with a weighted prompt-compute volume defined from input and output tokens. Output generation is weighted more heavily than input processing, so output-heavy scenarios and repeated LLM calls raise the estimate materially.', 'Le proxy actuel ajuste l’ancrage avec un volume de calcul par prompt pondéré, défini à partir des tokens d’entrée et de sortie. La génération de sortie est plus fortement pondérée que le traitement de l’entrée, de sorte que les scénarios riches en sortie et les appels LLM répétés augmentent sensiblement l’estimation.'],
      ['The current prompt-level branch is a screening proxy, not an audited benchmark. For this reason, the application returns a bounded low-central-high result rather than one falsely precise deterministic value.', 'La branche actuelle au niveau prompt est un proxy de screening, pas un benchmark audité. Pour cette raison, l’application renvoie un résultat borné bas-central-haut plutôt qu’une valeur déterministe faussement précise.'],
      ['4. Carbon derived from context.', '4. Un carbone dérivé du contexte.'],
      ['Carbon is not copied mechanically from the source paper. It is recalculated from the retained energy estimate using the electricity mix associated with the selected country context.', 'Le carbone n’est pas repris mécaniquement depuis l’article source. Il est recalculé à partir de l’estimation énergétique retenue en utilisant le mix électrique associé au contexte pays sélectionné.'],
      ['5. A research-oriented estimator.', '5. Un estimateur orienté recherche.'],
      ['The result is an auditable estimate intended for comparison, software design, and methodological discussion. It is useful precisely because the assumptions, factors, and retained sources remain visible and inspectable.', 'Le résultat est une estimation auditable destinée à la comparaison, à la conception logicielle et à la discussion méthodologique. Son intérêt vient précisément du fait que les hypothèses, les facteurs et les sources retenues restent visibles et inspectables.'],
      ['Scientific paper.', 'Article scientifique.'],
      ['Download the PDF', 'Télécharger le PDF'],
      ['Download BibTeX entry', 'Télécharger l’entrée BibTeX'],
      ['Download the associated scientific publication PDF', 'Télécharger le PDF de la publication scientifique associée'],
      ['ImpactLLM', 'ImpactLLM'],
      ['An open tool for exploring and estimating the environmental footprint of large language models.', 'Un outil libre pour estimer l’empreinte environnementale des LLMs.'],
      ['An open tool for estimating the environmental footprint of LLMs.', 'Un outil libre pour estimer l’empreinte environnementale des LLMs.'],
      ['Describe your application in natural language to obtain an inference estimate, its assumptions, and its source-linked calculation details.', 'Décrivez votre application en langage naturel pour obtenir une estimation d’inférence, ses hypothèses et les détails du calcul reliés aux sources.'],
      ['Install and run the project', 'Installer et lancer le projet'],
      ['GitHub repository.', 'Dépôt GitHub.'],
      ['OpenAI key and .env files', 'Clé OpenAI et fichiers .env'],
      ['Available endpoints', 'Endpoints disponibles'],
      ['curl examples', 'Exemples curl'],
      ['Tools exposed to agents', 'Outils exposés aux agents'],
      ['Dataset, validation, and exports', 'Jeu de données, validation et exports'],
      ['LaTeX compilation', 'Compilation LaTeX'],
      ['Reuse conditions', 'Conditions de réutilisation'],
      ['Email addresses are obfuscated in the page source to reduce basic scraping.', 'Les adresses e-mail sont obfusquées dans le code source de la page pour limiter le scraping basique.'],
      ['Calculation details', 'Détails du calcul'],
      ['Retained scope: only LLM inference externalities are included. Model training, software-system consumption, and ancillary infrastructure are excluded from the displayed estimate.', 'Périmètre retenu : seules les externalités d’inférence du LLM sont incluses. L’entraînement du modèle, la consommation du système logiciel et l’infrastructure annexe sont exclus de l’estimation affichée.'],
      ['Retained assumptions', 'Hypothèses retenues'],
      ['Inference-only estimate: training and software-system overheads excluded', 'Estimation limitée à l’inférence : entraînement et surcoûts du système logiciel exclus'],
      ['Inférence-only estimate: training and software-system overheads excluded', 'Estimation limitée à l’inférence : entraînement et surcoûts du système logiciel exclus'],
      ['Market multi-factor prompt proxy', 'Proxy prompt multi-facteurs du marché'],
      ['Multi-factor prompt proxy', 'Proxy prompt multi-facteurs'],
      ['Prompt-energy screening proxy calibrated on Elsworth et al. (2025), then adjusted by effective active parameters, serving assumptions, architecture overhead, and weighted token volume.', 'Proxy de screening en énergie par prompt calibré sur Elsworth et al. (2025), puis ajusté par les paramètres actifs effectifs, les hypothèses de service, l’overhead d’architecture et un volume de tokens pondéré.'],
      ['Prompt-energy estimate calibrated on Elsworth et al. (2025) at 0.24 Wh/prompt for Gemini Apps.', 'Estimation en énergie par prompt calibrée sur Elsworth et al. (2025) à 0.24 Wh/prompt pour Gemini Apps.'],
      ['Weighted prompt compute uses input tokens + 1.8 x output tokens relative to the project reference scenario.', 'Le calcul pondéré par prompt utilise les tokens d’entrée + 1.8 x les tokens de sortie par rapport au scénario de référence du projet.'],
      ['Effective active parameters adjust the raw model size with context window, serving mode, modality support, and architecture overhead.', 'Les paramètres actifs effectifs ajustent la taille brute du modèle avec la fenêtre de contexte, le mode de service, le support multimodal et l’overhead d’architecture.'],
      ['Exact market-model profile retained for', 'Profil exact du modèle marché retenu pour'],
      ['Exact market-model profile unavailable; synthetic multifactor fallback built from the target parameter estimate.', 'Profil exact du modèle marché indisponible ; fallback multifactoriel synthétique construit à partir de l’estimation des paramètres cible.'],
      ['Request type classified as chat_generation', 'Type de requête classé comme chat_generation'],
      ['1 LLM request(s) per feature use', '1 requête LLM par usage de fonctionnalité'],
      ['feature uses per year', 'usages de fonctionnalité par an'],
      ['Energy is the primary quantity, and carbon is contextualized with the retained electricity mix.', 'L’énergie est la quantité primaire, et le carbone est contextualisé avec le mix électrique retenu.'],
      ['Energy is the primary quantity, and carbon is derived from the electricity mix of the retained country.', 'L’énergie est la quantité primaire, et le carbone est dérivé du mix électrique du pays retenu.'],
      ['Model-size scaling enabled with target profile at', 'Mise à l’échelle par taille de modèle activée avec un profil cible à'],
      ['active parameters', 'paramètres actifs'],
      ['Carbon is recalculated using the publisher-country electricity mix for', 'Le carbone est recalculé à partir du mix électrique du pays de l’éditeur pour'],
      ['because the model is treated as a hosted proprietary service.', 'car le modèle est traité comme un service propriétaire hébergé.'],
      ['A prompt/query proxy is calibrated from 1 prompt/query anchor(s), then adjusted by a token ratio relative to 1550 tokens', 'Un proxy prompt/requête est calibré à partir de 1 ancrage prompt/requête, puis ajusté par un ratio de tokens relatif à 1550 tokens'],
      ['The page-family method was marked as not applicable for this scenario by the parser.', 'La méthode de la famille page a été marquée comme non applicable à ce scénario par le parseur.'],
      ['Wh/prompt|request proxy', 'Proxy Wh/prompt|requête'],
      ['Parametric Wh/prompt|request proxy calibrated on literature anchors, with a simple token-volume adjustment.', 'Proxy paramétrique Wh/prompt|requête calibré sur les ancrages de la littérature, avec ajustement simple au volume de tokens.'],
      ['1. Scenario input data', '1. Données d’entrée du scénario'],
      ['The interpreted scenario uses', 'Le scénario interprété utilise'],
      ['input tokens and', 'tokens en entrée et'],
      ['output tokens per call.', 'tokens en sortie par appel.'],
      ['The annual call volume is calculated as:', 'Le volume annuel d’appels est calculé ainsi :'],
      ['In the Wh/prompt|request family, one LLM request remains the base inference unit, then the proxy is adjusted with a simple token ratio relative to the project reference prompt. Annualization therefore relies on the number of LLM calls per year.', 'Dans la famille Wh/prompt|requête, une requête LLM reste l’unité d’inférence de base, puis le proxy est ajusté avec un ratio simple de tokens par rapport au prompt de référence du projet. L’annualisation repose donc sur le nombre d’appels LLM par an.'],
      ['In this method, one LLM request remains the base inference unit, then the prompt-energy anchor is adjusted with weighted token volume and multi-factor effective parameters.', 'Dans cette méthode, une requête LLM reste l’unité d’inférence de base, puis l’ancrage en énergie par prompt est ajusté avec un volume de tokens pondéré et des paramètres actifs effectifs multi-facteurs.'],
      ['2. Literature anchors and extrapolation', '2. Ancrages bibliographiques et extrapolation'],
      ['2. Prompt anchor and multi-factor extrapolation', '2. Ancrage prompt et extrapolation multi-facteurs'],
      ['The method starts from literature energy values published for the', 'La méthode part de valeurs d’énergie publiées dans la littérature pour la famille'],
      ['family, then applies scaling by parameter count.', ', puis applique une mise à l’échelle selon le nombre de paramètres.'],
      ['Observed literature value:', 'Valeur observée dans la littérature :'],
      ['Observed literature anchor:', 'Ancrage observé dans la littérature :'],
      ['Source parameter count:', 'Nombre de paramètres de la source :'],
      ['Target parameter count:', 'Nombre de paramètres cible :'],
      ['Target raw parameter count:', 'Nombre brut de paramètres cible :'],
      ['Applied parameter factor:', 'Facteur de paramètres appliqué :'],
      ['Extrapolated energy for one inference unit:', 'Énergie extrapolée pour une unité d’inférence :'],
      ['Central effective active parameters:', 'Paramètres actifs effectifs centraux :'],
      ['See the table ', 'Voir le tableau '],
      [' in Sources for the retained screening values by model.', ' dans Sources pour les valeurs de screening retenues par modèle.'],
      ['Where \\(P^{{eff}}_c\\) is the central effective active-parameter proxy, \\(P_t\\) the retained raw active-parameter count, \\(F_{{ctx}}\\) the context-window factor, \\(F_{{srv}}\\) the serving-mode factor, \\(F_{{mod}}\\) the modality factor, and \\(F_{{arch}}\\) the architecture-overhead factor.', 'Où \\(P^{{eff}}_c\\) désigne le proxy central de paramètres actifs effectifs, \\(P_t\\) le nombre brut de paramètres actifs retenu, \\(F_{{ctx}}\\) le facteur de fenêtre de contexte, \\(F_{{srv}}\\) le facteur de mode de service, \\(F_{{mod}}\\) le facteur de modalité, et \\(F_{{arch}}\\) le facteur d’overhead d’architecture.'],
      ['Central token factor:', 'Facteur central de tokens :'],
      ['Central per-request energy:', 'Énergie centrale par requête :'],
      ['The displayed range widens the scaling exponent and the contextual overhead factors between low, central, and high scenarios.', 'La plage affichée élargit l’exposant de mise à l’échelle et les facteurs d’overhead contextuels entre les scénarios bas, central et haut.'],
      ['This family currently relies on a single literature anchor, so the displayed central value is a calibrated proxy rather than a cross-study average.', 'Cette famille repose actuellement sur un seul ancrage bibliographique, de sorte que la valeur centrale affichée est un proxy calibré plutôt qu’une moyenne inter-études.'],
      ['The table below compares the models tracked by the project under the same inference scenario. For each model, the application shows the central values produced by the project’s multi-factor prompt proxy, both per hour of standardized use and per request.', 'Le tableau ci-dessous compare les modèles suivis par le projet dans le même scénario d’inférence. Pour chaque modèle, l’application affiche les valeurs centrales produites par le proxy prompt multi-facteurs du projet, à la fois par heure d’usage standardisé et par requête.'],
      ['The chart below shows the estimated central values for all catalog models under a standardized inference scenario corresponding to 1 hour of active use: 34.6 interactions/hour, 1000 input tokens, 550 output tokens, and one LLM request per use. The hourly pace is derived from an average reading speed of 238.0 words/min (Brysbaert, 2019) and a project convention of 1 token ≈ 0.75 word.', 'Le graphique ci-dessous présente les valeurs centrales estimées pour tous les modèles du catalogue dans un scénario d’inférence standardisé correspondant à 1 heure d’usage actif : 34,6 interactions/heure, 1000 tokens en entrée, 550 tokens en sortie et une requête LLM par usage. Le rythme horaire est dérivé d’une vitesse moyenne de lecture de 238,0 mots/min (Brysbaert, 2019) et d’une convention du projet de 1 token ≈ 0,75 mot.'],
      ['Benchmarks integrated into the chart, all expressed over one hour or rescaled to a comparable order of magnitude: household electricity from Purdue Extension measurements (fluorescent lamp ≈ 9.3 Wh over 1 h; laptop ≈ 32 Wh over 1 h) and a 1500 W electric space heater rescaled here to <strong>4.1 minutes</strong> to obtain ≈ <strong>103.5 Wh</strong>, close to the order of magnitude of Claude Opus 4.1 in the inference scenario; for carbon, an average gasoline car benchmark derived from the ICCT (2025) factor retained by the project (235 gCO2e/km), here rescaled to <strong>0.17 km</strong> to obtain ≈ <strong>40.0 gCO2e</strong>, close to the order of magnitude of Claude Opus 4.1 in the inference scenario.', 'Repères intégrés au graphique, tous exprimés sur une heure ou redimensionnés à un ordre de grandeur comparable : consommation électrique domestique issue des mesures de Purdue Extension (lampe fluorescente ≈ 9,3 Wh sur 1 h ; ordinateur portable ≈ 32 Wh sur 1 h) et radiateur électrique de 1500 W ramené ici à <strong>4,1 minutes</strong> pour obtenir ≈ <strong>103,5 Wh</strong>, proche de l’ordre de grandeur de Claude Opus 4.1 dans le scénario d’inférence ; pour le carbone, un repère de voiture essence moyenne dérivé du facteur ICCT (2025) retenu par le projet (235 gCO2e/km), ici ramené à <strong>0,17 km</strong> pour obtenir ≈ <strong>40,0 gCO2e</strong>, proche de l’ordre de grandeur de Claude Opus 4.1 dans le scénario d’inférence.'],
      ['Average gasoline car for 0.17 km', 'Voiture essence moyenne sur 0,17 km'],
      ['Trade-off', 'Arbitrage'],
      ['Inference model landscape', 'Paysage des modèles en inférence'],
      ['Inference vs. training impact map', 'Carte des impacts inférence vs entraînement'],
      ['Inference screening factor heatmap', 'Heatmap des facteurs de screening en inférence'],
      ['Training screening factor heatmap', 'Heatmap des facteurs de screening en entraînement'],
      ['Training uncertainty span by model', 'Étendue d’incertitude d’entraînement par modèle'],
      ['Inference carbon vs. parameter count', 'Carbone d’inférence vs nombre de paramètres'],
      ['Inference carbon by model release date', 'Carbone d’inférence par date de sortie du modèle'],
      ['Country-mix sensitivity', 'Sensibilité au mix pays'],
      ['Training model landscape', 'Paysage des modèles en entraînement'],
      ['Training carbon vs. parameter count', 'Carbone d’entraînement vs nombre de paramètres'],
      ['Training carbon by model release date', 'Carbone d’entraînement par date de sortie du modèle'],
      ['This landscape view clusters the catalog models from the characteristics retained by the project for inference screening: active and effective parameters, context window, serving mode, modality support, architecture notes, and central energy and carbon outputs. Nearby points indicate models with similar retained screening profiles, not a simple one-metric ranking.', 'Cette vue de paysage regroupe les modèles du catalogue à partir des caractéristiques retenues par le projet pour le screening en inférence : paramètres actifs et effectifs, fenêtre de contexte, mode de service, support multimodal, notes d’architecture, ainsi que sorties centrales d’énergie et de carbone. Des points proches indiquent des profils de screening retenus similaires, et non un classement sur une seule métrique.'],
      ['This scatter plot compares each catalog model on two axes at once: standardized inference impact over one hour on the horizontal axis and retained training impact on the vertical axis. Point size follows the retained active parameter count, while colors distinguish providers.', 'Ce nuage de points compare chaque modèle du catalogue sur deux axes à la fois : l’impact d’inférence standardisé sur une heure sur l’axe horizontal et l’impact d’entraînement retenu sur l’axe vertical. La taille des points suit le nombre de paramètres actifs retenu, tandis que les couleurs distinguent les fournisseurs.'],
      ['This heatmap exposes the central screening factors retained for each market model. It shows the four multiplicative factors used by the project’s prompt proxy and the resulting ratio between effective and raw active parameters.', 'Cette heatmap rend visibles les facteurs centraux de screening retenus pour chaque modèle du marché. Elle montre les quatre facteurs multiplicatifs utilisés par le proxy prompt du projet ainsi que le ratio résultant entre paramètres actifs effectifs et paramètres actifs bruts.'],
      ['This heatmap exposes the central screening factors retained for each market model in the training proxy. It shows the regime, architecture, and hardware factors together with the retained training-token ratio per parameter.', 'Cette heatmap rend visibles les facteurs centraux de screening retenus pour chaque modèle du marché dans le proxy d’entraînement. Elle montre les facteurs de régime, d’architecture et de matériel, ainsi que le ratio retenu de tokens d’entraînement par paramètre.'],
      ['This view shows the low, central, and high direct training CO2e values retained by the project for each market model. It makes explicit how widely the training proxy can vary once the parameter and token exponents and contextual factors are widened.', 'Cette vue montre, pour chaque modèle du marché, les valeurs basse, centrale et haute de CO2e direct d’entraînement retenues par le projet. Elle rend explicite l’ampleur de variation possible du proxy d’entraînement lorsque les exposants sur les paramètres et les tokens, ainsi que les facteurs contextuels, sont élargis.'],
      ['This complementary view places models by retained active parameter count on the horizontal axis and by central inference carbon over one hour on the vertical axis, using logarithmic scaling on both axes.', 'Cette vue complémentaire positionne les modèles selon leur nombre de paramètres actifs retenus sur l’axe horizontal et leur carbone central d’inférence sur une heure sur l’axe vertical, avec une échelle logarithmique sur les deux axes.'],
      ['This timeline follows the evolution of the project’s central inference CO2e estimate over time for the OpenAI, Claude, Grok, and Mistral families, using the release month of each model as the horizontal axis.', 'Cette chronologie suit l’évolution dans le temps de l’estimation centrale du CO2e d’inférence du projet pour les familles OpenAI, Claude, Grok et Mistral, en utilisant le mois de sortie de chaque modèle comme axe horizontal.'],
      ['This view compares central inference energy and carbon over one hour, while coloring each model by the retained electricity-mix country used for carbon recalculation. It helps separate model-size effects from country-mix effects.', 'Cette vue compare l’énergie et le carbone centraux d’inférence sur une heure, en colorant chaque modèle selon le pays de mix électrique retenu pour le recalcul du carbone. Elle aide à distinguer les effets de taille de modèle des effets de mix pays.'],
      ['This landscape view clusters the catalog models from the characteristics retained by the project for training screening: retained parameter count, training-token prior, training regime, hardware-class proxy, modality support, architecture notes, and central training energy and carbon outputs. Nearby points indicate similar retained screening profiles rather than a direct ranking on one axis.', 'Cette vue de paysage regroupe les modèles du catalogue à partir des caractéristiques retenues par le projet pour le screening en entraînement : nombre de paramètres retenu, prior sur les tokens d’entraînement, régime d’entraînement, proxy de classe matérielle, support multimodal, notes d’architecture, ainsi que sorties centrales d’énergie et de carbone d’entraînement. Des points proches indiquent des profils de screening retenus similaires plutôt qu’un classement direct sur un seul axe.'],
      ['This complementary view places models by retained parameter count on the horizontal axis and by direct training CO2e on the vertical axis, using logarithmic scaling on both axes.', 'Cette vue complémentaire positionne les modèles selon leur nombre de paramètres retenu sur l’axe horizontal et leur CO2e direct d’entraînement sur l’axe vertical, avec une échelle logarithmique sur les deux axes.'],
      ['This timeline follows the evolution of the project’s retained direct training CO2e estimate over time for the OpenAI, Claude, Grok, and Mistral families, using the release month of each model as the horizontal axis.', 'Cette chronologie suit l’évolution dans le temps de l’estimation retenue du CO2e direct d’entraînement du projet pour les familles OpenAI, Claude, Grok et Mistral, en utilisant le mois de sortie de chaque modèle comme axe horizontal.'],
      ['3. Carbon derivation from the country mix', '3. Dérivation du carbone à partir du mix pays'],
      ['Carbon is not reused directly from the literature. It is derived from extrapolated energy using the retained country electricity mix, here', 'Le carbone n’est pas réutilisé directement depuis la littérature. Il est dérivé de l’énergie extrapolée à partir du mix électrique du pays retenu, ici'],
      ['The unit result retained for this method then leads to the following annualized values: energy', 'Le résultat unitaire retenu pour cette méthode conduit ensuite aux valeurs annualisées suivantes : énergie'],
      ['and carbon', 'et carbone'],
      ['4. Final annualization', '4. Annualisation finale'],
      ['The final annual projection is based on', 'La projection annuelle finale est basée sur'],
      ['inference unit(s) per year.', 'unité(s) d’inférence par an.'],
      ['Bibliography for this analysis', 'Bibliographie de cette analyse'],
      ['Active-parameter characteristic', 'Caractéristique des paramètres actifs'],
      ['Context-window characteristic', 'Caractéristique de la fenêtre de contexte'],
      ['Serving-mode characteristic', 'Caractéristique du mode de service'],
      ['Vision characteristic', 'Caractéristique de la vision'],
      ['Observed literature value', 'Valeur observée dans la littérature'],
      ['Electricity mix', 'Mix électrique'],
      ['Evidence level: ', 'Niveau de preuve : '],
      ['Method: ', 'Méthode : '],
      ['Reference model: ', 'Modèle de référence : '],
      ['Approx. active parameters: ', 'Paramètres actifs approx. : '],
      ['Electricity mix: ', 'Mix électrique : '],
      ['publisher country', 'pays de l’éditeur'],
      ['project country', 'pays du projet'],
      ['reference country', 'pays de référence'],
      ['explicit country', 'pays explicite'],
      ['retained country', 'pays retenu'],
      ['Search for a model', 'Rechercher un modèle'],
      ['Comparative environmental impact of models', 'Impact environnemental comparatif des modèles'],
      ['current models tracked by the project', 'modèles actuels suivis par le projet'],
      ['Comparative training impacts of models', 'Impacts d’entraînement comparés des modèles'],
      ['The chart below shows the central values retained for all catalog models across two training indicator families: training energy and direct training CO2e. The current screening method combines retained parameter count, a training-token prior, a training-regime prior, architecture features, and a hardware-class proxy. Under these central screening assumptions, frontier models can reach very large training orders of magnitude. Everyday benchmarks are inserted directly into the list to situate those scales, not to imply direct observed equivalence.', 'Le graphique ci-dessous présente les valeurs centrales retenues pour tous les modèles du catalogue selon deux familles d’indicateurs d’entraînement : l’énergie d’entraînement et le CO2e direct d’entraînement. La méthode actuelle de screening combine le nombre de paramètres retenu, un prior sur les tokens d’entraînement, un prior sur le régime d’entraînement, des caractéristiques d’architecture et un proxy de classe matérielle. Sous ces hypothèses centrales de screening, les modèles de frontière peuvent atteindre des ordres de grandeur d’entraînement très élevés. Des repères du quotidien sont intégrés directement à la liste pour situer ces échelles, sans prétendre à une équivalence observée directe.'],
      ['Benchmarks integrated into the chart: household electricity for <strong>2,760,139 households</strong> over one year of domestic use, i.e. ≈ <strong>6.90 TWh</strong> based on an average consumption of 2,500 kWh per household (RTE, 2021 estimate), and full-flight aviation derived from Klöwer et al. (2025) from 577.97 MtCO2 and 27.45 million commercial flights observed in 2023, i.e. ≈ <strong>6,166,824.9 tCO2e</strong> for <strong>292,210 full flights</strong>. These comparison points are aligned with the current central screening order of magnitude of Claude Opus 4.1 in the training chart, not with a direct provider-side measurement.', 'Repères intégrés au graphique : électricité domestique pour <strong>2 760 139 foyers</strong> sur une année d’usage résidentiel, soit ≈ <strong>6,90 TWh</strong> sur la base d’une consommation moyenne de 2 500 kWh par foyer (estimation RTE 2021), et aviation commerciale complète dérivée de Klöwer et al. (2025) à partir de 577.97 MtCO2 et 27.45 millions de vols commerciaux observés en 2023, soit ≈ <strong>6 166 824,9 tCO2e</strong> pour <strong>292 210 vols complets</strong>. Ces repères sont alignés sur l’ordre de grandeur central actuel du screening pour Claude Opus 4.1 dans le graphique d’entraînement, et non sur une mesure directe côté fournisseur.'],
      ['current models with estimated training impacts', 'modèles actuels avec impacts d’entraînement estimés'],
      ['This table projects the training orders of magnitude of current models from the indicator families actually available in the literature: <strong>training energy</strong> derived from emissions when the source country is documented in the electricity-mix table, and <strong>direct training CO2e</strong>. The current screening proxy combines retained parameter count, a training-token prior, a training-regime prior, architecture features, and a hardware-class proxy. Training energy therefore remains a more fragile screening reconstruction than direct carbon.', 'Ce tableau projette les ordres de grandeur d’entraînement des modèles actuels à partir des familles d’indicateurs réellement disponibles dans la littérature : <strong>l’énergie d’entraînement</strong>, dérivée des émissions lorsque le pays source est documenté dans la table des mixes électriques, et le <strong>CO2e direct d’entraînement</strong>. Le proxy de screening actuel combine le nombre de paramètres retenu, un prior sur les tokens d’entraînement, un prior sur le régime d’entraînement, des caractéristiques d’architecture et un proxy de classe matérielle. L’énergie d’entraînement reste donc une reconstruction de screening plus fragile que le carbone direct.'],
      ['`*` indicates an estimated parameter count rather than a provider-published value.', '`*` indique un nombre de paramètres estimé plutôt qu’une valeur publiée par le fournisseur.'],
      ['Source annex used in the site', 'Annexe des sources utilisées sur le site'],
      ['Inference reference set', 'Jeu de références pour l’inférence'],
      ['Training reference set', 'Jeu de références pour l’entraînement'],
      ['Real-world comparison benchmarks', 'Repères de comparaison du monde réel'],
      ['Central screening factors retained for market models', 'Facteurs centraux de screening retenus pour les modèles du marché'],
      ['This table documents the central values retained by the project for the multi-factor prompt proxy of each catalog model: raw active parameters, context window, serving mode, modality support, the resulting central factors <code>F_ctx</code>, <code>F_srv</code>, <code>F_mod</code>, <code>F_arch</code>, and the resulting central effective active-parameter proxy <code>P_eff,c</code>. These are project screening factors, not provider-published measurements.', 'Ce tableau documente les valeurs centrales retenues par le projet pour le proxy prompt multi-facteurs de chaque modèle du catalogue : paramètres actifs bruts, fenêtre de contexte, mode de service, support de modalité, facteurs centraux résultants <code>F_ctx</code>, <code>F_srv</code>, <code>F_mod</code>, <code>F_arch</code>, ainsi que le proxy central de paramètres actifs effectifs <code>P_eff,c</code>. Il s’agit de facteurs de screening du projet, et non de mesures publiées par les fournisseurs.'],
      ['Central training screening factors retained for market models', 'Facteurs centraux de screening retenus pour l’entraînement des modèles du marché'],
      ['This table documents the central values retained by the project for the multi-factor training proxy of each catalog model: retained training parameter count, training-token prior, training regime, multimodal training flag, hardware-class proxy, and the resulting central factors <code>F_reg</code>, <code>F_arch-tr</code>, and <code>F_hw</code>. These are project screening factors, not provider-published measurements.', 'Ce tableau documente les valeurs centrales retenues par le projet pour le proxy multi-facteurs d’entraînement de chaque modèle du catalogue : nombre de paramètres d’entraînement retenu, prior sur les tokens d’entraînement, régime d’entraînement, indicateur d’entraînement multimodal, proxy de classe matérielle, ainsi que les facteurs centraux résultants <code>F_reg</code>, <code>F_arch-tr</code> et <code>F_hw</code>. Il s’agit de facteurs de screening du projet, et non de mesures publiées par les fournisseurs.'],
      ['Numbered source list for retained screening characteristics', 'Liste numérotée des sources des caractéristiques de screening retenues'],
      ['The numbered references used in the retained inference and training screening-characteristic tables are listed below.', 'Les références numérotées utilisées dans les tableaux de caractéristiques de screening retenues pour l’inférence et l’entraînement sont listées ci-dessous.'],
      ['Country factors for carbon and water recalculation', 'Facteurs pays pour le recalcul du carbone et de l’eau'],
      ['Model', 'Modèle'],
      ['Parameters', 'Paramètres'],
      ['Active parameters', 'Paramètres actifs'],
      ['Retained parameters', 'Paramètres retenus'],
      ['Context window', 'Fenêtre de contexte'],
      ['Serving mode', 'Mode de service'],
      ['Vision', 'Vision'],
      ['Training tokens', 'Tokens d’entraînement'],
      ['Training regime', 'Régime d’entraînement'],
      ['Multimodal', 'Multimodal'],
      ['Hardware class', 'Classe matérielle'],
      ['Server country', 'Pays du serveur'],
      ['Retained country', 'Pays retenu'],
      ['Energy / h', 'Énergie / h'],
      ['Carbon / h', 'Carbone / h'],
      ['Energy / request', 'Énergie / requête'],
      ['Carbon / request', 'Carbone / requête'],
      ['Context: ', 'Contexte : '],
      ['Modalities: ', 'Modalités : '],
      ['Architecture: ', 'Architecture : '],
      ['The market-model comparison now relies on <code>market_multifactor_prompt_proxy_v1</code>: a prompt-energy screening proxy whose main prompt-level calibration anchor comes from Elsworth et al. (2025), then adjusted by active parameters, context window, serving mode, modality support, architecture overhead, and standardized token volume, and interpreted alongside other inference references.', 'La comparaison des modèles du marché repose désormais sur <code>market_multifactor_prompt_proxy_v1</code> : un proxy de screening en énergie par prompt dont l’ancrage principal de calibration au niveau prompt vient de Elsworth et al. (2025), puis ajusté selon les paramètres actifs, la fenêtre de contexte, le mode de service, le support multimodal, l’overhead d’architecture et un volume de tokens standardisé, et interprété aux côtés d’autres références d’inférence.'],
      ['The market-model comparison now relies on market_multifactor_prompt_proxy_v1: a prompt-energy screening proxy whose main prompt-level calibration anchor comes from Elsworth et al. (2025), then adjusted by active parameters, context window, serving mode, modality support, architecture overhead, and standardized token volume, and interpreted alongside other inference references.', 'La comparaison des modèles du marché repose désormais sur market_multifactor_prompt_proxy_v1 : un proxy de screening en énergie par prompt dont l’ancrage principal de calibration au niveau prompt vient de Elsworth et al. (2025), puis ajusté selon les paramètres actifs, la fenêtre de contexte, le mode de service, le support multimodal, l’overhead d’architecture et un volume de tokens standardisé, et interprété aux côtés d’autres références d’inférence.'],
      ['Training energy', 'Énergie d’entraînement'],
      ['Direct training CO2e', 'CO2e direct d’entraînement'],
      ['Training parameters', 'Paramètres d’entraînement'],
      ['Training modality', 'Modalité d’entraînement'],
      ['Training hardware', 'Matériel d’entraînement'],
      ['Provider', 'Fournisseur'],
      ['Confidence', 'Confiance'],
      ['Notes', 'Notes'],
      ['No.', 'N°'],
      ['Domain', 'Domaine'],
      ['Indicator', 'Indicateur'],
      ['Reference', 'Référence'],
      ['Data type', 'Type de donnée'],
      ['LLM model', 'Modèle LLM'],
      ['Country', 'Pays'],
      ['Value', 'Valeur'],
      ['Citation', 'Citation'],
      ['Active parameters', 'Paramètres actifs'],
      ['Total parameters', 'Paramètres totaux'],
      ['Status', 'Statut'],
      ['Source', 'Source'],
      ['Electricity mix reference table', 'Table de référence du mix électrique'],
      ['Model reference table', 'Table de référence des modèles'],
      ['Annual energy', 'Énergie annuelle'],
      ['Annual carbon', 'Carbone annuel'],
      ['Fluorescent lamp 1 h', 'Lampe fluorescente 1 h'],
      ['Laptop 1 h', 'Ordinateur portable 1 h'],
      ['Electric heater 4.1 min', 'Radiateur électrique 4,1 min'],
      ['Electric heater 10 min (US mix)', 'Radiateur électrique 10 min (mix US)'],
      ['2,760,139 households (annual domestic use)', '2 760 139 foyers (usage domestique annuel)'],
      ['292,210 full commercial flights', '292 210 vols commerciaux complets'],
    ];
    const attributeTranslations = [
      ['#description', 'placeholder', 'estimatePlaceholder'],
      ['.example-prompts-label', 'textContent', 'examplePromptsLabel'],
      ['#market-model-search', 'placeholder', 'marketSearchPlaceholder'],
      ['#training-model-search', 'placeholder', 'trainingSearchPlaceholder'],
      ['nav.tabs', 'aria-label', 'navAriaLabel'],
      ['[data-tab-target="home"]', 'aria-label', 'homeAriaLabel'],
      ['.language-label', 'textContent', 'languageLabel'],
    ];
    const htmlTranslations = [
      ['[data-i18n-html="method-anchor-body"]', 'methodAnchorBody'],
      ['[data-i18n-html="method-proxy-body"]', 'methodProxyBody'],
      ['[data-i18n-html="method-tokens-body"]', 'methodTokensBody'],
      ['[data-i18n-html="method-bound-body"]', 'methodBoundBody'],
      ['[data-i18n-html="method-carbon-body"]', 'methodCarbonBody'],
      ['[data-i18n-html="method-research-body"]', 'methodResearchBody'],
    ];
    const benchmarkLabelMap = {{
      'Lampe fluorescente 1 h': 'Fluorescent lamp 1 h',
      'Ordinateur portable 1 h': 'Laptop 1 h',
      'Fluorescent lamp 1 h': 'Fluorescent lamp 1 h',
      'Laptop 1 h': 'Laptop 1 h',
      'Electric heater 4.1 min': 'Electric heater 4.1 min',
      'Electric heater 10 min (US mix)': 'Electric heater 10 min (US mix)',
      '2,760,139 households (annual domestic use)': '2,760,139 households (annual domestic use)',
      '292,210 full commercial flights': '292,210 full commercial flights',
    }};
    function normalizeLanguage(value) {{
      return value === 'fr' ? 'fr' : 'en';
    }}
    function safeStorageGet(key) {{
      try {{
        return window.localStorage.getItem(key);
      }} catch (error) {{
        return null;
      }}
    }}
    function safeStorageSet(key, value) {{
      try {{
        window.localStorage.setItem(key, value);
      }} catch (error) {{
        // Ignore browsers or privacy modes that block storage access.
      }}
    }}
    function translateText(text, lang) {{
      if (!text || lang === 'en') return text;
      let translated = text;
      textReplacements.forEach(([english, french]) => {{
        translated = translated.split(english).join(french);
      }});
      return translated;
    }}
    function translateBenchmarkLabel(label, lang) {{
      const canonical = benchmarkLabelMap[label] || label;
      return translateText(canonical, lang);
    }}
    function applyAttributeTranslations(lang) {{
      const locale = uiText[lang];
      attributeTranslations.forEach(([selector, kind, key]) => {{
        const element = document.querySelector(selector);
        if (!element) return;
        if (kind === 'textContent') {{
          element.textContent = locale[key];
          return;
        }}
        element.setAttribute(kind, locale[key]);
      }});
      document.title = locale.title;
      document.documentElement.lang = lang;
    }}
    function applyHtmlTranslations(lang) {{
      const locale = uiText[lang];
      htmlTranslations.forEach(([selector, key]) => {{
        const element = document.querySelector(selector);
        if (!element) return;
        element.innerHTML = locale[key];
      }});
    }}
    function updateExamplePrompts(lang) {{
      const locale = uiText[lang] || uiText.en;
      const prompts = [locale.examplePrompt1, locale.examplePrompt2, locale.examplePrompt3];
      examplePromptButtons.forEach((button, index) => {{
        const value = prompts[index] || '';
        button.textContent = value;
        button.setAttribute('data-example-prompt', value);
      }});
    }}
    function applyTextTranslations(lang) {{
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {{
        acceptNode(node) {{
          if (!node.parentElement) return NodeFilter.FILTER_REJECT;
          if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          if (node.parentElement.closest('script, style, textarea, pre, code, svg, .notranslate')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }}
      }});
      const nodes = [];
      let currentNode;
      while ((currentNode = walker.nextNode())) {{
        nodes.push(currentNode);
      }}
      nodes.forEach((node) => {{
        if (!textNodeOriginals.has(node)) {{
          textNodeOriginals.set(node, node.nodeValue);
        }}
        node.nodeValue = translateText(textNodeOriginals.get(node), lang);
      }});
    }}
    function preferredLanguage() {{
      const stored = safeStorageGet(LANGUAGE_STORAGE_KEY);
      if (stored === 'fr' || stored === 'en') return stored;
      return (navigator.language || '').toLowerCase().startsWith('fr') ? 'fr' : 'en';
    }}
    function applyLanguage(lang) {{
      const normalized = normalizeLanguage(lang);
      currentLanguage.value = normalized;
      languageLinks.forEach((link) => {{
        const isActive = link.getAttribute('data-language-option') === normalized;
        link.classList.toggle('is-active', isActive);
        link.setAttribute('aria-current', isActive ? 'true' : 'false');
      }});
      safeStorageSet(LANGUAGE_STORAGE_KEY, normalized);
      applyAttributeTranslations(normalized);
      applyHtmlTranslations(normalized);
      updateExamplePrompts(normalized);
      applyTextTranslations(normalized);
      renderModelsChart();
      renderParamsChart();
      renderScatterChart();
      renderFactorHeatmap();
      renderScatterLinearChart();
      renderCountryMixChart();
      renderTrainingChart();
      renderTrainingScatterChart();
      renderTrainingScatterLogChart();
      if (currentModelDetailKey) {{
        renderModelDetail(currentModelDetailKey);
      }}
      if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {{
        window.MathJax.typesetPromise();
      }}
    }}
    function hasEnoughDescriptionContent() {{
      return (descriptionInput?.value || '').trim().length >= MIN_DESCRIPTION_LENGTH;
    }}
    function updateSubmitState() {{
      if (!submitButton) return;
      if (submitButton.classList.contains('is-loading')) {{
        submitButton.disabled = true;
        return;
      }}
      submitButton.disabled = !hasEnoughDescriptionContent();
    }}
    if (estimateForm && submitButton) {{
      estimateForm.addEventListener('submit', function (event) {{
        if (!hasEnoughDescriptionContent()) {{
          event.preventDefault();
          submitButton.disabled = true;
          return;
        }}
        submitButton.disabled = true;
        submitButton.classList.add('is-loading');
      }});
    }}
    if (descriptionInput) {{
      descriptionInput.addEventListener('input', updateSubmitState);
    }}
    examplePromptButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        if (!descriptionInput) return;
        const value = button.getAttribute('data-example-prompt') || '';
        descriptionInput.value = value;
        descriptionInput.focus();
        descriptionInput.setSelectionRange(descriptionInput.value.length, descriptionInput.value.length);
        updateSubmitState();
      }});
    }});
    document.addEventListener('click', (event) => {{
      const trigger = event.target.closest('[data-model-detail-key]');
      if (trigger) {{
        renderModelDetail(trigger.getAttribute('data-model-detail-key'));
      }}
    }});
    if (modelDetailClose) {{
      modelDetailClose.addEventListener('click', closeModelDetail);
    }}
    if (modelDetailOverlay) {{
      modelDetailOverlay.addEventListener('click', closeModelDetail);
    }}
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape') {{
        closeModelDetail();
      }}
    }});
    updateSubmitState();
    function activateTab(target) {{
      let matched = false;
      tabButtons.forEach((item) => {{
        const isActive = item.getAttribute('data-tab-target') === target;
        item.classList.toggle('is-active', isActive);
        if (isActive) matched = true;
      }});
      tabPanels.forEach((panel) => {{
        panel.classList.toggle('is-active', panel.getAttribute('data-tab-panel') === target);
      }});
      return matched;
    }}
    function activateSubtab(target) {{
      let matched = false;
      subtabButtons.forEach((item) => {{
        const isActive = item.getAttribute('data-subtab-target') === target;
        item.classList.toggle('is-active', isActive);
        if (item.hasAttribute('aria-selected')) {{
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }}
        if (isActive) matched = true;
      }});
      subtabPanels.forEach((panel) => {{
        panel.classList.toggle('is-active', panel.getAttribute('data-subtab-panel') === target);
      }});
      return matched;
    }}
    function setTabHash(value) {{
      const nextHash = `${{TAB_HASH_PREFIX}}${{value}}`;
      if (window.location.hash === nextHash) return;
      window.history.replaceState(null, '', nextHash);
    }}
    function applyHashNavigation() {{
      const hash = window.location.hash || '';
      if (!hash.startsWith(TAB_HASH_PREFIX)) return false;
      const target = hash.slice(TAB_HASH_PREFIX.length);
      if (!target) return false;
      const parentTab = target.includes('-') ? target.split('-')[0] : '';
      if (parentTab && tabDefaultSubtabs[parentTab]) {{
        activateTab(parentTab);
        return activateSubtab(target);
      }}
      return activateTab(target);
    }}
    tabButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        const target = button.getAttribute('data-tab-target');
        activateTab(target);
        if (tabDefaultSubtabs[target]) {{
          const activeSubtab = document.querySelector(`[data-subtab-target^="${{target}}-"].is-active`);
          const nextSubtab = activeSubtab ? activeSubtab.getAttribute('data-subtab-target') : tabDefaultSubtabs[target];
          activateSubtab(nextSubtab);
          setTabHash(nextSubtab);
          return;
        }}
        setTabHash(target);
      }});
    }});
    subtabButtons.forEach((button) => {{
      button.addEventListener('click', function () {{
        const target = button.getAttribute('data-subtab-target');
        const parentTab = target.includes('-') ? target.split('-')[0] : 'observatory';
        activateTab(parentTab);
        activateSubtab(target);
        setTabHash(target);
      }});
    }});
    window.addEventListener('hashchange', () => {{
      applyHashNavigation();
    }});
    applyHashNavigation();
    if (resultsAnchor && resultsAnchor.textContent.trim()) {{
      window.requestAnimationFrame(() => {{
        resultsAnchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }});
    }}
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
    languageLinks.forEach((link) => {{
      link.addEventListener('click', function (event) {{
        event.preventDefault();
        applyLanguage(link.getAttribute('data-language-option'));
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
      const locale = uiText[currentLanguage.value];
      if (!rows.length) {{
        return `<p class="lead">${{locale.noData}}</p>`;
      }}
      const key = `${{family}}_${{metric}}_${{metric === 'energy' ? 'wh' : metric === 'carbon' ? 'gco2e' : 'ml'}}`;
      const sorted = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          value: Number(row[key] || 0),
          kind: row.kind || 'model',
        }}))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value);
      if (!sorted.length) {{
        return `<p class="lead">${{locale.noUsableValue}}</p>`;
      }}
      const maxValue = sorted[0].value || 1;
      const barHeight = 28;
      const rowGap = 22;
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
          <text x="0" y="${{y + 18}}" font-size="15" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 34}}" font-size="13" fill="#6c757d">${{row.provider}}</text>
          <rect x="${{barStart}}" y="${{y}}" width="${{width}}" height="${{barHeight}}" rx="4" fill="${{fill}}"></rect>
          <text x="${{barStart + width + 10}}" y="${{y + 18}}" font-size="14" fill="#212529">${{valueText}}</text>
        `;
      }}).join('');
      const titleMetric = metric === 'energy' ? locale.chartMetricEnergy : locale.chartMetricCarbon;
      const titleFamily = family === 'prompt' ? locale.chartFamilyPrompt : locale.chartFamilyPage;
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{locale.comparisonEstimatedPrefix}}<strong>${{titleMetric}}</strong>${{locale.comparisonEstimatedMiddle}}<strong>${{titleFamily}}</strong>${{locale.comparisonEstimatedSuffix}}</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="${{locale.modelsAriaLabel}}">${{bars}}</svg>
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
    const formatParamsChartValue = (value) => {{
      if (value >= 1000) return `${{(value / 1000).toFixed(2)}} T active params`;
      if (value >= 100) return `${{value.toFixed(0)}}B active params`;
      if (value >= 10) return `${{value.toFixed(1)}}B active params`;
      return `${{value.toFixed(3)}}B active params`;
    }};
    const buildParamsChartMarkup = (rows) => {{
      const locale = uiText[currentLanguage.value];
      if (!rows.length) {{
        return `<p class="lead">${{locale.noData}}</p>`;
      }}
      const sorted = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          value: Number(row.effective_active_parameters_billion || row.value || 0),
          kind: row.kind || 'model',
        }}))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value);
      if (!sorted.length) {{
        return `<p class="lead">${{locale.noUsableValue}}</p>`;
      }}
      const maxValue = sorted[0].value || 1;
      const barHeight = 28;
      const rowGap = 22;
      const chartWidth = 980;
      const labelWidth = 320;
      const valueWidth = 170;
      const barStart = labelWidth + 12;
      const barMaxWidth = chartWidth - labelWidth - valueWidth - 40;
      const chartHeight = sorted.length * (barHeight + rowGap) + 24;
      const bars = sorted.map((row, index) => {{
        const y = 12 + index * (barHeight + rowGap);
        const width = Math.max(2, (row.value / maxValue) * barMaxWidth);
        return `
          <text x="0" y="${{y + 18}}" font-size="15" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 34}}" font-size="13" fill="#6c757d">${{row.provider}}</text>
          <rect x="${{barStart}}" y="${{y}}" width="${{width}}" height="${{barHeight}}" rx="4" fill="#3f5a49"></rect>
          <text x="${{barStart + width + 10}}" y="${{y + 18}}" font-size="14" fill="#212529">${{formatParamsChartValue(row.value)}}</text>
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Comparaison des valeurs centrales de <strong>P_eff,c</strong> retenues par le proxy multi-facteurs pour les modèles du catalogue.'
        : 'Comparison of the central <strong>P_eff,c</strong> values retained by the multi-factor proxy for catalog models.';
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="Effective active-parameter proxy chart">${{bars}}</svg>
      `;
    }};
    const renderParamsChart = () => {{
      if (!paramsChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(paramsChart.getAttribute('data-params-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      paramsChart.innerHTML = buildParamsChartMarkup(rows);
    }};
    renderParamsChart();
    const normalizeFeatureMatrix = (rows, featureKeys) => {{
      const matrix = rows.map((row) => featureKeys.map((key) => Number(row[key] || 0)));
      const means = featureKeys.map((_, columnIndex) => matrix.reduce((sum, vector) => sum + vector[columnIndex], 0) / Math.max(matrix.length, 1));
      const stdevs = featureKeys.map((_, columnIndex) => {{
        const variance = matrix.reduce((sum, vector) => sum + ((vector[columnIndex] - means[columnIndex]) ** 2), 0) / Math.max(matrix.length - 1, 1);
        return Math.sqrt(variance) || 1;
      }});
      return matrix.map((vector) => vector.map((value, columnIndex) => (value - means[columnIndex]) / stdevs[columnIndex]));
    }};
    const covarianceMatrix = (matrix) => {{
      const dimension = matrix[0]?.length || 0;
      const covariance = Array.from({{ length: dimension }}, () => Array.from({{ length: dimension }}, () => 0));
      matrix.forEach((vector) => {{
        for (let i = 0; i < dimension; i += 1) {{
          for (let j = 0; j < dimension; j += 1) {{
            covariance[i][j] += vector[i] * vector[j];
          }}
        }}
      }});
      const divisor = Math.max(matrix.length - 1, 1);
      for (let i = 0; i < dimension; i += 1) {{
        for (let j = 0; j < dimension; j += 1) {{
          covariance[i][j] /= divisor;
        }}
      }}
      return covariance;
    }};
    const matrixVectorMultiply = (matrix, vector) => matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
    const normalizeVector = (vector) => {{
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + (value ** 2), 0)) || 1;
      return vector.map((value) => value / norm);
    }};
    const dotProduct = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
    const powerIteration = (matrix, orthogonalTo = null) => {{
      const dimension = matrix.length;
      let vector = normalizeVector(Array.from({{ length: dimension }}, (_, index) => 1 + index));
      for (let iteration = 0; iteration < 40; iteration += 1) {{
        let next = matrixVectorMultiply(matrix, vector);
        if (orthogonalTo) {{
          const projection = dotProduct(next, orthogonalTo);
          next = next.map((value, index) => value - projection * orthogonalTo[index]);
        }}
        vector = normalizeVector(next);
      }}
      return vector;
    }};
    const projectRows = (matrix, vectors) => matrix.map((row) => vectors.map((vector) => dotProduct(row, vector)));
    const kMeans = (matrix, k) => {{
      const centroids = matrix.slice(0, k).map((row) => row.slice());
      const assignments = Array.from({{ length: matrix.length }}, () => 0);
      for (let iteration = 0; iteration < 12; iteration += 1) {{
        matrix.forEach((row, rowIndex) => {{
          let bestIndex = 0;
          let bestDistance = Number.POSITIVE_INFINITY;
          centroids.forEach((centroid, centroidIndex) => {{
            const distance = row.reduce((sum, value, index) => sum + ((value - centroid[index]) ** 2), 0);
            if (distance < bestDistance) {{
              bestDistance = distance;
              bestIndex = centroidIndex;
            }}
          }});
          assignments[rowIndex] = bestIndex;
        }});
        centroids.forEach((_, centroidIndex) => {{
          const members = matrix.filter((_, rowIndex) => assignments[rowIndex] === centroidIndex);
          if (!members.length) return;
          centroids[centroidIndex] = centroids[centroidIndex].map((_, dimensionIndex) => members.reduce((sum, row) => sum + row[dimensionIndex], 0) / members.length);
        }});
      }}
      return assignments;
    }};
    const buildLandscapeMarkup = (rows, config) => {{
      const locale = uiText[currentLanguage.value];
      const points = rows.map((row) => ({{
        label: translateBenchmarkLabel(row.label, currentLanguage.value),
        provider: row.provider,
        features: row,
      }}));
      if (!points.length) {{
        return `<p class="lead">${{locale.noData}}</p>`;
      }}
      const matrix = normalizeFeatureMatrix(points.map((point) => point.features), config.featureKeys);
      if (!matrix.length || !matrix[0]?.length) {{
        return `<p class="lead">${{locale.noUsableValue}}</p>`;
      }}
      const covariance = covarianceMatrix(matrix);
      const pc1 = powerIteration(covariance);
      const pc2 = powerIteration(covariance, pc1);
      const coordinates = projectRows(matrix, [pc1, pc2]);
      const clusters = kMeans(matrix, Math.min(4, Math.max(2, Math.floor(Math.sqrt(points.length / 2)))));
      const palette = ['#3f5a49', '#8c7a5b', '#243b63', '#b85c38'];
      const width = 980;
      const height = 620;
      const padding = {{ top: 24, right: 24, bottom: 58, left: 58 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const xs = coordinates.map((point) => point[0]);
      const ys = coordinates.map((point) => point[1]);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      const scaleX = (value) => padding.left + ((value - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((value - yMin) / Math.max(yMax - yMin, 1e-9)) * plotHeight;
      const renderedPoints = points.map((point, index) => {{
        const cx = scaleX(coordinates[index][0]);
        const cy = scaleY(coordinates[index][1]);
        const fill = palette[clusters[index] % palette.length];
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="6" fill="${{fill}}" opacity="0.9"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{point.label}}</text>
        `;
      }}).join('');
      const legend = palette.slice(0, Math.max(...clusters) + 1).map((color, index) => `
        <g transform="translate(${{padding.left + index * 130}}, ${{height - 28}})">
          <rect width="14" height="14" rx="3" fill="${{color}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{currentLanguage.value === 'fr' ? `Cluster ${{index + 1}}` : `Cluster ${{index + 1}}`}}</text>
        </g>
      `).join('');
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel}}">
          <rect x="${{padding.left}}" y="${{padding.top}}" width="${{plotWidth}}" height="${{plotHeight}}" fill="rgba(140, 122, 91, 0.04)" stroke="rgba(0,0,0,0.08)"></rect>
          ${{renderedPoints}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{config.axisLabels.x[currentLanguage.value]}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{config.axisLabels.y[currentLanguage.value]}}</text>
        </svg>
      `;
    }};
    const renderScatterChart = () => {{
      if (!scatterChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(scatterChart.getAttribute('data-scatter-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      scatterChart.innerHTML = buildLandscapeMarkup(rows, {{
        featureKeys: [
          'active_parameters_billion',
          'effective_active_parameters_billion',
          'context_window_tokens',
          'serving_mode_score',
          'vision_support_score',
          'moe_score',
          'reasoning_score',
          'hour_energy_wh',
          'hour_carbon_gco2e',
          'request_energy_wh',
          'request_carbon_gco2e',
        ],
        intro: {{
          en: 'This clustered landscape is derived from the full inference screening profile retained by the project. Points that appear close share similar combinations of size, context, serving assumptions, modality support, architecture notes, and central impact outputs.',
          fr: 'Cette carte groupée est dérivée du profil complet de screening retenu par le projet pour l’inférence. Les points proches partagent des combinaisons similaires de taille, contexte, hypothèses de service, support multimodal, notes d’architecture et impacts centraux.'
        }},
        axisLabels: {{
          x: {{ en: 'Landscape dimension 1 (composite projection)', fr: 'Dimension 1 du paysage (projection composite)' }},
          y: {{ en: 'Landscape dimension 2 (composite projection)', fr: 'Dimension 2 du paysage (projection composite)' }},
        }},
        ariaLabel: 'Inference model landscape chart',
      }});
    }};
    renderScatterChart();
    const renderFactorHeatmap = () => {{
      if (!factorHeatmap) return;
      let rows = [];
      try {{
        rows = JSON.parse(factorHeatmap.getAttribute('data-factor-heatmap-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      if (!rows.length) {{
        factorHeatmap.innerHTML = `<p class="lead">${{locale.noData}}</p>`;
        return;
      }}
      const metrics = [
        {{ key: 'f_ctx', label: currentLanguage.value === 'fr' ? 'F_ctx' : 'F_ctx' }},
        {{ key: 'f_srv', label: currentLanguage.value === 'fr' ? 'F_srv' : 'F_srv' }},
        {{ key: 'f_mod', label: currentLanguage.value === 'fr' ? 'F_mod' : 'F_mod' }},
        {{ key: 'f_arch', label: currentLanguage.value === 'fr' ? 'F_arch' : 'F_arch' }},
        {{ key: 'p_eff_ratio', label: currentLanguage.value === 'fr' ? 'P_eff / P_raw' : 'P_eff / P_raw' }},
      ];
      const width = 980;
      const rowHeight = 34;
      const headerHeight = 34;
      const labelWidth = 280;
      const cellWidth = 120;
      const height = headerHeight + rows.length * rowHeight + 20;
      const colorForValue = (value) => {{
        const normalized = Math.max(0, Math.min(1, (Number(value || 0) - 0.8) / 0.7));
        const lightness = 94 - normalized * 42;
        return `hsl(147, 28%, ${{lightness}}%)`;
      }};
      const header = metrics.map((metric, index) => `
        <text x="${{labelWidth + index * cellWidth + cellWidth / 2}}" y="22" text-anchor="middle" font-size="13" fill="#495057">${{metric.label}}</text>
      `).join('');
      const body = rows.map((row, rowIndex) => {{
        const y = headerHeight + rowIndex * rowHeight;
        const label = translateBenchmarkLabel(row.label, currentLanguage.value);
        const provider = row.provider || '';
        const cells = metrics.map((metric, metricIndex) => {{
          const value = Number(row[metric.key] || 0);
          const x = labelWidth + metricIndex * cellWidth;
          return `
            <rect x="${{x}}" y="${{y}}" width="${{cellWidth - 10}}" height="${{rowHeight - 6}}" rx="4" fill="${{colorForValue(value)}}" stroke="rgba(0,0,0,0.08)"></rect>
            <text x="${{x + (cellWidth - 10) / 2}}" y="${{y + 19}}" text-anchor="middle" font-size="12" fill="#212529">${{value.toFixed(3)}}</text>
          `;
        }}).join('');
        return `
          <text x="0" y="${{y + 15}}" font-size="13" fill="#212529">${{label}}</text>
          <text x="0" y="${{y + 29}}" font-size="11" fill="#6c757d">${{provider}}</text>
          ${{cells}}
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Plus la cellule est foncée, plus le facteur central retenu par le proxy est élevé.'
        : 'Darker cells indicate higher central screening factors retained by the proxy.';
      factorHeatmap.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Inference screening factor heatmap">
          ${{header}}
          ${{body}}
        </svg>
      `;
    }};
    renderFactorHeatmap();
    const renderTrainingFactorHeatmap = () => {{
      if (!trainingFactorHeatmap) return;
      let rows = [];
      try {{
        rows = JSON.parse(trainingFactorHeatmap.getAttribute('data-training-factor-heatmap-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      if (!rows.length) {{
        trainingFactorHeatmap.innerHTML = `<p class="lead">${{locale.noData}}</p>`;
        return;
      }}
      const metrics = [
        {{ key: 'f_reg', label: currentLanguage.value === 'fr' ? 'F_reg' : 'F_reg' }},
        {{ key: 'f_arch', label: currentLanguage.value === 'fr' ? 'F_arch' : 'F_arch' }},
        {{ key: 'f_hw', label: currentLanguage.value === 'fr' ? 'F_hw' : 'F_hw' }},
        {{ key: 'token_ratio', label: currentLanguage.value === 'fr' ? 'Tok / Param' : 'Tok / Param' }},
      ];
      const width = 980;
      const rowHeight = 34;
      const headerHeight = 34;
      const labelWidth = 280;
      const cellWidth = 140;
      const height = headerHeight + rows.length * rowHeight + 20;
      const colorForValue = (value, metricKey) => {{
        let normalized = 0;
        if (metricKey === 'token_ratio') {{
          normalized = Math.max(0, Math.min(1, Number(value || 0) / 0.04));
        }} else {{
          normalized = Math.max(0, Math.min(1, (Number(value || 0) - 0.7) / 0.6));
        }}
        const lightness = 94 - normalized * 42;
        return `hsl(205, 35%, ${{lightness}}%)`;
      }};
      const header = metrics.map((metric, index) => `
        <text x="${{labelWidth + index * cellWidth + cellWidth / 2}}" y="22" text-anchor="middle" font-size="13" fill="#495057">${{metric.label}}</text>
      `).join('');
      const body = rows.map((row, rowIndex) => {{
        const y = headerHeight + rowIndex * rowHeight;
        const label = translateBenchmarkLabel(row.label, currentLanguage.value);
        const provider = row.provider || '';
        const cells = metrics.map((metric, metricIndex) => {{
          const value = Number(row[metric.key] || 0);
          const x = labelWidth + metricIndex * cellWidth;
          return `
            <rect x="${{x}}" y="${{y}}" width="${{cellWidth - 10}}" height="${{rowHeight - 6}}" rx="4" fill="${{colorForValue(value, metric.key)}}" stroke="rgba(0,0,0,0.08)"></rect>
            <text x="${{x + (cellWidth - 10) / 2}}" y="${{y + 19}}" text-anchor="middle" font-size="12" fill="#212529">${{metric.key === 'token_ratio' ? value.toFixed(3) : value.toFixed(3)}}</text>
          `;
        }}).join('');
        return `
          <text x="0" y="${{y + 15}}" font-size="13" fill="#212529">${{label}}</text>
          <text x="0" y="${{y + 29}}" font-size="11" fill="#6c757d">${{provider}}</text>
          ${{cells}}
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Plus la cellule est foncée, plus le facteur central retenu par le proxy d’entraînement est élevé.'
        : 'Darker cells indicate higher central screening factors retained by the training proxy.';
      trainingFactorHeatmap.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Training screening factor heatmap">
          ${{header}}
          ${{body}}
        </svg>
      `;
    }};
    renderTrainingFactorHeatmap();
    const renderTrainingUncertaintyChart = () => {{
      if (!trainingUncertaintyChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(trainingUncertaintyChart.getAttribute('data-training-uncertainty-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider || '',
          low: Number(row.low || 0),
          central: Number(row.central || 0),
          high: Number(row.high || 0),
        }}))
        .filter((row) => row.central > 0 && row.high > 0)
        .sort((a, b) => b.central - a.central);
      if (!points.length) {{
        trainingUncertaintyChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const width = 980;
      const rowHeight = 34;
      const headerHeight = 10;
      const labelWidth = 280;
      const valueWidth = 620;
      const paddingRight = 40;
      const height = headerHeight + points.length * rowHeight + 24;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const minValue = Math.min(...points.map((row) => Math.max(row.low, 1e-9)));
      const maxValue = Math.max(...points.map((row) => row.high));
      const minLog = safeLog(minValue);
      const maxLog = safeLog(maxValue);
      const scaleX = (value) => labelWidth + ((safeLog(value) - minLog) / Math.max(maxLog - minLog, 1e-9)) * valueWidth;
      const tickValues = (() => {{
        const ticks = [];
        const start = Math.floor(minLog);
        const end = Math.ceil(maxLog);
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= minValue && value <= maxValue);
      }})();
      const grid = tickValues.map((value) => {{
        const x = scaleX(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}} kt` : `${{value.toFixed(1)}} t`;
        return `
          <line x1="${{x}}" y1="0" x2="${{x}}" y2="${{height - 18}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 2}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const rowsMarkup = points.map((row, index) => {{
        const y = headerHeight + index * rowHeight + 16;
        const xLow = scaleX(row.low);
        const xCentral = scaleX(row.central);
        const xHigh = scaleX(row.high);
        return `
          <text x="0" y="${{y - 2}}" font-size="13" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 11}}" font-size="11" fill="#6c757d">${{row.provider}}</text>
          <line x1="${{xLow}}" y1="${{y}}" x2="${{xHigh}}" y2="${{y}}" stroke="#8c7a5b" stroke-width="3" stroke-linecap="round"></line>
          <circle cx="${{xLow}}" cy="${{y}}" r="4" fill="#d6c9b5"></circle>
          <circle cx="${{xCentral}}" cy="${{y}}" r="5.5" fill="#243b63"></circle>
          <circle cx="${{xHigh}}" cy="${{y}}" r="4" fill="#b85c38"></circle>
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Chaque ligne montre la borne basse, la valeur centrale et la borne haute du CO2e direct d’entraînement retenu pour un modèle.'
        : 'Each line shows the low, central, and high retained values for direct training CO2e.';
      const xLabel = currentLanguage.value === 'fr' ? 'CO2e direct d’entraînement, tCO2e (échelle logarithmique)' : 'Direct training CO2e, tCO2e (log scale)';
      trainingUncertaintyChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Training uncertainty span by model">
          ${{grid}}
          ${{rowsMarkup}}
          <text x="${{labelWidth + valueWidth / 2}}" y="${{height - 2}}" text-anchor="middle" font-size="13" fill="#495057">${{xLabel}}</text>
        </svg>
      `;
    }};
    renderTrainingUncertaintyChart();
    const renderScatterLinearChart = () => {{
      if (!scatterLinearChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(scatterLinearChart.getAttribute('data-scatter-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          x: Number(row.active_parameters_billion || 0),
          y: Number(row.hour_carbon_gco2e || 0),
        }}))
        .filter((row) => row.x > 0 && row.y > 0);
      if (!points.length) {{
        scatterLinearChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const padding = {{ top: 24, right: 24, bottom: 58, left: 74 }};
      const width = 980;
      const height = 620;
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const xMin = Math.min(...points.map((row) => row.x));
      const xMax = Math.max(...points.map((row) => row.x));
      const yMin = Math.min(...points.map((row) => row.y));
      const yMax = Math.max(...points.map((row) => row.y));
      const xMinLog = safeLog(xMin);
      const xMaxLog = safeLog(xMax);
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((safeLog(value) - xMinLog) / Math.max(xMaxLog - xMinLog, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const tickValues = (min, max) => {{
        const ticks = [];
        const start = Math.floor(safeLog(min));
        const end = Math.ceil(safeLog(max));
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= min && value <= max);
      }};
      const xGrid = tickValues(xMin, xMax).map((value) => {{
        const x = scaleX(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}}T` : `${{value >= 10 ? value.toFixed(0) : value.toFixed(1)}}B`;
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yGrid = tickValues(yMin, yMax).map((value) => {{
        const y = scaleY(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`;
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const dots = points.map((row) => {{
        const cx = scaleX(row.x);
        const cy = scaleY(row.y);
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="#243b63" opacity="0.9"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{row.label}}</text>
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Positionnement des modèles selon leurs paramètres actifs retenus et leur carbone central d’inférence sur une heure, en échelle logarithmique.'
        : 'Positioning of models by retained active parameter count and central inference carbon over one hour, on logarithmic axes.';
      const xLabel = currentLanguage.value === 'fr' ? 'Paramètres actifs retenus' : 'Retained active parameters';
      const yLabel = currentLanguage.value === 'fr' ? 'Carbone d’inférence central, gCO2e/h' : 'Central inference carbon, gCO2e/h';
      scatterLinearChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Inference carbon versus parameter count chart">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{dots}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{xLabel}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{yLabel}}</text>
        </svg>
      `;
    }};
    renderScatterLinearChart();
    const renderCountryMixChart = () => {{
      if (!countryMixChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(countryMixChart.getAttribute('data-country-mix-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          country: row.country_code || 'n.d.',
          x: Number(row.hour_energy_wh || 0),
          y: Number(row.hour_carbon_gco2e || 0),
        }}))
        .filter((row) => row.x > 0 && row.y > 0);
      if (!points.length) {{
        countryMixChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const countries = Array.from(new Set(points.map((row) => row.country))).sort();
      const palette = ['#243b63', '#3f5a49', '#8c7a5b', '#b85c38', '#6c5b7b', '#2f7f92', '#7a9e2f', '#a33d5e'];
      const colorByCountry = Object.fromEntries(countries.map((country, index) => [country, palette[index % palette.length]]));
      const padding = {{ top: 24, right: 24, bottom: 58, left: 74 }};
      const width = 980;
      const height = 620;
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const xMax = Math.max(...points.map((row) => row.x));
      const yMax = Math.max(...points.map((row) => row.y));
      const scaleX = (value) => padding.left + (value / Math.max(xMax, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - (value / Math.max(yMax, 1e-9)) * plotHeight;
      const xGrid = Array.from({{ length: 6 }}, (_, index) => {{
        const value = (xMax / 5) * index;
        const x = scaleX(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}} kWh` : `${{value.toFixed(1)}} Wh`;
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yGrid = Array.from({{ length: 6 }}, (_, index) => {{
        const value = (yMax / 5) * index;
        const y = scaleY(value);
        const label = `${{value.toFixed(1)}} g`;
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const dots = points.map((row) => {{
        const cx = scaleX(row.x);
        const cy = scaleY(row.y);
        const fill = colorByCountry[row.country];
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="${{fill}}" opacity="0.9"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{row.label}}</text>
        `;
      }}).join('');
      const legend = countries.map((country, index) => `
        <g transform="translate(${{padding.left + (index % 4) * 150}}, ${{height - 28 - Math.floor(index / 4) * 18}})">
          <rect width="14" height="14" rx="3" fill="${{colorByCountry[country]}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{country}}</text>
        </g>
      `).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Les points de même couleur partagent le même pays de mix électrique retenu pour le recalcul du carbone.'
        : 'Points with the same color share the same retained electricity-mix country for carbon recalculation.';
      const xLabel = currentLanguage.value === 'fr' ? 'Énergie centrale d’inférence, Wh/h' : 'Central inference energy, Wh/h';
      const yLabel = currentLanguage.value === 'fr' ? 'Carbone central d’inférence, gCO2e/h' : 'Central inference carbon, gCO2e/h';
      countryMixChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Country mix sensitivity chart">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{dots}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{xLabel}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{yLabel}}</text>
        </svg>
      `;
    }};
    renderCountryMixChart();
    function providerDisplayName(provider) {{
      const mapping = {{
        openai: 'OpenAI',
        anthropic: 'Claude',
        xai: 'Grok',
        mistral: 'Mistral',
        google: 'Google',
        meta: 'Meta',
        deepseek: 'DeepSeek',
        alibaba: 'Alibaba',
        microsoft: 'Microsoft',
        ai21: 'AI21',
        nvidia: 'NVIDIA',
      }};
      return mapping[provider] || provider || '';
    }}
    function escapeHtml(value) {{
      return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
    }}
    function normalizeModelDetailKey(value) {{
      return String(value ?? '')
        .toLowerCase()
        .replace(/[\\s\\-_.:,/();]/g, '');
    }}
    function renderModelDetailRow(label, value) {{
      return `
      <div class="model-detail-row">
        <span class="model-detail-label">${{escapeHtml(label)}}</span>
        <span class="model-detail-value">${{escapeHtml(value || 'n.d.')}}</span>
      </div>
    `;
    }}
    function resolveModelDetail(key) {{
      if (!key) return null;
      if (modelDetailIndex[key]) return [key, modelDetailIndex[key]];
      const normalizedKey = normalizeModelDetailKey(key);
      const entries = Object.entries(modelDetailIndex);
      for (const [candidateKey, detail] of entries) {{
        if (normalizeModelDetailKey(candidateKey) === normalizedKey) return [candidateKey, detail];
        if (normalizeModelDetailKey(detail.model_id || '') === normalizedKey) return [candidateKey, detail];
        if (normalizeModelDetailKey(detail.display_name || '') === normalizedKey) return [candidateKey, detail];
      }}
      for (const [candidateKey, detail] of entries) {{
        const candidateValues = [
          normalizeModelDetailKey(candidateKey),
          normalizeModelDetailKey(detail.model_id || ''),
          normalizeModelDetailKey(detail.display_name || ''),
        ];
        if (candidateValues.some((value) => value && (value.includes(normalizedKey) || normalizedKey.includes(value)))) {{
          return [candidateKey, detail];
        }}
      }}
      return null;
    }}
    function closeModelDetail() {{
      if (!modelDetailDrawer || !modelDetailOverlay) return;
      modelDetailDrawer.classList.remove('is-open');
      modelDetailDrawer.setAttribute('aria-hidden', 'true');
      modelDetailOverlay.hidden = true;
      document.body.style.overflow = '';
      currentModelDetailKey = null;
    }}
    function renderModelDetail(key) {{
      if (!modelDetailDrawer || !modelDetailOverlay || !modelDetailContent || !modelDetailTitle) return;
      const resolved = resolveModelDetail(key);
      if (!resolved) return;
      const [resolvedKey, detail] = resolved;
      const text = currentLanguage.value === 'fr'
        ? {{
            overview: 'Vue d’ensemble',
            inference: 'Inférence',
            training: 'Entraînement',
            method: 'Méthode et hypothèses',
            sources: 'Sources',
            status: 'Statut',
            release: 'Sortie',
            country: 'Pays retenu',
            serverCountry: 'Pays serveur',
            parameters: 'Paramètres',
            effectiveParameters: 'Paramètres effectifs',
            context: 'Contexte',
            vision: 'Vision',
            inputs: 'Entrées',
            outputs: 'Sorties',
            energyLow: 'Énergie basse',
            energyCentral: 'Énergie centrale',
            energyHigh: 'Énergie haute',
            carbonLow: 'Carbone bas',
            carbonCentral: 'Carbone central',
            carbonHigh: 'Carbone haut',
            tokens: 'Tokens d’entraînement',
            regime: 'Régime',
            hardware: 'Matériel',
            multimodal: 'Multimodal',
            methodId: 'Méthode',
            anchor: 'Ancrage',
            notes: 'Notes',
          }}
        : {{
            overview: 'Overview',
            inference: 'Inference',
            training: 'Training',
            method: 'Method and assumptions',
            sources: 'Sources',
            status: 'Status',
            release: 'Release',
            country: 'Retained country',
            serverCountry: 'Server country',
            parameters: 'Parameters',
            effectiveParameters: 'Effective parameters',
            context: 'Context',
            vision: 'Vision',
            inputs: 'Inputs',
            outputs: 'Outputs',
            energyLow: 'Energy low',
            energyCentral: 'Energy central',
            energyHigh: 'Energy high',
            carbonLow: 'Carbon low',
            carbonCentral: 'Carbon central',
            carbonHigh: 'Carbon high',
            tokens: 'Training tokens',
            regime: 'Regime',
            hardware: 'Hardware',
            multimodal: 'Multimodal',
            methodId: 'Method',
            anchor: 'Anchor',
            notes: 'Notes',
          }};
      modelDetailTitle.textContent = detail.display_name || detail.model_id || 'Model detail';
      const sourcesMarkup = (detail.sources || []).map((entry) => `
        <li>
          <span class="model-detail-source-meta">${{escapeHtml(entry.label)}} | ${{escapeHtml(entry.status || 'n.d.')}}</span>
          ${{entry.url ? `<a href="${{escapeHtml(entry.url)}}" target="_blank" rel="noopener noreferrer">${{escapeHtml(entry.citation)}}</a>` : `<span>${{escapeHtml(entry.citation)}}</span>`}}
        </li>
      `).join('');
      modelDetailContent.innerHTML = `
        <p class="model-detail-copy">${{escapeHtml(detail.provider || '')}} | ${{escapeHtml(detail.model_id || '')}}</p>
        <div class="model-detail-pillbar">
          <span class="model-detail-pill">${{escapeHtml(detail.market_status || 'n.d.')}}</span>
          <span class="model-detail-pill">${{escapeHtml(detail.serving_mode || 'n.d.')}}</span>
          <span class="model-detail-pill">${{escapeHtml(detail.parameter_status || 'n.d.')}}</span>
        </div>
        <section class="model-detail-section">
          <h4>${{text.overview}}</h4>
          <div class="model-detail-grid">
            <div class="model-detail-card">
              <h4>${{text.overview}}</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.release, detail.release_date)}}
                ${{renderModelDetailRow(text.country, `${{detail.retained_country}} (${{detail.retained_country_status}})`)}}
                ${{renderModelDetailRow(text.serverCountry, detail.server_country)}}
                ${{renderModelDetailRow(text.parameters, detail.parameter_display)}}
                ${{renderModelDetailRow(text.effectiveParameters, `${{detail.effective_active_parameters_billion}}B`)}}
                ${{renderModelDetailRow(text.context, detail.context_window_tokens === 'n.d.' ? 'n.d.' : `${{detail.context_window_tokens}} tokens`)}}
              </div>
            </div>
            <div class="model-detail-card">
              <h4>${{text.overview}}</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.vision, detail.vision_support)}}
                ${{renderModelDetailRow(text.inputs, detail.input_modalities)}}
                ${{renderModelDetailRow(text.outputs, detail.output_modalities)}}
                ${{renderModelDetailRow(text.regime, detail.training_regime)}}
                ${{renderModelDetailRow(text.hardware, detail.training_hardware)}}
                ${{renderModelDetailRow(text.multimodal, detail.training_multimodal)}}
              </div>
            </div>
          </div>
        </section>
        <section class="model-detail-section">
          <h4>${{text.inference}}</h4>
          <div class="model-detail-grid">
            <div class="model-detail-card">
              <h4>Energy / h</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.energyLow, detail.inference.energy.low)}}
                ${{renderModelDetailRow(text.energyCentral, detail.inference.energy.central)}}
                ${{renderModelDetailRow(text.energyHigh, detail.inference.energy.high)}}
              </div>
            </div>
            <div class="model-detail-card">
              <h4>Carbon / h</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.carbonLow, detail.inference.carbon.low)}}
                ${{renderModelDetailRow(text.carbonCentral, detail.inference.carbon.central)}}
                ${{renderModelDetailRow(text.carbonHigh, detail.inference.carbon.high)}}
              </div>
            </div>
          </div>
        </section>
        <section class="model-detail-section">
          <h4>${{text.training}}</h4>
          <div class="model-detail-grid">
            <div class="model-detail-card">
              <h4>Energy</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.energyLow, detail.training.energy.low)}}
                ${{renderModelDetailRow(text.energyCentral, detail.training.energy.central)}}
                ${{renderModelDetailRow(text.energyHigh, detail.training.energy.high)}}
              </div>
            </div>
            <div class="model-detail-card">
              <h4>CO2e</h4>
              <div class="model-detail-list">
                ${{renderModelDetailRow(text.carbonLow, detail.training.carbon.low)}}
                ${{renderModelDetailRow(text.carbonCentral, detail.training.carbon.central)}}
                ${{renderModelDetailRow(text.carbonHigh, detail.training.carbon.high)}}
              </div>
            </div>
          </div>
        </section>
        <section class="model-detail-section">
          <h4>${{text.method}}</h4>
          <div class="model-detail-list">
            ${{renderModelDetailRow(text.methodId, detail.screening_method_id)}}
            ${{renderModelDetailRow(text.anchor, detail.screening_reference_anchor)}}
            ${{renderModelDetailRow(text.tokens, detail.training_tokens_estimate_trillion === 'n.d.' ? 'n.d.' : `${{detail.training_tokens_estimate_trillion}}T`)}}
            ${{renderModelDetailRow(text.methodId, detail.training_method_id)}}
            ${{renderModelDetailRow(text.anchor, detail.training_multifactor_anchor)}}
            ${{renderModelDetailRow(text.notes, detail.notes)}}
          </div>
          <p class="model-detail-copy" style="margin-top:0.9rem;">${{escapeHtml(detail.architecture_notes || 'n.d.')}}</p>
        </section>
        <section class="model-detail-section">
          <h4>${{text.sources}}</h4>
          <ul class="model-detail-source-list">${{sourcesMarkup || `<li><span>${{currentLanguage.value === 'fr' ? 'Aucune source disponible.' : 'No source available.'}}</span></li>`}}</ul>
        </section>
      `;
      currentModelDetailKey = resolvedKey;
      modelDetailOverlay.hidden = false;
      modelDetailDrawer.classList.add('is-open');
      modelDetailDrawer.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }}
    const renderInferenceTrainingTradeoffChart = () => {{
      if (!inferenceTrainingTradeoffChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(inferenceTrainingTradeoffChart.getAttribute('data-cross-impact-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const metricControl = document.querySelector('[data-cross-impact-control="metric-tab"].is-active');
      const metric = metricControl ? metricControl.getAttribute('data-metric-value') : 'energy';
      const config = metric === 'carbon'
        ? {{
            xKey: 'hour_carbon_gco2e',
            yKey: 'direct_training_carbon_tco2e',
            intro: {{
              en: 'Positioning of models by central inference carbon over one standardized hour of use and retained direct training CO2e. Both axes use logarithmic scaling because the model catalog spans several orders of magnitude.',
              fr: 'Positionnement des modèles selon leur carbone central d’inférence sur une heure d’usage standardisée et leur CO2e direct d’entraînement retenu. Les deux axes utilisent une échelle logarithmique car le catalogue couvre plusieurs ordres de grandeur.',
            }},
            xLabel: {{
              en: 'Central inference carbon, gCO2e/h (log scale)',
              fr: 'Carbone central d’inférence, gCO2e/h (échelle logarithmique)',
            }},
            yLabel: {{
              en: 'Direct training CO2e, tCO2e (log scale)',
              fr: 'CO2e direct d’entraînement, tCO2e (échelle logarithmique)',
            }},
            formatX: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`,
            formatY: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kt` : `${{value.toFixed(1)}} t`,
            ariaLabel: {{
              en: 'Inference versus training carbon chart',
              fr: 'Graphique du carbone d’inférence versus entraînement',
            }},
          }}
        : {{
            xKey: 'hour_energy_wh',
            yKey: 'direct_training_energy_wh',
            intro: {{
              en: 'Positioning of models by central inference energy over one standardized hour of use and retained direct training energy. Both axes use logarithmic scaling because the model catalog spans several orders of magnitude.',
              fr: 'Positionnement des modèles selon leur énergie centrale d’inférence sur une heure d’usage standardisée et leur énergie d’entraînement retenue. Les deux axes utilisent une échelle logarithmique car le catalogue couvre plusieurs ordres de grandeur.',
            }},
            xLabel: {{
              en: 'Central inference energy, Wh/h (log scale)',
              fr: 'Énergie centrale d’inférence, Wh/h (échelle logarithmique)',
            }},
            yLabel: {{
              en: 'Direct training energy, Wh (log scale)',
              fr: 'Énergie directe d’entraînement, Wh (échelle logarithmique)',
            }},
            formatX: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kWh` : `${{value.toFixed(1)}} Wh`,
            formatY: (value) => value >= 1e9 ? `${{(value / 1e9).toFixed(1)}} GWh` : value >= 1e6 ? `${{(value / 1e6).toFixed(1)}} MWh` : `${{value.toFixed(1)}} Wh`,
            ariaLabel: {{
              en: 'Inference versus training energy chart',
              fr: 'Graphique de l’énergie d’inférence versus entraînement',
            }},
          }};
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider || '',
          providerLabel: providerDisplayName(row.provider || ''),
          activeParameters: Number(row.active_parameters_billion || 0),
          x: Number(row[config.xKey] || 0),
          y: Number(row[config.yKey] || 0),
        }}))
        .filter((row) => row.x > 0 && row.y > 0);
      if (!points.length) {{
        inferenceTrainingTradeoffChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const providerOrder = Array.from(new Set(points.map((point) => point.provider))).sort((a, b) => providerDisplayName(a).localeCompare(providerDisplayName(b)));
      const paletteValues = ['#243b63', '#8c7a5b', '#b85c38', '#3f5a49', '#6c5b7b', '#2f7f92', '#a33d5e', '#7a9e2f', '#b06c1f', '#4b5563'];
      const palette = Object.fromEntries(providerOrder.map((provider, index) => [provider, paletteValues[index % paletteValues.length]]));
      const width = 980;
      const height = 640;
      const padding = {{ top: 24, right: 24, bottom: 86, left: 92 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const xMin = Math.min(...points.map((row) => row.x));
      const xMax = Math.max(...points.map((row) => row.x));
      const yMin = Math.min(...points.map((row) => row.y));
      const yMax = Math.max(...points.map((row) => row.y));
      const sizeMin = Math.min(...points.map((row) => row.activeParameters || 1));
      const sizeMax = Math.max(...points.map((row) => row.activeParameters || 1));
      const xMinLog = safeLog(xMin);
      const xMaxLog = safeLog(xMax);
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((safeLog(value) - xMinLog) / Math.max(xMaxLog - xMinLog, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const scaleR = (value) => {{
        const safeValue = Math.max(value || 1, 1e-9);
        return 4.5 + ((safeLog(safeValue) - safeLog(sizeMin || 1)) / Math.max(safeLog(sizeMax || 1) - safeLog(sizeMin || 1), 1e-9)) * 7.5;
      }};
      const tickValues = (min, max) => {{
        const ticks = [];
        const start = Math.floor(safeLog(min));
        const end = Math.ceil(safeLog(max));
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= min && value <= max);
      }};
      const xGrid = tickValues(xMin, xMax).map((value) => {{
        const x = scaleX(value);
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#6c757d">${{config.formatX(value)}}</text>
        `;
      }}).join('');
      const yGrid = tickValues(yMin, yMax).map((value) => {{
        const y = scaleY(value);
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{config.formatY(value)}}</text>
        `;
      }}).join('');
      const dots = points.map((row) => {{
        const cx = scaleX(row.x);
        const cy = scaleY(row.y);
        const radius = scaleR(row.activeParameters);
        const fill = palette[row.provider] || '#495057';
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" fill="${{fill}}" opacity="0.84"></circle>
          <text x="${{cx + radius + 5}}" y="${{cy - radius - 3}}" font-size="12" fill="#212529">${{row.label}}</text>
        `;
      }}).join('');
      const legend = providerOrder.map((provider, index) => `
        <g transform="translate(${{padding.left + (index % 4) * 170}}, ${{height - 48 + Math.floor(index / 4) * 22}})">
          <rect width="14" height="14" rx="3" fill="${{palette[provider] || '#495057'}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{providerDisplayName(provider)}}</text>
        </g>
      `).join('');
      inferenceTrainingTradeoffChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel[currentLanguage.value]}}">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{dots}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{config.xLabel[currentLanguage.value]}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{config.yLabel[currentLanguage.value]}}</text>
        </svg>
      `;
    }};
    const renderInferenceUncertaintyChart = () => {{
      if (!inferenceUncertaintyChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(inferenceUncertaintyChart.getAttribute('data-inference-uncertainty-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const metricControl = document.querySelector('[data-inference-uncertainty-control="metric-tab"].is-active');
      const metric = metricControl ? metricControl.getAttribute('data-metric-value') : 'energy';
      const config = metric === 'carbon'
        ? {{
            lowKey: 'carbon_low',
            centralKey: 'carbon_central',
            highKey: 'carbon_high',
            intro: {{
              en: 'Each line shows the low, central, and high retained values for inference carbon over one standardized hour of use.',
              fr: 'Chaque ligne montre la borne basse, la valeur centrale et la borne haute du carbone d’inférence retenu sur une heure d’usage standardisée.',
            }},
            xLabel: {{
              en: 'Central inference carbon, gCO2e/h (log scale)',
              fr: 'Carbone central d’inférence, gCO2e/h (échelle logarithmique)',
            }},
            formatTick: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`,
            ariaLabel: {{
              en: 'Inference carbon uncertainty span by model',
              fr: 'Étendue d’incertitude du carbone d’inférence par modèle',
            }},
          }}
        : {{
            lowKey: 'energy_low',
            centralKey: 'energy_central',
            highKey: 'energy_high',
            intro: {{
              en: 'Each line shows the low, central, and high retained values for inference energy over one standardized hour of use.',
              fr: 'Chaque ligne montre la borne basse, la valeur centrale et la borne haute de l’énergie d’inférence retenue sur une heure d’usage standardisée.',
            }},
            xLabel: {{
              en: 'Central inference energy, Wh/h (log scale)',
              fr: 'Énergie centrale d’inférence, Wh/h (échelle logarithmique)',
            }},
            formatTick: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kWh` : `${{value.toFixed(1)}} Wh`,
            ariaLabel: {{
              en: 'Inference energy uncertainty span by model',
              fr: 'Étendue d’incertitude de l’énergie d’inférence par modèle',
            }},
          }};
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider || '',
          low: Number(row[config.lowKey] || 0),
          central: Number(row[config.centralKey] || 0),
          high: Number(row[config.highKey] || 0),
        }}))
        .filter((row) => row.central > 0 && row.high > 0)
        .sort((a, b) => b.central - a.central);
      if (!points.length) {{
        inferenceUncertaintyChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const width = 980;
      const rowHeight = 34;
      const headerHeight = 10;
      const labelWidth = 280;
      const valueWidth = 620;
      const height = headerHeight + points.length * rowHeight + 24;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const minValue = Math.min(...points.map((row) => Math.max(row.low, 1e-9)));
      const maxValue = Math.max(...points.map((row) => row.high));
      const minLog = safeLog(minValue);
      const maxLog = safeLog(maxValue);
      const scaleX = (value) => labelWidth + ((safeLog(value) - minLog) / Math.max(maxLog - minLog, 1e-9)) * valueWidth;
      const tickValues = (() => {{
        const ticks = [];
        const start = Math.floor(minLog);
        const end = Math.ceil(maxLog);
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= minValue && value <= maxValue);
      }})();
      const grid = tickValues.map((value) => {{
        const x = scaleX(value);
        return `
          <line x1="${{x}}" y1="0" x2="${{x}}" y2="${{height - 18}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 2}}" text-anchor="middle" font-size="12" fill="#6c757d">${{config.formatTick(value)}}</text>
        `;
      }}).join('');
      const rowsMarkup = points.map((row, index) => {{
        const y = headerHeight + index * rowHeight + 16;
        const xLow = scaleX(row.low);
        const xCentral = scaleX(row.central);
        const xHigh = scaleX(row.high);
        return `
          <text x="0" y="${{y - 2}}" font-size="13" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 11}}" font-size="11" fill="#6c757d">${{row.provider}}</text>
          <line x1="${{xLow}}" y1="${{y}}" x2="${{xHigh}}" y2="${{y}}" stroke="#8c7a5b" stroke-width="3" stroke-linecap="round"></line>
          <circle cx="${{xLow}}" cy="${{y}}" r="4" fill="#d6c9b5"></circle>
          <circle cx="${{xCentral}}" cy="${{y}}" r="5.5" fill="#243b63"></circle>
          <circle cx="${{xHigh}}" cy="${{y}}" r="4" fill="#b85c38"></circle>
        `;
      }}).join('');
      inferenceUncertaintyChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel[currentLanguage.value]}}">
          ${{grid}}
          ${{rowsMarkup}}
          <text x="${{labelWidth + valueWidth / 2}}" y="${{height - 2}}" text-anchor="middle" font-size="13" fill="#495057">${{config.xLabel[currentLanguage.value]}}</text>
        </svg>
      `;
    }};
    const renderInferenceBubbleChart = () => {{
      if (!inferenceBubbleChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(inferenceBubbleChart.getAttribute('data-scatter-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const metricControl = document.querySelector('[data-inference-bubble-control="metric-tab"].is-active');
      const metric = metricControl ? metricControl.getAttribute('data-metric-value') : 'energy';
      const config = metric === 'carbon'
        ? {{
            yKey: 'hour_carbon_gco2e',
            intro: {{
              en: 'Models are positioned by retained effective active parameters on the horizontal axis and by central inference carbon over one standardized hour on the vertical axis. Bubble size follows the retained context window.',
              fr: 'Les modèles sont positionnés selon leurs paramètres actifs effectifs retenus sur l’axe horizontal et leur carbone central d’inférence sur une heure standardisée sur l’axe vertical. La taille des bulles suit la fenêtre de contexte retenue.',
            }},
            yLabel: {{
              en: 'Central inference carbon, gCO2e/h (log scale)',
              fr: 'Carbone central d’inférence, gCO2e/h (échelle logarithmique)',
            }},
            formatY: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`,
            ariaLabel: {{
              en: 'Inference bubble chart by carbon and effective parameters',
              fr: 'Nuage de bulles d’inférence par carbone et paramètres effectifs',
            }},
          }}
        : {{
            yKey: 'hour_energy_wh',
            intro: {{
              en: 'Models are positioned by retained effective active parameters on the horizontal axis and by central inference energy over one standardized hour on the vertical axis. Bubble size follows the retained context window.',
              fr: 'Les modèles sont positionnés selon leurs paramètres actifs effectifs retenus sur l’axe horizontal et leur énergie centrale d’inférence sur une heure standardisée sur l’axe vertical. La taille des bulles suit la fenêtre de contexte retenue.',
            }},
            yLabel: {{
              en: 'Central inference energy, Wh/h (log scale)',
              fr: 'Énergie centrale d’inférence, Wh/h (échelle logarithmique)',
            }},
            formatY: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kWh` : `${{value.toFixed(1)}} Wh`,
            ariaLabel: {{
              en: 'Inference bubble chart by energy and effective parameters',
              fr: 'Nuage de bulles d’inférence par énergie et paramètres effectifs',
            }},
          }};
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider || '',
          x: Number(row.effective_active_parameters_billion || 0),
          y: Number(row[config.yKey] || 0),
          context: Number(row.context_window_tokens || 0),
        }}))
        .filter((row) => row.x > 0 && row.y > 0 && row.context > 0);
      if (!points.length) {{
        inferenceBubbleChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const providerOrder = Array.from(new Set(points.map((point) => point.provider))).sort((a, b) => providerDisplayName(a).localeCompare(providerDisplayName(b)));
      const paletteValues = ['#243b63', '#8c7a5b', '#b85c38', '#3f5a49', '#6c5b7b', '#2f7f92', '#a33d5e', '#7a9e2f', '#b06c1f', '#4b5563'];
      const palette = Object.fromEntries(providerOrder.map((provider, index) => [provider, paletteValues[index % paletteValues.length]]));
      const width = 980;
      const height = 640;
      const padding = {{ top: 24, right: 24, bottom: 86, left: 92 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const xMin = Math.min(...points.map((row) => row.x));
      const xMax = Math.max(...points.map((row) => row.x));
      const yMin = Math.min(...points.map((row) => row.y));
      const yMax = Math.max(...points.map((row) => row.y));
      const cMin = Math.min(...points.map((row) => row.context));
      const cMax = Math.max(...points.map((row) => row.context));
      const xMinLog = safeLog(xMin);
      const xMaxLog = safeLog(xMax);
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((safeLog(value) - xMinLog) / Math.max(xMaxLog - xMinLog, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const scaleR = (value) => 5 + ((safeLog(value) - safeLog(cMin || 1)) / Math.max(safeLog(cMax || 1) - safeLog(cMin || 1), 1e-9)) * 11;
      const tickValues = (min, max) => {{
        const ticks = [];
        const start = Math.floor(safeLog(min));
        const end = Math.ceil(safeLog(max));
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= min && value <= max);
      }};
      const xGrid = tickValues(xMin, xMax).map((value) => {{
        const x = scaleX(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}}T` : `${{value >= 10 ? value.toFixed(0) : value.toFixed(1)}}B`;
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yGrid = tickValues(yMin, yMax).map((value) => {{
        const y = scaleY(value);
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{config.formatY(value)}}</text>
        `;
      }}).join('');
      const dots = points.map((row) => {{
        const cx = scaleX(row.x);
        const cy = scaleY(row.y);
        const radius = scaleR(row.context);
        const fill = palette[row.provider] || '#495057';
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" fill="${{fill}}" opacity="0.72"></circle>
          <text x="${{cx + radius + 5}}" y="${{cy - radius - 3}}" font-size="12" fill="#212529">${{row.label}}</text>
        `;
      }}).join('');
      const legend = providerOrder.map((provider, index) => `
        <g transform="translate(${{padding.left + (index % 4) * 170}}, ${{height - 48 + Math.floor(index / 4) * 22}})">
          <rect width="14" height="14" rx="3" fill="${{palette[provider] || '#495057'}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{providerDisplayName(provider)}}</text>
        </g>
      `).join('');
      const xLabel = currentLanguage.value === 'fr' ? 'Paramètres actifs effectifs retenus (échelle logarithmique)' : 'Retained effective active parameters (log scale)';
      const sizeNote = currentLanguage.value === 'fr' ? 'Taille des bulles : fenêtre de contexte retenue' : 'Bubble size: retained context window';
      inferenceBubbleChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel[currentLanguage.value]}}">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{dots}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{xLabel}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{config.yLabel[currentLanguage.value]}}</text>
          <text x="${{width - 24}}" y="20" text-anchor="end" font-size="12" fill="#6c757d">${{sizeNote}}</text>
        </svg>
      `;
    }};
    const renderReleaseTimelineChart = (container, attrName, config) => {{
      if (!container) return;
      let rows = [];
      try {{
        rows = JSON.parse(container.getAttribute(attrName) || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const points = rows
        .map((row) => {{
          const date = new Date(`${{row.release_date}}T00:00:00Z`);
          const value = Number(row[config.valueKey] || 0);
          return {{
            label: translateBenchmarkLabel(row.label, currentLanguage.value),
            provider: row.provider || '',
            providerLabel: providerDisplayName(row.provider || ''),
            releaseDate: row.release_date,
            date,
            value,
          }};
        }})
        .filter((row) => !Number.isNaN(row.date.getTime()) && row.value > 0)
        .sort((a, b) => a.date - b.date || a.provider.localeCompare(b.provider));
      if (!points.length) {{
        container.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const providerOrder = ['openai', 'anthropic', 'xai', 'mistral'];
      const palette = {{
        openai: '#243b63',
        anthropic: '#8c7a5b',
        xai: '#b85c38',
        mistral: '#3f5a49',
      }};
      const width = 980;
      const height = 620;
      const padding = {{ top: 24, right: 24, bottom: 86, left: 86 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const xMin = Math.min(...points.map((row) => row.date.getTime()));
      const xMax = Math.max(...points.map((row) => row.date.getTime()));
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const yMin = Math.min(...points.map((row) => row.value));
      const yMax = Math.max(...points.map((row) => row.value));
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const monthTicks = (() => {{
        const ticks = [];
        const start = new Date(xMin);
        const end = new Date(xMax);
        const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1));
        while (cursor <= end) {{
          ticks.push(new Date(cursor.getTime()));
          cursor.setUTCMonth(cursor.getUTCMonth() + 2);
        }}
        return ticks;
      }})();
      const xGrid = monthTicks.map((tick) => {{
        const x = scaleX(tick.getTime());
        const label = tick.toLocaleDateString(currentLanguage.value === 'fr' ? 'fr-FR' : 'en-US', {{
          year: 'numeric',
          month: 'short',
          timeZone: 'UTC',
        }});
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 40}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yTickValues = (() => {{
        const ticks = [];
        const start = Math.floor(yMinLog);
        const end = Math.ceil(yMaxLog);
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= yMin && value <= yMax);
      }})();
      const yGrid = yTickValues.map((value) => {{
        const y = scaleY(value);
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{config.formatValue(value)}}</text>
        `;
      }}).join('');
      const byProvider = providerOrder
        .map((provider) => [provider, points.filter((point) => point.provider === provider)])
        .filter(([, providerPoints]) => providerPoints.length);
      const lines = byProvider.map(([provider, providerPoints]) => {{
        const d = providerPoints
          .map((point, index) => `${{index === 0 ? 'M' : 'L'}}${{scaleX(point.date.getTime()).toFixed(2)}},${{scaleY(point.value).toFixed(2)}}`)
          .join(' ');
        return `<path d="${{d}}" fill="none" stroke="${{palette[provider] || '#495057'}}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"></path>`;
      }}).join('');
      const dots = points.map((point) => {{
        const cx = scaleX(point.date.getTime());
        const cy = scaleY(point.value);
        const fill = palette[point.provider] || '#495057';
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="${{fill}}" opacity="0.95"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{point.label}}</text>
        `;
      }}).join('');
      const legend = byProvider.map(([provider], index) => `
        <g transform="translate(${{padding.left + index * 150}}, ${{height - 20}})">
          <rect width="14" height="14" rx="3" fill="${{palette[provider] || '#495057'}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{providerDisplayName(provider)}}</text>
        </g>
      `).join('');
      container.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel[currentLanguage.value]}}">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{lines}}
          ${{dots}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 52}}" text-anchor="middle" font-size="13" fill="#495057">${{config.xLabel[currentLanguage.value]}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{config.yLabel[currentLanguage.value]}}</text>
        </svg>
      `;
    }};
    const renderDoublingTimelineChart = (container, attrName, config) => {{
      if (!container) return;
      let rows = [];
      try {{
        rows = JSON.parse(container.getAttribute(attrName) || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const keepLabels = new Set(config.keepLabels);
      const points = rows
        .map((row) => {{
          const date = new Date(`${{row.release_date}}T00:00:00Z`);
          const value = Number(row[config.valueKey] || 0);
          return {{
            rawLabel: row.label,
            label: translateBenchmarkLabel(row.label, currentLanguage.value),
            provider: row.provider || '',
            date,
            value,
          }};
        }})
        .filter((row) => keepLabels.has(row.rawLabel) && !Number.isNaN(row.date.getTime()) && row.value > 0)
        .sort((a, b) => a.date - b.date || a.provider.localeCompare(b.provider));
      if (!points.length) {{
        container.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const computeDoublingMonths = (series) => {{
        if (series.length < 2) return null;
        const baseDate = series[0].date;
        const xs = series.map((point) => (point.date - baseDate) / (1000 * 60 * 60 * 24 * 30.4375));
        const ys = series.map((point) => Math.log2(Math.max(point.value, 1e-9)));
        const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
        const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
        const num = xs.reduce((sum, value, index) => sum + ((value - meanX) * (ys[index] - meanY)), 0);
        const den = xs.reduce((sum, value) => sum + ((value - meanX) ** 2), 0);
        if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return null;
        const slope = num / den;
        if (!Number.isFinite(slope) || slope <= 0) return null;
        return 1 / slope;
      }};
      const doublingMonths = computeDoublingMonths(points);
      const providerOrder = ['openai', 'anthropic', 'xai'];
      const palette = {{
        openai: '#243b63',
        anthropic: '#8c7a5b',
        xai: '#b85c38',
      }};
      const width = 980;
      const height = 620;
      const padding = {{ top: 24, right: 24, bottom: 86, left: 86 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const xMin = Math.min(...points.map((row) => row.date.getTime()));
      const xMax = Math.max(...points.map((row) => row.date.getTime()));
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const yMin = Math.min(...points.map((row) => row.value));
      const yMax = Math.max(...points.map((row) => row.value));
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const monthTicks = (() => {{
        const ticks = [];
        const start = new Date(xMin);
        const end = new Date(xMax);
        const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1));
        while (cursor <= end) {{
          ticks.push(new Date(cursor.getTime()));
          cursor.setUTCMonth(cursor.getUTCMonth() + 2);
        }}
        return ticks;
      }})();
      const xGrid = monthTicks.map((tick) => {{
        const x = scaleX(tick.getTime());
        const label = tick.toLocaleDateString(currentLanguage.value === 'fr' ? 'fr-FR' : 'en-US', {{
          year: 'numeric',
          month: 'short',
          timeZone: 'UTC',
        }});
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 40}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yTickValues = (() => {{
        const ticks = [];
        const start = Math.floor(yMinLog);
        const end = Math.ceil(yMaxLog);
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= yMin && value <= yMax);
      }})();
      const yGrid = yTickValues.map((value) => {{
        const y = scaleY(value);
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{config.formatValue(value)}}</text>
        `;
      }}).join('');
      const byProvider = providerOrder
        .map((provider) => [provider, points.filter((point) => point.provider === provider)])
        .filter(([, providerPoints]) => providerPoints.length);
      const lines = byProvider.map(([provider, providerPoints]) => {{
        const d = providerPoints
          .map((point, index) => `${{index === 0 ? 'M' : 'L'}}${{scaleX(point.date.getTime()).toFixed(2)}},${{scaleY(point.value).toFixed(2)}}`)
          .join(' ');
        return `<path d="${{d}}" fill="none" stroke="${{palette[provider] || '#495057'}}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"></path>`;
      }}).join('');
      const dots = points.map((point) => {{
        const cx = scaleX(point.date.getTime());
        const cy = scaleY(point.value);
        const fill = palette[point.provider] || '#495057';
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="${{fill}}" opacity="0.95"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{point.label}}</text>
        `;
      }}).join('');
      const legend = byProvider.map(([provider], index) => `
        <g transform="translate(${{padding.left + index * 150}}, ${{height - 20}})">
          <rect width="14" height="14" rx="3" fill="${{palette[provider] || '#495057'}}"></rect>
          <text x="20" y="11" font-size="12" fill="#495057">${{providerDisplayName(provider)}}</text>
        </g>
      `).join('');
      const doublingText = doublingMonths
        ? (currentLanguage.value === 'fr'
          ? `Temps de doublement estimé : ${{doublingMonths.toFixed(1)}} mois`
          : `Estimated doubling time: ${{doublingMonths.toFixed(1)}} months`)
        : '';
      container.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{config.intro[currentLanguage.value]}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{config.ariaLabel[currentLanguage.value]}}">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{lines}}
          ${{dots}}
          ${{legend}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 52}}" text-anchor="middle" font-size="13" fill="#495057">${{config.xLabel[currentLanguage.value]}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{config.yLabel[currentLanguage.value]}}</text>
          ${{doublingText ? `<text x="${{width - 24}}" y="20" text-anchor="end" font-size="12" fill="#6c757d">${{doublingText}}</text>` : ''}}
        </svg>
      `;
    }};
    renderReleaseTimelineChart(inferenceReleaseTimelineChart, 'data-release-timeline-rows', {{
      valueKey: 'hour_carbon_gco2e',
      formatValue: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`,
      intro: {{
        en: 'Connected points trace the release sequence of OpenAI, Claude, Grok, and Mistral models against the project’s central inference CO2e estimate. The vertical axis uses a logarithmic scale to keep small and large models readable together.',
        fr: 'Les points reliés retracent la séquence de sortie des modèles OpenAI, Claude, Grok et Mistral face à l’estimation centrale de CO2e d’inférence du projet. L’axe vertical utilise une échelle logarithmique pour garder lisibles ensemble les petits et les grands modèles.'
      }},
      xLabel: {{
        en: 'Model release month',
        fr: 'Mois de sortie du modèle',
      }},
      yLabel: {{
        en: 'Central inference carbon, gCO2e/h (log scale)',
        fr: 'Carbone central d’inférence, gCO2e/h (échelle logarithmique)',
      }},
      ariaLabel: {{
        en: 'Inference carbon by model release date chart',
        fr: 'Graphique du carbone d’inférence par date de sortie du modèle',
      }},
    }});
    renderDoublingTimelineChart(inferenceDoublingTimelineChart, 'data-release-timeline-rows', {{
      keepLabels: ['GPT-3.5 Turbo', 'GPT-4', 'GPT-5.2', 'Claude 2', 'Claude 3.5 Sonnet', 'Claude Sonnet 4', 'Claude Opus 4.1', 'Grok 1', 'Grok 2', 'Grok 4'],
      valueKey: 'hour_carbon_gco2e',
      formatValue: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kg` : `${{value.toFixed(1)}} g`,
      intro: {{
        en: 'Flagship GPT, Claude, and Grok models are isolated here to visualize the rate at which the current central inference-screening values appear to rise over time.',
        fr: 'Les modèles phares GPT, Claude et Grok sont isolés ici pour visualiser la vitesse à laquelle les valeurs centrales actuelles de screening d’inférence semblent augmenter dans le temps.',
      }},
      xLabel: {{
        en: 'Model release month',
        fr: 'Mois de sortie du modèle',
      }},
      yLabel: {{
        en: 'Central inference carbon, gCO2e/h (log scale)',
        fr: 'Carbone central d’inférence, gCO2e/h (échelle logarithmique)',
      }},
      ariaLabel: {{
        en: 'Inference CO2 doubling view',
        fr: 'Vue du doublement du CO2 d’inférence',
      }},
    }});
    crossImpactControls.forEach((control) => {{
      control.addEventListener('click', () => {{
        crossImpactControls.forEach((item) => {{
          const isActive = item === control;
          item.classList.toggle('is-active', isActive);
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        renderInferenceTrainingTradeoffChart();
      }});
    }});
    renderInferenceTrainingTradeoffChart();
    inferenceBubbleControls.forEach((control) => {{
      control.addEventListener('click', () => {{
        inferenceBubbleControls.forEach((item) => {{
          const isActive = item === control;
          item.classList.toggle('is-active', isActive);
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        renderInferenceBubbleChart();
      }});
    }});
    renderInferenceBubbleChart();
    inferenceUncertaintyControls.forEach((control) => {{
      control.addEventListener('click', () => {{
        inferenceUncertaintyControls.forEach((item) => {{
          const isActive = item === control;
          item.classList.toggle('is-active', isActive);
          item.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        renderInferenceUncertaintyChart();
      }});
    }});
    renderInferenceUncertaintyChart();
    const renderTrainingScatterLogChart = () => {{
      if (!trainingScatterLogChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(trainingScatterLogChart.getAttribute('data-training-scatter-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      const locale = uiText[currentLanguage.value];
      const points = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          x: Number(row.active_parameters_billion || 0),
          y: Number(row.direct_training_carbon_tco2e || 0),
        }}))
        .filter((row) => row.x > 0 && row.y > 0);
      if (!points.length) {{
        trainingScatterLogChart.innerHTML = `<p class="lead">${{locale.noUsableValue}}</p>`;
        return;
      }}
      const padding = {{ top: 24, right: 24, bottom: 58, left: 84 }};
      const width = 980;
      const height = 620;
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const safeLog = (value) => Math.log10(Math.max(value, 1e-9));
      const xMin = Math.min(...points.map((row) => row.x));
      const xMax = Math.max(...points.map((row) => row.x));
      const yMin = Math.min(...points.map((row) => row.y));
      const yMax = Math.max(...points.map((row) => row.y));
      const xMinLog = safeLog(xMin);
      const xMaxLog = safeLog(xMax);
      const yMinLog = safeLog(yMin);
      const yMaxLog = safeLog(yMax);
      const scaleX = (value) => padding.left + ((safeLog(value) - xMinLog) / Math.max(xMaxLog - xMinLog, 1e-9)) * plotWidth;
      const scaleY = (value) => padding.top + plotHeight - ((safeLog(value) - yMinLog) / Math.max(yMaxLog - yMinLog, 1e-9)) * plotHeight;
      const tickValues = (min, max) => {{
        const ticks = [];
        const start = Math.floor(safeLog(min));
        const end = Math.ceil(safeLog(max));
        for (let exponent = start; exponent <= end; exponent += 1) {{
          ticks.push(10 ** exponent);
        }}
        return ticks.filter((value) => value >= min && value <= max);
      }};
      const xGrid = tickValues(xMin, xMax).map((value) => {{
        const x = scaleX(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}}T` : `${{value >= 10 ? value.toFixed(0) : value.toFixed(1)}}B`;
        return `
          <line x1="${{x}}" y1="${{padding.top}}" x2="${{x}}" y2="${{padding.top + plotHeight}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{x}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const yGrid = tickValues(yMin, yMax).map((value) => {{
        const y = scaleY(value);
        const label = value >= 1000 ? `${{(value / 1000).toFixed(1)}} kt` : `${{value.toFixed(1)}} t`;
        return `
          <line x1="${{padding.left}}" y1="${{y}}" x2="${{padding.left + plotWidth}}" y2="${{y}}" stroke="rgba(0,0,0,0.08)" />
          <text x="${{padding.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="12" fill="#6c757d">${{label}}</text>
        `;
      }}).join('');
      const dots = points.map((row) => {{
        const cx = scaleX(row.x);
        const cy = scaleY(row.y);
        return `
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="#8c7a5b" opacity="0.9"></circle>
          <text x="${{cx + 8}}" y="${{cy - 8}}" font-size="12" fill="#212529">${{row.label}}</text>
        `;
      }}).join('');
      const intro = currentLanguage.value === 'fr'
        ? 'Positionnement des modèles selon leur nombre de paramètres retenu et leur CO2e direct d’entraînement, en échelle logarithmique.'
        : 'Positioning of models by retained parameter count and direct training CO2e, on logarithmic axes.';
      const xLabel = currentLanguage.value === 'fr' ? 'Nombre de paramètres retenu' : 'Retained parameter count';
      const yLabel = currentLanguage.value === 'fr' ? 'CO2e direct d’entraînement, tCO2e' : 'Direct training CO2e, tCO2e';
      trainingScatterLogChart.innerHTML = `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{intro}}</div>
        <svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Training carbon versus parameter count chart">
          ${{xGrid}}
          ${{yGrid}}
          <line x1="${{padding.left}}" y1="${{padding.top + plotHeight}}" x2="${{padding.left + plotWidth}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          <line x1="${{padding.left}}" y1="${{padding.top}}" x2="${{padding.left}}" y2="${{padding.top + plotHeight}}" stroke="#495057" />
          ${{dots}}
          <text x="${{padding.left + plotWidth / 2}}" y="${{height - 4}}" text-anchor="middle" font-size="13" fill="#495057">${{xLabel}}</text>
          <text x="18" y="${{padding.top + plotHeight / 2}}" text-anchor="middle" font-size="13" fill="#495057" transform="rotate(-90 18 ${{padding.top + plotHeight / 2}})">${{yLabel}}</text>
        </svg>
      `;
    }};
    renderReleaseTimelineChart(trainingReleaseTimelineChart, 'data-training-release-timeline-rows', {{
      valueKey: 'direct_training_carbon_tco2e',
      formatValue: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kt` : `${{value.toFixed(1)}} t`,
      intro: {{
        en: 'Connected points trace the release sequence of OpenAI, Claude, Grok, and Mistral models against the project’s retained direct training CO2e estimate. The vertical axis uses a logarithmic scale because training orders of magnitude remain widely spread.',
        fr: 'Les points reliés retracent la séquence de sortie des modèles OpenAI, Claude, Grok et Mistral face à l’estimation retenue du CO2e direct d’entraînement du projet. L’axe vertical utilise une échelle logarithmique car les ordres de grandeur d’entraînement restent très dispersés.'
      }},
      xLabel: {{
        en: 'Model release month',
        fr: 'Mois de sortie du modèle',
      }},
      yLabel: {{
        en: 'Direct training CO2e, tCO2e (log scale)',
        fr: 'CO2e direct d’entraînement, tCO2e (échelle logarithmique)',
      }},
      ariaLabel: {{
        en: 'Training carbon by model release date chart',
        fr: 'Graphique du carbone d’entraînement par date de sortie du modèle',
      }},
    }});
    renderDoublingTimelineChart(trainingDoublingTimelineChart, 'data-training-release-timeline-rows', {{
      keepLabels: ['GPT-3.5 Turbo', 'GPT-4', 'GPT-5.2', 'Claude 2', 'Claude 3.5 Sonnet', 'Claude Sonnet 4', 'Claude Opus 4.1', 'Grok 1', 'Grok 2', 'Grok 4'],
      valueKey: 'direct_training_carbon_tco2e',
      formatValue: (value) => value >= 1000 ? `${{(value / 1000).toFixed(1)}} kt` : `${{value.toFixed(1)}} t`,
      intro: {{
        en: 'Flagship GPT, Claude, and Grok models are isolated here to visualize the apparent acceleration of the current central training-screening values over time.',
        fr: 'Les modèles phares GPT, Claude et Grok sont isolés ici pour visualiser l’accélération apparente des valeurs centrales actuelles de screening d’entraînement dans le temps.',
      }},
      xLabel: {{
        en: 'Model release month',
        fr: 'Mois de sortie du modèle',
      }},
      yLabel: {{
        en: 'Direct training CO2e, tCO2e (log scale)',
        fr: 'CO2e direct d’entraînement, tCO2e (échelle logarithmique)',
      }},
      ariaLabel: {{
        en: 'Training CO2 doubling view',
        fr: 'Vue du doublement du CO2 d’entraînement',
      }},
    }});
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
      const locale = uiText[currentLanguage.value];
      if (!rows.length) {{
        return `<p class="lead">${{locale.noData}}</p>`;
      }}
      const key = metric === 'direct_training_energy'
        ? 'direct_training_energy_wh'
        : 'direct_training_carbon_tco2e';
      const sorted = rows
        .map((row) => ({{
          label: translateBenchmarkLabel(row.label, currentLanguage.value),
          provider: row.provider,
          value: Number(row[key] || 0),
          kind: row.kind || 'model',
        }}))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value);
      if (!sorted.length) {{
        return `<p class="lead">${{locale.noUsableValue}}</p>`;
      }}
      const maxValue = sorted[0].value || 1;
      const barHeight = 28;
      const rowGap = 22;
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
          <text x="0" y="${{y + 18}}" font-size="15" fill="#212529">${{row.label}}</text>
          <text x="0" y="${{y + 34}}" font-size="13" fill="#6c757d">${{row.provider}}</text>
          <rect x="${{barStart}}" y="${{y}}" width="${{width}}" height="${{barHeight}}" rx="4" fill="${{fill}}"></rect>
          <text x="${{barStart + width + 10}}" y="${{y + 18}}" font-size="14" fill="#212529">${{valueText}}</text>
        `;
      }}).join('');
      const titleMetric = metric === 'direct_training_energy'
        ? locale.chartMetricEnergy
        : locale.trainingMetricCarbon;
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">${{locale.comparisonTrainingPrefix}}<strong>${{titleMetric}}</strong>${{locale.comparisonTrainingSuffix}}</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="${{locale.trainingAriaLabel}}">${{bars}}</svg>
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
    const renderTrainingScatterChart = () => {{
      if (!trainingScatterChart) return;
      let rows = [];
      try {{
        rows = JSON.parse(trainingScatterChart.getAttribute('data-training-scatter-chart-rows') || '[]');
      }} catch (error) {{
        rows = [];
      }}
      trainingScatterChart.innerHTML = buildLandscapeMarkup(rows, {{
        featureKeys: [
          'active_parameters_billion',
          'training_tokens_estimate_trillion',
          'training_regime_score',
          'training_hardware_score',
          'vision_support_score',
          'moe_score',
          'reasoning_score',
          'direct_training_energy_wh',
          'direct_training_carbon_tco2e',
        ],
        intro: {{
          en: 'This clustered landscape is derived from the full training screening profile retained by the project. Nearby points indicate models with similar retained combinations of size, training-token prior, training regime, hardware proxy, architecture notes, and central training outcomes.',
          fr: 'Cette carte groupée est dérivée du profil complet de screening retenu par le projet pour l’entraînement. Les points proches indiquent des modèles aux combinaisons retenues similaires de taille, prior sur les tokens d’entraînement, régime d’entraînement, proxy matériel, notes d’architecture et résultats centraux d’entraînement.'
        }},
        axisLabels: {{
          x: {{ en: 'Landscape dimension 1 (composite projection)', fr: 'Dimension 1 du paysage (projection composite)' }},
          y: {{ en: 'Landscape dimension 2 (composite projection)', fr: 'Dimension 2 du paysage (projection composite)' }},
        }},
        ariaLabel: 'Training model landscape chart',
      }});
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
    renderTrainingScatterChart();
    applyLanguage(preferredLanguage());
  </script>
</body>
</html>
"""
class Handler(BaseHTTPRequestHandler):
    def _normalized_path(self):
        request_path = urlsplit(self.path).path
        if URL_PREFIX:
            if request_path == URL_PREFIX:
                return "/"
            if request_path.startswith(f"{URL_PREFIX}/"):
                stripped = request_path[len(URL_PREFIX):]
                return stripped or "/"
            return None
        return request_path

    def _write_bytes(self, body, content_type, filename=None, status=200, send_body=True):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Last-Modified", formatdate(usegmt=True))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _write_html(self, html, status=200, send_body=True):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _handle_get_like(self, send_body=True):
        normalized_path = self._normalized_path()
        if normalized_path is None:
            self._write_html(render_page(error_message="Page not found."), status=404, send_body=send_body)
            return
        if normalized_path == "/favicon.svg":
            if LOGO_MARK_PATH.exists():
                self._write_bytes(
                    LOGO_MARK_PATH.read_bytes(),
                    "image/svg+xml",
                    send_body=send_body,
                )
                return
            self._write_html(render_page(error_message="Favicon not found."), status=404, send_body=send_body)
            return
        if normalized_path in {"/downloads/ImpactLLM_paper.pdf", "/downloads/llm_environment_opendata_paper.pdf"}:
            if PAPER_PDF_PATH.exists():
                self._write_bytes(
                    PAPER_PDF_PATH.read_bytes(),
                    "application/pdf",
                    filename="ImpactLLM_paper.pdf",
                    send_body=send_body,
                )
                return
            self._write_html(render_page(error_message="Publication PDF not found."), status=404, send_body=send_body)
            return
        if normalized_path in {"/downloads/ImpactLLM_paper.bib", "/downloads/llm_environment_opendata_paper.bib"}:
            self._write_bytes(
                PROJECT_PAPER_BIBTEX.encode("utf-8"),
                "application/x-bibtex; charset=utf-8",
                filename="ImpactLLM_paper.bib",
                send_body=send_body,
            )
            return
        if normalized_path in {"/downloads/ImpactLLM_paper_preview.png", "/downloads/llm_environment_opendata_paper_preview.png"}:
            if PAPER_PREVIEW_PATH.exists():
                self._write_bytes(
                    PAPER_PREVIEW_PATH.read_bytes(),
                    "image/png",
                    send_body=send_body,
                )
                return
            self._write_html(render_page(error_message="Publication preview not found."), status=404, send_body=send_body)
            return
        if normalized_path == "/downloads/linkedin_training_co2_doubling_en.png":
            if TRAINING_DOUBLING_FIGURE_PATH.exists():
                self._write_bytes(
                    TRAINING_DOUBLING_FIGURE_PATH.read_bytes(),
                    "image/png",
                    send_body=send_body,
                )
                return
            self._write_html(render_page(error_message="Training doubling figure not found."), status=404, send_body=send_body)
            return
        if normalized_path == "/downloads/linkedin_inference_co2_doubling_en.png":
            if INFERENCE_DOUBLING_FIGURE_PATH.exists():
                self._write_bytes(
                    INFERENCE_DOUBLING_FIGURE_PATH.read_bytes(),
                    "image/png",
                    send_body=send_body,
                )
                return
            self._write_html(render_page(error_message="Inference doubling figure not found."), status=404, send_body=send_body)
            return
        self._write_html(render_page(), send_body=send_body)

    def do_GET(self):
        self._handle_get_like()

    def do_HEAD(self):
        self._handle_get_like(send_body=False)

    def do_POST(self):
        normalized_path = self._normalized_path()
        if normalized_path != "/":
            self._write_html(render_page(error_message="Page not found."), status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw)
        description = form.get("description", [""])[0]

        try:
            description, parsed_payload, parser_notes, parser_meta, result, rows, method_comparisons = process_description(form)
        except (OpenAIModerationError, OpenAIParserError) as exc:
            self._write_html(render_page(description=description, error_message=str(exc)), status=502)
            return

        if not ((parser_meta or {}).get("cache") or {}).get("hit"):
            persist_analysis_run(description, parsed_payload, parser_notes, parser_meta, result, rows, method_comparisons)

        self._write_html(
            render_page(
                result=result,
                description=description,
                parsed_payload=parsed_payload,
                parser_notes=parser_notes,
                parser_meta=parser_meta,
                factor_rows=rows,
                method_comparisons=method_comparisons,
            )
        )


def apply_overrides(payload, form):
    return payload


if __name__ == "__main__":
    host = os.environ.get("LLM_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("LLM_WEB_PORT", "8080"))
    server = HTTPServer((host, port), Handler)
    print(f"{PROJECT_NAME} web app running on http://{host}:{port}")
    server.serve_forever()
