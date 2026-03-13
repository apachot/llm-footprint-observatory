#!/usr/bin/env python3
from datetime import datetime
import json
import re
import sys
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


PROJECT_NAME = "EcoTrace LLM"
BIB_PATH = ROOT.parent / "llm-environment-opendata-paper" / "references_llm_environment_opendata.bib"
ANALYSIS_LOG_PATH = ROOT / "data" / "analysis_runs.json"
REFERENCE_PAGE_TOKENS = 750.0
DEFAULT_PROMPT_TOKENS = 1550.0


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
            "label": "Proxy scientifique",
            "description": "L'estimation repose sur des facteurs de littérature applicables a une famille d'usages, sans mesure attribuable a un modele cible specifique.",
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
            "label": "Mesure directe",
            "description": "Au moins un facteur selectionne correspond explicitement au modele mentionne dans la demande.",
        }
    if family_match:
        return {
            "level": "proxy_scientifique",
            "label": "Proxy scientifique",
            "description": "Les facteurs retenus sont proches de la famille de service ou du fournisseur mentionne, mais ne constituent pas une mesure directe du modele cible.",
        }
    return {
        "level": "extrapolation",
        "label": "Extrapolation",
        "description": "Aucune mesure directe du modele cible n'est disponible dans le corpus mobilise; l'estimation est derivee de facteurs de reference et d'ajustements de contexte.",
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
        "Parametric extrapolation enabled for target model": "Extrapolation parametrique appliquee au modele cible",
        "Reference inference scaling derived from Ren et al. 2024 with page-level measurements at": "Mise a l'echelle fondee sur Ren et al. (2024), a partir de mesures par page de",
        "Token scaling applied relative to": "Ajustement realise en fonction du volume de tokens, avec une reference de",
        "Carbon contextualized using country electricity carbon intensity": "Les emissions carbone sont ajustées avec l'intensite carbone du mix electrique du pays retenu.",
        "Water contextualized using country electricity water intensity": "La consommation d'eau est ajustée avec l'intensite hydrique du mix electrique retenu.",
        "Request type classified as": "Type d'usage interprete comme",
        "Country mix fallback applied for": "Mix electrique par defaut applique pour",
        "Carbon and water recalculated with the publisher-country mix for": "Le calcul carbone/eau utilise le mix electrique du pays de l'editeur pour",
        "because the model is treated as a proprietary hosted service": "car le modele est traite comme un service proprietaire heberge.",
        "Carbon and water recalculated with the project country mix for": "Le calcul carbone/eau utilise le mix electrique du pays du projet pour",
        "because the model is treated as open-weight or self-hosted": "car le modele est traite comme open weight ou auto-heberge.",
        "Page-based method anchored on the nearest available source model in parameter count:": "La methode par page retient comme ancrage le modele le plus proche en nombre de parametres :",
        "Final central result uses the prompt-based method only because no empirical prompt-to-page conversion is available and the target model is treated as a hosted proprietary service": "Le resultat central final utilise uniquement la methode par prompt, car aucune conversion empirique prompt-vers-page n'est disponible et le modele cible est traite comme un service proprietaire heberge.",
        "Final central result uses the page-based method only because no empirical prompt-to-page conversion is available and the target model is treated as open-weight or self-hosted": "Le resultat central final utilise uniquement la methode par page, car aucune conversion empirique prompt-vers-page n'est disponible et le modele cible est traite comme open weight ou auto-heberge.",
        "Final central result averages the available inference methods": "Le resultat central final moyenne les methodes d'inference disponibles.",
        "Unified inference model calibrated from the nearest literature energy anchor after harmonizing the observed value to Wh per request": "Le modele d'inference utilise l'ancrage energetique de la litterature le plus proche, harmonise en Wh par requete.",
        "Nearest calibration anchor:": "Ancrage de calibration le plus proche :",
        "LLM request(s) per feature use": "appel(s) au LLM par usage de la fonctionnalite",
        "feature uses per year": "usages annuels de la fonctionnalite",
        "A page calibration is computed from the mean Wh per parameter observed in page-generation inference records": "La methode par page est calculee a partir de la moyenne des intensites energetiques Wh/parametre observees dans les publications par page.",
        "Page-family annualization uses generated page equivalents, with": "L'annualisation de la methode par page repose sur des pages equivalentes generees, avec",
        "tokens per reference page when no explicit page count is provided": "tokens par page de reference lorsqu'aucun nombre de pages n'est fourni.",
        "Page-family method marked as not applicable for this scenario by the parser": "La methode par page a ete jugee non pertinente pour ce scenario par l'interpretation du cas d'usage.",
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
            f"<li><strong>{escape(anchor.get('source_model', 'modele source'))}</strong> : valeur d'origine "
            f"<code>{escape(formatted_source)}</code>{citation_link} × facteur applique <code>{factor_value:.3f}</code> "
            f"= valeur extrapolee <code>{escape(formatted_extrapolated)}</code></li>"
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
        <span>Voir le détail du calcul</span>
      </summary>
      <div class="metric-detail-body">
        <p class="math-detail">
          <strong>{escape(title)}</strong> :
          <code>{escape(format_range_display(annual_llm[unit_key], metric_label))}</code>
        </p>
        {render_extrapolation_details(result, metric_label, source_rows)}
        <p class="math-detail">
          Impact estime pour une requete LLM :
          <code>{escape(format_range_display(per_request[unit_key], metric_label))}</code>
        </p>
        <p class="math-detail">
          Impact estime pour un usage de la fonctionnalite :
          <code>{escape(format_range_display(per_feature[unit_key], metric_label))}</code>
        </p>
        <p class="math-detail">
          En projetant cette valeur sur <code>{format_count(annual_requests)}</code> appels par an,
          on obtient <code>{escape(format_range_display(annual_llm[unit_key], metric_label))}</code>.
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
      <span class="math-label">Hypotheses retenues</span>
      <ul class="assumptions-list">
        {''.join(f'<li>{escape(humanize_assumption(item))}</li>' for item in assumptions)}
      </ul>
    </div>
    """


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
    target_country = target_mix.get("country_name") or method.get("target_country") or "non spécifié"
    target_carbon = method.get("target_grid_carbon_intensity")
    target_water = method.get("target_water_intensity")
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
            f"<p>Le volume annuel d'appels est calculé par :</p>"
            f"<p>\\["
            f"N_{{appels/an}} = {format_count(feature_uses_per_month)} \\times {format_scalar(months_per_year, 0)} \\times {format_scalar(requests_per_feature, 0)} = {format_count(annual_requests)}"
            f"\\]</p>"
        )
        if family == "page":
            source_note = "issus de la description utilisateur" if token_source_note in {"user_tokens", "parser_page_equivalent", "output_tokens"} else "valeur par défaut du projet"
            annualization_sentence += (
                f"<p>Pour la famille <code>Wh/page</code>, le moteur convertit ensuite les sorties en pages équivalentes :</p>"
                f"<p>\\["
                f"P_{{eq/appel}} = \\frac{{{format_scalar(standard_request.get('output_tokens', 0), 0)}}}{{{format_scalar(reference_page_tokens, 0)}}} = {format_scalar(pages_per_request_equivalent, 3)}"
                f"\\]</p>"
                f"<p>Cette conversion repose sur {source_note}.</p>"
                f"<p>\\["
                f"P_{{eq/an}} = {format_count(annual_requests)} \\times {format_scalar(pages_per_request_equivalent, 3)} = {format_count(annual_page_equivalents)}"
                f"\\]</p>"
            )
        else:
            annualization_sentence += (
                "<p>Dans la famille <code>Wh/prompt|requête</code>, une requête LLM correspond directement à une unité d'inférence. "
                "L'annualisation repose donc sur le nombre d'appels LLM par an.</p>"
            )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">1. Données d'entrée du scénario</div>
              <p>Le scénario interprété retient <code>{format_scalar(standard_request.get('input_tokens', 0), 0)}</code> tokens en entrée et <code>{format_scalar(standard_request.get('output_tokens', 0), 0)}</code> tokens en sortie par appel.</p>
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
                  <p>Valeur observée dans la littérature : <code>{escape(anchor.get('source_energy', 'n.d.'))}</code> {ref}</p>
                  <p>Nombre de paramètres source : <code>{format_scalar(anchor.get('source_params'))}B</code>. Nombre de paramètres cible : <code>{format_scalar(anchor.get('target_params'))}B</code>.</p>
                  <p>Facteur paramétrique appliqué :</p>
                  <p>\\[
                  r_P = \\frac{{P_t}}{{P_s}} = \\frac{{{format_scalar(anchor.get('target_params'))}}}{{{format_scalar(anchor.get('source_params'))}}} = {format_scalar(anchor.get('parameter_factor'), 4)}
                  \\]</p>
                  <p>Énergie extrapolée pour une unité d'inférence :</p>
                  <p>\\[
                  E_t = E_s \\times r_P = {escape(anchor.get('source_energy', 'n.d.'))} \\times {format_scalar(anchor.get('parameter_factor'), 4)} = {escape(format_range_display(anchor.get('per_request_energy', {'low':0,'high':0}), 'energy'))}
                  \\]</p>
                </li>
                """
            )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">2. Ancrages de littérature et extrapolation</div>
              <p>La méthode part des valeurs énergétiques publiées dans la littérature pour la famille <code>{escape(detail.get('unit_basis', 'Wh'))}</code>, puis applique une mise à l'échelle par le nombre de paramètres.</p>
              <ul class="extrapolation-list">{''.join(anchor_lines) or '<li>n.d.</li>'}</ul>
              <p>Lorsque plusieurs ancrages existent dans la même famille, le moteur calcule ensuite une moyenne des intensités énergétiques par milliard de paramètres pour obtenir la valeur centrale affichée dans le bloc résultat.</p>
            </div>
            """
        )
        sections.append(
            f"""
            <div class="method-modal-section">
              <div class="math-label">3. Dérivation CO2 et eau à partir du mix pays</div>
              <p>Le carbone et l'eau ne sont pas repris tels quels depuis la littérature. Ils sont dérivés de l'énergie extrapolée avec le mix électrique du pays retenu, ici <strong>{escape(target_country)}</strong>.</p>
              <p>\\[
              CO2_{{unitaire}} = \\frac{{E_{{unitaire}}}}{{1000}} \\times CI_c
              \\qquad
              Water_{{unitaire}} = \\frac{{E_{{unitaire}}}}{{1000}} \\times WI_c \\times 1000
              \\]</p>
              <p>Avec \\(CI_c = {format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\) et \\(WI_c = {format_scalar(target_water)}\\ \\text{{L/kWh}}\\).</p>
              <p>Le résultat unitaire retenu pour cette méthode conduit ensuite aux valeurs annualisées suivantes : énergie <code>{escape(method['energy'])}</code>, carbone <code>{escape(method['carbon'])}</code>, eau <code>{escape(method['water'])}</code>.</p>
            </div>
            <div class="method-modal-section">
              <div class="math-label">4. Annualisation finale</div>
              <p>La projection annuelle finale repose sur <code>{format_count(annual_multiplier)}</code> unité(s) d'inférence par an.</p>
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
          <div class="math-label">1. Annualisation des appels</div>
          <p><code>{format_count(feature_uses_per_month)}</code> usages/mois × <code>{format_scalar(months_per_year, 0)}</code> mois = <code>{format_count(annual_feature_uses)}</code> usages/an</p>
          <p><code>{format_count(annual_feature_uses)}</code> usages/an × <code>{format_scalar(requests_per_feature, 0)}</code> appel(s) LLM/usage = <code>{format_count(annual_requests)}</code> appels LLM/an</p>
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
                f" intensité source ≈ \\({format_scalar(anchor['source_carbon_intensity'])}\\ \\text{{gCO2e/kWh}}\\) "
                f"remplacée par \\({format_scalar(target_carbon)}\\ \\text{{gCO2e/kWh}}\\)"
            )
        water_note = ""
        if anchor.get("source_water_intensity") is not None and target_water is not None:
            water_note = (
                f" intensité source ≈ \\({format_scalar(anchor['source_water_intensity'])}\\ \\text{{L/kWh}}\\) "
                f"remplacée par \\({format_scalar(target_water)}\\ \\text{{L/kWh}}\\)"
            )
        anchor_lines.append(
            f"""
            <li>
              <strong>{escape(anchor.get('source_model', 'source'))}</strong> ({escape(anchor.get('source_country', 'n.d.'))}) :
              énergie publiée <code>{escape(anchor.get('source_energy', 'n.d.'))}</code>,
              ratio appliqué <code>{format_scalar(ratio_value)}</code>,
              énergie par requête <code>{escape(format_range_display(anchor.get('per_request_energy', {'low':0,'high':0}), 'energy'))}</code>,
              carbone par requête <code>{escape(format_range_display(anchor.get('per_request_carbon', {'low':0,'high':0}), 'carbon'))}</code> ({carbon_note.strip() or 'mix cible appliqué'}),
              eau par requête <code>{escape(format_range_display(anchor.get('per_request_water', {'low':0,'high':0}), 'water'))}</code> ({water_note.strip() or 'mix cible appliqué'}).
            </li>
            """
        )
    sections.append(
        f"""
        <div class="method-modal-section">
          <div class="math-label">2. Méthode des multiples sur les indicateurs {escape(unit_basis)}</div>
          <p>Chaque indicateur scientifique disponible dans cette famille d'unités est recalculé pour le scénario cible. Quand plusieurs ancrages existent, la méthode retient d'abord le modèle le plus proche en nombre de paramètres, puis construit la valeur centrale sur cet ancrage sélectionné.</p>
          <ul class="extrapolation-list">{''.join(anchor_lines) or '<li>n.d.</li>'}</ul>
        </div>
        """
    )
    sections.append(
        f"""
        <div class="method-modal-section">
          <div class="math-label">3. Moyenne et annualisation</div>
          <p>Énergie annuelle retenue : moyenne des indicateurs recalculés × <code>{format_count(annual_requests)}</code> appels/an = <code>{escape(method['energy'])}</code>.</p>
          <p>Carbone annuel retenu : moyenne des indicateurs recalculés pour <strong>{escape(target_country)}</strong> = <code>{escape(method['carbon'])}</code>.</p>
          <p>Eau annuelle retenue : moyenne des indicateurs recalculés pour <strong>{escape(target_country)}</strong> = <code>{escape(method['water'])}</code>.</p>
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
    target_water = target_mix.get("water_intensity_l_per_kwh") or parsed_payload.get("water_intensity_l_per_kwh")

    for method in result.get("method_results", []):
        rows = factor_details(records, method.get("record_ids", []))
        detail = dict(method.get("detail", {}))
        methods.append(
            {
                "label": method.get("label", "Méthode"),
                "basis": method.get("basis", ""),
                "energy": format_result_card_display(method["annual_energy_wh"], "energy"),
                "carbon": format_result_card_display(method["annual_carbon_gco2e"], "carbon"),
                "water": format_result_card_display(method["annual_water_ml"], "water"),
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
                "target_water_intensity": float(target_water) if target_water not in (None, "") else None,
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
              <div class="result-method-kicker">Résultat</div>
              <h4>{escape(method['label'])}</h4>
            </div>
            <div class="result-method-refs">{method['refs']}</div>
          </div>
          <p class="result-method-basis">{escape(method['basis'])}</p>
          <div class="result-method-metrics">
            <div class="result-method-metric">
              <span class="result-method-label">Énergie annuelle</span>
              <strong>{escape(method['energy'])}</strong>
            </div>
            <div class="result-method-metric">
              <span class="result-method-label">Carbone annuel</span>
              <strong>{escape(method['carbon'])}</strong>
            </div>
            <div class="result-method-metric">
              <span class="result-method-label">Eau annuelle</span>
              <strong>{escape(method['water'])}</strong>
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
              <div class="summary-kicker">Détail du calcul</div>
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
        ("training", "training_emissions"): "Émissions de gaz à effet de serre liées à l'entraînement",
        ("training", "compute_time"): "Temps de calcul mobilisé pour l'entraînement",
        ("training", "training_tokens"): "Volume de tokens utilisés pour l'entraînement",
        ("lifecycle", "creation_lifecycle_emissions"): "Émissions sur l'ensemble du cycle de création du modèle",
        ("lifecycle", "creation_lifecycle_water"): "Consommation d'eau sur l'ensemble du cycle de création du modèle",
        ("lifecycle", "development_share"): "Part de l'impact attribuée à la phase de développement",
        ("lifecycle", "power_utilization_range"): "Plage de facteur d'utilisation électrique de l'infrastructure",
        ("lifecycle", "training_water_total"): "Consommation totale d'eau associée à l'entraînement",
        ("lifecycle", "training_water_onsite"): "Consommation d'eau sur site associée à l'entraînement",
        ("infrastructure", "ai_share_of_datacenter_electricity"): "Part de l'IA dans la consommation électrique des centres de données",
        ("infrastructure", "annual_electricity"): "Consommation électrique annuelle des centres de données",
        ("infrastructure", "electricity_share"): "Part de la demande électrique attribuée aux centres de données",
        ("infrastructure", "annual_growth_rate"): "Taux de croissance annuel de la consommation électrique",
        ("infrastructure", "ai_power_demand"): "Puissance électrique appelée par les systèmes d'IA",
        ("infrastructure", "annualized_energy"): "Consommation énergétique annualisée des systèmes d'IA",
        ("inference", "query_energy"): "Consommation énergétique par requête",
        ("inference", "prompt_energy"): "Consommation énergétique par prompt",
        ("inference", "prompt_emissions"): "Émissions par prompt",
        ("inference", "prompt_water"): "Consommation d'eau par prompt",
        ("inference", "efficiency_gain"): "Gain d'efficacité annoncé entre deux configurations d'inférence",
        ("inference", "energy_component"): "Répartition de la consommation énergétique par composant d'inférence",
        ("inference", "page_generation_energy"): "Consommation énergétique pour générer une page",
        ("inference", "page_generation_emissions"): "Émissions pour générer une page",
        ("inference", "page_generation_water"): "Consommation d'eau pour générer une page",
        ("inference", "response_water_equivalent"): "Consommation d'eau pour un petit ensemble de réponses",
    }

    label = labels.get((phase, metric_name))
    if not label:
        phase_label = {
            "training": "Entraînement",
            "inference": "Inférence",
            "infrastructure": "Infrastructure",
            "lifecycle": "Cycle de vie",
        }.get(phase, phase.capitalize() if phase else "Indicateur")
        metric_label = metric_name.replace("_", " ") if metric_name else "non précisé"
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
        return f"""
        <div class="reference-subtable">
          <h4>{escape(title)}</h4>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Réf.</th>
                  <th>Type de donnée</th>
                  <th>Modèle LLM</th>
                  <th>Paramètres</th>
                  <th>Pays</th>
                  <th>Valeur</th>
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
        "training": render_reference_table("".join(grouped["training"]), "Apprentissage"),
        "inference": render_reference_table("".join(grouped["inference"]), "Inférence"),
        "counts": {
            "total": len(rows),
            "training": len(grouped["training"]),
            "inference": len(grouped["inference"]),
        },
    }


def render_model_reference_table():
    rows = load_models()
    if not rows:
        return ""
    body = "".join(
        f"""
        <tr>
          <td>{escape(row.get('model_id', ''))}</td>
          <td>{escape(row.get('provider', ''))}</td>
          <td>{escape((row.get('active_parameters_billion') or 'n.d.') + ('B' if row.get('active_parameters_billion') else ''))}</td>
          <td>{escape((row.get('total_parameters_billion') or 'n.d.') + ('B' if row.get('total_parameters_billion') else ''))}</td>
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
          <div class="summary-kicker">Modèles</div>
          <h3>Table de référence des modèles</h3>
        </div>
      </div>
      <p class="summary-intro">Cette table recense les modèles de référence utilisés par le projet, leur nombre de paramètres et la source de la valeur observée ou estimée.</p>
      <div class="reference-table-wrap">
        <table class="reference-table">
          <thead>
            <tr>
              <th>Modèle</th>
              <th>Provider</th>
              <th>Paramètres actifs</th>
              <th>Paramètres totaux</th>
              <th>Statut</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
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
          <td>{escape(row.get('water_intensity_l_per_kwh', ''))} L/kWh</td>
          <td>{escape(row.get('source_citation', 'n.d.'))}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Pays</div>
          <h3>Table de référence des mix énergétiques</h3>
        </div>
      </div>
      <p class="summary-intro">Cette table recense les facteurs pays utilisés pour contextualiser le carbone et l'eau, avec la source associée à chaque valeur.</p>
      <div class="reference-table-wrap">
        <table class="reference-table">
          <thead>
            <tr>
              <th>Code</th>
              <th>Pays</th>
              <th>Année</th>
              <th>Intensité carbone</th>
              <th>Intensité eau</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </section>
    """


def format_market_country_status(value):
    labels = {
        "multi_region": "Multi-région documentée",
        "documented_multi_region": "Multi-région documentée",
        "self_hosted_variable": "Variable selon l'hébergeur",
        "provider_country_proxy": "Proxy pays fournisseur",
        "screening_proxy": "Proxy de screening",
        "comparative_reference": "Pays de référence comparative",
        "documented_region_proxy": "Région documentée, pays retenu à titre de référence",
        "non_specified": "Non spécifié",
    }
    return labels.get(value, value or "n.d.")


def format_market_parameter_display(row):
    active = str(row.get("active_parameters_billion", "") or "").strip()
    total = str(row.get("total_parameters_billion", "") or "").strip()
    if active and total and active != total:
        return f"{active}B actifs / {total}B totaux"
    if active:
        return f"{active}B"
    if total:
        return f"{total}B"
    return "n.d."


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
        parameter_title = escape(str(row.get("parameter_source", "") or "Source non précisée"))
        server_title = escape(str(row.get("server_country_source", "") or "Source non précisée"))
        estimation_title = escape(str(row.get("estimation_country_source", "") or "Source non précisée"))
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
                "prompt_water_ml": float((prompt_method.get("annual_water_ml") or {}).get("central", 0.0) or 0.0),
                "page_water_ml": float((page_method.get("annual_water_ml") or {}).get("central", 0.0) or 0.0),
            }
        )
        body.append(
            f"""
            <tr>
              <td><strong>{escape(row.get('display_name', row.get('model_id', '')))}</strong><div class="method-basis">{escape(row.get('provider', ''))}</div></td>
              <td>{escape(format_market_parameter_display(row))}</td>
              <td>{escape(row.get('server_country', 'n.d.') or 'n.d.')}<div class="reference-locator">{escape(format_market_country_status(row.get('server_country_status')))}</div></td>
              <td>{escape(row.get('estimation_country_code', 'n.d.') or 'n.d.')}<div class="reference-locator">{escape(format_market_country_status(row.get('estimation_country_status')))}</div></td>
              <td>{escape(prompt_energy)}</td>
              <td>{escape(page_energy)}</td>
              <td>{escape(prompt_carbon)}</td>
              <td>{escape(page_carbon)}</td>
              <td>
                <div><a href="{escape(str(row.get('parameter_source_url', '') or '#'), quote=True)}" target="_blank" rel="noopener noreferrer" title="{parameter_title}">Paramètres</a></div>
                <div><a href="{escape(str(row.get('server_country_source_url', '') or '#'), quote=True)}" target="_blank" rel="noopener noreferrer" title="{server_title}">Pays serveur</a></div>
                <div><a href="{escape(str(row.get('estimation_country_source_url', '') or '#'), quote=True)}" target="_blank" rel="noopener noreferrer" title="{estimation_title}">Pays retenu</a></div>
              </td>
            </tr>
            """
        )

    chart_rows.extend(
        [
            {
                "label": "Lampe fluorescente 1 h",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "prompt_energy_wh": 9.3,
                "page_energy_wh": 9.3,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
                "prompt_water_ml": 0.0,
                "page_water_ml": 0.0,
            },
            {
                "label": "Ordinateur portable 1 h",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "prompt_energy_wh": 32.0,
                "page_energy_wh": 32.0,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
                "prompt_water_ml": 0.0,
                "page_water_ml": 0.0,
            },
            {
                "label": "Chauffage électrique 1 h",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "prompt_energy_wh": 1500.0,
                "page_energy_wh": 1500.0,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
                "prompt_water_ml": 0.0,
                "page_water_ml": 0.0,
            },
            {
                "label": "Douche 1 h",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "prompt_energy_wh": 0.0,
                "page_energy_wh": 0.0,
                "prompt_carbon_gco2e": 0.0,
                "page_carbon_gco2e": 0.0,
                "prompt_water_ml": 600000.0,
                "page_water_ml": 600000.0,
            },
            {
                "label": "Voiture thermique 1 h",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "prompt_energy_wh": 0.0,
                "page_energy_wh": 0.0,
                "prompt_carbon_gco2e": 23500.0,
                "page_carbon_gco2e": 23500.0,
                "prompt_water_ml": 0.0,
                "page_water_ml": 0.0,
            },
        ]
    )

    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Impact environnemental comparé des modèles</h3>
        </div>
      </div>
      <p class="summary-intro">Le graphique ci-dessous représente les valeurs centrales estimées pour tous les modèles du catalogue sur un scénario standardisé d'inférence correspondant à <strong>1 heure d'usage actif</strong> : <strong>{requests_per_hour} interactions/heure</strong>, <strong>1000 tokens en entrée</strong>, <strong>550 tokens en sortie</strong>, une requête LLM par usage. La cadence horaire est dérivée d'une vitesse moyenne de lecture de <strong>{reading_wpm} mots/min</strong> (Brysbaert, 2019) et d'une convention de conversion de travail <strong>1 token ≈ {words_per_token} mot</strong>.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Indicateur du graphique d'inférence">
        <button type="button" class="chart-tab-button is-active" data-model-chart-control="metric-tab" data-metric-value="energy" aria-selected="true">Énergie</button>
        <button type="button" class="chart-tab-button" data-model-chart-control="metric-tab" data-metric-value="carbon" aria-selected="false">Carbone</button>
        <button type="button" class="chart-tab-button" data-model-chart-control="metric-tab" data-metric-value="water" aria-selected="false">Eau</button>
      </div>
      <div id="models-impact-chart" class="models-impact-chart" data-chart-rows='{escape(json.dumps(chart_rows, ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Repères intégrés dans le graphique, tous exprimés sur une durée d'une heure : énergie domestique issue de mesures Purdue Extension (lampe fluorescente ≈ 9,3 Wh sur 1 h ; ordinateur portable ≈ 32 Wh sur 1 h) et d'un chauffage électrique d'appoint de puissance nominale 1500 W (≈ 1,5 kWh sur 1 h) ; eau domestique issue de l'ADEME/INC et de l'Anses (douche ≈ 10 L/min, soit ≈ 600 L sur 1 h) ; carbone automobile issu de l'ICCT (2025, 235 gCO2e/km pour une voiture essence moyenne) avec une convention de comparaison fixée à 100 km parcourus en 1 h, soit ≈ 23,5 kgCO2e sur 1 h.</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Modèles</div>
          <h3>{len(rows)} modèles actuels suivis par le projet</h3>
        </div>
      </div>
      <p class="summary-intro">Le tableau ci-dessous compare les modèles suivis par le projet sur ce même scénario d'inférence. Pour chaque modèle, l'application affiche séparément l'estimation dérivée de la moyenne <strong>Wh/prompt|requête</strong> et l'estimation dérivée de la moyenne <strong>Wh/page</strong>.</p>
      <div class="table-toolbar">
        <label class="table-search-label" for="market-model-search">Rechercher un modèle</label>
        <input id="market-model-search" class="table-search-input" type="search" placeholder="Exemple: GPT, Claude, Mistral, US, 70B" data-table-search="market-models-table">
      </div>
      <div class="reference-table-wrap">
        <table class="reference-table sortable-table" id="market-models-table">
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="0" data-sort-type="text">Modèle</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="1" data-sort-type="text">Paramètres</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="2" data-sort-type="text">Pays serveur</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="3" data-sort-type="text">Pays retenu</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="4" data-sort-type="number">Énergie / h prompt-réq</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="5" data-sort-type="number">Énergie / h page</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="6" data-sort-type="number">Carbone / h prompt-réq</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="7" data-sort-type="number">Carbone / h page</button></th>
              <th><button type="button" class="sort-button" data-sort-table="market-models-table" data-sort-index="8" data-sort-type="text">Sources</button></th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
      <p class="summary-intro">`Pays serveur` décrit l'information publiée sur l'hébergement du service ou, pour les open weights, le fait que l'hébergement dépend du déploiement. `Pays retenu` correspond au pays effectivement utilisé pour recalculer le CO2 via le mix électrique. Quand le pays exact n'est pas publié, le projet utilise un proxy explicite de screening plutôt qu'une localisation présentée comme certaine.</p>
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
                "creation_lifecycle_water_kl": float(lifecycle_water.get("value", 0.0) or 0.0),
            }
        )
        body.append(
            f"""
            <tr>
              <td><strong>{escape(row.get('display_name', row.get('model_id', '')))}</strong><div class="method-basis">{escape(row.get('provider', ''))}</div></td>
              <td>{escape(format_market_parameter_display(row))}</td>
              <td>{escape(format_training_estimate(direct_energy.get('value'), direct_energy.get('unit')))}</td>
              <td>{escape(format_training_estimate(direct_carbon.get('value'), direct_carbon.get('unit')))}</td>
              <td>{escape(format_training_estimate(lifecycle_water.get('value'), lifecycle_water.get('unit')))}</td>
            </tr>
            """
        )

    chart_rows.extend(
        [
            {
                "label": "10 000 foyers (usages domestiques annuels)",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "direct_training_energy_wh": 25000000000.0,
                "direct_training_carbon_tco2e": 235.0,
                "creation_lifecycle_water_kl": 0.0,
            },
            {
                "label": "4 955 vols commerciaux complets",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 104560.5,
                "creation_lifecycle_water_kl": 0.0,
            },
            {
                "label": "1 000 douches courtes",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 0.0,
                "creation_lifecycle_water_kl": 50.0,
            },
            {
                "label": "1 000 bains",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 0.0,
                "creation_lifecycle_water_kl": 150.0,
            },
            {
                "label": "1,5 million de journées d'eau potable",
                "provider": "Repère de vie courante",
                "kind": "reference",
                "direct_training_energy_wh": 0.0,
                "direct_training_carbon_tco2e": 0.0,
                "creation_lifecycle_water_kl": 225000.0,
            },
        ]
    )

    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Visualisation</div>
          <h3>Impacts d'apprentissage comparés des modèles</h3>
        </div>
      </div>
      <p class="summary-intro">Le graphique ci-dessous représente les valeurs centrales extrapolées pour tous les modèles du catalogue sur trois familles d'indicateurs d'apprentissage : énergie d'entraînement, CO2e d'entraînement direct et eau du cycle de création. Des repères de vie courante sont insérés directement dans la liste pour situer les ordres de grandeur.</p>
      <div class="chart-tabbar" role="tablist" aria-label="Indicateur du graphique d'apprentissage">
        <button type="button" class="chart-tab-button is-active" data-training-chart-control="metric-tab" data-metric-value="direct_training_energy" aria-selected="true">Énergie</button>
        <button type="button" class="chart-tab-button" data-training-chart-control="metric-tab" data-metric-value="direct_training_carbon" aria-selected="false">Carbone</button>
        <button type="button" class="chart-tab-button" data-training-chart-control="metric-tab" data-metric-value="creation_lifecycle_water" aria-selected="false">Eau cycle de création</button>
      </div>
      <div id="training-impact-chart" class="models-impact-chart" data-training-chart-rows='{escape(json.dumps(chart_rows, ensure_ascii=False), quote=True)}'></div>
      <p class="summary-intro models-benchmark-note">Repères intégrés dans le graphique : énergie domestique avec 10 000 foyers sur une année d'usages domestiques, soit ≈ 25 GWh sur la base d'une consommation moyenne de 2 500 kWh par ménage (RTE, estimation 2021), automobile issue de l'ICCT (2025, 235 gCO2e/km pour une voiture essence moyenne), avion complet dérivé de Klöwer et al. (2025) à partir de 577,97 MtCO2 et 27,45 millions de vols commerciaux observés en 2023, soit ≈ 104 560,5 tCO2 pour 4 955 vols complets, et eau domestique issue de l'ADEME/INC et de l'Anses (douche courte ≈ 50 L ; bain ≈ 150 L ; consommation quotidienne moyenne d'eau potable ≈ 150 L par personne en France, soit ≈ 225 000 kL pour 1,5 million de journées).</p>
    </section>
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Modèles</div>
          <h3>{len(rows)} modèles actuels avec estimation des impacts d'apprentissage</h3>
        </div>
      </div>
      <p class="summary-intro">Ce tableau projette les ordres de grandeur d'apprentissage des modèles actuels à partir des familles d'indicateurs réellement disponibles dans la littérature: <strong>énergie d'entraînement</strong> dérivée des émissions quand le pays source est documenté dans la table des mix, <strong>CO2e d'entraînement direct</strong> et <strong>eau du cycle de création</strong>. Les valeurs sont extrapolées par nombre de paramètres. L'énergie d'entraînement reste donc une reconstruction de screening plus fragile que le carbone direct et l'eau.</p>
      <div class="table-toolbar">
        <label class="table-search-label" for="training-model-search">Rechercher un modèle</label>
        <input id="training-model-search" class="table-search-input" type="search" placeholder="Exemple: GPT, Claude, 70B, Meta" data-table-search="training-models-table">
      </div>
      <div class="reference-table-wrap">
        <table class="reference-table sortable-table" id="training-models-table">
          <thead>
            <tr>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="0" data-sort-type="text">Modèle</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="1" data-sort-type="text">Paramètres</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="2" data-sort-type="number">Énergie d'entraînement</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="3" data-sort-type="number">CO2e entraînement direct</button></th>
              <th><button type="button" class="sort-button" data-sort-table="training-models-table" data-sort-index="4" data-sort-type="number">Eau cycle de création</button></th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
    </section>
    """


def render_how_it_works_tab(records):
    anchors = build_energy_inference_anchors(records)
    prompt_query_anchors = [anchor for anchor in anchors if anchor.get("metric_name") in {"prompt_energy", "query_energy"}]
    page_anchors = [anchor for anchor in anchors if anchor.get("metric_name") == "page_generation_energy"]

    def render_anchor_rows(anchor_rows):
        if not anchor_rows:
            return "<tr><td colspan=\"5\">Aucun ancrage exploitable dans cette famille.</td></tr>"
        rows = []
        for anchor in anchor_rows:
            record = get_record(records, anchor.get("record_id"))
            rows.append(
                f"""
                <tr>
                  <td>{escape(anchor.get('source_model', 'n.d.'))}</td>
                  <td>{escape(format_reference_parameters(record.get('model_parameters_normalized') if record else 'n.d.'))}</td>
                  <td>{escape(anchor.get('source_country', 'n.d.'))}</td>
                  <td>{escape(anchor.get('source_energy', 'n.d.'))}</td>
                  <td><a href="{escape(str((record or {}).get('source_url', '#')), quote=True)}" target="_blank" rel="noopener noreferrer" title="{escape(format_apa_hover(record or {}))}">{escape(format_apa_citation(record or {}))}</a></td>
                </tr>
                """
            )
        return "".join(rows)

    return f"""
    <section class="tab-panel" id="tab-how-panel" data-tab-panel="how">
      <section class="panel summary-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">How it works</div>
            <h3>Méthode d'estimation d'inférence</h3>
          </div>
        </div>
        <p class="summary-intro">EcoTrace LLM ne prétend pas mesurer des données télémétriques propriétaires qui ne sont pas publiées. L'application construit une extrapolation de screening à partir des quelques indicateurs d'inférence quantifiés réellement observés dans la littérature scientifique.</p>
        <div class="summary-body">
          <p><strong>1. Ancrages observés.</strong> Le moteur prédictif part uniquement des ancrages énergétiques d'inférence pour lesquels le nombre de paramètres du modèle source est connu ou estimé. Dans l'état actuel du corpus, cela donne une famille <code>Wh/prompt|requête</code> et une famille <code>Wh/page</code>.</p>
          <p><strong>2. Mise à l'échelle par modèle.</strong> Pour chaque famille, l'application calcule une intensité moyenne <code>Wh / milliard de paramètres</code>, puis applique cette intensité au modèle cible à partir de son nombre de paramètres publiés ou estimés.</p>
          <p><strong>3. Conversion pays.</strong> Le carbone et l'eau ne sont pas extrapolés directement depuis la littérature. Ils sont dérivés de l'énergie via le mix électrique du pays retenu. Pour un service propriétaire hébergé, on retient le pays de l'éditeur dans le catalogue. Pour un modèle open weight, on retient le pays du projet quand il est fourni.</p>
          <p><strong>4. Annualisation.</strong> Les résultats par requête ou par page sont ensuite projetés à l'année en fonction du nombre d'usages, du nombre d'appels LLM par usage et de la fréquence mensuelle.</p>
        </div>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Formules</div>
            <h3>Chaîne de calcul retenue</h3>
          </div>
        </div>
        <div class="summary-body">
          <p>\\[
          I_f = \\frac{{1}}{{|A_f|}} \\sum_{{a \\in A_f}} \\frac{{E_a}}{{P_a}}
          \\]
          avec \\( f \\in \\{{\\text{{prompt|requête}}, \\text{{page}}\\}} \\).</p>
          <p>\\[
          E_{{t,f}} = I_f \\times P_t
          \\]
          pour estimer l'énergie d'inférence du modèle cible.</p>
          <p>\\[
          CO2_{{t,f,c}} = \\frac{{E_{{t,f}}}}{{1000}} \\times CI_c
          \\qquad
          Water_{{t,f,c}} = \\frac{{E_{{t,f}}}}{{1000}} \\times WI_c \\times 1000
          \\]</p>
          <p>\\[
          N_{{year}} = usages_{{mensuels}} \\times 12 \\times appels_{{LLM/usage}}
          \\qquad
          Impact_{{year}} = Impact_{{unitaire}} \\times N_{{year}}
          \\]</p>
          <p><strong>Définition des variables.</strong></p>
          <p>\\(A_f\\) désigne l'ensemble des ancrages de littérature disponibles pour la famille d'unités \\(f\\). \\(E_a\\) est la valeur énergétique observée pour l'ancrage \\(a\\), convertie en Wh dans son unité native. \\(P_a\\) est le nombre de paramètres du modèle source de l'ancrage, exprimé en milliards. \\(I_f\\) est l'intensité énergétique moyenne par milliard de paramètres pour la famille \\(f\\).</p>
          <p>\\(P_t\\) désigne le nombre de paramètres du modèle cible. \\(E_{{t,f}}\\) est l'énergie estimée pour ce modèle cible dans la famille \\(f\\). \\(CI_c\\) est l'intensité carbone de l'électricité du pays \\(c\\), exprimée en gCO2e/kWh. \\(WI_c\\) est l'intensité hydrique de l'électricité du pays \\(c\\), exprimée en L/kWh.</p>
          <p>\\(N_{{year}}\\) est le nombre annuel d'unités d'inférence retenues pour le scénario. \\(Impact_{{unitaire}}\\) est la valeur énergétique, carbone ou eau associée à une unité d'inférence. \\(Impact_{{year}}\\) est la projection annuelle obtenue après multiplication par le volume d'usage.</p>
        </div>
        <p class="summary-intro">Les deux familles <code>Wh/prompt|requête</code> et <code>Wh/page</code> restent affichées séparément. Le projet ne fusionne pas ces deux mesures en une pseudo-valeur unique, car le corpus ne contient pas encore de conversion empirique robuste entre elles pour un même modèle.</p>
      </section>

      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Corpus</div>
            <h3>Ancrages énergétiques mobilisés dans le prédicteur</h3>
          </div>
        </div>
        <p class="summary-intro">Seuls les indicateurs énergétiques d'inférence disposant d'un nombre de paramètres exploitable sont retenus dans le coeur prédictif. Les autres indicateurs restent visibles dans le référentiel, mais ne servent pas à la mise à l'échelle des modèles.</p>
        <div class="reference-subtable">
          <h4>Famille Wh/prompt|requête</h4>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Modèle source</th>
                  <th>Paramètres</th>
                  <th>Pays</th>
                  <th>Valeur observée</th>
                  <th>Citation</th>
                </tr>
              </thead>
              <tbody>{render_anchor_rows(prompt_query_anchors)}</tbody>
            </table>
          </div>
        </div>
        <div class="reference-subtable">
          <h4>Famille Wh/page</h4>
          <div class="reference-table-wrap">
            <table class="reference-table">
              <thead>
                <tr>
                  <th>Modèle source</th>
                  <th>Paramètres</th>
                  <th>Pays</th>
                  <th>Valeur observée</th>
                  <th>Citation</th>
                </tr>
              </thead>
              <tbody>{render_anchor_rows(page_anchors)}</tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel summary-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Limites</div>
            <h3>Ce que la méthode fait et ne fait pas</h3>
          </div>
        </div>
        <div class="summary-body">
          <p><strong>Oui :</strong> produire une estimation cohérente là où les fournisseurs ne publient rien, en documentant explicitement les ancrages, les paramètres et le pays retenu.</p>
          <p><strong>Non :</strong> prétendre fournir une mesure auditée ou un chiffre officiel d'empreinte du fournisseur. Le résultat doit être lu comme une estimation de screening, utile pour comparer des ordres de grandeur et guider l'écoconception.</p>
          <p><strong>Point de vigilance :</strong> les nombres de paramètres de plusieurs modèles propriétaires sont estimés à partir de sources tierces. Ils sont maintenus dans le catalogue avec leur source et leur statut (<code>observed</code>, <code>estimated</code> ou proxy de famille).</p>
        </div>
      </section>
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
            "Décris une application, une fonctionnalité ou un workflow utilisant un LLM, "
            "avec son usage, son volume ou son contexte technique."
        )
        raise OpenAIModerationError(
            f"Cette description ne correspond pas clairement à un logiciel ou à un usage de LLM exploitable par la plateforme. "
            f"{moderation['reason']} {guidance}"
        )
    parsed_payload, parser_notes, parser_meta = parse_application_description_with_openai(description)
    parser_meta["moderation"] = moderation
    parsed_payload["software_components"] = []
    parser_notes.append("L'estimation a ete restreinte aux consommations du LLM; les composants logiciels hors LLM ont ete exclus du calcul.")
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
          <h2>Description non exploitable</h2>
          <p class="lead">{escape(error_message)}</p>
          <p class="lead">EcoTrace LLM attend une description d'application, de fonctionnalité ou de workflow mobilisant un ou plusieurs LLMs.</p>
        </section>
        """

    reference_sections = render_reference_catalog_sections()
    training_reference_block = reference_sections["training"]
    inference_reference_block = reference_sections["inference"]
    reference_counts = reference_sections["counts"]
    all_records = load_records()
    market_models_block = render_market_models_table(all_records)
    training_models_block = render_training_models_table(all_records)
    how_it_works_tab = render_how_it_works_tab(all_records)
    result_block = ""
    result_methods_block = ""
    if result:
        annual = result["annual_llm"]
        method_comparisons = build_method_comparisons(load_records(), parsed_payload, result)
        result_methods_block = render_method_comparisons(method_comparisons)
        evidence = (parser_meta or {}).get("evidence", {})
        method_label = {
            "parametric_extrapolation": "Extrapolation parametrique",
            "literature_proxy": "Proxy de litterature",
            "literature_multiples": "Agrégation multi-indicateurs d'inférence",
            "wh_parameter_model": "Modèle unifié Wh -> paramètres",
        }.get(result.get("method"), "Methode non qualifiee")
        model_profile = result.get("model_profile") or {}
        country_mix = result.get("country_energy_mix") or {}
        country_resolution_label = {
            "publisher_country": "pays de l'éditeur",
            "project_country": "pays du projet",
            "fallback_reference_country": "pays de référence",
            "explicit_country": "pays explicite",
        }.get(result.get("country_resolution"), "pays retenu")
        result_block = f"""
        <section class="panel result hero-card">
          <div class="summary-header">
            <div>
              <div class="summary-kicker">Calcul</div>
              <h3>DÉTAIL DU CALCUL</h3>
            </div>
          </div>
          <p class="lead">Estimation d'inférence fondée sur des indicateurs scientifiques sourcés et un calcul traçable.</p>
          <p class="scope-note">Périmètre retenu: seules les externalités d'inférence du LLM sont prises en compte. L'apprentissage du modèle, les consommations du système logiciel et les infrastructures annexes sont exclus de cette estimation affichée.</p>
          <p class="meta-inline">Niveau de preuve: <strong>{escape(evidence.get('label', 'Non qualifie'))}</strong></p>
          <p class="meta-inline">Methode: <strong>{escape(method_label)}</strong></p>
          <p class="meta-inline">Modele de reference: <strong>{escape(model_profile.get('model_id', parsed_payload.get('model_id', 'non specifie')))}</strong>{' | Parametres actifs approx.: <strong>' + escape(model_profile.get('active_parameters_billion', '')) + 'B</strong>' if model_profile.get('active_parameters_billion') else ''}</p>
          <p class="meta-inline">Mix electrique: <strong>{escape(country_mix.get('country_code', parsed_payload.get('country', 'non specifie')))}</strong> <span class="method-basis">({escape(country_resolution_label)})</span>{' | ' + escape(country_mix.get('grid_carbon_intensity_gco2_per_kwh', '')) + ' gCO2e/kWh' if country_mix.get('grid_carbon_intensity_gco2_per_kwh') else ''}</p>
          {render_assumptions_summary(result)}
          {render_method_calculation_details(method_comparisons)}
        </section>
        """

    home_tab = f"""
    <section class="tab-panel is-active" id="tab-home-panel" data-tab-panel="home">
      <header class="hero">
        <div class="eyebrow">Projet {PROJECT_NAME}</div>
        <h1>Estimer l’empreinte environnementale d’inférence d’une application utilisant des LLMs</h1>
        <p class="subtitle">Décris ton application en langage naturel pour obtenir une estimation d’inférence, ses hypothèses et son détail de calcul sourcé.</p>
      </header>
      <form class="panel" method="post" action="/" id="estimate-form">
        <label for="description">Description libre de l'application</label>
        <textarea id="description" name="description" placeholder="Exemple: Nous avons un assistant RAG sur GPT-4 via API, utilisé 4000 fois par mois en France. Chaque requête envoie 2200 input tokens et reçoit 500 output tokens. Il y a une base vectorielle, des embeddings et du logging.">{escape(description)}</textarea>
        <button type="submit" id="submit-button">
          <span class="spinner" aria-hidden="true"></span>
          <span class="default-text">Evaluer l'application</span>
          <span class="loading-text">Evaluation en cours...</span>
        </button>
      </form>
      {result_methods_block}
      {error_block}
      {result_block}
    </section>
    """
    training_tab = f"""
    <section class="tab-panel" id="tab-training-panel" data-tab-panel="training">
      {training_models_block}
      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Référentiel</div>
            <h3>{reference_counts['training']} indicateurs d'apprentissage issus de la littérature scientifique</h3>
          </div>
        </div>
        <p class="summary-intro">Cette page présente des indicateurs documentés sur les externalités environnementales liées à l'apprentissage et au cycle de vie de création des modèles LLM. Les références macro d'infrastructure et les sources institutionnelles hors périmètre LLM ont été exclues.</p>
        {training_reference_block}
        <p class="summary-intro">`*` indique une valeur de nombre de paramètres estimée et non officiellement publiée par le fournisseur.</p>
      </section>
    </section>
    """
    models_tab = f"""
    <section class="tab-panel" id="tab-models-panel" data-tab-panel="models">
      {market_models_block}
      <section class="panel reference-panel">
        <div class="summary-header">
          <div>
            <div class="summary-kicker">Référentiel</div>
            <h3>{reference_counts['inference']} indicateurs d'inférence issus de la littérature scientifique</h3>
          </div>
        </div>
        <p class="summary-intro">Ce tableau regroupe les indicateurs quantifiés relatifs à l'inférence des LLMs dans les publications retenues.</p>
        {inference_reference_block}
        <p class="summary-intro">`*` indique une valeur de nombre de paramètres estimée et non officiellement publiée par le fournisseur.</p>
      </section>
    </section>
    """

    return f"""<!doctype html>
<html lang="fr">
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
      --ink: #20261f;
      --muted: #667065;
      --line: #d7dbd2;
      --accent: #3f5a49;
      --accent-soft: #eff2ec;
      --error: #dc3545;
      --shadow: none;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      scroll-behavior: smooth;
    }}
    .wrap {{ max-width: 960px; margin: 0 auto; padding: 24px 16px 40px; }}
    .hero {{ margin-bottom: 20px; }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 0;
      border-radius: 0;
      background: transparent;
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(1.8rem, 4vw, 2.35rem); line-height: 1.18; font-weight: 650; }}
    h2, h3 {{ margin: 0 0 0.75rem; font-weight: 700; }}
    .subtitle {{ max-width: 760px; color: var(--muted); font-size: 0.98rem; line-height: 1.55; margin: 0; }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 0.4rem;
      padding: 1rem 1rem;
      box-shadow: var(--shadow);
      margin-bottom: 1rem;
    }}
    .hero-card {{
      border-color: var(--line);
      background: var(--paper);
    }}
    .alert-panel.error {{
      border-color: rgba(220,53,69,0.3);
      background: #fff5f6;
    }}
    textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      padding: 0.875rem 1rem;
      font: inherit;
      background: #fdfdf9;
    }}
    textarea:focus, input:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: none;
    }}
    textarea {{ min-height: 200px; resize: vertical; }}
    label {{ display: block; font-size: 0.95rem; font-weight: 600; color: var(--ink); margin-bottom: 0.5rem; }}
    button {{
      margin-top: 1rem;
      border: 1px solid var(--accent);
      border-radius: 0.35rem;
      background: var(--paper);
      color: var(--accent);
      padding: 0.75rem 1.15rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.65rem;
    }}
    button:hover {{ background: var(--accent-soft); }}
    button:disabled {{
      background: #f1f3ef;
      color: #8a948b;
      border-color: #b9c0b7;
      cursor: wait;
    }}
    .spinner {{
      width: 1rem;
      height: 1rem;
      border: 2px solid rgba(255,255,255,0.45);
      border-top-color: #fff;
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
      margin-top: 0.9rem;
      padding: 0.9rem;
      border-radius: 0.65rem;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 0.5rem;
    }}
    .metric strong {{
      font-size: 1.12rem;
      line-height: 1.35;
    }}
    .submetrics-block {{
      margin-top: 1rem;
      padding-top: 0.9rem;
      border-top: 1px solid var(--line);
    }}
    .submetrics-title {{
      margin: 0 0 0.7rem;
      color: #495057;
      font-size: 0.92rem;
      font-weight: 700;
    }}
    .submetrics-title-secondary {{
      margin-top: 1rem;
    }}
    .submetrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.7rem;
    }}
    .submetric {{
      padding: 0.75rem 0.8rem;
      border: 1px solid var(--line);
      border-radius: 0.6rem;
      background: #f7f7f3;
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
      margin-top: 1rem;
      padding-top: 0.9rem;
      border-top: 1px solid var(--line);
    }}
    .result-panel {{
      margin-bottom: 1.5rem;
      padding-top: 0;
      border-top: 0;
    }}
    .result-method-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 0.9rem;
    }}
    .result-method-card {{
      border: 1px solid var(--line);
      border-radius: 0.4rem;
      background: var(--paper);
      padding: 1rem;
      box-shadow: none;
    }}
    .result-method-head {{
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: flex-start;
      margin-bottom: 0.6rem;
    }}
    .result-method-head h4 {{
      margin: 0;
      font-size: 1.02rem;
      line-height: 1.35;
    }}
    .result-method-kicker {{
      margin-bottom: 0.2rem;
      color: var(--accent);
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .result-method-refs {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 0.3rem;
      min-width: 3rem;
    }}
    .result-method-basis {{
      margin: 0 0 0.85rem;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
    }}
    .result-method-metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.6rem;
    }}
    .result-method-metric {{
      padding: 0.75rem 0.8rem;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: #f8f8f4;
    }}
    .result-method-label {{
      display: block;
      margin-bottom: 0.3rem;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .result-method-metric strong {{
      font-size: 1rem;
      line-height: 1.35;
    }}
    .result-method-actions {{
      margin-top: 0.9rem;
    }}
    .result-method-detail {{
      margin-top: 0.95rem;
      padding-top: 0.9rem;
      border-top: 1px solid var(--line);
    }}
    .result-method-detail-card {{
      border: 1px solid var(--line);
      border-radius: 0.4rem;
      background: var(--paper);
      padding: 1rem;
      box-shadow: none;
    }}
    .result-method-detail-card + .result-method-detail-card {{
      margin-top: 0.9rem;
    }}
    .result-method-detail .method-modal-section + .method-modal-section {{
      margin-top: 0.85rem;
      padding-top: 0.85rem;
      border-top: 1px solid var(--line);
    }}
    .method-basis {{
      margin-top: 0.25rem;
      color: var(--muted);
      font-size: 0.83rem;
      line-height: 1.45;
    }}
    .method-detail-button,
    .modal-close-button {{
      appearance: none;
      border: 1px solid var(--line);
      background: transparent;
      color: var(--accent);
      border-radius: 0.35rem;
      padding: 0.45rem 0.75rem;
      font: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      margin: 0;
    }}
    .method-detail-button:hover,
    .modal-close-button:hover {{
      background: var(--accent-soft);
    }}
    .method-modal {{
      border: 0;
      padding: 0;
      background: transparent;
      max-width: min(920px, calc(100vw - 2rem));
      width: 100%;
    }}
    .method-modal::backdrop {{
      background: rgba(33,37,41,0.45);
    }}
    .method-modal-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 0.9rem;
      box-shadow: var(--shadow);
      padding: 1rem 1.1rem;
    }}
    .method-modal-section + .method-modal-section {{
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--line);
    }}
    .metric-detail {{
      margin-top: 0.85rem;
      border-top: 1px solid var(--line);
      padding-top: 0.7rem;
    }}
    .metric-detail-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      cursor: pointer;
      color: var(--accent);
      font-size: 0.9rem;
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
      padding: 0.85rem 0.9rem;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: #f8f8f4;
    }}
    .lead {{ color: var(--muted); line-height: 1.6; margin: 0; }}
    .scope-note {{
      margin: 0.6rem 0 0;
      padding: 0.7rem 0.85rem;
      border-left: 2px solid var(--line);
      background: #f8f8f4;
      color: #495057;
      font-size: 0.93rem;
      line-height: 1.55;
    }}
    .meta-inline {{
      margin: 0.45rem 0 0;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .summary-panel {{
      border-color: rgba(13,110,253,0.18);
      background: var(--paper);
    }}
    .tabs {{
      display: flex;
      gap: 0.6rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }}
    .tab-button {{
      appearance: none;
      border: 0;
      border-bottom: 1px solid transparent;
      background: transparent;
      color: #495057;
      border-radius: 0;
      padding: 0.4rem 0 0.45rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      margin: 0 1rem 0 0;
    }}
    .tab-button:hover {{
      border-color: rgba(63,90,73,0.35);
      background: transparent;
    }}
    .tab-button.is-active {{
      background: transparent;
      color: var(--accent);
      border-color: var(--accent);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.is-active {{
      display: block;
    }}
    .summary-header {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.5rem;
    }}
    .summary-kicker {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.78rem;
      font-weight: 700;
      margin-bottom: 0.35rem;
    }}
    .summary-intro {{
      margin: 0 0 0.85rem;
      color: var(--muted);
      max-width: 72ch;
      line-height: 1.5;
      font-size: 0.94rem;
    }}
    .summary-body {{
      border: 1px solid var(--line);
      padding: 0.9rem 1rem;
      background: #f8f8f4;
      border-radius: 0.35rem;
      line-height: 1.7;
      font-size: 0.96rem;
    }}
    .reference-panel {{
      border-color: rgba(70,103,86,0.14);
    }}
    .table-toolbar {{
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      margin: 0 0 0.85rem;
    }}
    .table-search-label {{
      margin: 0;
      font-size: 0.88rem;
      font-weight: 600;
      color: #495057;
    }}
    .table-search-input {{
      max-width: 420px;
      padding: 0.7rem 0.9rem;
    }}
    .models-chart-panel {{
      margin-bottom: 1rem;
      padding: 1rem;
      border: 1px solid var(--line);
      border-radius: 0.4rem;
      background: var(--paper);
    }}
    .models-chart-toolbar {{
      display: flex;
      gap: 0.8rem;
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
      color: #495057;
    }}
    .models-chart-field select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      padding: 0.7rem 0.8rem;
      font: inherit;
      background: #fdfdf9;
    }}
    .chart-tabbar {{
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }}
    .chart-tab-button {{
      margin-top: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      padding: 0.45rem 0.8rem;
      font-size: 0.88rem;
      font-weight: 600;
    }}
    .chart-tab-button:hover {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(63,90,73,0.24);
    }}
    .chart-tab-button.is-active {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: rgba(63,90,73,0.24);
    }}
    .models-impact-chart {{
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: #f8f8f4;
      padding: 0.9rem;
      min-height: 480px;
    }}
    .models-benchmark-note {{
      margin-top: 0.85rem;
      font-size: 0.88rem;
    }}
    .models-impact-chart svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .reference-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: var(--paper);
    }}
    .reference-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    .reference-table th,
    .reference-table td {{
      padding: 0.75rem 0.85rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .reference-table th {{
      background: #f0f2ee;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #495057;
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
      background: #f3ecd7;
    }}
    .reference-locator {{
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.83rem;
      line-height: 1.45;
    }}
    .source-tag {{
      display: inline-block;
      margin-left: 0.15rem;
      padding: 0.05rem 0.35rem;
      border-radius: 0.25rem;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      text-decoration: none;
      vertical-align: baseline;
    }}
    .source-tag:hover {{
      background: rgba(70,103,86,0.18);
      text-decoration: none;
    }}
    .math-panel {{
      border-color: rgba(108,117,125,0.18);
      background: #fff;
    }}
    .assumptions-box {{
      margin-bottom: 0.85rem;
      padding: 0.9rem;
      border: 1px solid var(--line);
      border-radius: 0.35rem;
      background: #f8f8f4;
    }}
    .assumptions-box-compact {{
      margin-top: 0.9rem;
      margin-bottom: 0;
    }}
    .assumptions-list {{
      margin: 0;
      padding-left: 1.15rem;
      color: #495057;
      line-height: 1.6;
    }}
    .assumptions-list li + li {{
      margin-top: 0.35rem;
    }}
    .math-label {{
      display: block;
      margin-bottom: 0.5rem;
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #495057;
    }}
    .math-formula {{
      margin-bottom: 0.75rem;
      font-size: 1.05rem;
    }}
    .math-formula code,
    .metric-detail-body code,
    .math-panel code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #fff;
      border: 1px solid var(--line);
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
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.74rem;
      font-weight: 700;
      text-decoration: none;
    }}
    .inline-ref:hover {{
      background: rgba(70,103,86,0.18);
      text-decoration: none;
    }}
    .metric-detail-body p,
    .math-panel p {{
      margin: 0;
      color: #495057;
      line-height: 1.7;
    }}
    .extrapolation-list {{
      margin: 0 0 0.75rem;
      padding-left: 1.2rem;
      color: #495057;
      line-height: 1.6;
      font-size: 0.92rem;
    }}
    .extrapolation-list li + li {{
      margin-top: 0.35rem;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 900px) {{
      .submetrics {{ grid-template-columns: 1fr; }}
      .summary-header {{ flex-direction: column; }}
    }}
  </style>
</head>
  <body>
  <main class="wrap">
    <nav class="tabs" aria-label="Navigation principale">
      <button type="button" class="tab-button is-active" data-tab-target="home">Accueil</button>
      <button type="button" class="tab-button" data-tab-target="how">How it works</button>
      <button type="button" class="tab-button" data-tab-target="training">Apprentissage</button>
      <button type="button" class="tab-button" data-tab-target="models">Inférence</button>
    </nav>

    {home_tab}
    {how_it_works_tab}
    {training_tab}
    {models_tab}
  </main>
  <script>
    const estimateForm = document.getElementById('estimate-form');
    const submitButton = document.getElementById('submit-button');
    const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
    const tabPanels = Array.from(document.querySelectorAll('[data-tab-panel]'));
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
          const textA = cellA ? cellA.textContent.trim() : '';
          const textB = cellB ? cellB.textContent.trim() : '';
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
        return '<p class="lead">Aucune donnée disponible pour cette sélection.</p>';
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
        return '<p class="lead">Aucune valeur exploitable pour cette sélection.</p>';
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
      const titleMetric = metric === 'energy' ? 'Énergie' : metric === 'carbon' ? 'Carbone' : 'Eau';
      const titleFamily = family === 'prompt' ? 'prompt|requête' : 'page';
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">Comparaison des valeurs centrales estimées pour l'indicateur <strong>${{titleMetric}}</strong>, selon la famille <strong>${{titleFamily}}</strong>.</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="Graphique comparatif des modèles">${{bars}}</svg>
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
      if (metric === 'creation_lifecycle_water') {{
        if (value >= 1000) return `${{value.toFixed(0)}} kL`;
        if (value >= 100) return `${{value.toFixed(1)}} kL`;
        return `${{value.toFixed(2)}} kL`;
      }}
      if (value >= 1000) return `${{value.toFixed(0)}} tCO2e`;
      if (value >= 100) return `${{value.toFixed(1)}} tCO2e`;
      return `${{value.toFixed(2)}} tCO2e`;
    }};
    const buildTrainingChartMarkup = (rows, metric) => {{
      if (!rows.length) {{
        return '<p class="lead">Aucune donnée disponible pour cette sélection.</p>';
      }}
      const key = metric === 'direct_training_energy'
        ? 'direct_training_energy_wh'
        : metric === 'direct_training_carbon'
          ? 'direct_training_carbon_tco2e'
          : 'creation_lifecycle_water_kl';
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
        return '<p class="lead">Aucune valeur exploitable pour cette sélection.</p>';
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
        ? "Énergie d'entraînement"
        : metric === 'direct_training_carbon'
          ? "CO2e entraînement direct"
          : "Eau cycle de création";
      return `
        <div class="summary-intro" style="margin-bottom:0.75rem;">Comparaison des valeurs centrales extrapolées pour l'indicateur <strong>${{titleMetric}}</strong>.</div>
        <svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" role="img" aria-label="Graphique comparatif des impacts d'apprentissage">${{bars}}</svg>
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
    def _write_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
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
