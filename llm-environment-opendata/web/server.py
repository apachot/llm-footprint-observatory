#!/usr/bin/env python3
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

from core.estimator import estimate_feature_externalities, get_record, load_records
from core.openai_parser import (
    OpenAIModerationError,
    OpenAIParserError,
    OpenAISummaryError,
    generate_evaluation_summary,
    moderate_application_description_with_openai,
    parse_application_description_with_openai,
)


PROJECT_NAME = "EcoTrace LLM"
BIB_PATH = ROOT.parent / "llm-environment-opendata-paper" / "references_llm_environment_opendata.bib"


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


def html_id_attr(value):
    if not value:
        return ""
    return f' id="{escape(value, quote=True)}"'


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
        return f"{value:.1f}", "Wh"

    if unit_kind == "carbon":
        if abs_value >= 1000:
            return f"{value / 1000.0:.2f}", "kgCO2e"
        return f"{value:.1f}", "gCO2e"

    if unit_kind == "water":
        if abs_value >= 1000:
            return f"{value / 1000.0:.1f}", "L"
        return f"{value:.1f}", "mL"

    return f"{value:.1f}", ""


def format_range_display(range_obj, unit_kind):
    low_value, unit = format_scaled_value(range_obj["low"], unit_kind)
    high_value, _ = format_scaled_value(range_obj["high"], unit_kind)
    return f"{low_value} - {high_value} {unit}"


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
        "LLM request(s) per feature use": "appel(s) au LLM par usage de la fonctionnalite",
        "feature uses per year": "usages annuels de la fonctionnalite",
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
    refs = []
    for index, row in enumerate(rows, start=1):
        title = escape(format_apa_hover(row))
        href = f"#{reference_anchor_id(row)}" if reference_anchor_id(row) else "#"
        refs.append(
            f'<a class="inline-ref" href="{href}" title="{title}">[{index}]</a>'
        )
    return " ".join(refs)


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


def render_math_demo(result, factor_rows):
    annual_llm = result["annual_llm"]
    scope = result["feature_scope"]
    assumptions = result.get("assumptions", [])
    annual_uses = float(scope["annual_feature_uses"])
    annual_requests = float(scope["annual_llm_requests"])
    requests_per_use = float(scope["requests_per_feature"])
    monthly_uses = float(scope["feature_uses_per_month"])
    months_per_year = float(scope["months_per_year"])
    per_request = result["per_request_llm"]
    per_feature = result["per_feature_llm"]
    energy_rows = matching_factor_rows(factor_rows, ["energy", "wh"])
    carbon_rows = matching_factor_rows(factor_rows, ["carbon", "emission", "gco2"])
    water_rows = matching_factor_rows(factor_rows, ["water", "ml", "liter", "litre"])

    return f"""
    <section class="panel math-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Demonstration</div>
          <h3>Detail du calcul</h3>
        </div>
        <div class="summary-badge">Lecture pas a pas</div>
      </div>
      <p class="summary-intro">Le calcul part d'un impact unitaire par requete, puis le projette sur ton volume d'usage annuel. Le bloc ci-dessous montre cette logique et les hypotheses retenues pour l'extrapolation.</p>
      <div class="assumptions-box">
        <span class="math-label">Hypotheses retenues</span>
        <ul class="assumptions-list">
          {''.join(f'<li>{escape(humanize_assumption(item))}</li>' for item in assumptions)}
        </ul>
      </div>
      <div class="math-steps">
        <div class="math-step">
          <span class="math-step-index">1</span>
          <div>
            <strong>Calculer le volume annuel d'usage</strong>
            <p>
              On part de <code>{format_count(monthly_uses)}</code> usages par mois.
              En le multipliant par <code>{format_count(months_per_year)}</code> mois,
              on obtient <code>{format_count(annual_uses)}</code> usages par an.
            </p>
          </div>
        </div>
        <div class="math-step">
          <span class="math-step-index">2</span>
          <div>
            <strong>Calculer le nombre annuel d'appels au LLM</strong>
            <p>
              Chaque usage mobilise <code>{requests_per_use:g}</code> appel(s) au LLM.
              Donc <code>{format_count(annual_uses)}</code> usages par an
              correspondent a <code>{format_count(annual_requests)}</code> appels LLM par an.
            </p>
          </div>
        </div>
        <div class="math-step">
          <span class="math-step-index">3</span>
          <div>
            <strong>Projeter l'impact unitaire a l'annee</strong>
            <p>
              On applique les facteurs estimes par requete au volume annuel d'appels LLM
              pour obtenir une fourchette annuelle d'energie, de carbone et d'eau.
            </p>
          </div>
        </div>
      </div>
      <div class="math-grid">
        <div class="math-card">
          <span class="math-label">Energie annuelle totale</span>
          <div class="math-formula">
            {render_sourced_value(format_range_display(annual_llm['energy_wh'], 'energy'), energy_rows)}
          </div>
          {render_extrapolation_details(result, "energy", energy_rows)}
          <p class="math-detail">
            Impact estime pour une requete LLM:
            {render_sourced_value(format_range_display(per_request['energy_wh'], 'energy'), energy_rows)}
          </p>
          <p class="math-detail">
            Impact estime pour un usage de la fonctionnalite:
            {render_sourced_value(format_range_display(per_feature['energy_wh'], 'energy'), energy_rows)}
          </p>
          <p class="math-detail">
            En projetant cette valeur sur
            <code>{format_count(annual_requests)}</code> appels par an,
            on obtient:
            {render_sourced_value(format_range_display(annual_llm['energy_wh'], 'energy'), energy_rows)}
          </p>
          <p class="math-total">
            Resultat retenu pour l'energie annuelle du LLM:
            {render_sourced_value(format_range_display(annual_llm['energy_wh'], 'energy'), energy_rows)}
          </p>
        </div>
        <div class="math-card">
          <span class="math-label">Carbone annuel total</span>
          <div class="math-formula">
            {render_sourced_value(format_range_display(annual_llm['carbon_gco2e'], 'carbon'), carbon_rows)}
          </div>
          {render_extrapolation_details(result, "carbon", carbon_rows)}
          <p class="math-detail">
            Impact estime pour une requete LLM:
            {render_sourced_value(format_range_display(per_request['carbon_gco2e'], 'carbon'), carbon_rows)}
          </p>
          <p class="math-detail">
            Impact estime pour un usage de la fonctionnalite:
            {render_sourced_value(format_range_display(per_feature['carbon_gco2e'], 'carbon'), carbon_rows)}
          </p>
          <p class="math-detail">
            En projetant cette valeur sur
            <code>{format_count(annual_requests)}</code> appels par an,
            on obtient:
            {render_sourced_value(format_range_display(annual_llm['carbon_gco2e'], 'carbon'), carbon_rows)}
          </p>
          <p class="math-total">
            Resultat retenu pour le carbone annuel du LLM:
            {render_sourced_value(format_range_display(annual_llm['carbon_gco2e'], 'carbon'), carbon_rows)}
          </p>
        </div>
        <div class="math-card">
          <span class="math-label">Eau annuelle totale</span>
          <div class="math-formula">
            {render_sourced_value(format_range_display(annual_llm['water_ml'], 'water'), water_rows)}
          </div>
          {render_extrapolation_details(result, "water", water_rows)}
          <p class="math-detail">
            Impact estime pour une requete LLM:
            {render_sourced_value(format_range_display(per_request['water_ml'], 'water'), water_rows)}
          </p>
          <p class="math-detail">
            Impact estime pour un usage de la fonctionnalite:
            {render_sourced_value(format_range_display(per_feature['water_ml'], 'water'), water_rows)}
          </p>
          <p class="math-detail">
            En projetant cette valeur sur
            <code>{format_count(annual_requests)}</code> appels par an,
            on obtient:
            {render_sourced_value(format_range_display(annual_llm['water_ml'], 'water'), water_rows)}
          </p>
          <p class="math-total">
            Resultat retenu pour l'eau annuelle du LLM:
            {render_sourced_value(format_range_display(annual_llm['water_ml'], 'water'), water_rows)}
          </p>
        </div>
      </div>
    </section>
    """


def render_summary_html(summary_text, factor_rows):
    text = escape(summary_text or "")
    source_map = {}
    for index, row in enumerate(factor_rows or [], start=1):
        source_map[str(index)] = row
        source_map[f"SRC{index}"] = row

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


def build_literature_catalog_rows():
    rows = []
    for record in load_records():
        source_url = str(record.get("source_url", ""))
        if "iea.org" in source_url or "publicpower.org" in source_url:
            continue
        rows.append(
            {
                "record_id": record.get("record_id", ""),
                "study_key": record.get("study_key", ""),
                "data_type": f"{record.get('phase', '')} / {record.get('metric_name', '')}",
                "metric_value": f"{record.get('metric_value', '')} {record.get('metric_unit', '')}".strip(),
                "citation": record.get("citation", ""),
                "source_locator": record.get("source_locator", ""),
                "source_url": record.get("source_url", "#"),
            }
        )
    return rows


def render_reference_catalog():
    rows = build_literature_catalog_rows()
    if not rows:
        return ""
    rendered_rows = []
    for row in rows:
        locator_html = ""
        if row["source_locator"]:
            locator_html = f'<div class="reference-locator">{escape(row["source_locator"])}</div>'
        row_id = reference_anchor_id(row)
        rendered_rows.append(
            f"<tr{html_id_attr(row_id)}>"
            f"<td>{escape(row['data_type'])}</td>"
            f"<td>{escape(row['metric_value'])}</td>"
            f"<td>"
            f"<a href=\"{escape(row['source_url'], quote=True)}\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"{escape(format_apa_hover(row))}\">{escape(format_apa_citation(row))}</a>"
            f"{locator_html}"
            f"</td>"
            f"</tr>"
        )
    table_rows = "".join(rendered_rows)

    return f"""
    <section class="panel reference-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Referentiel</div>
          <h3>44 indicateurs issus de la litterature scientifique</h3>
        </div>
        <div class="summary-badge">Tableau de reference</div>
      </div>
      <p class="summary-intro">Ce tableau presente les indicateurs quantifies extraits du corpus scientifique mobilise par le projet. Les sources institutionnelles hors litterature scientifique directe ont ete exclues de ce referentiel.</p>
      <div class="reference-table-wrap">
        <table class="reference-table">
          <thead>
            <tr>
              <th>Type de donnée</th>
              <th>Valeur</th>
              <th>Citation</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
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
    summary = generate_evaluation_summary(description, parsed_payload, result, rows, parser_meta)
    return description, parsed_payload, parser_notes, parser_meta, result, rows, summary


def render_page(result=None, description="", parsed_payload=None, parser_notes=None, parser_meta=None, factor_rows=None, summary_text=None, error_message=None):
    error_block = ""
    if error_message:
        error_block = f"""
        <section class="panel alert-panel error">
          <h2>Description non exploitable</h2>
          <p class="lead">{escape(error_message)}</p>
          <p class="lead">EcoTrace LLM attend une description d'application, de fonctionnalité ou de workflow mobilisant un ou plusieurs LLMs.</p>
        </section>
        """

    reference_block = render_reference_catalog()
    result_block = ""
    if result:
        annual = result["annual_llm"]
        evidence = (parser_meta or {}).get("evidence", {})
        method_label = {
            "parametric_extrapolation": "Extrapolation parametrique",
            "literature_proxy": "Proxy de litterature",
        }.get(result.get("method"), "Methode non qualifiee")
        model_profile = result.get("model_profile") or {}
        country_mix = result.get("country_energy_mix") or {}
        result_block = f"""
        <section class="panel result hero-card">
          <h2>Evaluation environnementale</h2>
          <p class="lead">Estimation fondee sur des indicateurs scientifiques sourcés et un calcul traceable.</p>
          <p class="scope-note">Perimetre retenu: seules les consommations du LLM sont prises en compte. Les autres consommations du systeme logiciel, de l'infrastructure applicative et des services annexes sont exclues de cette estimation.</p>
          <p class="meta-inline">Niveau de preuve: <strong>{escape(evidence.get('label', 'Non qualifie'))}</strong></p>
          <p class="meta-inline">Methode: <strong>{escape(method_label)}</strong></p>
          <p class="meta-inline">Modele de reference: <strong>{escape(model_profile.get('model_id', parsed_payload.get('model_id', 'non specifie')))}</strong>{' | Parametres actifs approx.: <strong>' + escape(model_profile.get('active_parameters_billion', '')) + 'B</strong>' if model_profile.get('active_parameters_billion') else ''}</p>
          <p class="meta-inline">Mix electrique: <strong>{escape(country_mix.get('country_code', parsed_payload.get('country', 'non specifie')))}</strong>{' | ' + escape(country_mix.get('grid_carbon_intensity_gco2_per_kwh', '')) + ' gCO2e/kWh' if country_mix.get('grid_carbon_intensity_gco2_per_kwh') else ''}</p>
          <div class="metrics">
            <div class="metric"><span class="label">Energie annuelle totale</span><strong>{format_range_display(annual['energy_wh'], 'energy')}</strong></div>
            <div class="metric"><span class="label">Carbone annuel total</span><strong>{format_range_display(annual['carbon_gco2e'], 'carbon')}</strong></div>
            <div class="metric"><span class="label">Eau annuelle totale</span><strong>{format_range_display(annual['water_ml'], 'water')}</strong></div>
          </div>
        </section>

        <section class="panel summary-panel">
          <div class="summary-header">
            <div>
              <div class="summary-kicker">Analyse</div>
              <h3>Synthèse automatique</h3>
            </div>
            <div class="summary-badge">Sources scientifiques</div>
          </div>
          <p class="summary-intro">Cette synthèse reformule le scénario interprété, les principaux facteurs retenus dans la littérature et la logique de calcul appliquée à ton cas d'usage.</p>
          <div class="summary-body">{render_summary_html(summary_text, factor_rows)}</div>
        </section>

        {render_math_demo(result, factor_rows)}
        """

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PROJECT_NAME}</title>
  <style>
    :root {{
      --bg: #f8f9fa;
      --paper: #ffffff;
      --ink: #212529;
      --muted: #6c757d;
      --line: #dee2e6;
      --accent: #0d6efd;
      --accent-soft: #e7f1ff;
      --error: #dc3545;
      --shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.08);
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
    .hero {{ margin-bottom: 16px; }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 0.25rem 0.55rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.74rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(1.8rem, 4vw, 2.5rem); line-height: 1.15; font-weight: 700; }}
    h2, h3 {{ margin: 0 0 0.75rem; font-weight: 700; }}
    .subtitle {{ max-width: 760px; color: var(--muted); font-size: 0.98rem; line-height: 1.55; margin: 0; }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 0.75rem;
      padding: 1rem 1.1rem;
      box-shadow: var(--shadow);
      margin-bottom: 1rem;
    }}
    .hero-card {{
      border-color: rgba(13,110,253,0.18);
      background: #ffffff;
    }}
    .alert-panel.error {{
      border-color: rgba(220,53,69,0.3);
      background: #fff5f6;
    }}
    textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 0.5rem;
      padding: 0.875rem 1rem;
      font: inherit;
      background: #fff;
    }}
    textarea:focus, input:focus {{
      outline: none;
      border-color: rgba(13,110,253,0.55);
      box-shadow: 0 0 0 0.2rem rgba(13,110,253,0.15);
    }}
    textarea {{ min-height: 200px; resize: vertical; }}
    label {{ display: block; font-size: 0.95rem; font-weight: 600; color: var(--ink); margin-bottom: 0.5rem; }}
    button {{
      margin-top: 1rem;
      border: 0;
      border-radius: 0.5rem;
      background: var(--accent);
      color: white;
      padding: 0.75rem 1.15rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.65rem;
    }}
    button:hover {{ background: #0b5ed7; }}
    button:disabled {{
      background: #6ea8fe;
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
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
      margin-top: 0.9rem;
    }}
    .metric {{
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
    .lead {{ color: var(--muted); line-height: 1.6; margin: 0; }}
    .scope-note {{
      margin: 0.6rem 0 0;
      padding: 0.7rem 0.85rem;
      border-left: 3px solid var(--accent);
      background: #f8fbff;
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
    .summary-badge {{
      flex: 0 0 auto;
      border: 1px solid rgba(13,110,253,0.18);
      border-radius: 999px;
      padding: 0.4rem 0.75rem;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 600;
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
      background: #f8fbff;
      border-radius: 0.75rem;
      line-height: 1.7;
      font-size: 0.96rem;
    }}
    .reference-panel {{
      border-color: rgba(13,110,253,0.14);
    }}
    .reference-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 0.65rem;
      background: #fff;
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
      background: #f8f9fa;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #495057;
    }}
    .reference-table tbody tr:last-child td {{
      border-bottom: 0;
    }}
    .reference-table tbody tr:target td {{
      background: #fff8db;
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
      border-radius: 999px;
      background: rgba(13,110,253,0.1);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      text-decoration: none;
      vertical-align: baseline;
    }}
    .source-tag:hover {{
      background: rgba(13,110,253,0.18);
      text-decoration: none;
    }}
    .math-panel {{
      border-color: rgba(108,117,125,0.18);
      background: #fff;
    }}
    .math-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }}
    .assumptions-box {{
      margin-bottom: 0.85rem;
      padding: 0.9rem;
      border: 1px solid var(--line);
      border-radius: 0.65rem;
      background: #f8f9fa;
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
    .math-card {{
      border: 1px solid var(--line);
      border-radius: 0.65rem;
      padding: 0.9rem;
      background: #f8f9fa;
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
    .math-card code {{
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
      border-radius: 999px;
      background: rgba(13,110,253,0.1);
      color: var(--accent);
      font-size: 0.74rem;
      font-weight: 700;
      text-decoration: none;
    }}
    .inline-ref:hover {{
      background: rgba(13,110,253,0.18);
      text-decoration: none;
    }}
    .math-card p {{
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
      .metrics {{ grid-template-columns: 1fr; }}
      .summary-header {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <div class="eyebrow">Projet {PROJECT_NAME}</div>
      <h1>Estimer l’empreinte environnementale d’une application utilisant des LLMs</h1>
      <p class="subtitle">Décris ton application en langage naturel pour obtenir une estimation environnementale, son détail de calcul et une synthèse sourcée.</p>
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
    {error_block}
    {result_block}
    {reference_block}
  </main>
  <script>
    const estimateForm = document.getElementById('estimate-form');
    const submitButton = document.getElementById('submit-button');
    if (estimateForm && submitButton) {{
      estimateForm.addEventListener('submit', function () {{
        submitButton.disabled = true;
        submitButton.classList.add('is-loading');
      }});
    }}
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
            description, parsed_payload, parser_notes, parser_meta, result, rows, summary_text = process_description(form)
        except (OpenAIModerationError, OpenAIParserError, OpenAISummaryError) as exc:
            self._write_html(render_page(description=description, error_message=str(exc)), status=502)
            return

        self._write_html(
            render_page(
                result=result,
                description=description,
                parsed_payload=parsed_payload,
                parser_notes=parser_notes,
                parser_meta=parser_meta,
                factor_rows=rows,
                summary_text=summary_text,
            )
        )


def apply_overrides(payload, form):
    return payload


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8080), Handler)
    print(f"{PROJECT_NAME} web app running on http://127.0.0.1:8080")
    server.serve_forever()
