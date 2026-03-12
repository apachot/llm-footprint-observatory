#!/usr/bin/env python3
import json
import re
import sys
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


def normalize_model_label(value):
    if not value:
        return ""
    lowered = str(value).lower()
    for char in (" ", "-", "_", ".", ",", ":", ";", "/", "(", ")"):
        lowered = lowered.replace(char, "")
    return lowered


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


def render_math_demo(result):
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

    return f"""
    <section class="panel math-panel">
      <div class="summary-header">
        <div>
          <div class="summary-kicker">Demonstration</div>
          <h3>Detail du calcul</h3>
        </div>
        <div class="summary-badge">Mode mathematique</div>
      </div>
      <p class="summary-intro">Le calcul suit trois etapes: volume d'usage annuel, impact d'une requete LLM, puis annualisation a l'echelle de la fonctionnalite.</p>
      <div class="assumptions-box">
        <span class="math-label">Hypotheses retenues</span>
        <ul class="assumptions-list">
          {''.join(f'<li>{escape(item)}</li>' for item in assumptions)}
        </ul>
      </div>
      <div class="math-steps">
        <div class="math-step">
          <span class="math-step-index">1</span>
          <div>
            <strong>Volume annuel de la fonctionnalite</strong>
            <p>
              <code>{int(monthly_uses):,}</code> usages/mois × <code>{int(months_per_year)}</code> mois
              = <code>{int(annual_uses):,}</code> usages/an
            </p>
          </div>
        </div>
        <div class="math-step">
          <span class="math-step-index">2</span>
          <div>
            <strong>Nombre annuel d'appels LLM</strong>
            <p>
              <code>{int(annual_uses):,}</code> usages/an × <code>{requests_per_use:g}</code> appel(s) LLM/usage
              = <code>{int(annual_requests):,}</code> appels LLM/an
            </p>
          </div>
        </div>
      </div>
      <div class="math-grid">
        <div class="math-card">
          <span class="math-label">Energie annuelle totale</span>
          <div class="math-formula">
            <code>{format_range_display(annual_llm['energy_wh'], 'energy')}</code>
          </div>
          <p class="math-detail">
            Par requete LLM: <code>{format_range_display(per_request['energy_wh'], 'energy')}</code>
          </p>
          <p class="math-detail">
            Par usage de la fonctionnalite: <code>{format_range_display(per_feature['energy_wh'], 'energy')}</code>
          </p>
          <p class="math-detail">
            Impact annuel LLM: <code>{format_range_display(annual_llm['energy_wh'], 'energy')}</code>
          </p>
          <p class="math-total">
            Total = impact annuel du LLM = <code>{format_range_display(annual_llm['energy_wh'], 'energy')}</code>
          </p>
        </div>
        <div class="math-card">
          <span class="math-label">Carbone annuel total</span>
          <div class="math-formula">
            <code>{format_range_display(annual_llm['carbon_gco2e'], 'carbon')}</code>
          </div>
          <p class="math-detail">
            Par requete LLM: <code>{format_range_display(per_request['carbon_gco2e'], 'carbon')}</code>
          </p>
          <p class="math-detail">
            Par usage de la fonctionnalite: <code>{format_range_display(per_feature['carbon_gco2e'], 'carbon')}</code>
          </p>
          <p class="math-detail">
            Impact annuel LLM: <code>{format_range_display(annual_llm['carbon_gco2e'], 'carbon')}</code>
          </p>
          <p class="math-total">
            Total = impact annuel du LLM = <code>{format_range_display(annual_llm['carbon_gco2e'], 'carbon')}</code>
          </p>
        </div>
        <div class="math-card">
          <span class="math-label">Eau annuelle totale</span>
          <div class="math-formula">
            <code>{format_range_display(annual_llm['water_ml'], 'water')}</code>
          </div>
          <p class="math-detail">
            Par requete LLM: <code>{format_range_display(per_request['water_ml'], 'water')}</code>
          </p>
          <p class="math-detail">
            Par usage de la fonctionnalite: <code>{format_range_display(per_feature['water_ml'], 'water')}</code>
          </p>
          <p class="math-detail">
            Impact annuel LLM: <code>{format_range_display(annual_llm['water_ml'], 'water')}</code>
          </p>
          <p class="math-total">
            Total = impact annuel du LLM = <code>{format_range_display(annual_llm['water_ml'], 'water')}</code>
          </p>
        </div>
      </div>
    </section>
    """


def render_summary_html(summary_text, factor_rows):
    text = escape(summary_text or "")
    source_map = {}
    for index, row in enumerate(factor_rows or [], start=1):
        source_map[f"SRC{index}"] = row

    def replace_source_tag(match):
        tag = match.group(1)
        row = source_map.get(tag)
        if not row:
            return f"[{escape(tag)}]"
        title = escape(
            f"{row.get('citation', '')} | {row.get('metric_name', '')} | {row.get('source_locator', '')}"
        )
        href = escape(row.get("source_url", "#"), quote=True)
        return (
            f'<a class="source-tag" href="{href}" target="_blank" rel="noopener noreferrer" '
            f'title="{title}">[{escape(tag)}]</a>'
        )

    html = re.sub(r"\[(SRC\d+)\]", replace_source_tag, text)
    return html.replace("\n", "<br>")


def factor_details(records, factor_ids):
    rows = []
    for factor_id in factor_ids:
        record = get_record(records, factor_id)
        if not record:
            continue
        rows.append(
            {
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

        {render_math_demo(result)}

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
    .math-card p {{
      margin: 0;
      color: #495057;
      line-height: 1.7;
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
      <h1>Estimer l'impact environnemental d'une application utilisant des LLMs</h1>
      <p class="subtitle">Décris ton application en langage naturel pour obtenir un resultat, sa demonstration mathematique et une synthese sourcée.</p>
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
