#!/usr/bin/env python3
import json
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
    apply_overrides(parsed_payload, form)
    records = load_records()
    result = estimate_feature_externalities(records, parsed_payload)
    rows = factor_details(records, result["selected_factors"])
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
        annual = result["annual_total"]
        overhead = result["software_overhead"]
        result_block = f"""
        <section class="panel result hero-card">
          <h2>Evaluation environnementale</h2>
          <p class="lead">EcoTrace LLM interprète le scénario d'usage, sélectionne les facteurs du corpus scientifique, puis calcule une estimation expliquée et traçable.</p>
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
          <div class="summary-body">{escape(summary_text or "")}</div>
        </section>

        <section class="panel table-panel">
          <h3>Bilan logiciel détaillé</h3>
          <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Poste</th>
                <th>Description</th>
                <th>Energie / usage</th>
                <th>Energie / an</th>
                <th>Carbone / an</th>
                <th>Eau / an</th>
              </tr>
            </thead>
            <tbody>
              {''.join(render_component_row(row) for row in overhead['components'])}
            </tbody>
          </table>
          </div>
        </section>

        <section class="panel table-panel">
          <h3>Sources mobilisées</h3>
          <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Facteur</th>
                <th>Valeur</th>
                <th>Source</th>
                <th>Localisation</th>
              </tr>
            </thead>
            <tbody>
              {''.join(render_factor_row(row) for row in (factor_rows or []))}
            </tbody>
          </table>
          </div>
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
      --success-soft: #f1f8ff;
      --error: #dc3545;
      --error-soft: #f8d7da;
      --shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .wrap {{ max-width: 1140px; margin: 0 auto; padding: 32px 16px 56px; }}
    .hero {{ margin-bottom: 24px; }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 12px;
      padding: 0.35rem 0.65rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.15; font-weight: 700; }}
    h2, h3 {{ margin: 0 0 0.75rem; font-weight: 700; }}
    .subtitle {{ max-width: 820px; color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin: 0; }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 0.75rem;
      padding: 1.25rem;
      box-shadow: var(--shadow);
      margin-bottom: 1.25rem;
    }}
    .hero-card {{
      border-color: rgba(13,110,253,0.18);
      background: linear-gradient(180deg, #ffffff, #fbfdff);
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
      gap: 1rem;
      margin-top: 1rem;
    }}
    .metric {{
      padding: 1rem;
      border-radius: 0.75rem;
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
      font-size: 1.25rem;
      line-height: 1.35;
    }}
    .lead {{ color: var(--muted); line-height: 1.6; margin: 0; }}
    .summary-panel {{
      border-color: rgba(13,110,253,0.18);
      background: var(--paper);
    }}
    .summary-header {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.75rem;
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
      margin: 0 0 1rem;
      color: var(--muted);
      max-width: 76ch;
      line-height: 1.5;
    }}
    .summary-body {{
      border: 1px solid var(--line);
      padding: 1rem 1.1rem;
      background: #f8fbff;
      border-radius: 0.75rem;
      white-space: pre-line;
      line-height: 1.75;
      font-size: 0.99rem;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.88rem;
    }}
    .table-panel h3 {{ margin-bottom: 1rem; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 0.75rem;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.95rem;
      background: #fff;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 0.9rem 0.85rem;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f8f9fa;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #495057;
    }}
    tbody tr:nth-child(even) td {{ background: #fcfdff; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
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
      <p class="subtitle">Décris ton application en langage naturel. EcoTrace LLM s'appuie sur l'état de l'art scientifique pour mobiliser des indicateurs environnementaux sourcés, construire un bilan logiciel par postes techniques, puis retourner une estimation expliquée avec sa méthode et ses références.</p>
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


def render_factor_row(row):
    return (
        "<tr>"
        f"<td>{escape(row['metric_name'])}</td>"
        f"<td>{escape(row['metric_value'])} {escape(row['metric_unit'])}</td>"
        f"<td><a href=\"{escape(row['source_url'])}\">{escape(row['citation'])}</a></td>"
        f"<td>{escape(row['source_locator'])}</td>"
        "</tr>"
    )


def render_component_row(row):
    return (
        "<tr>"
        f"<td>{escape(row['component_type'])}</td>"
        f"<td>{escape(row['description'])}</td>"
        f"<td>{format_value_display(row['energy_wh_per_feature'], 'energy')}</td>"
        f"<td>{format_value_display(row['annual_energy_wh'], 'energy')}</td>"
        f"<td>{format_value_display(0.0 if row['annual_carbon_gco2e'] is None else row['annual_carbon_gco2e'], 'carbon')}</td>"
        f"<td>{format_value_display(0.0 if row['annual_water_ml'] is None else row['annual_water_ml'], 'water')}</td>"
        "</tr>"
    )


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
