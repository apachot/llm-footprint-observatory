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
            f"Description refusée par le filtre d'usage ({moderation['decision']}) via {moderation['model']}: "
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
        <section class="panel error">
          <h2>Erreur d'interprétation OpenAI</h2>
          <p class="lead">{escape(error_message)}</p>
          <p class="lead">EcoTrace LLM fonctionne désormais uniquement avec l'analyse structurée du modèle OpenAI configuré dans <code>.env</code>.</p>
        </section>
        """

    result_block = ""
    if result:
        annual = result["annual_total"]
        overhead = result["software_overhead"]
        result_block = f"""
        <section class="panel result">
          <h2>Evaluation environnementale</h2>
          <p class="lead">EcoTrace LLM confie l'interprétation du scénario à OpenAI, sélectionne les facteurs du corpus scientifique, puis calcule une estimation expliquée et traçable.</p>
          <div class="metrics">
            <div class="metric"><span class="label">Energie annuelle totale</span><strong>{annual['energy_wh']['low']:.1f} - {annual['energy_wh']['high']:.1f} Wh</strong></div>
            <div class="metric"><span class="label">Carbone annuel total</span><strong>{annual['carbon_gco2e']['low']:.1f} - {annual['carbon_gco2e']['high']:.1f} gCO2e</strong></div>
            <div class="metric"><span class="label">Eau annuelle totale</span><strong>{annual['water_ml']['low']:.1f} - {annual['water_ml']['high']:.1f} mL</strong></div>
          </div>
        </section>

        <section class="panel">
          <h3>Synthèse automatique</h3>
          <p class="lead">{escape(summary_text or "")}</p>
        </section>

        <section class="panel">
          <h3>Bilan logiciel détaillé</h3>
          <table>
            <thead>
              <tr>
                <th>Poste</th>
                <th>Description</th>
                <th>Wh / usage</th>
                <th>Wh / an</th>
                <th>gCO2e / an</th>
                <th>mL / an</th>
              </tr>
            </thead>
            <tbody>
              {''.join(render_component_row(row) for row in overhead['components'])}
            </tbody>
          </table>
        </section>

        <section class="panel">
          <h3>Sources mobilisées</h3>
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
      --bg: #f3efe7;
      --paper: #fffdf8;
      --ink: #17212b;
      --muted: #5c6773;
      --line: #d9d2c4;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --error: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 28%),
        radial-gradient(circle at bottom right, rgba(194,65,12,0.10), transparent 22%),
        var(--bg);
    }}
    .wrap {{ max-width: 1160px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{ margin-bottom: 24px; }}
    .eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.82rem; }}
    h1 {{ margin: 6px 0 10px; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.02; }}
    .subtitle {{ max-width: 780px; color: var(--muted); font-size: 1.05rem; line-height: 1.5; }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 14px 40px rgba(23,33,43,0.06);
    }}
    .error {{ border-color: rgba(185,28,28,0.35); background: #fff7f7; }}
    form.panel {{ margin-bottom: 24px; }}
    textarea, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      background: #fff;
    }}
    textarea {{ min-height: 200px; resize: vertical; }}
    .row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    label {{ display: block; font-size: 0.92rem; color: var(--muted); margin-bottom: 6px; }}
    button {{
      margin-top: 16px;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), #155e75);
      color: white;
      padding: 12px 20px;
      font: inherit;
      cursor: pointer;
    }}
    .ghost-button {{ background: linear-gradient(135deg, var(--accent-2), #9a3412); margin-top: 0; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .metric {{
      padding: 14px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(15,118,110,0.06), rgba(15,118,110,0.02));
      border: 1px solid rgba(15,118,110,0.12);
    }}
    .metric .label {{ display: block; color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }}
    .lead {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.88rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      vertical-align: top;
    }}
    a {{ color: var(--accent-2); }}
    @media (max-width: 900px) {{
      .row, .metrics, .grid {{ grid-template-columns: 1fr; }}
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

    <form class="panel" method="post" action="/">
      <label for="description">Description libre de l'application</label>
      <textarea id="description" name="description" placeholder="Exemple: Nous avons un assistant RAG sur GPT-4 via API, utilisé 4000 fois par mois en France. Chaque requête envoie 2200 input tokens et reçoit 500 output tokens. Il y a une base vectorielle, des embeddings et du logging.">{escape(description)}</textarea>
      <button type="submit">Evaluer l'application</button>
    </form>
    {error_block}
    {result_block}
  </main>
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
        f"<td>{row['energy_wh_per_feature']:.3f}</td>"
        f"<td>{row['annual_energy_wh']:.1f}</td>"
        f"<td>{0.0 if row['annual_carbon_gco2e'] is None else row['annual_carbon_gco2e']:.1f}</td>"
        f"<td>{0.0 if row['annual_water_ml'] is None else row['annual_water_ml']:.1f}</td>"
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
