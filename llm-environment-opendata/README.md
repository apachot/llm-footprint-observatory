# ImpactLLM

Open dataset, local API, MCP server, observatory, and OpenAI-backed parsing frontend for quantified environmental indicators extracted from the scientific and technical literature on LLMs.

## Scope

This project supports the first publication on LLMs and environment by turning the literature review into:

- a reusable open dataset;
- a local query API;
- a local MCP server for downstream agents and tools.

The dataset stores only extracted values with source attribution.

## Project Layout

- `data/records.csv`: canonical tabular dataset
- `data/records.json`: JSON export of the same records
- `schema/llm_environment_record.schema.json`: record schema
- `api/server.py`: local HTTP API
- `mcp/server.py`: stdio MCP server
- `web/server.py`: user-facing web frontend for natural-language application descriptions
- `core/openai_parser.py`: OpenAI-backed structured parser for application descriptions
- `scripts/export_json.py`: regenerate the JSON export from CSV
- `scripts/validate_dataset.py`: basic dataset validation
- `docs/publication_integration.md`: how to describe the dataset and infrastructure in the paper
- `docs/llm_externalities_estimator_method.md`: method for estimating prompt, feature, and software externalities
- `docs/publication_plan_llm_externalities_method.md`: draft publication plan for the estimation method
- `docs/deployment_vps.md`: production deployment notes for a Linux VPS
- `deploy/systemd/llm-environment-opendata-web.service`: example `systemd` unit
- `deploy/nginx/llm-environment-opendata.conf`: example `nginx` reverse-proxy config
- `requirements.txt`: minimal Python dependency list for the web parser integration

## Data Model

Each record captures one quantified value from one source and one source locator.

Core fields:

- `record_id`
- `study_key`
- `citation`
- `publication_year`
- `phase`
- `impact_category`
- `metric_name`
- `metric_value`
- `metric_unit`
- `model_or_scope`
- `geography`
- `time_scope`
- `system_boundary`
- `data_type`
- `source_locator`
- `source_url`

## Run the API

```bash
cd "llm-environment-opendata"
python3 api/server.py
```

Default address: `http://127.0.0.1:8000`

Available endpoints:

- `GET /health`
- `GET /records`
- `GET /records/<record_id>`
- `GET /sources`
- `GET /stats`
- `POST /estimate`
- `POST /estimate_feature`

Optional query parameters on `/records`:

- `phase`
- `impact_category`
- `study_key`
- `geography`

## Run the MCP Server

```bash
cd "llm-environment-opendata"
python3 mcp/server.py
```

Supported tools:

- `list_records`
- `get_record`
- `aggregate_by_phase`
- `list_sources`
- `estimate_externalities`
- `estimate_feature_externalities`

## Run the Web Frontend

```bash
cd "llm-environment-opendata"
python3 -m pip install -r requirements.txt
python3 web/server.py
```

Default address: `http://127.0.0.1:8080`

Optional environment variables:

- `LLM_WEB_HOST`: bind address for the HTTP server
- `LLM_WEB_PORT`: bind port for the HTTP server

The frontend lets a user:

- describe an application in natural language;
- infer a usage scenario through an OpenAI model;
- compute a feature-level environmental estimate;
- inspect the method, assumptions, software breakdown, and source-backed factors used;
- export a standalone HTML report.

## OpenAI Configuration

The web frontend now requires an OpenAI API key.

Supported `.env` locations:

- `llm-environment-opendata/.env`
- `llm-environment-opendata/web/.env`

Minimal configuration:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Deploy on a VPS

For a simple production setup with `systemd` and `nginx`, use [`docs/deployment_vps.md`](./docs/deployment_vps.md).

## Validation

The API and MCP server use the CSV dataset as the single source of truth. The JSON file is an export for easier reuse.

```bash
cd "llm-environment-opendata"
python3 scripts/validate_dataset.py
python3 scripts/export_json.py
```

## GitHub Repository

The dedicated GitHub repository for this project is:

- `https://github.com/apachot/ImpactLLM`

To clone it directly:

```bash
git clone https://github.com/apachot/ImpactLLM.git
```
