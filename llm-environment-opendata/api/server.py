#!/usr/bin/env python3
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.estimator import (
    DATASET_PATH,
    COUNTRY_MIX_PATH,
    EXTRAPOLATION_RULES_PATH,
    MODELS_PATH,
    compute_stats,
    estimate_externalities,
    estimate_feature_externalities,
    filter_records,
    get_record,
    get_country_mix,
    get_model_profile,
    list_sources,
    load_country_energy_mix,
    load_extrapolation_rules,
    load_models,
    load_records,
)


class Handler(BaseHTTPRequestHandler):
    def _write_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        records = load_records()

        if path == "/health":
            self._write_json(
                {
                    "status": "ok",
                    "dataset": str(DATASET_PATH),
                    "models": str(MODELS_PATH),
                    "country_mix": str(COUNTRY_MIX_PATH),
                    "extrapolation_rules": str(EXTRAPOLATION_RULES_PATH),
                }
            )
            return

        if path == "/records":
            params = parse_qs(parsed.query)
            self._write_json({"records": filter_records(records, params)})
            return

        if path.startswith("/records/"):
            record_id = path.split("/", 2)[2]
            record = get_record(records, record_id)
            if record:
                self._write_json(record)
                return
            self._write_json({"error": "record not found", "record_id": record_id}, status=404)
            return

        if path == "/sources":
            self._write_json({"sources": list_sources(records)})
            return

        if path == "/models":
            self._write_json({"models": load_models()})
            return

        if path.startswith("/models/"):
            model_id = path.split("/", 2)[2]
            profile = get_model_profile(model_id=model_id)
            if profile:
                self._write_json(profile)
                return
            self._write_json({"error": "model not found", "model_id": model_id}, status=404)
            return

        if path == "/energy-mix":
            self._write_json({"country_energy_mix": load_country_energy_mix()})
            return

        if path.startswith("/energy-mix/"):
            country_code = path.split("/", 2)[2]
            country_mix = get_country_mix(country_code)
            if country_mix:
                self._write_json(country_mix)
                return
            self._write_json({"error": "country mix not found", "country_code": country_code}, status=404)
            return

        if path == "/extrapolation-rules":
            self._write_json({"extrapolation_rules": load_extrapolation_rules()})
            return

        if path == "/stats":
            self._write_json(compute_stats(records))
            return

        self._write_json(
            {
                "error": "not found",
                "available_endpoints": [
                    "/health",
                    "/records",
                    "/records/<record_id>",
                    "/sources",
                    "/models",
                    "/models/<model_id>",
                    "/energy-mix",
                    "/energy-mix/<country_code>",
                    "/extrapolation-rules",
                    "/stats",
                    "POST /estimate",
                    "POST /estimate_feature",
                ],
            },
            status=404,
        )

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/estimate", "/estimate_feature"):
            self._write_json(
                {"error": "not found", "available_endpoints": ["POST /estimate", "POST /estimate_feature"]},
                status=404,
            )
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._write_json({"error": "empty body"}, status=400)
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json({"error": "invalid json body"}, status=400)
            return

        records = load_records()
        if parsed.path == "/estimate":
            estimate = estimate_externalities(records, payload)
        else:
            estimate = estimate_feature_externalities(records, payload)
        self._write_json(estimate)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving LLM environment dataset API on http://127.0.0.1:8000")
    server.serve_forever()
