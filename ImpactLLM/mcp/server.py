#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.estimator import (
    build_market_model_predictions,
    compute_stats,
    estimate_externalities,
    estimate_feature_externalities,
    get_country_mix,
    get_model_profile,
    get_record,
    list_sources,
    load_country_energy_mix,
    load_extrapolation_rules,
    load_models,
    load_records,
    predict_inference_externalities,
)


def tool_definitions():
    return [
        {
            "name": "list_records",
            "description": "List extracted environmental records with optional filters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "impact_category": {"type": "string"},
                    "study_key": {"type": "string"},
                },
            },
        },
        {
            "name": "get_record",
            "description": "Get one record by record_id.",
            "inputSchema": {
                "type": "object",
                "required": ["record_id"],
                "properties": {
                    "record_id": {"type": "string"},
                },
            },
        },
        {
            "name": "aggregate_by_phase",
            "description": "Count records by analytical phase.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_sources",
            "description": "List distinct source studies represented in the dataset.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_models",
            "description": "List model profiles available for parameter-based extrapolation.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_market_models",
            "description": "List the maintained catalog of current market models with standardized inference estimates.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_model_profile",
            "description": "Get one model profile by model identifier.",
            "inputSchema": {
                "type": "object",
                "required": ["model_id"],
                "properties": {"model_id": {"type": "string"}},
            },
        },
        {
            "name": "list_country_energy_mix",
            "description": "List country-level default electricity carbon and water intensity factors.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_country_energy_mix",
            "description": "Get one country-level default electricity mix profile.",
            "inputSchema": {
                "type": "object",
                "required": ["country_code"],
                "properties": {"country_code": {"type": "string"}},
            },
        },
        {
            "name": "list_extrapolation_rules",
            "description": "List parameter-based extrapolation rules used by the estimator.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "predict_inference_externalities",
            "description": "Predict annualized inference-only environmental externalities using a unified Wh-based calibration from literature anchors, scaled by model size and country electricity mix.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "request_type": {"type": "string"},
                    "input_tokens": {"type": "number"},
                    "output_tokens": {"type": "number"},
                    "requests_per_feature": {"type": "number"},
                    "feature_uses_per_month": {"type": "number"},
                    "months_per_year": {"type": "number"},
                    "estimated_active_parameters_billion": {"type": "number"},
                    "country": {"type": "string"},
                    "grid_carbon_intensity_gco2_per_kwh": {"type": "number"},
                    "water_intensity_l_per_kwh": {"type": "number"},
                },
            },
        },
        {
            "name": "estimate_externalities",
            "description": "Estimate prompt- or scenario-level environmental externalities for an LLM use case.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "deployment_mode": {"type": "string"},
                    "request_type": {"type": "string"},
                    "input_tokens": {"type": "number"},
                    "output_tokens": {"type": "number"},
                    "requests_count": {"type": "number"},
                    "country": {"type": "string"},
                    "estimated_active_parameters_billion": {"type": "number"},
                    "grid_carbon_intensity_gco2_per_kwh": {"type": "number"},
                    "water_intensity_l_per_kwh": {"type": "number"},
                },
            },
        },
        {
            "name": "estimate_feature_externalities",
            "description": "Estimate annualized inference-only environmental externalities for software using an LLM, using the unified Wh-based calibration predictor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "model_id": {"type": "string"},
                    "deployment_mode": {"type": "string"},
                    "request_type": {"type": "string"},
                    "input_tokens": {"type": "number"},
                    "output_tokens": {"type": "number"},
                    "requests_per_feature": {"type": "number"},
                    "feature_uses_per_month": {"type": "number"},
                    "months_per_year": {"type": "number"},
                    "estimated_active_parameters_billion": {"type": "number"},
                    "software_overhead_wh_per_feature": {"type": "number"},
                    "country": {"type": "string"},
                    "grid_carbon_intensity_gco2_per_kwh": {"type": "number"},
                    "water_intensity_l_per_kwh": {"type": "number"},
                },
            },
        },
    ]


def make_text_payload(obj):
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False, indent=2)}]}


def handle_call(name, arguments):
    records = load_records()

    if name == "list_records":
        filtered = records
        for field in ("phase", "impact_category", "study_key"):
            value = arguments.get(field)
            if value:
                filtered = [record for record in filtered if record.get(field) == value]
        return make_text_payload({"records": filtered})

    if name == "get_record":
        record_id = arguments["record_id"]
        record = get_record(records, record_id)
        if record:
            return make_text_payload(record)
        return make_text_payload({"error": "record not found", "record_id": record_id})

    if name == "aggregate_by_phase":
        return make_text_payload(compute_stats(records))

    if name == "list_sources":
        return make_text_payload({"sources": list_sources(records)})

    if name == "list_models":
        return make_text_payload({"models": load_models()})

    if name == "list_market_models":
        return make_text_payload({"market_models": build_market_model_predictions(records)})

    if name == "get_model_profile":
        return make_text_payload({"model_profile": get_model_profile(model_id=arguments["model_id"])})

    if name == "list_country_energy_mix":
        return make_text_payload({"country_energy_mix": load_country_energy_mix()})

    if name == "get_country_energy_mix":
        return make_text_payload({"country_energy_mix": get_country_mix(arguments["country_code"])})

    if name == "list_extrapolation_rules":
        return make_text_payload({"extrapolation_rules": load_extrapolation_rules()})

    if name == "estimate_externalities":
        return make_text_payload(estimate_externalities(records, arguments))

    if name == "predict_inference_externalities":
        return make_text_payload(predict_inference_externalities(records, arguments))

    if name == "estimate_feature_externalities":
        return make_text_payload(estimate_feature_externalities(records, arguments))

    return make_text_payload({"error": "unknown tool", "tool": name})


def send(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "ImpactLLM", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    },
                }
            )
            continue

        if method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tool_definitions()}})
            continue

        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            send({"jsonrpc": "2.0", "id": request_id, "result": handle_call(tool_name, arguments)})
            continue

        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )


if __name__ == "__main__":
    main()
