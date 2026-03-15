#!/usr/bin/env python3
"""Recompute screening columns for the market model catalog."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.estimator import (
    MARKET_REFERENCE_REQUESTS_PER_HOUR,
    build_market_model_predictions,
    build_training_market_predictions,
    load_records,
)

MARKET_MODELS_PATH = ROOT / "data" / "market_models.csv"


def format_float(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def update_inference_columns(row, prediction):
    if not prediction:
        return

    per_request_energy = prediction.get("per_request_energy_wh") or {}
    per_request_carbon = prediction.get("per_request_carbon_gco2e") or {}
    requests_per_hour = MARKET_REFERENCE_REQUESTS_PER_HOUR

    for scenario in ("low", "central", "high"):
        energy_value = per_request_energy.get(scenario)
        carbon_value = per_request_carbon.get(scenario)
        row[f"screening_per_request_energy_wh_{scenario}"] = format_float(energy_value)
        row[f"screening_per_request_carbon_gco2e_{scenario}"] = format_float(carbon_value)
        row[f"screening_per_hour_energy_wh_{scenario}"] = format_float(
            (energy_value or 0.0) * requests_per_hour
        )
        row[f"screening_per_hour_carbon_gco2e_{scenario}"] = format_float(
            (carbon_value or 0.0) * requests_per_hour
        )

    method = prediction.get("method_results_by_id", {}).get("market_multifactor_prompt_proxy_v1") or {}
    detail = method.get("detail", {})
    eff_params = detail.get("effective_active_parameters_billion", {})
    for scenario in ("low", "central", "high"):
        row[f"screening_effective_active_parameters_billion_{scenario}"] = format_float(eff_params.get(scenario))

    if detail.get("reference_anchor"):
        row["screening_reference_anchor"] = detail["reference_anchor"]
    if method.get("basis"):
        row["screening_notes"] = method["basis"]


def update_training_factors(row, prediction):
    if not prediction:
        return

    factors = prediction.get("training_proxy_profile", {}).get("factors", {})
    for scenario in ("low", "central", "high"):
        profile = factors.get(scenario, {})
        row[f"training_f_params_{scenario}"] = format_float(profile.get("parameter_exponent"))
        row[f"training_f_tokens_{scenario}"] = format_float(profile.get("token_exponent"))
        row[f"training_f_regime_{scenario}"] = format_float(profile.get("regime_factor"))
        row[f"training_f_arch_{scenario}"] = format_float(profile.get("architecture_factor"))
        row[f"training_f_hardware_{scenario}"] = format_float(profile.get("hardware_factor"))


def main():
    with MARKET_MODELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise SystemExit("market_models.csv has no header")
        rows = [dict(row) for row in reader]

    inference_predictions = build_market_model_predictions(rows)
    training_predictions = build_training_market_predictions(load_records())
    inference_by_id = {pred["model_id"]: pred for pred in inference_predictions}
    training_by_id = {pred["model_id"]: pred for pred in training_predictions}

    for row in rows:
        model_id = row.get("model_id")
        update_inference_columns(row, inference_by_id.get(model_id))
        update_training_factors(row, training_by_id.get(model_id))

    with MARKET_MODELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: (row.get(col) or "") for col in header})

    print("Screening columns regenerated from current market-model predictions.")


if __name__ == "__main__":
    main()
