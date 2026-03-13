#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "records.csv"
METADATA_PATH = ROOT / "data" / "record_metadata.csv"
MODELS_PATH = ROOT / "data" / "models.csv"
COUNTRY_MIX_PATH = ROOT / "data" / "country_energy_mix.csv"
EXTRAPOLATION_RULES_PATH = ROOT / "data" / "extrapolation_rules.csv"
MARKET_MODELS_PATH = ROOT / "data" / "market_models.csv"

REFERENCE_PROMPT_TOKENS = 1550.0
REFERENCE_PAGE_TOKENS = 750.0
MARKET_REFERENCE_REQUESTS_PER_YEAR = 1_000_000.0


def load_records():
    metadata = load_record_metadata()
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [normalize_record(record, metadata.get(record["record_id"], {})) for record in csv.DictReader(handle)]

def load_record_metadata():
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def load_models():
    with MODELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def load_country_energy_mix():
    with COUNTRY_MIX_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def load_market_models():
    with MARKET_MODELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def load_extrapolation_rules():
    with EXTRAPOLATION_RULES_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def normalize_record(record, metadata):
    normalized = dict(record)
    normalized["unit_of_analysis"] = metadata.get("unit_of_analysis") or infer_unit_of_analysis(record)
    normalized["source_type"] = metadata.get("source_type") or infer_source_type(record)
    normalized["uncertainty_level"] = metadata.get("uncertainty_level") or infer_uncertainty_level(record)
    normalized["raw_or_derived"] = metadata.get("raw_or_derived") or infer_raw_or_derived(record)
    normalized["applicability_domain"] = metadata.get("applicability_domain") or infer_applicability_domain(record)
    return normalized


def infer_unit_of_analysis(record):
    metric_unit = record.get("metric_unit", "").lower()
    if "/prompt" in metric_unit:
        return "prompt"
    if "/query" in metric_unit:
        return "query"
    if "/page" in metric_unit:
        return "page"
    if "/year" in metric_unit:
        return "year"
    if "gpu-hours" in metric_unit:
        return "gpu-hour"
    if record.get("phase") == "training":
        return "training_run"
    if record.get("phase") == "infrastructure":
        return "infrastructure_scope"
    return "record"


def infer_source_type(record):
    url = record.get("source_url", "")
    citation = record.get("citation", "").lower()
    notes = record.get("notes", "").lower()
    if "huggingface.co" in url or "model card" in citation or "model card" in notes:
        return "model_card"
    if "arxiv.org" in url:
        return "preprint"
    if "iea.org" in url or "lbl.gov" in url or "publicpower.org" in url:
        return "institutional_or_report"
    return "article"


def infer_uncertainty_level(record):
    data_type = record.get("data_type", "")
    source_type = infer_source_type(record)
    if data_type == "measured" and source_type in ("article", "preprint", "model_card"):
        return "medium"
    if data_type in ("calculated", "statistical"):
        return "medium"
    if data_type in ("estimated", "projected", "modeled"):
        return "high"
    return "medium"


def infer_raw_or_derived(record):
    locator = record.get("source_locator", "").lower()
    notes = record.get("notes", "").lower()
    if "converted from" in locator or "derived" in notes:
        return "derived"
    return "raw"


def infer_applicability_domain(record):
    return {
        "phase": record.get("phase"),
        "impact_category": record.get("impact_category"),
        "unit_of_analysis": infer_unit_of_analysis(record),
        "model_or_scope": record.get("model_or_scope"),
        "geography": record.get("geography"),
        "system_boundary": record.get("system_boundary"),
    }


def filter_records(records, params):
    filtered = records
    for field in ("phase", "impact_category", "study_key", "geography"):
        value = extract_single_value(params, field)
        if value:
            filtered = [record for record in filtered if record.get(field) == value]
    return filtered


def extract_single_value(params, key):
    value = params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def compute_stats(records):
    by_phase = {}
    by_impact = {}
    for record in records:
        by_phase[record["phase"]] = by_phase.get(record["phase"], 0) + 1
        by_impact[record["impact_category"]] = by_impact.get(record["impact_category"], 0) + 1
    return {
        "record_count": len(records),
        "studies": sorted({record["study_key"] for record in records}),
        "by_phase": by_phase,
        "by_impact_category": by_impact,
    }


def list_sources(records):
    sources = []
    seen = set()
    for record in records:
        key = (record["study_key"], record["citation"], record["source_url"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "study_key": record["study_key"],
                "citation": record["citation"],
                "publication_year": record["publication_year"],
                "source_url": record["source_url"],
                "source_type": record["source_type"],
            }
        )
    return sources


def get_record(records, record_id):
    for record in records:
        if record["record_id"] == record_id:
            return record
    return None


def normalize_identifier(value):
    if not value:
        return ""
    normalized = str(value).strip().lower()
    for char in (" ", "-", "_", ".", ",", ":", ";", "/", "(", ")"):
        normalized = normalized.replace(char, "")
    return normalized


def get_model_profile(model_id=None, provider=None, estimated_active_parameters_billion=None):
    if estimated_active_parameters_billion not in (None, ""):
        return {
            "model_id": model_id or "user-estimated-model",
            "provider": provider or "unknown",
            "active_parameters_billion": str(estimated_active_parameters_billion),
            "parameter_confidence": "user_provided",
            "parameter_source": "user_input",
            "matching_strategy": "user_input",
        }

    normalized_model = normalize_identifier(model_id)
    normalized_provider = normalize_identifier(provider)
    if not normalized_model and not normalized_provider:
        return None

    for profile in load_models():
        aliases = [normalize_identifier(profile.get("model_id"))]
        aliases.extend(normalize_identifier(part) for part in profile.get("aliases", "").split("|") if part.strip())
        if normalized_model and normalized_model in aliases:
            result = dict(profile)
            result["matching_strategy"] = "model_id"
            return result

    for profile in load_models():
        if normalized_provider and normalize_identifier(profile.get("provider")) == normalized_provider:
            result = dict(profile)
            result["matching_strategy"] = "provider_family"
            return result

    return None


def get_country_mix(country_code):
    normalized = normalize_identifier(country_code)
    if not normalized:
        return None
    for row in load_country_energy_mix():
        if normalize_identifier(row.get("country_code")) == normalized or normalize_identifier(row.get("country_name")) == normalized:
            return row
    return None


def get_market_model_profile(model_id):
    normalized_model = normalize_identifier(model_id)
    if not normalized_model:
        return None
    for profile in load_market_models():
        aliases = [normalize_identifier(profile.get("model_id"))]
        aliases.extend(normalize_identifier(part) for part in profile.get("display_name", "").split("|") if part.strip())
        if normalized_model in aliases:
            return dict(profile)
    return None


def resolve_inference_country_mix(payload):
    explicit_country = payload.get("country")
    explicit_mix = get_country_mix(explicit_country) if explicit_country else None
    market_profile = get_market_model_profile(payload.get("model_id"))
    deployment_mode = normalize_identifier(payload.get("deployment_mode"))

    if not market_profile:
        return explicit_mix, "explicit_country" if explicit_mix else "none", market_profile

    market_status = str(market_profile.get("market_status", "")).strip().lower()
    serving_mode = str(market_profile.get("serving_mode", "")).strip().lower()
    provider_mix = get_country_mix(market_profile.get("estimation_country_code"))

    hosted_deployment = deployment_mode in {"api", "saas", "cloud", "hosted", "managed"}
    self_hosted_deployment = deployment_mode in {"selfhosted", "self_hosted", "local", "onprem", "onpremise"}

    if market_status == "api" or (serving_mode == "closed" and not self_hosted_deployment):
        return provider_mix, "publisher_country", market_profile

    if market_status == "api_and_open_weight" and hosted_deployment:
        return provider_mix, "publisher_country", market_profile

    if explicit_mix:
        return explicit_mix, "project_country", market_profile

    if market_status in {"open_weight", "api_and_open_weight"} or serving_mode in {"open", "hybrid"}:
        return provider_mix, "fallback_reference_country", market_profile

    return explicit_mix or provider_mix, "fallback_reference_country" if provider_mix else "none", market_profile


def get_extrapolation_rule(metric_kind):
    for row in load_extrapolation_rules():
        if row.get("metric_kind") == metric_kind:
            return row
    return None


def get_record_by_prefix(records, prefix):
    for record in records:
        if record["record_id"].startswith(prefix):
            return record
    return None


def infer_parametric_request_estimate(records, payload, grid_carbon_intensity, water_intensity):
    model_profile = get_model_profile(
        model_id=payload.get("model_id"),
        provider=payload.get("provider"),
        estimated_active_parameters_billion=payload.get("estimated_active_parameters_billion"),
    )
    if not model_profile:
        return None

    try:
        target_params = to_float(model_profile.get("active_parameters_billion"), default=None)
    except ValueError:
        return None
    if target_params is None:
        return None

    requests_count = to_float(payload.get("requests_count", 1.0), default=1.0)
    input_tokens = to_float(payload.get("input_tokens", 0.0), default=0.0)
    output_tokens = to_float(payload.get("output_tokens", 0.0), default=0.0)
    total_tokens = input_tokens + output_tokens

    energy_rule = get_extrapolation_rule("energy")
    carbon_rule = get_extrapolation_rule("carbon")
    water_rule = get_extrapolation_rule("water")
    if not energy_rule:
        return None

    reference_tokens = to_float(energy_rule.get("reference_tokens"), default=750.0)
    token_ratio = clamp((total_tokens / reference_tokens) if total_tokens > 0 else (REFERENCE_PROMPT_TOKENS / reference_tokens), 0.25, 6.0)

    anchors = [
        {
            "profile": get_model_profile("gemma-2b-it"),
            "energy_record": get_record(records, "ren2024_gemma2b_energy"),
            "carbon_record": get_record(records, "ren2024_gemma2b_carbon"),
            "water_record": get_record(records, "ren2024_gemma2b_water"),
        },
        {
            "profile": get_model_profile("llama-3-70b"),
            "energy_record": get_record(records, "ren2024_llama70b_energy"),
            "carbon_record": get_record(records, "ren2024_llama70b_carbon"),
            "water_record": get_record(records, "ren2024_llama70b_water"),
        },
    ]
    anchors = [item for item in anchors if item["profile"]]
    if len(anchors) < 2:
        return None

    assumptions = [
        f"Parametric extrapolation enabled for target model {model_profile.get('model_id')} ({target_params:g}B active parameters)",
        f"Reference inference scaling derived from Ren et al. 2024 with page-level measurements at {reference_tokens:g} tokens",
    ]
    if total_tokens > 0:
        assumptions.append(f"Token scaling applied relative to {reference_tokens:g} tokens per reference page")
    else:
        assumptions.append(f"No token counts provided; default prompt size approximated from {int(REFERENCE_PROMPT_TOKENS)} tokens")

    selected_factors = []
    results = {}
    extrapolation_details = {}

    for metric_kind, rule, unit_kind, scale_factor in (
        ("energy", energy_rule, "energy_wh", 1000.0),
        ("carbon", carbon_rule, "carbon_gco2e", 1.0),
        ("water", water_rule, "water_ml", 1000.0),
    ):
        if not rule:
            continue
        central_exp = to_float(rule.get("exponent_central"), default=1.0)
        low_exp = to_float(rule.get("exponent_low"), default=central_exp)
        high_exp = to_float(rule.get("exponent_high"), default=central_exp)
        anchor_values = []
        selected_record_ids = []
        detail_rows = []
        for anchor in anchors:
            record = anchor[f"{metric_kind}_record"]
            profile = anchor["profile"]
            if not record or not profile:
                continue
            anchor_params = to_float(profile.get("active_parameters_billion"), default=None)
            if anchor_params in (None, 0):
                continue
            metric_value = to_float(record["metric_value"], default=None)
            if metric_value is None:
                continue
            ratio = target_params / anchor_params
            low_factor = (ratio ** low_exp) * token_ratio * requests_count
            central_factor = (ratio ** central_exp) * token_ratio * requests_count
            high_factor = (ratio ** high_exp) * token_ratio * requests_count
            low_value = metric_value * low_factor * scale_factor
            central_value = metric_value * central_factor * scale_factor
            high_value = metric_value * high_factor * scale_factor
            anchor_values.append((low_value, central_value, high_value))
            selected_record_ids.append(record["record_id"])
            detail_rows.append(
                {
                    "record_id": record["record_id"],
                    "source_model": profile.get("model_id"),
                    "source_active_parameters_billion": anchor_params,
                    "source_value": metric_value,
                    "source_unit": record.get("metric_unit"),
                    "parameter_ratio": round(ratio, 6),
                    "factor_low": round(low_factor, 6),
                    "factor_central": round(central_factor, 6),
                    "factor_high": round(high_factor, 6),
                    "extrapolated_value_low": round(low_value, 6),
                    "extrapolated_value_central": round(central_value, 6),
                    "extrapolated_value_high": round(high_value, 6),
                }
            )
        if not anchor_values:
            continue
        lows = [item[0] for item in anchor_values]
        centrals = [item[1] for item in anchor_values]
        highs = [item[2] for item in anchor_values]
        results[unit_kind] = rounded_range(min(lows), sum(centrals) / len(centrals), max(highs))
        selected_factors.extend(selected_record_ids)
        extrapolation_details[unit_kind] = {
            "reference_tokens": reference_tokens,
            "token_ratio": round(token_ratio, 6),
            "exponent_low": low_exp,
            "exponent_central": central_exp,
            "exponent_high": high_exp,
            "anchors": detail_rows,
        }

    if grid_carbon_intensity is not None and "energy_wh" in results and "carbon_gco2e" in results:
        energy = results["energy_wh"]
        derived_central = wh_to_gco2e(energy["central"], grid_carbon_intensity)
        derived_high = wh_to_gco2e(energy["high"], grid_carbon_intensity)
        results["carbon_gco2e"] = rounded_range(
            min(results["carbon_gco2e"]["low"], derived_central),
            derived_central,
            max(results["carbon_gco2e"]["high"], derived_high),
        )
        assumptions.append("Carbon contextualized using country electricity carbon intensity")

    if water_intensity is not None and "energy_wh" in results and "water_ml" in results:
        energy = results["energy_wh"]
        derived_central = wh_to_liters(energy["central"], water_intensity) * 1000.0
        derived_high = wh_to_liters(energy["high"], water_intensity) * 1000.0
        results["water_ml"] = rounded_range(
            min(results["water_ml"]["low"], derived_central),
            derived_central,
            max(results["water_ml"]["high"], derived_high),
        )
        assumptions.append("Water contextualized using country electricity water intensity")

    return {
        "model_profile": model_profile,
        "results": results,
        "selected_factors": dedupe(selected_factors),
        "assumptions": assumptions,
        "method": "parametric_extrapolation",
        "rule_ids": [row.get("rule_id") for row in load_extrapolation_rules()],
        "extrapolation_details": extrapolation_details,
    }


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

    if metric_kind == "carbon" and "gco2" in metric_unit:
        return metric_value / energy_kwh

    if metric_kind == "water":
        if "ml/" in metric_unit:
            water_l = metric_value / 1000.0
        elif metric_unit.startswith("l/"):
            water_l = metric_value
        else:
            return None
        return water_l / energy_kwh

    return None


def format_scalar(value, decimals=3):
    if value is None:
        return "n.d."
    text = f"{float(value):.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_raw_metric(value, unit):
    return f"{format_scalar(value)} {unit}".strip()


def aggregate_method_ranges(methods):
    if not methods:
        empty = rounded_range(0.0, 0.0, 0.0)
        return {
            "energy_wh": empty,
            "carbon_gco2e": empty,
            "water_ml": empty,
        }

    def aggregate_metric(metric_key):
        lows = [float(method[metric_key]["low"]) for method in methods if method.get(metric_key)]
        centrals = [float(method[metric_key]["central"]) for method in methods if method.get(metric_key)]
        highs = [float(method[metric_key]["high"]) for method in methods if method.get(metric_key)]
        if not centrals:
            return rounded_range(0.0, 0.0, 0.0)
        return rounded_range(min(lows), sum(centrals) / len(centrals), max(highs))

    return {
        "energy_wh": aggregate_metric("annual_energy_wh"),
        "carbon_gco2e": aggregate_metric("annual_carbon_gco2e"),
        "water_ml": aggregate_metric("annual_water_ml"),
    }


def build_inference_method_set(records, payload):
    request_type = payload.get("request_type", "chat_generation")
    input_tokens = to_float(payload.get("input_tokens", 0.0), default=0.0)
    output_tokens = to_float(payload.get("output_tokens", 0.0), default=0.0)
    requests_per_feature = to_float(payload.get("requests_per_feature", 1.0), default=1.0)
    feature_uses_per_month = to_float(payload.get("feature_uses_per_month", 0.0), default=0.0)
    months_per_year = to_float(payload.get("months_per_year", 12.0), default=12.0)
    annual_feature_uses = feature_uses_per_month * months_per_year
    annual_requests = annual_feature_uses * requests_per_feature
    model_profile = get_model_profile(
        model_id=payload.get("model_id"),
        provider=payload.get("provider"),
        estimated_active_parameters_billion=payload.get("estimated_active_parameters_billion"),
    )
    target_params = to_float((model_profile or {}).get("active_parameters_billion"), default=None)

    country_mix, country_resolution, market_profile = resolve_inference_country_mix(payload)
    grid_carbon_intensity = to_float(payload.get("grid_carbon_intensity_gco2_per_kwh"), default=None)
    water_intensity = to_float(payload.get("water_intensity_l_per_kwh"), default=None)
    if grid_carbon_intensity is None and country_mix:
        grid_carbon_intensity = to_float(country_mix.get("grid_carbon_intensity_gco2_per_kwh"), default=None)
    if water_intensity is None and country_mix:
        water_intensity = to_float(country_mix.get("water_intensity_l_per_kwh"), default=None)

    token_ratio = compute_token_ratio(input_tokens, output_tokens)
    total_tokens = input_tokens + output_tokens
    page_ratio = (total_tokens / REFERENCE_PAGE_TOKENS) if total_tokens > 0 else (REFERENCE_PROMPT_TOKENS / REFERENCE_PAGE_TOKENS)

    methods = []
    selected_factors = []
    assumptions = [
        "Inference-only estimate: training and software-system overheads excluded",
        f"Request type classified as {request_type}",
        f"{requests_per_feature} LLM request(s) per feature use",
        f"{annual_feature_uses} feature uses per year",
    ]
    if target_params is not None:
        assumptions.append(f"Model-size scaling enabled with target profile at {target_params:g}B active parameters")
    else:
        assumptions.append("No parameter count available for the target model; multiplicative scaling disabled")
    if country_mix:
        if country_resolution == "publisher_country":
            assumptions.append(
                f"Carbon and water recalculated with the publisher-country mix for {country_mix.get('country_code')} because the model is treated as a proprietary hosted service"
            )
        elif country_resolution == "project_country":
            assumptions.append(
                f"Carbon and water recalculated with the project country mix for {country_mix.get('country_code')} because the model is treated as open-weight or self-hosted"
            )
        else:
            assumptions.append(
                f"Country mix fallback applied for {country_mix.get('country_code')} ({country_mix.get('source_citation')})"
            )

    def add_method(
        method_id,
        label,
        basis,
        record_ids,
        per_request_energy_wh,
        per_request_carbon_g,
        per_request_water_ml,
        detail,
    ):
        annual_energy = rounded_range(
            per_request_energy_wh["low"] * annual_requests,
            per_request_energy_wh["central"] * annual_requests,
            per_request_energy_wh["high"] * annual_requests,
        )
        annual_carbon = rounded_range(
            per_request_carbon_g["low"] * annual_requests,
            per_request_carbon_g["central"] * annual_requests,
            per_request_carbon_g["high"] * annual_requests,
        )
        annual_water = rounded_range(
            per_request_water_ml["low"] * annual_requests,
            per_request_water_ml["central"] * annual_requests,
            per_request_water_ml["high"] * annual_requests,
        )
        methods.append(
            {
                "method_id": method_id,
                "label": label,
                "basis": basis,
                "record_ids": dedupe(record_ids),
                "annual_energy_wh": annual_energy,
                "annual_carbon_gco2e": annual_carbon,
                "annual_water_ml": annual_water,
                "per_request_energy_wh": per_request_energy_wh,
                "per_request_carbon_gco2e": per_request_carbon_g,
                "per_request_water_ml": per_request_water_ml,
                "detail": detail,
            }
        )

    prompt_energy = get_record(records, "elsworth2025_prompt_energy")
    prompt_carbon = get_record(records, "elsworth2025_prompt_carbon")
    prompt_water = get_record(records, "elsworth2025_prompt_water")
    if prompt_energy:
        source_params = to_float(prompt_energy.get("model_parameters_normalized", "").replace("B", ""), default=None)
        param_ratio = (target_params / source_params) if (target_params is not None and source_params not in (None, 0)) else 1.0
        source_energy_wh = to_float(prompt_energy["metric_value"]) * token_ratio * param_ratio
        per_request_energy = rounded_range(source_energy_wh, source_energy_wh, source_energy_wh)
        if grid_carbon_intensity is not None:
            carbon_val = wh_to_gco2e(source_energy_wh, grid_carbon_intensity)
            per_request_carbon = rounded_range(carbon_val, carbon_val, carbon_val)
        elif prompt_carbon:
            carbon_val = to_float(prompt_carbon["metric_value"]) * token_ratio
            per_request_carbon = rounded_range(carbon_val, carbon_val, carbon_val)
        else:
            per_request_carbon = rounded_range(0.0, 0.0, 0.0)
        if water_intensity is not None:
            water_val = wh_to_liters(source_energy_wh, water_intensity) * 1000.0
            per_request_water = rounded_range(water_val, water_val, water_val)
        elif prompt_water:
            water_val = to_float(prompt_water["metric_value"]) * token_ratio
            per_request_water = rounded_range(water_val, water_val, water_val)
        else:
            per_request_water = rounded_range(0.0, 0.0, 0.0)
        record_ids = ["elsworth2025_prompt_energy", "elsworth2025_prompt_carbon", "elsworth2025_prompt_water"]
        selected_factors.extend(record_ids)
        add_method(
            "prompt_average",
            "Méthode par prompt",
            "Moyenne des indicateurs connus exprimés par prompt, recalculés pour le pays d'inférence",
            record_ids,
            per_request_energy,
            per_request_carbon,
            per_request_water,
            {
                "kind": "multiples",
                "unit_basis": "prompt",
                "ratio": token_ratio,
                "anchors": [
                    {
                        "source_model": prompt_energy.get("llm_normalized") or prompt_energy.get("model_or_scope"),
                        "source_country": prompt_energy.get("country_normalized") or "Non spécifié",
                        "source_params": source_params,
                        "target_params": target_params,
                        "parameter_factor": param_ratio,
                        "source_energy": format_raw_metric(prompt_energy["metric_value"], prompt_energy["metric_unit"]),
                        "source_carbon": format_raw_metric(prompt_carbon["metric_value"], prompt_carbon["metric_unit"]) if prompt_carbon else None,
                        "source_water": format_raw_metric(prompt_water["metric_value"], prompt_water["metric_unit"]) if prompt_water else None,
                        "source_carbon_intensity": infer_source_intensity(prompt_energy, prompt_carbon, "carbon") if prompt_carbon else None,
                        "source_water_intensity": infer_source_intensity(prompt_energy, prompt_water, "water") if prompt_water else None,
                        "per_request_energy": rounded_range(source_energy_wh, source_energy_wh, source_energy_wh),
                        "per_request_carbon": per_request_carbon,
                        "per_request_water": per_request_water,
                    }
                ],
                "target_country": payload.get("country"),
                "target_mix": country_mix,
                "target_grid_carbon_intensity": grid_carbon_intensity,
                "target_water_intensity": water_intensity,
            },
        )

    page_anchors = []
    page_record_ids = []
    for prefix in ("ren2024_gemma2b", "ren2024_llama70b"):
        energy_record = get_record(records, f"{prefix}_energy")
        carbon_record = get_record(records, f"{prefix}_carbon")
        water_record = get_record(records, f"{prefix}_water")
        if not energy_record:
            continue
        source_params = to_float(energy_record.get("model_parameters_normalized", "").replace("B", ""), default=None)
        param_ratio = (target_params / source_params) if (target_params is not None and source_params not in (None, 0)) else 1.0
        source_energy_wh = to_float(energy_record["metric_value"]) * 1000.0 * page_ratio * param_ratio
        if grid_carbon_intensity is not None:
            carbon_val = wh_to_gco2e(source_energy_wh, grid_carbon_intensity)
            per_request_carbon = rounded_range(carbon_val, carbon_val, carbon_val)
        elif carbon_record:
            carbon_val = to_float(carbon_record["metric_value"]) * page_ratio
            per_request_carbon = rounded_range(carbon_val, carbon_val, carbon_val)
        else:
            per_request_carbon = rounded_range(0.0, 0.0, 0.0)
        if water_intensity is not None:
            water_val = wh_to_liters(source_energy_wh, water_intensity) * 1000.0
            per_request_water = rounded_range(water_val, water_val, water_val)
        elif water_record:
            water_val = to_float(water_record["metric_value"]) * 1000.0 * page_ratio
            per_request_water = rounded_range(water_val, water_val, water_val)
        else:
            per_request_water = rounded_range(0.0, 0.0, 0.0)
        anchor_record_ids = [f"{prefix}_energy", f"{prefix}_carbon", f"{prefix}_water"]
        page_record_ids.extend(anchor_record_ids)
        page_anchors.append(
            {
                "source_model": energy_record.get("llm_normalized") or energy_record.get("model_or_scope"),
                "source_country": energy_record.get("country_normalized") or "Non spécifié",
                "source_params": source_params,
                "target_params": target_params,
                "parameter_factor": param_ratio,
                "source_energy": format_raw_metric(energy_record["metric_value"], energy_record["metric_unit"]),
                "source_carbon": format_raw_metric(carbon_record["metric_value"], carbon_record["metric_unit"]) if carbon_record else None,
                "source_water": format_raw_metric(water_record["metric_value"], water_record["metric_unit"]) if water_record else None,
                "source_carbon_intensity": infer_source_intensity(energy_record, carbon_record, "carbon") if carbon_record else None,
                "source_water_intensity": infer_source_intensity(energy_record, water_record, "water") if water_record else None,
                "per_request_energy": rounded_range(source_energy_wh, source_energy_wh, source_energy_wh),
                "per_request_carbon": per_request_carbon,
                "per_request_water": per_request_water,
            }
        )
    if page_anchors:
        selected_factors.extend(page_record_ids)
        def avg_range(anchors, key):
            lows = [anchor[key]["low"] for anchor in anchors]
            centrals = [anchor[key]["central"] for anchor in anchors]
            highs = [anchor[key]["high"] for anchor in anchors]
            return rounded_range(min(lows), sum(centrals) / len(centrals), max(highs))
        add_method(
            "page_average",
            "Méthode par page",
            "Moyenne des indicateurs connus exprimés par page, ajustés au volume de tokens puis recalculés pour le pays d'inférence",
            page_record_ids,
            avg_range(page_anchors, "per_request_energy"),
            avg_range(page_anchors, "per_request_carbon"),
            avg_range(page_anchors, "per_request_water"),
            {
                "kind": "multiples",
                "unit_basis": "page",
                "ratio": page_ratio,
                "anchors": page_anchors,
                "target_country": payload.get("country"),
                "target_mix": country_mix,
                "target_grid_carbon_intensity": grid_carbon_intensity,
                "target_water_intensity": water_intensity,
            },
        )

    aggregated = aggregate_method_ranges(methods)
    per_request_aggregate = {
        "energy_wh": rounded_range(
            aggregated["energy_wh"]["low"] / annual_requests if annual_requests else 0.0,
            aggregated["energy_wh"]["central"] / annual_requests if annual_requests else 0.0,
            aggregated["energy_wh"]["high"] / annual_requests if annual_requests else 0.0,
        ),
        "carbon_gco2e": rounded_range(
            aggregated["carbon_gco2e"]["low"] / annual_requests if annual_requests else 0.0,
            aggregated["carbon_gco2e"]["central"] / annual_requests if annual_requests else 0.0,
            aggregated["carbon_gco2e"]["high"] / annual_requests if annual_requests else 0.0,
        ),
        "water_ml": rounded_range(
            aggregated["water_ml"]["low"] / annual_requests if annual_requests else 0.0,
            aggregated["water_ml"]["central"] / annual_requests if annual_requests else 0.0,
            aggregated["water_ml"]["high"] / annual_requests if annual_requests else 0.0,
        ),
    }

    return {
        "scenario_id": payload.get("scenario_id", "inference-estimate"),
        "estimate_level": "inference_feature",
        "method": "literature_multiples",
        "uncertainty_level": "high",
        "applicability_note": "Inference-only screening estimate aggregated across multiple literature indicators.",
        "inputs": {
            "provider": payload.get("provider"),
            "model_id": payload.get("model_id"),
            "request_type": request_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "country": payload.get("country"),
            "effective_country": country_mix.get("country_code") if country_mix else None,
        },
        "feature_scope": {
            "requests_per_feature": requests_per_feature,
            "feature_uses_per_month": feature_uses_per_month,
            "months_per_year": months_per_year,
            "annual_feature_uses": annual_feature_uses,
            "annual_llm_requests": annual_requests,
        },
        "per_request_llm": per_request_aggregate,
        "per_feature_llm": {
            "energy_wh": scale_range(per_request_aggregate["energy_wh"], requests_per_feature),
            "carbon_gco2e": scale_range(per_request_aggregate["carbon_gco2e"], requests_per_feature),
            "water_ml": scale_range(per_request_aggregate["water_ml"], requests_per_feature),
        },
        "annual_llm": aggregated,
        "annual_total": aggregated,
        "method_results": methods,
        "selected_factors": dedupe(selected_factors),
        "assumptions": assumptions,
        "country_energy_mix": country_mix,
        "country_resolution": country_resolution,
        "model_profile": get_model_profile(
            model_id=payload.get("model_id"),
            provider=payload.get("provider"),
            estimated_active_parameters_billion=payload.get("estimated_active_parameters_billion"),
        ),
        "market_model_profile": market_profile,
        "extrapolation_rules": [],
        "extrapolation_details": {},
        "software_overhead": {"components": [], "annual_energy_wh": 0.0, "annual_carbon_gco2e": 0.0, "annual_water_ml": 0.0},
    }


def estimate_externalities(records, payload):
    request_type = payload.get("request_type", "chat_generation")
    requests_count = to_float(payload.get("requests_count", 1.0), default=1.0)
    input_tokens = to_float(payload.get("input_tokens", 0.0), default=0.0)
    output_tokens = to_float(payload.get("output_tokens", 0.0), default=0.0)
    country_mix, country_resolution, market_profile = resolve_inference_country_mix(payload)
    grid_carbon_intensity = to_float(payload.get("grid_carbon_intensity_gco2_per_kwh"), default=None)
    water_intensity = to_float(payload.get("water_intensity_l_per_kwh"), default=None)
    if grid_carbon_intensity is None and country_mix:
        grid_carbon_intensity = to_float(country_mix.get("grid_carbon_intensity_gco2_per_kwh"), default=None)
    if water_intensity is None and country_mix:
        water_intensity = to_float(country_mix.get("water_intensity_l_per_kwh"), default=None)

    extrapolated = infer_parametric_request_estimate(records, payload, grid_carbon_intensity, water_intensity)
    if extrapolated and extrapolated.get("results"):
        assumptions = list(extrapolated["assumptions"])
        assumptions.append(f"Request type classified as {request_type}")
        if country_mix:
            if country_resolution == "publisher_country":
                assumptions.append(
                    f"Carbon and water recalculated with the publisher-country mix for {country_mix.get('country_code')} because the model is treated as a proprietary hosted service"
                )
            elif country_resolution == "project_country":
                assumptions.append(
                    f"Carbon and water recalculated with the project country mix for {country_mix.get('country_code')} because the model is treated as open-weight or self-hosted"
                )
            else:
                assumptions.append(
                    f"Country mix fallback applied for {country_mix.get('country_code')} ({country_mix.get('source_citation')})"
                )
        return {
            "scenario_id": payload.get("scenario_id", "unspecified"),
            "estimate_level": "request" if requests_count == 1 else "scenario",
            "inputs": {
                "provider": payload.get("provider"),
                "model_id": payload.get("model_id"),
                "deployment_mode": payload.get("deployment_mode"),
                "request_type": request_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "requests_count": requests_count,
                "country": payload.get("country"),
            },
            "results": extrapolated["results"],
            "selected_factors": extrapolated["selected_factors"],
            "assumptions": assumptions,
            "uncertainty_level": "high",
            "applicability_note": "Screening-level estimate based on parametric extrapolation from literature factors; not suitable as-is for audited declarations.",
            "method": extrapolated["method"],
            "model_profile": extrapolated["model_profile"],
            "country_energy_mix": country_mix,
            "country_resolution": country_resolution,
            "market_model_profile": market_profile,
            "extrapolation_rules": extrapolated["rule_ids"],
            "extrapolation_details": extrapolated.get("extrapolation_details", {}),
        }

    token_ratio = compute_token_ratio(input_tokens, output_tokens)
    assumptions = []
    if input_tokens or output_tokens:
        assumptions.append(
            f"Token scaling heuristic applied with reference prompt size of {int(REFERENCE_PROMPT_TOKENS)} tokens"
        )
    else:
        assumptions.append("No token counts provided; literature factors used without prompt-size scaling")

    assumptions.append(f"Request type classified as {request_type}")
    if country_mix:
        if country_resolution == "publisher_country":
            assumptions.append(
                f"Carbon and water recalculated with the publisher-country mix for {country_mix.get('country_code')} because the model is treated as a proprietary hosted service"
            )
        elif country_resolution == "project_country":
            assumptions.append(
                f"Carbon and water recalculated with the project country mix for {country_mix.get('country_code')} because the model is treated as open-weight or self-hosted"
            )
        else:
            assumptions.append(
                f"Country mix fallback applied for {country_mix.get('country_code')} ({country_mix.get('source_citation')})"
            )

    prompt_energy = get_record(records, "elsworth2025_prompt_energy")
    prompt_carbon = get_record(records, "elsworth2025_prompt_carbon")
    prompt_water = get_record(records, "elsworth2025_prompt_water")
    query_energy_high = get_record(records, "epri2024_chatgpt_query")

    selected_factors = []
    results = {}

    if prompt_energy:
        central_energy = to_float(prompt_energy["metric_value"]) * token_ratio * requests_count
        low_energy = central_energy
        if query_energy_high:
            high_energy = to_float(query_energy_high["metric_value"]) * token_ratio * requests_count
            selected_factors.append(query_energy_high["record_id"])
        else:
            high_energy = central_energy * 2.0
        results["energy_wh"] = rounded_range(low_energy, central_energy, high_energy)
        selected_factors.append(prompt_energy["record_id"])

    if prompt_carbon:
        raw_carbon = to_float(prompt_carbon["metric_value"]) * token_ratio * requests_count
        central_carbon = raw_carbon
        low_carbon = raw_carbon
        high_carbon = raw_carbon
        if grid_carbon_intensity is not None and "energy_wh" in results:
            derived_central = wh_to_gco2e(results["energy_wh"]["central"], grid_carbon_intensity)
            derived_high = wh_to_gco2e(results["energy_wh"]["high"], grid_carbon_intensity)
            central_carbon = derived_central
            low_carbon = min(raw_carbon, derived_central)
            high_carbon = max(raw_carbon, derived_high)
            assumptions.append("Carbon adjusted using provided electricity carbon intensity")
        results["carbon_gco2e"] = rounded_range(low_carbon, central_carbon, high_carbon)
        selected_factors.append(prompt_carbon["record_id"])

    if prompt_water:
        raw_water = to_float(prompt_water["metric_value"]) * token_ratio * requests_count
        central_water = raw_water
        low_water = raw_water
        high_water = raw_water
        if water_intensity is not None and "energy_wh" in results:
            derived_central = wh_to_liters(results["energy_wh"]["central"], water_intensity) * 1000.0
            derived_high = wh_to_liters(results["energy_wh"]["high"], water_intensity) * 1000.0
            central_water = max(raw_water, derived_central)
            low_water = min(raw_water, derived_central)
            high_water = max(raw_water, derived_high)
            assumptions.append("Water adjusted using provided electricity water intensity")
        results["water_ml"] = rounded_range(low_water, central_water, high_water)
        selected_factors.append(prompt_water["record_id"])

    uncertainty_level = "high"
    applicability_note = (
        "Screening-level estimate based on literature factors; not suitable as-is for audited declarations."
    )

    return {
        "scenario_id": payload.get("scenario_id", "unspecified"),
        "estimate_level": "request" if requests_count == 1 else "scenario",
        "inputs": {
            "provider": payload.get("provider"),
            "model_id": payload.get("model_id"),
            "deployment_mode": payload.get("deployment_mode"),
            "request_type": request_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests_count": requests_count,
            "country": payload.get("country"),
        },
        "results": results,
        "selected_factors": dedupe(selected_factors),
        "assumptions": assumptions,
        "uncertainty_level": uncertainty_level,
        "applicability_note": applicability_note,
        "method": "literature_proxy",
        "model_profile": get_model_profile(
            model_id=payload.get("model_id"),
            provider=payload.get("provider"),
            estimated_active_parameters_billion=payload.get("estimated_active_parameters_billion"),
        ),
        "country_energy_mix": country_mix,
        "country_resolution": country_resolution,
        "market_model_profile": market_profile,
        "extrapolation_rules": [],
        "extrapolation_details": {},
    }


def estimate_feature_externalities(records, payload):
    return build_inference_method_set(records, payload)


def predict_inference_externalities(records, payload):
    return build_inference_method_set(records, payload)


def build_market_model_predictions(records):
    predictions = []
    monthly_uses = MARKET_REFERENCE_REQUESTS_PER_YEAR / 12.0
    for row in load_market_models():
        active_parameters = to_float(row.get("active_parameters_billion"), default=None)
        payload = {
            "scenario_id": f"market-{row.get('model_id')}",
            "provider": row.get("provider"),
            "model_id": row.get("model_id"),
            "request_type": "chat_generation",
            "input_tokens": 1000.0,
            "output_tokens": 550.0,
            "requests_per_feature": 1.0,
            "feature_uses_per_month": monthly_uses,
            "months_per_year": 12.0,
            "country": row.get("estimation_country_code") or "US",
        }
        if active_parameters is not None:
            payload["estimated_active_parameters_billion"] = active_parameters
        estimate = predict_inference_externalities(records, payload)
        predictions.append(
            {
                **row,
                "standard_scenario": {
                    "request_type": "chat_generation",
                    "input_tokens": 1000.0,
                    "output_tokens": 550.0,
                    "requests_per_year": MARKET_REFERENCE_REQUESTS_PER_YEAR,
                },
                "estimate_status": "estimated",
                "estimate_method": estimate.get("method"),
                "evidence_level": "measured_proxy" if active_parameters is not None else "generic_proxy",
                "per_request_energy_wh": estimate.get("per_request_llm", {}).get("energy_wh", {}),
                "per_request_carbon_gco2e": estimate.get("per_request_llm", {}).get("carbon_gco2e", {}),
                "per_request_water_ml": estimate.get("per_request_llm", {}).get("water_ml", {}),
                "annual_energy_wh": estimate.get("annual_llm", {}).get("energy_wh", {}),
                "annual_carbon_gco2e": estimate.get("annual_llm", {}).get("carbon_gco2e", {}),
                "annual_water_ml": estimate.get("annual_llm", {}).get("water_ml", {}),
                "country_energy_mix": estimate.get("country_energy_mix") or {},
                "selected_factors": estimate.get("selected_factors", []),
                "assumptions": estimate.get("assumptions", []),
                "method_results": estimate.get("method_results", []),
            }
        )
    return predictions


def compute_token_ratio(input_tokens, output_tokens):
    total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return 1.0
    ratio = total_tokens / REFERENCE_PROMPT_TOKENS
    return clamp(ratio, 0.25, 4.0)


def clamp(value, low, high):
    return max(low, min(high, value))


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


def wh_to_gco2e(wh_value, grid_carbon_intensity_gco2_per_kwh):
    return (wh_value / 1000.0) * grid_carbon_intensity_gco2_per_kwh


def wh_to_liters(wh_value, water_intensity_l_per_kwh):
    return (wh_value / 1000.0) * water_intensity_l_per_kwh


def rounded_range(low, central, high):
    return {
        "low": round(low, 4),
        "central": round(central, 4),
        "high": round(high, 4),
    }


def scale_range(range_obj, factor):
    return {
        "low": round(range_obj["low"] * factor, 4),
        "central": round(range_obj["central"] * factor, 4),
        "high": round(range_obj["high"] * factor, 4),
    }


def add_scalar_to_range(range_obj, scalar):
    return {
        "low": round(range_obj["low"] + scalar, 4),
        "central": round(range_obj["central"] + scalar, 4),
        "high": round(range_obj["high"] + scalar, 4),
    }


def build_software_breakdown(components, annual_feature_uses, grid_carbon_intensity, water_intensity):
    breakdown = []
    annual_energy_wh = 0.0
    annual_carbon_gco2e = 0.0 if grid_carbon_intensity is not None else None
    annual_water_ml = 0.0 if water_intensity is not None else None

    for item in components:
        energy_wh_per_feature = to_float(item.get("energy_wh_per_feature", 0.0), default=0.0)
        item_energy_annual = energy_wh_per_feature * annual_feature_uses
        item_carbon_annual = wh_to_gco2e(item_energy_annual, grid_carbon_intensity) if grid_carbon_intensity is not None else None
        item_water_annual = wh_to_liters(item_energy_annual, water_intensity) * 1000.0 if water_intensity is not None else None
        annual_energy_wh += item_energy_annual
        if annual_carbon_gco2e is not None:
            annual_carbon_gco2e += item_carbon_annual
        if annual_water_ml is not None:
            annual_water_ml += item_water_annual
        breakdown.append(
            {
                "component_type": item.get("component_type", "component"),
                "description": item.get("description", ""),
                "energy_wh_per_feature": round(energy_wh_per_feature, 4),
                "annual_energy_wh": round(item_energy_annual, 4),
                "annual_carbon_gco2e": round(item_carbon_annual, 4) if item_carbon_annual is not None else None,
                "annual_water_ml": round(item_water_annual, 4) if item_water_annual is not None else None,
            }
        )

    return {
        "components": breakdown,
        "energy_wh_per_feature": round(sum(item["energy_wh_per_feature"] for item in breakdown), 4),
        "annual_energy_wh": round(annual_energy_wh, 4),
        "annual_carbon_gco2e": round(annual_carbon_gco2e, 4) if annual_carbon_gco2e is not None else None,
        "annual_water_ml": round(annual_water_ml, 4) if annual_water_ml is not None else None,
    }


def dedupe(values):
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
