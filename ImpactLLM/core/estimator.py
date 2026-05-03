#!/usr/bin/env python3
import csv
import math
import statistics
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
MARKET_REFERENCE_INPUT_TOKENS = 1000.0
MARKET_REFERENCE_OUTPUT_TOKENS = 550.0
MARKET_REFERENCE_READING_WORDS_PER_MINUTE = 238.0
MARKET_REFERENCE_WORDS_PER_TOKEN = 0.75
MARKET_PROMPT_ANCHOR_ENERGY_WH = 0.24
MARKET_PROMPT_ANCHOR_ACTIVE_PARAMS_B = 180.0
MARKET_PROMPT_OUTPUT_WEIGHT = 1.8
MARKET_REFERENCE_COMPUTE_TOKENS = MARKET_REFERENCE_INPUT_TOKENS + (MARKET_PROMPT_OUTPUT_WEIGHT * MARKET_REFERENCE_OUTPUT_TOKENS)
MARKET_REFERENCE_REQUESTS_PER_HOUR = round(
    (MARKET_REFERENCE_READING_WORDS_PER_MINUTE * 60.0)
    / (MARKET_REFERENCE_OUTPUT_TOKENS * MARKET_REFERENCE_WORDS_PER_TOKEN),
    1,
)
TRAINING_REFERENCE_TOKENS_PER_PARAMETER = 20.0


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


def parse_parameter_count_billion(value):
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"n.d.", "nd", "na", "n/a"}:
        return None
    raw = raw.replace(",", ".")
    active_match = None
    if "active" in raw.lower():
        active_match = raw.split("/")[0].strip()
    token = active_match or raw
    match = None
    import re
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([BM])", token, re.IGNORECASE)
    if not match:
        try:
            return float(token)
        except ValueError:
            return None
    value_num = float(match.group(1))
    suffix = match.group(2).upper()
    if suffix == "M":
        return value_num / 1000.0
    return value_num


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

    market_profile = get_market_model_profile(model_id)
    if market_profile:
        return build_reference_profile_from_market_profile(market_profile, "market_model_id")

    if not normalized_model:
        for profile in load_models():
            if normalized_provider and normalize_identifier(profile.get("provider")) == normalized_provider:
                result = dict(profile)
                result["matching_strategy"] = "provider_family"
                return result

    market_provider_profile = get_market_provider_profile(provider)
    if market_provider_profile:
        return build_reference_profile_from_market_profile(market_provider_profile, "provider_market_family")

    return None


def build_reference_profile_from_market_profile(profile, matching_strategy):
    return {
        "model_id": profile.get("model_id"),
        "provider": profile.get("provider"),
        "active_parameters_billion": profile.get("active_parameters_billion"),
        "total_parameters_billion": profile.get("total_parameters_billion"),
        "parameter_value_status": profile.get("parameter_value_status"),
        "parameter_confidence": profile.get("parameter_confidence"),
        "parameter_source": profile.get("parameter_source"),
        "parameter_source_url": profile.get("parameter_source_url"),
        "notes": profile.get("notes"),
        "matching_strategy": matching_strategy,
    }


def get_country_mix(country_code):
    normalized = normalize_identifier(country_code)
    if not normalized:
        return None
    for row in load_country_energy_mix():
        if normalize_identifier(row.get("country_code")) == normalized or normalize_identifier(row.get("country_name")) == normalized:
            return row
    return None


def get_record_country_mix(record):
    candidates = [
        record.get("country_code"),
        record.get("country_normalized"),
        record.get("geography"),
    ]
    aliases = {
        "États-Unis": "US",
        "United States": "US",
        "France": "FR",
        "Germany": "DE",
        "Deutschland": "DE",
        "United Kingdom": "GB",
        "Great Britain": "GB",
        "Canada": "CA",
        "China": "CN",
    }
    for candidate in candidates:
        mix = get_country_mix(candidate)
        if mix:
            return mix
        alias = aliases.get(str(candidate or "").strip())
        if alias:
            mix = get_country_mix(alias)
            if mix:
                return mix
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


def get_market_provider_profile(provider):
    normalized_provider = normalize_identifier(provider)
    if not normalized_provider:
        return None
    candidates = []
    for profile in load_market_models():
        if normalize_identifier(profile.get("provider")) != normalized_provider:
            continue
        candidates.append(dict(profile))
    if not candidates:
        return None

    def sort_key(profile):
        market_status = str(profile.get("market_status", "")).strip().lower()
        serving_mode = str(profile.get("serving_mode", "")).strip().lower()
        if market_status == "api" or serving_mode == "closed":
            return (0, profile.get("model_id", ""))
        if market_status == "api_and_open_weight":
            return (1, profile.get("model_id", ""))
        return (2, profile.get("model_id", ""))

    candidates.sort(key=sort_key)
    provider_profile = candidates[0]
    provider_profile["matching_strategy"] = "provider_market_family"
    return provider_profile


def resolve_inference_country_mix(payload):
    explicit_country = payload.get("country")
    explicit_mix = get_country_mix(explicit_country) if explicit_country else None
    market_profile = get_market_model_profile(payload.get("model_id"))
    if not market_profile:
        market_profile = get_market_provider_profile(payload.get("provider"))
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


def select_nearest_parameter_anchors(anchors, target_params):
    if not anchors:
        return [], "none"
    if target_params in (None, 0):
        return anchors, "average_all"

    anchors_with_params = []
    anchors_without_params = []
    for anchor in anchors:
        source_params = anchor.get("source_params")
        if source_params in (None, 0):
            anchors_without_params.append(anchor)
            continue
        distance = abs((target_params - source_params) / source_params)
        enriched = dict(anchor)
        enriched["parameter_distance"] = distance
        anchors_with_params.append(enriched)

    if not anchors_with_params:
        return anchors, "average_all"

    anchors_with_params.sort(key=lambda item: item["parameter_distance"])
    nearest = anchors_with_params[0]
    return [nearest], "nearest_parameter_anchor"


def select_primary_inference_methods(methods, market_profile):
    if not methods:
        return [], "none"

    method_by_id = {method.get("method_id"): method for method in methods}
    market_status = str((market_profile or {}).get("market_status", "")).strip().lower()
    serving_mode = str((market_profile or {}).get("serving_mode", "")).strip().lower()

    if market_status == "api" or serving_mode == "closed":
        if method_by_id.get("prompt_average"):
            return [method_by_id["prompt_average"]], "primary_prompt_only"

    if market_status in {"open_weight", "api_and_open_weight"} or serving_mode in {"open", "hybrid"}:
        if method_by_id.get("page_average"):
            return [method_by_id["page_average"]], "primary_page_only"

    return methods, "average_all_methods"


def build_empirical_unit_conversions(records):
    conversions = {}
    by_model = {}
    for record in records:
        if record.get("phase") != "inference":
            continue
        model = record.get("llm_normalized")
        if not model or model == "n.d.":
            continue
        by_model.setdefault(model, []).append(record)

    for model, rows in by_model.items():
        prompt_energy = next((r for r in rows if r.get("metric_name") == "prompt_energy"), None)
        page_energy = next((r for r in rows if r.get("metric_name") == "page_generation_energy"), None)
        if not prompt_energy or not page_energy:
            continue
        try:
            prompt_wh = to_float(prompt_energy.get("metric_value"), default=None)
            page_wh = to_float(page_energy.get("metric_value"), default=None) * 1000.0
        except ValueError:
            continue
        if prompt_wh in (None, 0) or page_wh in (None, 0):
            continue
        conversions[model] = {
            "page_per_prompt_energy_ratio": page_wh / prompt_wh,
            "prompt_per_page_energy_ratio": prompt_wh / page_wh,
            "prompt_record_id": prompt_energy.get("record_id"),
            "page_record_id": page_energy.get("record_id"),
        }
    return conversions


def build_energy_inference_anchors(records):
    anchors = []
    for record in records:
        if record.get("phase") != "inference":
            continue
        metric_name = record.get("metric_name")
        if metric_name not in {"prompt_energy", "page_generation_energy", "query_energy"}:
            continue
        source_params_raw = str(record.get("model_parameters_normalized", "") or "").replace("B", "").strip()
        if source_params_raw.lower() in {"", "n.d.", "nd", "na", "n/a"}:
            continue
        source_params = to_float(source_params_raw, default=None)
        if source_params in (None, 0):
            continue
        energy_wh = None
        unit = str(record.get("metric_unit", "") or "")
        if metric_name == "prompt_energy":
            energy_wh = to_float(record.get("metric_value"), default=None)
        elif metric_name == "page_generation_energy":
            energy_wh = to_float(record.get("metric_value"), default=None)
            if energy_wh is not None:
                energy_wh *= 1000.0
        elif metric_name == "query_energy":
            energy_wh = to_float(record.get("metric_value"), default=None)
        if energy_wh in (None, 0):
            continue
        anchors.append(
            {
                "record_id": record.get("record_id"),
                "source_model": record.get("llm_normalized") or record.get("model_or_scope"),
                "source_country": record.get("country_normalized") or "Non spécifié",
                "source_params": source_params,
                "source_energy_wh": energy_wh,
                "metric_name": metric_name,
                "metric_unit": unit,
                "source_energy": format_literature_metric(record.get("metric_value"), unit),
            }
        )
    return anchors


def get_anchor_family(metric_name):
    if metric_name in {"prompt_energy", "query_energy"}:
        return "prompt_query"
    if metric_name == "page_generation_energy":
        return "page"
    return "other"


def select_nearest_energy_anchors(anchors, target_params, limit=2):
    if not anchors:
        return []
    if target_params in (None, 0):
        return anchors[:limit]
    enriched = []
    for anchor in anchors:
        source_params = anchor.get("source_params")
        if source_params in (None, 0):
            continue
        distance = abs((target_params - source_params) / source_params)
        entry = dict(anchor)
        entry["parameter_distance"] = distance
        enriched.append(entry)
    enriched.sort(key=lambda item: item["parameter_distance"])
    return enriched[:limit]


def build_inference_method_set(records, payload):
    request_type = payload.get("request_type", "chat_generation")
    input_tokens = to_float(payload.get("input_tokens", 0.0), default=0.0)
    output_tokens = to_float(payload.get("output_tokens", 0.0), default=0.0)
    requests_per_feature = to_float(payload.get("requests_per_feature", 1.0), default=1.0)
    feature_uses_per_month = to_float(payload.get("feature_uses_per_month", 0.0), default=0.0)
    months_per_year = to_float(payload.get("months_per_year", 12.0), default=12.0)
    annual_feature_uses = feature_uses_per_month * months_per_year
    annual_requests = annual_feature_uses * requests_per_feature
    total_tokens = input_tokens + output_tokens
    prompt_token_ratio = compute_token_ratio(input_tokens, output_tokens)
    reference_page_tokens = REFERENCE_PAGE_TOKENS
    page_method_applicable = bool(payload.get("page_method_applicable", False))
    parser_page_equivalent = to_float(payload.get("output_page_equivalents_per_request", 0.0), default=0.0)
    if parser_page_equivalent > 0:
        pages_per_request_equivalent = parser_page_equivalent
        token_source_note = "parser_page_equivalent"
    elif output_tokens > 0:
        pages_per_request_equivalent = output_tokens / reference_page_tokens
        token_source_note = "output_tokens"
    else:
        pages_per_request_equivalent = 0.0
        token_source_note = "default_tokens"
    annual_page_equivalents = annual_requests * pages_per_request_equivalent
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
    standard_request = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    selected_factors = []
    assumptions = [
        "Inference-only estimate: training and software-system overheads excluded",
        f"Request type classified as {request_type}",
        f"{requests_per_feature} LLM request(s) per feature use",
        f"{annual_feature_uses} feature uses per year",
        "Energy is treated as the primary quantity; carbon and water are derived from the electricity mix of the retained country",
    ]
    if target_params is not None:
        assumptions.append(f"Model-size scaling enabled with target profile at {target_params:g}B active parameters")
    else:
        assumptions.append("No parameter count available for the target model; parametric scaling disabled")
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
    energy_anchors = build_energy_inference_anchors(records)
    methods = []
    family_groups = {"prompt_query": [], "page": []}
    for anchor in energy_anchors:
        family = get_anchor_family(anchor.get("metric_name"))
        if family in family_groups:
            family_groups[family].append(anchor)

    if family_groups["prompt_query"]:
        assumptions.append(
            f"A prompt/query proxy is calibrated from {len(family_groups['prompt_query'])} prompt/query anchor(s), then adjusted by a token ratio relative to {int(REFERENCE_PROMPT_TOKENS)} tokens"
        )
    if family_groups["page"]:
        if page_method_applicable:
            assumptions.append("A page calibration is computed from the mean Wh per parameter observed in page-generation inference records")
            assumptions.append(
                f"Page-family annualization uses generated page equivalents, with {reference_page_tokens:g} tokens per reference page when no explicit page count is provided"
            )
        else:
            assumptions.append("Page-family method marked as not applicable for this scenario by the parser")

    exact_market_profile = get_market_model_profile(payload.get("model_id"))
    multifactor_profile = dict(exact_market_profile) if exact_market_profile else None
    if not multifactor_profile and target_params not in (None, 0):
        inferred_serving_mode = "closed" if country_resolution == "publisher_country" else "open"
        multifactor_profile = {
            "model_id": payload.get("model_id") or (model_profile or {}).get("model_id") or "unspecified-model",
            "display_name": payload.get("model_id") or (model_profile or {}).get("model_id") or "unspecified-model",
            "provider": payload.get("provider") or "",
            "active_parameters_billion": target_params,
            "total_parameters_billion": to_float((model_profile or {}).get("total_parameters_billion"), default=target_params) or target_params,
            "serving_mode": inferred_serving_mode,
            "context_window_tokens": payload.get("context_window_tokens") or 131072,
            "vision_support": "no",
            "architecture_notes": "Synthetic fallback profile built from the target model parameter estimate because no exact market-model profile is available.",
            "estimation_country_code": (country_mix or {}).get("country_code") or payload.get("country") or "US",
            "matching_strategy": "synthetic_parameter_profile",
        }

    if multifactor_profile:
        multifactor_proxy = compute_market_screening_proxy(
            multifactor_profile,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests_per_hour=None,
        )
        if multifactor_proxy:
            selected_factors = dedupe(selected_factors + ["elsworth2025_prompt_energy"])
            token_factors = {
                scenario: round(market_token_factor(input_tokens, output_tokens, scenario=scenario), 4)
                for scenario in ("low", "central", "high")
            }
            context_factors = {
                scenario: round(market_context_factor(multifactor_profile.get("context_window_tokens"), scenario=scenario), 4)
                for scenario in ("low", "central", "high")
            }
            serving_factors = {
                scenario: round(market_serving_factor(multifactor_profile.get("serving_mode"), scenario=scenario), 4)
                for scenario in ("low", "central", "high")
            }
            modality_factors = {
                scenario: round(market_modality_factor(multifactor_profile, scenario=scenario), 4)
                for scenario in ("low", "central", "high")
            }
            architecture_factors = {
                scenario: round(market_architecture_factor(multifactor_profile, scenario=scenario), 4)
                for scenario in ("low", "central", "high")
            }
            assumptions = [
                "Inference-only estimate: training and software-system overheads excluded",
                f"Request type classified as {request_type}",
                f"{requests_per_feature} LLM request(s) per feature use",
                f"{annual_feature_uses} feature uses per year",
                "Energy is treated as the primary quantity; carbon is derived from the electricity mix of the retained country",
                "Prompt-energy estimate calibrated on Elsworth et al. (2025) at 0.24 Wh/prompt for Gemini Apps",
                "Weighted prompt compute uses input tokens + 1.8 x output tokens relative to the project reference scenario",
                "Effective active parameters adjust the raw model size with context window, serving mode, modality support, and architecture overhead",
            ]
            if target_params is not None:
                assumptions.append(f"Model-size scaling enabled with target profile at {target_params:g}B active parameters")
            if exact_market_profile:
                assumptions.append(f"Exact market-model profile retained for {multifactor_profile.get('model_id')}")
            else:
                assumptions.append("Exact market-model profile unavailable; synthetic multifactor fallback built from the target parameter estimate")
            if country_mix:
                if country_resolution == "publisher_country":
                    assumptions.append(
                        f"Carbon recalculated with the publisher-country mix for {country_mix.get('country_code')} because the model is treated as a proprietary hosted service"
                    )
                elif country_resolution == "project_country":
                    assumptions.append(
                        f"Carbon recalculated with the project country mix for {country_mix.get('country_code')} because the model is treated as open-weight or self-hosted"
                    )
                else:
                    assumptions.append(
                        f"Country mix fallback applied for {country_mix.get('country_code')} ({country_mix.get('source_citation')})"
                    )

            methods = [
                {
                    "method_id": multifactor_proxy["method_id"],
                    "label": "Proxy prompt multi-facteurs",
                    "basis": "Proxy de screening en énergie par prompt calibré sur Elsworth et al. (2025), puis ajusté par les paramètres actifs effectifs, les hypothèses de service, l’overhead d’architecture et un volume de tokens pondéré.",
                    "record_ids": ["elsworth2025_prompt_energy"],
                    "annual_energy_wh": scale_range(multifactor_proxy["per_request_energy_wh"], annual_requests),
                    "annual_carbon_gco2e": scale_range(multifactor_proxy["per_request_carbon_gco2e"], annual_requests),
                    "annual_water_ml": rounded_range(0.0, 0.0, 0.0),
                    "per_request_energy_wh": multifactor_proxy["per_request_energy_wh"],
                    "per_request_carbon_gco2e": multifactor_proxy["per_request_carbon_gco2e"],
                    "per_request_water_ml": rounded_range(0.0, 0.0, 0.0),
                    "detail": {
                        "kind": "market_multifactor_prompt_proxy",
                        "unit_basis": "Wh/prompt|request",
                        "reference_anchor": multifactor_proxy["reference_anchor"],
                        "standard_request": standard_request,
                        "annual_multiplier": annual_requests,
                        "target_country": payload.get("country"),
                        "target_mix": country_mix,
                        "target_grid_carbon_intensity": grid_carbon_intensity,
                        "target_params": target_params,
                        "effective_active_parameters_billion": multifactor_proxy["effective_active_parameters_billion"],
                        "token_factor": token_factors,
                        "context_factor": context_factors,
                        "serving_factor": serving_factors,
                        "modality_factor": modality_factors,
                        "architecture_factor": architecture_factors,
                        "serving_mode": multifactor_profile.get("serving_mode"),
                        "context_window_tokens": multifactor_profile.get("context_window_tokens"),
                        "vision_support": multifactor_profile.get("vision_support"),
                        "architecture_notes": multifactor_profile.get("architecture_notes"),
                        "matching_strategy": multifactor_profile.get("matching_strategy") or "exact_market_model",
                        "scaling_exponent": {"low": 0.85, "central": 0.95, "high": 1.05},
                    },
                }
            ]
            primary_aggregated = aggregate_method_ranges(methods)
            per_request_aggregate = {
                "energy_wh": methods[0]["per_request_energy_wh"],
                "carbon_gco2e": methods[0]["per_request_carbon_gco2e"],
                "water_ml": rounded_range(0.0, 0.0, 0.0),
            }
            return {
                "scenario_id": payload.get("scenario_id", "inference-estimate"),
                "estimate_level": "inference_feature",
                "method": multifactor_proxy["method_id"],
                "uncertainty_level": "high",
                "applicability_note": "Inference-only screening estimate based on a prompt-energy anchor and a multi-factor market-model proxy.",
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
                    "pages_per_request_equivalent": pages_per_request_equivalent,
                    "annual_page_equivalents": annual_page_equivalents,
                },
                "per_request_llm": per_request_aggregate,
                "per_feature_llm": {
                    "energy_wh": scale_range(per_request_aggregate["energy_wh"], requests_per_feature),
                    "carbon_gco2e": scale_range(per_request_aggregate["carbon_gco2e"], requests_per_feature),
                    "water_ml": rounded_range(0.0, 0.0, 0.0),
                },
                "annual_llm": primary_aggregated,
                "annual_total": primary_aggregated,
                "method_results": methods,
                "primary_method_results": methods,
                "aggregation_strategy": multifactor_proxy["method_id"],
                "selected_factors": selected_factors,
                "assumptions": assumptions,
                "country_energy_mix": country_mix,
                "country_resolution": country_resolution,
                "model_profile": model_profile,
                "market_model_profile": multifactor_profile,
                "extrapolation_rules": [],
                "extrapolation_details": {},
                "software_overhead": {"components": [], "annual_energy_wh": 0.0, "annual_carbon_gco2e": 0.0, "annual_water_ml": 0.0},
            }

    for family_name, family_anchors in family_groups.items():
        if not family_anchors:
            continue
        if family_name == "page" and not page_method_applicable:
            continue
        selected_factors.extend([anchor["record_id"] for anchor in family_anchors])
        intensities = [anchor["source_energy_wh"] / anchor["source_params"] for anchor in family_anchors if anchor.get("source_params")]
        if not intensities:
            continue
        mean_intensity = sum(intensities) / len(intensities)
        target_energy_wh = (mean_intensity * target_params) if target_params not in (None, 0) else sum(anchor["source_energy_wh"] for anchor in family_anchors) / len(family_anchors)
        if family_name == "prompt_query":
            target_energy_wh *= prompt_token_ratio
        target_carbon = wh_to_gco2e(target_energy_wh, grid_carbon_intensity) if grid_carbon_intensity is not None else 0.0
        target_water = wh_to_liters(target_energy_wh, water_intensity) * 1000.0 if water_intensity is not None else 0.0
        annual_multiplier = annual_requests if family_name == "prompt_query" else annual_page_equivalents
        scaled_family_anchors = []
        for anchor in family_anchors:
            parameter_factor = (target_params / anchor["source_params"]) if (target_params not in (None, 0) and anchor.get("source_params") not in (None, 0)) else 1.0
            token_factor = prompt_token_ratio if family_name == "prompt_query" else 1.0
            anchor_energy = anchor["source_energy_wh"] * parameter_factor * token_factor
            anchor_carbon = wh_to_gco2e(anchor_energy, grid_carbon_intensity) if grid_carbon_intensity is not None else 0.0
            anchor_water = wh_to_liters(anchor_energy, water_intensity) * 1000.0 if water_intensity is not None else 0.0
            scaled_family_anchors.append(
                {
                    **anchor,
                    "target_params": target_params,
                    "parameter_factor": parameter_factor,
                    "token_factor": token_factor,
                    "per_request_energy": rounded_range(anchor_energy, anchor_energy, anchor_energy),
                    "per_request_carbon": rounded_range(anchor_carbon, anchor_carbon, anchor_carbon),
                    "per_request_water": rounded_range(anchor_water, anchor_water, anchor_water),
                    "source_carbon_intensity": None,
                    "source_water_intensity": None,
                }
            )
        method_id = "prompt_query_average" if family_name == "prompt_query" else "page_average"
        methods.append(
            {
                "method_id": method_id,
                "label": "Proxy Wh/prompt|requête" if family_name == "prompt_query" else "Proxy Wh/page",
                "basis": "Proxy paramétrique Wh/prompt|requête calibré sur les ancrages de la littérature, avec ajustement simple au volume de tokens." if family_name == "prompt_query" else "Proxy paramétrique Wh/page calibré sur les ancrages de génération de pages de la littérature.",
                "record_ids": [anchor["record_id"] for anchor in family_anchors],
                "annual_energy_wh": scale_range(rounded_range(target_energy_wh, target_energy_wh, target_energy_wh), annual_multiplier),
                "annual_carbon_gco2e": scale_range(rounded_range(target_carbon, target_carbon, target_carbon), annual_multiplier),
                "annual_water_ml": scale_range(rounded_range(target_water, target_water, target_water), annual_multiplier),
                "per_request_energy_wh": rounded_range(target_energy_wh, target_energy_wh, target_energy_wh),
                "per_request_carbon_gco2e": rounded_range(target_carbon, target_carbon, target_carbon),
                "per_request_water_ml": rounded_range(target_water, target_water, target_water),
                "detail": {
                    "kind": "wh_parameter_model",
                    "unit_basis": "Wh/prompt|requête" if family_name == "prompt_query" else "Wh/page",
                    "anchors": scaled_family_anchors,
                    "target_country": payload.get("country"),
                    "target_mix": country_mix,
                    "target_grid_carbon_intensity": grid_carbon_intensity,
                    "target_water_intensity": water_intensity,
                    "standard_request": standard_request,
                    "aggregation": "mean_wh_per_parameter",
                    "family": family_name,
                    "family_anchor_count": len(family_anchors),
                    "mean_intensity_wh_per_billion": mean_intensity,
                    "prompt_token_ratio": prompt_token_ratio if family_name == "prompt_query" else None,
                    "reference_page_tokens": reference_page_tokens,
                    "pages_per_request_equivalent": pages_per_request_equivalent,
                    "annual_page_equivalents": annual_page_equivalents,
                    "annual_multiplier": annual_multiplier,
                    "token_source_note": token_source_note,
                },
            }
        )

    primary_methods = methods
    prompt_query_methods = [method for method in methods if method.get("method_id") == "prompt_query_average"]
    if prompt_query_methods:
        primary_methods = prompt_query_methods
    primary_aggregated = aggregate_method_ranges(primary_methods)
    per_request_aggregate = {
        "energy_wh": rounded_range(
            primary_aggregated["energy_wh"]["low"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["energy_wh"]["central"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["energy_wh"]["high"] / annual_requests if annual_requests else 0.0,
        ),
        "carbon_gco2e": rounded_range(
            primary_aggregated["carbon_gco2e"]["low"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["carbon_gco2e"]["central"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["carbon_gco2e"]["high"] / annual_requests if annual_requests else 0.0,
        ),
        "water_ml": rounded_range(
            primary_aggregated["water_ml"]["low"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["water_ml"]["central"] / annual_requests if annual_requests else 0.0,
            primary_aggregated["water_ml"]["high"] / annual_requests if annual_requests else 0.0,
        ),
    }

    return {
        "scenario_id": payload.get("scenario_id", "inference-estimate"),
        "estimate_level": "inference_feature",
        "method": "wh_parameter_model",
        "uncertainty_level": "high",
        "applicability_note": "Inference-only screening estimate based on Wh calibration anchors from the literature and parametric scaling by model size.",
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
            "pages_per_request_equivalent": pages_per_request_equivalent,
            "annual_page_equivalents": annual_page_equivalents,
        },
        "per_request_llm": per_request_aggregate,
        "per_feature_llm": {
            "energy_wh": scale_range(per_request_aggregate["energy_wh"], requests_per_feature),
            "carbon_gco2e": scale_range(per_request_aggregate["carbon_gco2e"], requests_per_feature),
            "water_ml": scale_range(per_request_aggregate["water_ml"], requests_per_feature),
        },
        "annual_llm": primary_aggregated,
        "annual_total": primary_aggregated,
        "method_results": methods,
        "primary_method_results": primary_methods,
        "aggregation_strategy": "wh_parameter_model",
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
    for row in load_market_models():
        active_parameters = to_float(row.get("active_parameters_billion"), default=None)
        payload = {
            "scenario_id": f"market-{row.get('model_id')}",
            "provider": row.get("provider"),
            "model_id": row.get("model_id"),
            "request_type": "chat_generation",
            "input_tokens": MARKET_REFERENCE_INPUT_TOKENS,
            "output_tokens": MARKET_REFERENCE_OUTPUT_TOKENS,
            "page_method_applicable": True,
            "output_page_equivalents_per_request": MARKET_REFERENCE_OUTPUT_TOKENS / REFERENCE_PAGE_TOKENS,
            "requests_per_feature": 1.0,
            "feature_uses_per_month": MARKET_REFERENCE_REQUESTS_PER_HOUR,
            "months_per_year": 1.0,
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
                    "input_tokens": MARKET_REFERENCE_INPUT_TOKENS,
                    "output_tokens": MARKET_REFERENCE_OUTPUT_TOKENS,
                    "requests_per_hour": MARKET_REFERENCE_REQUESTS_PER_HOUR,
                    "reading_words_per_minute": MARKET_REFERENCE_READING_WORDS_PER_MINUTE,
                    "words_per_token": MARKET_REFERENCE_WORDS_PER_TOKEN,
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
                "method_results_by_id": {method.get("method_id"): method for method in estimate.get("method_results", [])},
                "country_energy_mix": estimate.get("country_energy_mix") or {},
                "selected_factors": estimate.get("selected_factors", []),
                "assumptions": estimate.get("assumptions", []),
                "method_results": estimate.get("method_results", []),
            }
        )
    return predictions


def parse_market_bool(value):
    return normalize_identifier(value) in {"yes", "true", "1"}


def market_context_factor(context_window_tokens, scenario="central"):
    context_tokens = to_float(context_window_tokens, default=131072.0)
    coefficients = {
        "low": 0.02,
        "central": 0.035,
        "high": 0.05,
    }
    coefficient = coefficients.get(scenario, coefficients["central"])
    ratio = max(context_tokens, 32768.0) / 32768.0
    return clamp(1.0 + (coefficient * math.log(ratio, 2)), 0.95, 1.3)


def market_serving_factor(serving_mode, scenario="central"):
    normalized = normalize_identifier(serving_mode)
    factors = {
        "open": {"low": 1.0, "central": 1.0, "high": 1.02},
        "hybrid": {"low": 1.03, "central": 1.07, "high": 1.1},
        "closed": {"low": 1.08, "central": 1.14, "high": 1.2},
    }
    bucket = factors.get(normalized, {"low": 1.02, "central": 1.05, "high": 1.1})
    return bucket.get(scenario, bucket["central"])


def market_modality_factor(row, scenario="central"):
    if not parse_market_bool(row.get("vision_support")):
        return 1.0
    factors = {
        "low": 1.01,
        "central": 1.03,
        "high": 1.06,
    }
    return factors.get(scenario, factors["central"])


def market_architecture_factor(row, scenario="central"):
    active = to_float(row.get("active_parameters_billion"), default=None)
    total = to_float(row.get("total_parameters_billion"), default=active)
    if active in (None, 0):
        return 1.0
    total = total or active
    ratio = max(total / active, 1.0)
    coefficients = {
        "low": 0.04,
        "central": 0.08,
        "high": 0.12,
    }
    coefficient = coefficients.get(scenario, coefficients["central"])
    multiplier = 1.0 + (coefficient * math.log(ratio, 2)) if ratio > 1.0 else 1.0
    notes = str(row.get("architecture_notes", "") or "").lower()
    if "reasoning" in notes:
        reasoning_factors = {
            "low": 1.02,
            "central": 1.05,
            "high": 1.08,
        }
        multiplier *= reasoning_factors.get(scenario, reasoning_factors["central"])
    return multiplier


def market_token_factor(input_tokens, output_tokens, scenario="central"):
    input_value = to_float(input_tokens, default=0.0)
    output_value = to_float(output_tokens, default=0.0)
    compute_tokens = input_value + (MARKET_PROMPT_OUTPUT_WEIGHT * output_value)
    if compute_tokens <= 0:
        compute_tokens = MARKET_REFERENCE_COMPUTE_TOKENS
    exponents = {
        "low": 0.85,
        "central": 0.92,
        "high": 1.0,
    }
    exponent = exponents.get(scenario, exponents["central"])
    return (compute_tokens / MARKET_REFERENCE_COMPUTE_TOKENS) ** exponent


def compute_market_screening_proxy(row, input_tokens=None, output_tokens=None, requests_per_hour=None):
    active = to_float(row.get("active_parameters_billion"), default=None)
    if active in (None, 0):
        return None

    if input_tokens is None:
        input_tokens = MARKET_REFERENCE_INPUT_TOKENS
    if output_tokens is None:
        output_tokens = MARKET_REFERENCE_OUTPUT_TOKENS
    if requests_per_hour is None:
        requests_per_hour = MARKET_REFERENCE_REQUESTS_PER_HOUR

    country_mix = get_country_mix(row.get("estimation_country_code"))
    grid_carbon_intensity = to_float((country_mix or {}).get("grid_carbon_intensity_gco2_per_kwh"), default=None)

    ranges = {}
    effective_params = {}
    for scenario, exponent in (("low", 0.85), ("central", 0.95), ("high", 1.05)):
        effective_active = active
        effective_active *= market_context_factor(row.get("context_window_tokens"), scenario=scenario)
        effective_active *= market_serving_factor(row.get("serving_mode"), scenario=scenario)
        effective_active *= market_modality_factor(row, scenario=scenario)
        effective_active *= market_architecture_factor(row, scenario=scenario)
        effective_params[scenario] = round(effective_active, 4)
        token_factor = market_token_factor(input_tokens, output_tokens, scenario=scenario)
        energy = MARKET_PROMPT_ANCHOR_ENERGY_WH * ((effective_active / MARKET_PROMPT_ANCHOR_ACTIVE_PARAMS_B) ** exponent) * token_factor
        carbon = wh_to_gco2e(energy, grid_carbon_intensity) if grid_carbon_intensity is not None else None
        ranges[scenario] = {
            "per_request_energy_wh": round(energy, 4),
            "per_request_carbon_gco2e": round(carbon, 4) if carbon is not None else None,
            "per_hour_energy_wh": round(energy * requests_per_hour, 4),
            "per_hour_carbon_gco2e": round(carbon * requests_per_hour, 4) if carbon is not None else None,
        }

    effective_values = list(effective_params.values())
    request_energy_values = [ranges[item]["per_request_energy_wh"] for item in ("low", "central", "high")]
    request_carbon_values = [ranges[item]["per_request_carbon_gco2e"] or 0.0 for item in ("low", "central", "high")]
    hour_energy_values = [ranges[item]["per_hour_energy_wh"] for item in ("low", "central", "high")]
    hour_carbon_values = [ranges[item]["per_hour_carbon_gco2e"] or 0.0 for item in ("low", "central", "high")]

    return {
        "method_id": "market_multifactor_prompt_proxy_v1",
        "method_label": "Multi-factor prompt proxy calibrated on production prompt energy",
        "reference_anchor": "Elsworth et al. (2025) Gemini Apps median prompt = 0.24 Wh/prompt",
        "standard_input_tokens": input_tokens,
        "standard_output_tokens": output_tokens,
        "standard_requests_per_hour": requests_per_hour,
        "effective_active_parameters_billion": rounded_range(
            min(effective_values),
            effective_params["central"],
            max(effective_values),
        ),
        "per_request_energy_wh": rounded_range(
            min(request_energy_values),
            ranges["central"]["per_request_energy_wh"],
            max(request_energy_values),
        ),
        "per_request_carbon_gco2e": rounded_range(
            min(request_carbon_values),
            ranges["central"]["per_request_carbon_gco2e"] or 0.0,
            max(request_carbon_values),
        ),
        "per_hour_energy_wh": rounded_range(
            min(hour_energy_values),
            ranges["central"]["per_hour_energy_wh"],
            max(hour_energy_values),
        ),
        "per_hour_carbon_gco2e": rounded_range(
            min(hour_carbon_values),
            ranges["central"]["per_hour_carbon_gco2e"] or 0.0,
            max(hour_carbon_values),
        ),
        "notes": (
            "Central estimate = prompt-energy anchor scaled by active parameters, context window, serving mode, modality support, "
            "architecture overhead, and reference token volume. Low/high values widen the exponent and overhead assumptions."
        ),
    }


def training_parameter_count_billion(row):
    total = to_float(row.get("total_parameters_billion"), default=None)
    if total not in (None, 0):
        return total
    return to_float(row.get("active_parameters_billion"), default=None)


def training_tokens_estimate_trillion(row):
    explicit = to_float(row.get("training_tokens_estimate_trillion"), default=None)
    if explicit not in (None, 0):
        return explicit
    params = training_parameter_count_billion(row)
    if params in (None, 0):
        return None
    return round((params * TRAINING_REFERENCE_TOKENS_PER_PARAMETER) / 1000.0, 4)


def training_regime_factor(regime, scenario="central"):
    factors = {
        "pretraining": {"low": 0.8, "central": 1.0, "high": 1.2},
        "continued_pretraining": {"low": 0.08, "central": 0.15, "high": 0.3},
        "instruction_tuning": {"low": 0.01, "central": 0.03, "high": 0.08},
        "alignment_or_rl": {"low": 0.02, "central": 0.08, "high": 0.2},
        "unknown": {"low": 0.2, "central": 1.0, "high": 2.5},
    }
    normalized = str(regime or "unknown").strip().lower() or "unknown"
    return factors.get(normalized, factors["unknown"]).get(scenario, factors["unknown"]["central"])


def training_architecture_factor(row, scenario="central"):
    multiplier = 1.0
    notes = str(row.get("architecture_notes", "") or "").lower()
    active = to_float(row.get("active_parameters_billion"), default=0.0)
    total = to_float(row.get("total_parameters_billion"), default=0.0)
    multimodal = parse_market_bool(row.get("training_multimodal"))
    if not multimodal:
        multimodal = parse_market_bool(row.get("vision_support"))

    if "moe" in notes or (active > 0 and total > active * 1.5):
        moe_factors = {
            "low": 0.7,
            "central": 0.9,
            "high": 1.1,
        }
        multiplier *= moe_factors.get(scenario, moe_factors["central"])

    if multimodal:
        multimodal_factors = {
            "low": 1.05,
            "central": 1.15,
            "high": 1.35,
        }
        multiplier *= multimodal_factors.get(scenario, multimodal_factors["central"])

    return multiplier


def training_hardware_factor(hardware_class, scenario="central"):
    factors = {
        "modern_hyperscale_gpu": {"low": 0.8, "central": 0.9, "high": 1.0},
        "mixed_gpu_cluster": {"low": 0.9, "central": 1.0, "high": 1.1},
        "standard_gpu_cluster": {"low": 0.95, "central": 1.05, "high": 1.15},
        "older_or_unknown_cluster": {"low": 1.0, "central": 1.1, "high": 1.25},
        "unknown": {"low": 0.85, "central": 1.0, "high": 1.2},
    }
    normalized = str(hardware_class or "unknown").strip().lower() or "unknown"
    return factors.get(normalized, factors["unknown"]).get(scenario, factors["unknown"]["central"])


def normalize_training_metric_value(record):
    metric_name = record.get("metric_name")
    value = to_float(record.get("metric_value"), default=None)
    unit = str(record.get("metric_unit", "")).strip().lower()
    if value is None:
        return None, None

    if metric_name == "training_energy":
        if unit == "wh":
            return value, "Wh"
        if unit == "kwh":
            return value * 1000.0, "Wh"
        if unit == "mwh":
            return value * 1_000_000.0, "Wh"
        if unit == "gwh":
            return value * 1_000_000_000.0, "Wh"
    if metric_name in {"training_emissions", "creation_lifecycle_emissions"}:
        if "lb" in unit:
            return value * 0.00045359237, "tCO2e"
        if "tco2" in unit:
            return value, "tCO2e"
    if metric_name in {"creation_lifecycle_water", "training_water_total", "training_water_onsite"}:
        if "million liters" in unit:
            return value * 1000.0, "kL"
        if unit == "l":
            return value / 1000.0, "kL"
        if "kl" in unit:
            return value, "kL"
    return None, None


def build_training_token_lookup(records):
    lookup = {}
    for record in records:
        if record.get("metric_name") != "training_tokens":
            continue
        value = to_float(record.get("metric_value"), default=None)
        if value in (None, 0):
            continue
        unit = str(record.get("metric_unit", "") or "").strip().lower()
        if "trillion" in unit:
            normalized_value = value
        elif "billion" in unit:
            normalized_value = value / 1000.0
        elif "million" in unit:
            normalized_value = value / 1_000_000.0
        else:
            continue
        model_key = normalize_identifier(record.get("llm_normalized") or record.get("model_or_scope"))
        if not model_key:
            continue
        rounded_value = round(normalized_value, 4)
        current = lookup.get(model_key)
        if current is None or rounded_value > current:
            lookup[model_key] = rounded_value
    return lookup


def build_training_prediction_anchors(records):
    token_lookup = build_training_token_lookup(records)
    families = {
        "direct_training_energy": [],
        "direct_training_carbon": [],
        "creation_lifecycle_carbon": [],
        "creation_lifecycle_water": [],
    }
    for record in records:
        metric_name = record.get("metric_name")
        phase = record.get("phase")
        if metric_name == "training_energy" and phase == "training":
            family = "direct_training_energy"
        elif metric_name == "training_emissions" and phase == "training":
            family = "direct_training_carbon"
        elif metric_name == "creation_lifecycle_emissions" and phase == "lifecycle":
            family = "creation_lifecycle_carbon"
        elif metric_name == "creation_lifecycle_water" and phase == "lifecycle":
            family = "creation_lifecycle_water"
        else:
            continue

        source_params = parse_parameter_count_billion(record.get("model_parameters_normalized"))
        if source_params in (None, 0):
            continue
        normalized_value, normalized_unit = normalize_training_metric_value(record)
        if normalized_value is None:
            continue
        source_model = record.get("llm_normalized") or record.get("model_or_scope")
        anchor_payload = {
            "record_id": record.get("record_id"),
            "source_model": source_model,
            "source_params": source_params,
            "source_country": record.get("country_normalized") or "Non spécifié",
            "source_value": normalized_value,
            "source_unit": normalized_unit,
            "source_tokens_trillion": token_lookup.get(normalize_identifier(source_model)),
        }
        families[family].append(anchor_payload)

        if family == "direct_training_carbon":
            mix = get_record_country_mix(record)
            grid_carbon = to_float((mix or {}).get("grid_carbon_intensity_gco2_per_kwh"), default=None)
            if grid_carbon not in (None, 0):
                energy_kwh = (normalized_value * 1_000_000.0) / grid_carbon
                families["direct_training_energy"].append(
                    {
                        **anchor_payload,
                        "source_value": energy_kwh * 1000.0,
                        "source_unit": "Wh",
                        "source_country": (mix or {}).get("country_name") or anchor_payload["source_country"],
                    }
                )
    return families


def retain_comparable_training_anchors(anchors, target_params):
    if not anchors or target_params in (None, 0):
        return []

    # Discard anchors that are too far from the target scale to avoid
    # unstable frontier extrapolations from very small training runs.
    min_ratio = 0.05
    max_ratio = 20.0
    min_source_params = max(1.0, target_params * min_ratio)
    max_source_params = target_params * max_ratio

    filtered = [
        anchor
        for anchor in anchors
        if min_source_params <= to_float(anchor.get("source_params"), default=0.0) <= max_source_params
    ]
    if filtered:
        return filtered

    # Fallback to the closest anchors by parameter count if the comparable window is empty.
    ranked = sorted(
        anchors,
        key=lambda anchor: abs(math.log(max(to_float(anchor.get("source_params"), default=1e-9), 1e-9) / target_params)),
    )
    return ranked[: min(3, len(ranked))]


def build_training_market_predictions(records):
    anchors_by_family = build_training_prediction_anchors(records)
    predictions = []
    for row in load_market_models():
        target_params = training_parameter_count_billion(row)
        target_tokens = training_tokens_estimate_trillion(row)
        training_regime = str(row.get("training_regime") or "unknown").strip().lower() or "unknown"
        hardware_class = str(row.get("training_hardware_class_proxy") or "unknown").strip().lower() or "unknown"
        family_results = {}
        selected_factors = []
        proxy_profile = {
            "target_params_billion": round(target_params, 4) if target_params not in (None, 0) else None,
            "target_tokens_trillion": round(target_tokens, 4) if target_tokens not in (None, 0) else None,
            "training_regime": training_regime,
            "training_hardware_class_proxy": hardware_class,
            "factors": {},
        }
        for scenario, alpha, beta in (("low", 0.85, 0.70), ("central", 1.00, 1.00), ("high", 1.15, 1.20)):
            proxy_profile["factors"][scenario] = {
                "parameter_exponent": alpha,
                "token_exponent": beta,
                "regime_factor": round(training_regime_factor(training_regime, scenario=scenario), 4),
                "architecture_factor": round(training_architecture_factor(row, scenario=scenario), 4),
                "hardware_factor": round(training_hardware_factor(hardware_class, scenario=scenario), 4),
            }
        for family_name, anchors in anchors_by_family.items():
            if not anchors or target_params in (None, 0) or target_tokens in (None, 0):
                continue
            retained_anchor_pool = retain_comparable_training_anchors(anchors, target_params)
            if not retained_anchor_pool:
                continue
            scenario_estimates = {"low": [], "central": [], "high": []}
            retained_anchors = []
            for anchor in retained_anchor_pool:
                source_params = to_float(anchor.get("source_params"), default=None)
                if source_params in (None, 0):
                    continue
                source_tokens = to_float(anchor.get("source_tokens_trillion"), default=None)
                if source_tokens in (None, 0):
                    source_tokens = round((source_params * TRAINING_REFERENCE_TOKENS_PER_PARAMETER) / 1000.0, 4)
                if source_tokens in (None, 0):
                    continue
                retained_anchors.append({**anchor, "source_tokens_trillion": source_tokens})
                for scenario, alpha, beta in (("low", 0.85, 0.70), ("central", 1.00, 1.00), ("high", 1.15, 1.20)):
                    profile = proxy_profile["factors"][scenario]
                    estimate = anchor["source_value"]
                    estimate *= (target_params / source_params) ** alpha
                    estimate *= (target_tokens / source_tokens) ** beta
                    estimate *= profile["regime_factor"]
                    estimate *= profile["architecture_factor"]
                    estimate *= profile["hardware_factor"]
                    scenario_estimates[scenario].append(estimate)

            if not retained_anchors:
                continue
            central_estimate = statistics.median(scenario_estimates["central"])
            low_estimate = statistics.median(scenario_estimates["low"])
            high_estimate = statistics.median(scenario_estimates["high"])
            range_low = min(low_estimate, central_estimate, high_estimate)
            range_high = max(low_estimate, central_estimate, high_estimate)
            family_results[family_name] = {
                "value": central_estimate,
                "unit": "Wh" if family_name == "direct_training_energy" else ("tCO2e" if "carbon" in family_name else "kL"),
                "anchors": retained_anchors,
                "range": rounded_range(range_low, central_estimate, range_high),
                "method_id": "training_multifactor_proxy_v1",
                "method_label": "Multi-factor training proxy calibrated on literature training anchors",
                "reference_token_ratio": TRAINING_REFERENCE_TOKENS_PER_PARAMETER,
                "target_params": round(target_params, 4),
                "target_tokens_trillion": round(target_tokens, 4),
                "training_regime": training_regime,
                "training_hardware_class_proxy": hardware_class,
                "notes": (
                    "Central estimate = median of comparable literature training anchors scaled by retained parameter count, "
                    "training-token prior, training regime, architecture profile, and hardware-class proxy. Low/high values "
                    "widen the parameter and token exponents together with contextual factors."
                ),
            }
            selected_factors.extend(anchor["record_id"] for anchor in retained_anchors)
        predictions.append(
            {
                **row,
                "training_results_by_id": family_results,
                "selected_training_factors": dedupe(selected_factors),
                "training_proxy_profile": proxy_profile,
            }
        )
    return predictions


def compute_token_ratio(input_tokens, output_tokens):
    total_tokens = to_float(input_tokens, default=0.0) + to_float(output_tokens, default=0.0)
    if total_tokens <= 0:
        total_tokens = REFERENCE_PROMPT_TOKENS
    return clamp(total_tokens / REFERENCE_PROMPT_TOKENS, 0.25, 6.0)


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
