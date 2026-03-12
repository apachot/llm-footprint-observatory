#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "records.csv"
METADATA_PATH = ROOT / "data" / "record_metadata.csv"
MODELS_PATH = ROOT / "data" / "models.csv"
COUNTRY_MIX_PATH = ROOT / "data" / "country_energy_mix.csv"
EXTRAPOLATION_RULES_PATH = ROOT / "data" / "extrapolation_rules.csv"

REFERENCE_PROMPT_TOKENS = 1550.0


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
            low_value = metric_value * (ratio ** low_exp) * token_ratio * requests_count * scale_factor
            central_value = metric_value * (ratio ** central_exp) * token_ratio * requests_count * scale_factor
            high_value = metric_value * (ratio ** high_exp) * token_ratio * requests_count * scale_factor
            anchor_values.append((low_value, central_value, high_value))
            selected_record_ids.append(record["record_id"])
        if not anchor_values:
            continue
        lows = [item[0] for item in anchor_values]
        centrals = [item[1] for item in anchor_values]
        highs = [item[2] for item in anchor_values]
        results[unit_kind] = rounded_range(min(lows), sum(centrals) / len(centrals), max(highs))
        selected_factors.extend(selected_record_ids)

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
    }


def estimate_externalities(records, payload):
    request_type = payload.get("request_type", "chat_generation")
    requests_count = to_float(payload.get("requests_count", 1.0), default=1.0)
    input_tokens = to_float(payload.get("input_tokens", 0.0), default=0.0)
    output_tokens = to_float(payload.get("output_tokens", 0.0), default=0.0)
    country_mix = get_country_mix(payload.get("country"))
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
            "extrapolation_rules": extrapolated["rule_ids"],
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
        "extrapolation_rules": [],
    }


def estimate_feature_externalities(records, payload):
    per_request_payload = {
        "scenario_id": payload.get("scenario_id", "feature-estimate"),
        "provider": payload.get("provider"),
        "model_id": payload.get("model_id"),
        "estimated_active_parameters_billion": payload.get("estimated_active_parameters_billion"),
        "deployment_mode": payload.get("deployment_mode"),
        "request_type": payload.get("request_type", "chat_generation"),
        "input_tokens": payload.get("input_tokens", 0.0),
        "output_tokens": payload.get("output_tokens", 0.0),
        "requests_count": 1,
        "country": payload.get("country"),
        "grid_carbon_intensity_gco2_per_kwh": payload.get("grid_carbon_intensity_gco2_per_kwh"),
        "water_intensity_l_per_kwh": payload.get("water_intensity_l_per_kwh"),
    }
    per_request = estimate_externalities(records, per_request_payload)

    requests_per_feature = to_float(payload.get("requests_per_feature", 1.0), default=1.0)
    feature_uses_per_month = to_float(payload.get("feature_uses_per_month", 0.0), default=0.0)
    months_per_year = to_float(payload.get("months_per_year", 12.0), default=12.0)
    software_components = payload.get("software_components", [])
    annual_feature_uses = feature_uses_per_month * months_per_year
    llm_requests_per_year = annual_feature_uses * requests_per_feature

    llm_energy_per_request = per_request["results"].get("energy_wh", {"low": 0.0, "central": 0.0, "high": 0.0})
    llm_carbon_per_request = per_request["results"].get("carbon_gco2e", {"low": 0.0, "central": 0.0, "high": 0.0})
    llm_water_per_request = per_request["results"].get("water_ml", {"low": 0.0, "central": 0.0, "high": 0.0})

    llm_energy_per_feature = scale_range(llm_energy_per_request, requests_per_feature)
    llm_carbon_per_feature = scale_range(llm_carbon_per_request, requests_per_feature)
    llm_water_per_feature = scale_range(llm_water_per_request, requests_per_feature)

    annual_llm_energy = scale_range(llm_energy_per_feature, annual_feature_uses)
    annual_llm_carbon = scale_range(llm_carbon_per_feature, annual_feature_uses)
    annual_llm_water = scale_range(llm_water_per_feature, annual_feature_uses)

    grid_carbon_intensity = to_float(payload.get("grid_carbon_intensity_gco2_per_kwh"), default=None)
    water_intensity = to_float(payload.get("water_intensity_l_per_kwh"), default=None)
    software_breakdown = build_software_breakdown(software_components, annual_feature_uses, grid_carbon_intensity, water_intensity)
    overhead_energy_annual = software_breakdown["annual_energy_wh"]
    overhead_carbon_annual = software_breakdown["annual_carbon_gco2e"]
    overhead_water_annual = software_breakdown["annual_water_ml"]

    total_energy_annual = add_scalar_to_range(annual_llm_energy, overhead_energy_annual)
    total_carbon_annual = (
        add_scalar_to_range(annual_llm_carbon, overhead_carbon_annual)
        if overhead_carbon_annual is not None
        else annual_llm_carbon
    )
    total_water_annual = (
        add_scalar_to_range(annual_llm_water, overhead_water_annual)
        if overhead_water_annual is not None
        else annual_llm_water
    )

    assumptions = list(per_request["assumptions"])
    assumptions.append(f"{requests_per_feature} LLM request(s) per feature use")
    assumptions.append(f"{annual_feature_uses} feature uses per year")
    if software_components:
        assumptions.append("Software-system overhead split across explicit technical components")

    return {
        "scenario_id": payload.get("scenario_id", "feature-estimate"),
        "estimate_level": "feature",
        "feature_scope": {
            "requests_per_feature": requests_per_feature,
            "feature_uses_per_month": feature_uses_per_month,
            "months_per_year": months_per_year,
            "annual_feature_uses": annual_feature_uses,
            "annual_llm_requests": llm_requests_per_year,
        },
        "per_request_llm": per_request["results"],
        "per_feature_llm": {
            "energy_wh": llm_energy_per_feature,
            "carbon_gco2e": llm_carbon_per_feature,
            "water_ml": llm_water_per_feature,
        },
        "annual_llm": {
            "energy_wh": annual_llm_energy,
            "carbon_gco2e": annual_llm_carbon,
            "water_ml": annual_llm_water,
        },
        "annual_total": {
            "energy_wh": total_energy_annual,
            "carbon_gco2e": total_carbon_annual,
            "water_ml": total_water_annual,
        },
        "software_overhead": software_breakdown,
        "selected_factors": per_request["selected_factors"],
        "assumptions": assumptions,
        "uncertainty_level": "high",
        "applicability_note": "Feature-level annualization for screening and eco-design, not an audited footprint statement.",
        "method": per_request.get("method", "literature_proxy"),
        "model_profile": per_request.get("model_profile"),
        "country_energy_mix": per_request.get("country_energy_mix"),
        "extrapolation_rules": per_request.get("extrapolation_rules", []),
    }


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
