#!/usr/bin/env python3
"""Shared market-catalog transparency, quantification, and donor-prior helpers."""

import re
import statistics
from urllib.parse import urlparse
from datetime import date


THIRD_PARTY_ESTIMATE_DOMAINS = {
    "artificialanalysis.ai",
    "explodingtopics.com",
    "lifearchitect.ai",
    "nexos.ai",
}

STRICT_PARAMETER_AUDITS = {
    "exact_in_cited_source",
    "derivable_from_cited_source",
    "third_party_estimate",
}

PARTIAL_DATA_PARAMETER_AUDITS = {"partial_data_prior"}
LEGACY_PROXY_PARAMETER_AUDITS = {"project_proxy_no_url", "internal_proxy_no_url"}
QUANTIFIED_PARAMETER_AUDITS = STRICT_PARAMETER_AUDITS | PARTIAL_DATA_PARAMETER_AUDITS

PARTIAL_SINGLE_DONOR_LOW_FACTOR = 0.70
PARTIAL_SINGLE_DONOR_HIGH_FACTOR = 1.30
PARTIAL_MULTI_DONOR_LOW_PADDING = 0.85
PARTIAL_MULTI_DONOR_HIGH_PADDING = 1.15


def parameter_source_domain(url):
    if not url:
        return ""
    return urlparse(str(url)).netloc.replace("www.", "")


def _normalize_text(value):
    text = str(value or "").lower()
    text = text.replace("/", " ")
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = text.replace("4o-mini", "4o mini")
    text = text.replace("flash-lite", "flash lite")
    text = text.replace("gpt-4o", "gpt 4o")
    return re.sub(r"\s+", " ", text).strip()


def _row_text(row):
    parameter_source = str(row.get("parameter_source", "") or "")
    include_source_text = not parameter_source.startswith("Partial-data donor prior")
    return _normalize_text(
        " ".join(
            str(row.get(key, "") or "")
            for key in (
                "model_id",
                "display_name",
                "architecture_notes",
                "modalities_source",
            )
        )
        + (" " + parameter_source if include_source_text else "")
        + (" " + str(row.get("notes", "") or "") if include_source_text else "")
    )


def _has_phrase(text, phrase):
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def _to_float(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _format_decimal(value):
    if value in (None, ""):
        return ""
    rounded = round(float(value), 4)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _format_billions(value):
    if value in (None, ""):
        return "n.d."
    rounded = round(float(value), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}B"
    return f"{rounded:.1f}B"


def _parse_market_bool(value):
    return _normalize_text(value) in {"yes", "true", "1"}


def _parse_release_date(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        year, month, day = raw.split("-")
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def classify_market_parameter_source(row):
    model_id = str(row.get("model_id", "") or "")
    status = str(row.get("parameter_value_status", "") or "")
    source = str(row.get("parameter_source", "") or "")
    url = str(row.get("parameter_source_url", "") or "")
    notes = str(row.get("notes", "") or "").lower()
    url_domain = parameter_source_domain(url)

    if source.startswith("Partial-data donor prior"):
        return (
            "partial_data_prior",
            "The retained parameter value is a catalog-level partial-data prior derived from source-linked donor models sharing documented metadata.",
        )

    if model_id == "grok-1":
        return (
            "derivable_from_cited_source",
            "The cited xAI post publishes 314B total parameters and says 25% of weights are active per token; the retained 78.5B active value is derived from that ratio.",
        )

    if model_id == "lamda-1" and "research.google/pubs/lamda-language-models-for-dialog-applications/" in url:
        return (
            "exact_in_cited_source",
            "The cited Google Research summary explicitly states that the LaMDA family has up to 137B parameters.",
        )

    if status in {"observed", "documented"}:
        return (
            "exact_in_cited_source",
            "The catalog marks this parameter value as observed/documented and points to a source expected to publish it directly.",
        )

    if model_id == "grok-2" and url_domain == "huggingface.co":
        return (
            "third_party_estimate",
            "The retained value comes from a third-party/open-weight discussion rather than a provider-published parameter disclosure.",
        )

    if url_domain in THIRD_PARTY_ESTIMATE_DOMAINS:
        return (
            "third_party_estimate",
            "The retained parameter value is estimated from a third-party taxonomy or report, not from the provider's own model page.",
        )

    if source.startswith("Project screening") or source.startswith("Project frontier"):
        if url:
            return (
                "needs_cleanup",
                "A project proxy should not point to a parameter-source URL if the cited page does not literally publish the count.",
            )
        return (
            "project_proxy_no_url",
            "The retained parameter value is an explicit project screening proxy with no literal parameter-source URL attached.",
        )

    if source.startswith("Internal approximate profile"):
        return (
            "internal_proxy_no_url",
            "The retained parameter value is an internal approximation kept for extrapolation and intentionally has no external source URL.",
        )

    if "does not publish" in notes:
        if url:
            return (
                "needs_cleanup",
                "The notes say the provider does not publish the parameter count, so the parameter-source URL should not imply otherwise.",
            )
        return (
            "project_proxy_no_url",
            "The notes explicitly state that the provider does not publish the parameter count; the catalog retains a project proxy with no parameter-source URL.",
        )

    return (
        "manual_review",
        "This row does not match the standard exact / third-party / partial-data / proxy patterns and should be reviewed manually.",
    )


def is_market_model_strict_ready(row):
    status = str(row.get("parameter_source_audit", "") or "")
    if not status:
        status, _ = classify_market_parameter_source(row)
    return status in STRICT_PARAMETER_AUDITS


def is_market_model_quantified(row):
    status = str(row.get("parameter_source_audit", "") or "")
    if not status:
        status, _ = classify_market_parameter_source(row)
    return status in QUANTIFIED_PARAMETER_AUDITS


def _series_label(row):
    model_id = _normalize_text(row.get("model_id"))
    text = _row_text(row)
    if model_id.startswith("o1"):
        return "o1"
    if model_id.startswith("gpt") or "gpt " in text:
        return "gpt"
    if "claude" in text:
        return "claude"
    if "gemini" in text:
        return "gemini"
    if "grok" in text:
        return "grok"
    if "mistral" in text or "ministral" in text or "codestral" in text or "devstral" in text:
        return "mistral"
    if "llama" in text:
        return "llama"
    if "qwen" in text:
        return "qwen"
    if "deepseek" in text:
        return "deepseek"
    if "gopher" in text:
        return "gopher"
    if "jurassic" in text:
        return "jurassic"
    if "megatron" in text:
        return "megatron"
    return _normalize_text(row.get("provider"))


def _family_label(row):
    text = _row_text(row)
    model_id = _normalize_text(row.get("model_id"))
    series = _series_label(row)

    if _has_phrase(text, "flash lite") or _has_phrase(text, "tts"):
        return "flash_lite"
    if _has_phrase(text, "haiku"):
        return "haiku"
    if _has_phrase(text, "sonnet"):
        return "sonnet"
    if _has_phrase(text, "opus") or _has_phrase(text, "mythos"):
        return "opus"
    if _has_phrase(text, "4o mini"):
        return "mini"
    if _has_phrase(text, "mini") or _has_phrase(text, "nano"):
        return "mini"
    if _has_phrase(text, "flash") or _has_phrase(text, "fast") or _has_phrase(text, "live"):
        return "flash"
    if _has_phrase(text, "reasoning") or series == "o1":
        return "reasoning"
    if model_id.startswith("gpt-4o") or _has_phrase(text, "gpt 4o"):
        return "4o"
    if "grok 4" in text or model_id.startswith("grok-4"):
        return "grok4"
    if "grok 1.5" in text or model_id.startswith("grok-1.5"):
        return "grok_legacy"
    if series == "gemini" and _has_phrase(text, "pro"):
        return "pro"
    if _has_phrase(text, "large"):
        return "large"
    return ""


def _size_tier(row):
    source = _normalize_text(row.get("parameter_source"))
    text = _row_text(row)
    model_id = _normalize_text(row.get("model_id"))
    series = _series_label(row)
    family = _family_label(row)

    if family == "mini":
        return "compact"
    if family in {"haiku", "flash_lite"} or _has_phrase(source, "small model") or _has_phrase(source, "small reasoning family"):
        return "small"
    if family in {"flash", "sonnet"} or _has_phrase(source, "medium size") or _has_phrase(source, "mid size") or _has_phrase(source, "fast model"):
        return "mid"
    if family in {"pro", "opus", "reasoning", "large", "grok4", "4o"}:
        return "frontier"
    if "frontier" in source:
        return "frontier"
    if series == "gpt" and model_id.startswith("gpt-4o"):
        return "frontier"
    if series == "gpt" and (model_id.startswith("gpt-4") or model_id.startswith("gpt-5")):
        return "frontier"
    if series == "o1":
        return "frontier"
    if series == "grok" and model_id.startswith("grok-4"):
        return "frontier"
    if series == "grok" and model_id.startswith("grok-1.5"):
        return "mid"
    if series == "gemini" and _has_phrase(text, "pro"):
        return "frontier"
    return "mid"


def _context_bucket(row):
    context_tokens = _to_float(row.get("context_window_tokens")) or 0.0
    if context_tokens >= 1_000_000:
        return "xl"
    if context_tokens >= 200_000:
        return "long"
    if context_tokens >= 100_000:
        return "standard"
    return "compact"


def _modalities(row):
    raw = (
        f"{row.get('input_modalities', '')},"
        f"{row.get('output_modalities', '')}"
    )
    return {
        item.strip().lower()
        for item in re.split(r"[,\|]", raw)
        if item and item.strip()
    }


def _build_partial_signature(row):
    text = _row_text(row)
    return {
        "provider": _normalize_text(row.get("provider")),
        "series": _series_label(row),
        "family": _family_label(row),
        "size_tier": _size_tier(row),
        "reasoning": _has_phrase(text, "reasoning") or _series_label(row) == "o1",
        "vision": _parse_market_bool(row.get("vision_support")),
        "market_status": _normalize_text(row.get("market_status")),
        "serving_mode": _normalize_text(row.get("serving_mode")),
        "context_bucket": _context_bucket(row),
        "modalities": _modalities(row),
    }


def _donor_score(target_sig, donor_sig):
    score = 0.0
    if target_sig["provider"] == donor_sig["provider"]:
        score += 40.0
    if target_sig["series"] == donor_sig["series"]:
        score += 20.0
    if target_sig["family"] and target_sig["family"] == donor_sig["family"]:
        score += 60.0
    if target_sig["size_tier"] == donor_sig["size_tier"]:
        score += 20.0
    if target_sig["reasoning"] == donor_sig["reasoning"]:
        score += 8.0
    if target_sig["vision"] == donor_sig["vision"]:
        score += 4.0
    else:
        score -= 2.0
    if target_sig["market_status"] == donor_sig["market_status"]:
        score += 3.0
    else:
        score -= 3.0
    if target_sig["serving_mode"] == donor_sig["serving_mode"]:
        score += 3.0
    else:
        score -= 4.0
    if target_sig["context_bucket"] == donor_sig["context_bucket"]:
        score += 4.0
    score += 2.0 * len(target_sig["modalities"] & donor_sig["modalities"])
    return score


def _sort_donors(target_sig, candidates):
    return sorted(
        candidates,
        key=lambda item: (
            -_donor_score(target_sig, item["__partial_signature"]),
            -(_to_float(item.get("active_parameters_billion")) or 0.0),
            str(item.get("model_id", "") or ""),
        ),
    )


def _candidate_strict_rows(target_row, strict_rows):
    target_date = _parse_release_date(target_row.get("release_date"))
    if not target_date:
        return strict_rows
    earlier_or_same = []
    for row in strict_rows:
        donor_date = _parse_release_date(row.get("release_date"))
        if donor_date and donor_date <= target_date:
            earlier_or_same.append(row)
    return earlier_or_same or strict_rows


def _is_later_than_all_donors(target_row, donors):
    target_date = _parse_release_date(target_row.get("release_date"))
    if not target_date or not donors:
        return False
    donor_dates = [
        _parse_release_date(donor.get("release_date"))
        for donor in donors
        if _parse_release_date(donor.get("release_date")) is not None
    ]
    if not donor_dates:
        return False
    return target_date > max(donor_dates)


def _pick_partial_donors(target_row, strict_rows):
    target_sig = target_row["__partial_signature"]
    family = target_sig["family"]
    provider = target_sig["provider"]
    size_tier = target_sig["size_tier"]
    candidate_rows = _candidate_strict_rows(target_row, strict_rows)
    provider_required = (
        size_tier == "frontier"
        and target_sig["series"] in {"claude", "gemini", "gpt", "grok", "o1"}
    ) or target_sig["market_status"] == "research"

    same_provider_family = [
        row
        for row in candidate_rows
        if row["__partial_signature"]["provider"] == provider
        and family
        and row["__partial_signature"]["family"] == family
    ]
    if same_provider_family:
        ordered = _sort_donors(target_sig, same_provider_family)
        return ordered[:3], "same_provider_family"

    same_provider_tier = [
        row
        for row in candidate_rows
        if row["__partial_signature"]["provider"] == provider
        and row["__partial_signature"]["size_tier"] == size_tier
    ]
    if len(same_provider_tier) >= 2:
        ordered = _sort_donors(target_sig, same_provider_tier)
        return ordered[:3], "same_provider_size_tier"
    if len(same_provider_tier) == 1:
        global_same_tier = [
            row
            for row in candidate_rows
            if row["__partial_signature"]["size_tier"] == size_tier
            and row["model_id"] != same_provider_tier[0]["model_id"]
        ]
        ordered = same_provider_tier + _sort_donors(target_sig, global_same_tier)[:2]
        return ordered[:3], "provider_tier_plus_global_tier"

    same_provider_any = [
        row
        for row in candidate_rows
        if row["__partial_signature"]["provider"] == provider
    ]
    if provider_required:
        if same_provider_any:
            ordered = _sort_donors(target_sig, same_provider_any)
            return ordered[:3], "same_provider_fallback"
        return [], "insufficient_provider_link"

    global_same_family = [
        row
        for row in candidate_rows
        if family and row["__partial_signature"]["family"] == family
    ]
    if global_same_family:
        ordered = _sort_donors(target_sig, global_same_family)
        return ordered[:3], "global_family"

    global_same_tier = [
        row
        for row in candidate_rows
        if row["__partial_signature"]["size_tier"] == size_tier
    ]
    if global_same_tier:
        ordered = _sort_donors(target_sig, global_same_tier)
        return ordered[:3], "global_size_tier"

    if same_provider_any:
        ordered = _sort_donors(target_sig, same_provider_any)
        return ordered[:3], "same_provider_fallback"

    if provider_required:
        return [], "insufficient_provider_link"

    ordered = _sort_donors(target_sig, candidate_rows)
    return ordered[:3], "global_fallback"


def _donor_strategy_note(strategy):
    notes = {
        "same_provider_family": "same provider and same documented family/tier",
        "same_provider_size_tier": "same provider and same documented size tier",
        "provider_tier_plus_global_tier": "same provider when available, then same documented size tier across the strict catalog",
        "global_family": "same documented family/tier across the strict catalog",
        "global_size_tier": "same documented size tier across the strict catalog",
        "same_provider_fallback": "same provider in the strict catalog",
        "global_fallback": "nearest strict donor models in the catalog",
    }
    return notes.get(strategy, "nearest source-linked donor models in the catalog")


def derive_partial_parameter_prior(row, strict_rows):
    donors, strategy = _pick_partial_donors(row, strict_rows)
    donor_values = [
        _to_float(donor.get("active_parameters_billion"))
        for donor in donors
        if _to_float(donor.get("active_parameters_billion")) not in (None, 0)
    ]
    if not donor_values:
        return None

    donor_values.sort()
    central = statistics.median(donor_values)
    monotonic_frontier_floor = False
    target_sig = row["__partial_signature"]
    if (
        strategy == "same_provider_family"
        and target_sig.get("size_tier") == "frontier"
        and _is_later_than_all_donors(row, donors)
    ):
        central = max(central, max(donor_values))
        monotonic_frontier_floor = True
    if len(donor_values) == 1:
        low = central * PARTIAL_SINGLE_DONOR_LOW_FACTOR
        high = central * PARTIAL_SINGLE_DONOR_HIGH_FACTOR
    else:
        low = min(donor_values) * PARTIAL_MULTI_DONOR_LOW_PADDING
        high = max(donor_values) * PARTIAL_MULTI_DONOR_HIGH_PADDING

    low = round(low, 4)
    central = round(central, 4)
    high = round(high, 4)
    donor_model_ids = [str(donor.get("model_id", "") or "") for donor in donors]
    donor_labels = [
        f"{donor.get('display_name', donor.get('model_id', 'n.d.'))} ({_format_billions(_to_float(donor.get('active_parameters_billion')))})"
        for donor in donors
    ]
    strategy_note = _donor_strategy_note(strategy)
    note = (
        "Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. "
        f"Selection rule: {strategy_note}. "
        f"Donors: {', '.join(donor_labels)}. "
        f"Retained central active-parameter basis: {_format_billions(central)} "
        f"(low/high prior: {_format_billions(low)} - {_format_billions(high)})."
    )
    if monotonic_frontier_floor:
        note += (
            " Central prior floored to the largest same-family frontier donor because the target is a later "
            "revision in the same provider family with no direct public parameter disclosure."
        )
    return {
        "parameter_source": "Partial-data donor prior from source-linked market models",
        "parameter_source_url": "",
        "parameter_value_status": "estimated",
        "parameter_confidence": "low",
        "active_parameters_billion_low": low,
        "active_parameters_billion_central": central,
        "active_parameters_billion_high": high,
        "total_parameters_billion_central": central,
        "donor_model_ids": donor_model_ids,
        "donor_labels": donor_labels,
        "strategy": strategy,
        "note": note,
    }


def annotate_market_catalog(rows):
    annotated = [dict(row) for row in rows]
    for row in annotated:
        row["__partial_signature"] = _build_partial_signature(row)
        audit_status, audit_note = classify_market_parameter_source(row)
        row["parameter_source_audit"] = audit_status
        row["parameter_source_audit_note"] = audit_note
        row["quantification_tier"] = "tracked_only"
        row["benchmark_readiness"] = "tracked_only"
        row["benchmark_included"] = "no"
        row["benchmark_exclusion_reason"] = audit_note
        row["quantification_note"] = ""
        row["quantification_donor_models"] = ""
        row["quantified_active_parameters_billion_low"] = ""
        row["quantified_active_parameters_billion_central"] = ""
        row["quantified_active_parameters_billion_high"] = ""
        row["quantified_total_parameters_billion_central"] = ""

    strict_rows = []
    for row in annotated:
        if row["parameter_source_audit"] in STRICT_PARAMETER_AUDITS:
            active = _to_float(row.get("active_parameters_billion"))
            total = _to_float(row.get("total_parameters_billion")) or active
            if active in (None, 0):
                continue
            row["quantification_tier"] = "strict_benchmark"
            row["benchmark_readiness"] = "strict_benchmark"
            row["benchmark_included"] = "yes"
            row["benchmark_exclusion_reason"] = ""
            row["quantification_note"] = row["parameter_source_audit_note"]
            row["quantified_active_parameters_billion_low"] = _format_decimal(active)
            row["quantified_active_parameters_billion_central"] = _format_decimal(active)
            row["quantified_active_parameters_billion_high"] = _format_decimal(active)
            row["quantified_total_parameters_billion_central"] = _format_decimal(total)
            strict_rows.append(row)

    for row in annotated:
        if row["parameter_source_audit"] not in (LEGACY_PROXY_PARAMETER_AUDITS | PARTIAL_DATA_PARAMETER_AUDITS):
            continue
        prior = derive_partial_parameter_prior(row, strict_rows)
        if not prior:
            continue
        row["parameter_source"] = prior["parameter_source"]
        row["parameter_source_url"] = prior["parameter_source_url"]
        row["parameter_value_status"] = prior["parameter_value_status"]
        row["parameter_confidence"] = prior["parameter_confidence"]
        row["notes"] = prior["note"]
        row["active_parameters_billion"] = _format_decimal(prior["active_parameters_billion_central"])
        row["total_parameters_billion"] = _format_decimal(prior["total_parameters_billion_central"])
        row["parameter_source_audit"] = "partial_data_prior"
        row["parameter_source_audit_note"] = (
            "The retained parameter value is a catalog-level partial-data prior derived from benchmark-ready donor models sharing sourced metadata."
        )
        row["quantification_tier"] = "partial_data_benchmark"
        row["benchmark_readiness"] = "partial_data_benchmark"
        row["benchmark_included"] = "yes"
        row["benchmark_exclusion_reason"] = ""
        row["quantification_note"] = prior["note"]
        row["quantification_donor_models"] = "|".join(prior["donor_model_ids"])
        row["quantified_active_parameters_billion_low"] = _format_decimal(prior["active_parameters_billion_low"])
        row["quantified_active_parameters_billion_central"] = _format_decimal(prior["active_parameters_billion_central"])
        row["quantified_active_parameters_billion_high"] = _format_decimal(prior["active_parameters_billion_high"])
        row["quantified_total_parameters_billion_central"] = _format_decimal(prior["total_parameters_billion_central"])

    for row in annotated:
        row.pop("__partial_signature", None)

    return annotated


def annotate_market_model_profile(row):
    annotated = dict(row)
    audit_status, audit_note = classify_market_parameter_source(annotated)
    quantified = audit_status in QUANTIFIED_PARAMETER_AUDITS
    annotated["parameter_source_audit"] = audit_status
    annotated["parameter_source_audit_note"] = audit_note
    annotated["quantification_tier"] = "strict_benchmark" if audit_status in STRICT_PARAMETER_AUDITS else "tracked_only"
    annotated["benchmark_readiness"] = annotated["quantification_tier"]
    annotated["benchmark_included"] = "yes" if quantified else "no"
    annotated["benchmark_exclusion_reason"] = "" if quantified else audit_note
    return annotated
