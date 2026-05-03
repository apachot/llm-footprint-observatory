#!/usr/bin/env python3
"""Upsert the latest market-model and calculator reference profiles."""

import csv
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKET_MODELS_PATH = ROOT / "data" / "market_models.csv"
MODELS_PATH = ROOT / "data" / "models.csv"

TRAINING_TOKENS_PER_PARAMETER = Decimal("20")
TRAINING_TOKEN_QUANT = Decimal("0.0001")

DEFAULT_SCREENING_METHOD_ID = "market_multifactor_prompt_proxy_v1"
DEFAULT_TRAINING_METHOD_ID = "training_multifactor_proxy_v1"
DEFAULT_TRAINING_TOKENS_SOURCE = "Project screening prior aligned with Chinchilla (20 training tokens per retained parameter)"
DEFAULT_TRAINING_TOKENS_SOURCE_URL = "https://arxiv.org/abs/2203.15556"
DEFAULT_TRAINING_REGIME_SOURCE = "Project screening prior (foundation-model pretraining default)"
DEFAULT_TRAINING_HARDWARE_SOURCE = "Project screening prior from market status and serving mode"
DEFAULT_TRAINING_ANCHOR = (
    "Transformer (big) (strubell2019_co2_transformer_big); "
    "BLOOM 176B (luccioni2023_bloom_dynamic); "
    "BLOOM 176B (luccioni2023_bloom_extended)"
)
DEFAULT_ESTIMATION_SOURCE = "Project screening proxy when exact serving country is not publicly specified"


def stringify(value):
    if value in (None, ""):
        return ""
    return str(value)


def compute_training_tokens(row):
    retained = row.get("total_parameters_billion") or row.get("active_parameters_billion")
    if not retained:
        return ""
    value = (Decimal(str(retained)) * TRAINING_TOKENS_PER_PARAMETER / Decimal("1000")).quantize(
        TRAINING_TOKEN_QUANT
    )
    return f"{value:.4f}"


def build_alias_string(row):
    aliases = []
    for candidate in (row.get("display_name"), row.get("reference_aliases", "")):
        for part in str(candidate or "").split("|"):
            part = part.strip()
            if part and part not in aliases:
                aliases.append(part)
    return "|".join(aliases)


def market_row(**values):
    row = {
        "screening_method_id": DEFAULT_SCREENING_METHOD_ID,
        "training_regime": "pretraining",
        "training_regime_status": "screening_prior",
        "training_regime_source": DEFAULT_TRAINING_REGIME_SOURCE,
        "training_regime_source_url": "",
        "training_multimodal_source_url": "",
        "training_hardware_source": DEFAULT_TRAINING_HARDWARE_SOURCE,
        "training_hardware_source_url": "",
        "training_multifactor_method_id": DEFAULT_TRAINING_METHOD_ID,
        "training_multifactor_anchor": DEFAULT_TRAINING_ANCHOR,
    }
    row.update({key: stringify(value) for key, value in values.items()})

    if not row.get("training_tokens_estimate_trillion"):
        row["training_tokens_estimate_trillion"] = compute_training_tokens(row)
        if not row.get("training_tokens_status"):
            row["training_tokens_status"] = "screening_prior"
        if not row.get("training_tokens_source"):
            row["training_tokens_source"] = DEFAULT_TRAINING_TOKENS_SOURCE
        if not row.get("training_tokens_source_url"):
            row["training_tokens_source_url"] = DEFAULT_TRAINING_TOKENS_SOURCE_URL
    else:
        if not row.get("training_tokens_status"):
            row["training_tokens_status"] = "documented"
        row.setdefault("training_tokens_source_url", "")

    if not row.get("training_multimodal_source"):
        row["training_multimodal_source"] = row.get("modalities_source", "")
    if not row.get("training_multimodal_source_url"):
        row["training_multimodal_source_url"] = row.get("modalities_source_url", "")

    return row


def openai_row(
    model_id,
    display_name,
    release_date,
    release_url,
    doc_url,
    active_parameters_billion,
    total_parameters_billion,
    parameter_source,
    notes,
    *,
    context_window_tokens="400000",
    max_output_tokens="128000",
    reference_aliases="",
):
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="openai",
        market_status="api",
        serving_mode="closed",
        server_country="United States or Europe",
        server_country_status="multi_region",
        server_country_source="OpenAI data residency and models documentation",
        server_country_source_url="https://developers.openai.com/api/docs/models",
        estimation_country_code="US",
        estimation_country_status="screening_proxy",
        estimation_country_source=DEFAULT_ESTIMATION_SOURCE,
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="estimated",
        parameter_confidence="low",
        parameter_source=parameter_source,
        parameter_source_url=release_url or doc_url,
        notes=notes,
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        vision_support="yes",
        architecture_notes="Closed multimodal model; official docs expose capabilities and token limits but not architecture details or parameter count.",
        modalities_source=f"{display_name} model docs | OpenAI API",
        modalities_source_url=doc_url,
        context_source=f"{display_name} model docs | OpenAI API",
        context_source_url=doc_url,
        architecture_source=f"{display_name} model docs | OpenAI API",
        architecture_source_url=doc_url,
        release_date=release_date,
        release_source=f"Introducing {display_name} | OpenAI",
        release_source_url=release_url,
        training_multimodal="yes",
        training_hardware_class_proxy="modern_hyperscale_gpu",
    )


def anthropic_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    parameter_source,
    notes,
    *,
    market_status="api",
    serving_mode="closed",
    context_window_tokens="1000000",
    max_output_tokens="64000",
    reference_aliases="",
):
    docs_url = "https://platform.claude.com/docs/en/about-claude/models/overview"
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="anthropic",
        market_status=market_status,
        serving_mode=serving_mode,
        server_country="Not publicly specified",
        server_country_status="non_specified",
        server_country_source="Anthropic models overview",
        server_country_source_url=docs_url,
        estimation_country_code="US",
        estimation_country_status="screening_proxy",
        estimation_country_source=DEFAULT_ESTIMATION_SOURCE,
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="estimated",
        parameter_confidence="low",
        parameter_source=parameter_source,
        parameter_source_url=release_url or docs_url,
        notes=notes,
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        vision_support="yes",
        architecture_notes="Closed multimodal model; Anthropic documents token limits and vision support but does not publish architecture details or parameter count.",
        modalities_source="Models overview - Claude API Docs",
        modalities_source_url=docs_url,
        context_source="Models overview - Claude API Docs",
        context_source_url=docs_url,
        architecture_source="Models overview - Claude API Docs",
        architecture_source_url=docs_url,
        release_date=release_date,
        release_source=f"{display_name} | Anthropic",
        release_source_url=release_url,
        training_multimodal="yes",
        training_hardware_class_proxy="modern_hyperscale_gpu",
    )


def google_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    parameter_source,
    notes,
    *,
    context_window_tokens="1048576",
    max_output_tokens="65536",
    reference_aliases="",
):
    docs_url = "https://ai.google.dev/gemini-api/docs/models"
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="google",
        market_status="api",
        serving_mode="closed",
        server_country="Global multi-region",
        server_country_status="multi_region",
        server_country_source="Gemini API model documentation",
        server_country_source_url=docs_url,
        estimation_country_code="US",
        estimation_country_status="screening_proxy",
        estimation_country_source=DEFAULT_ESTIMATION_SOURCE,
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="estimated",
        parameter_confidence="low",
        parameter_source=parameter_source,
        parameter_source_url=release_url or docs_url,
        notes=notes,
        input_modalities="text,image,audio,video,pdf",
        output_modalities="text",
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        vision_support="yes",
        architecture_notes="Closed multimodal model; Google documents token limits and modalities but not architecture details or parameter count.",
        modalities_source="Gemini models | Gemini API",
        modalities_source_url=docs_url,
        context_source="Gemini models | Gemini API",
        context_source_url=docs_url,
        architecture_source="Gemini models | Gemini API",
        architecture_source_url=docs_url,
        release_date=release_date,
        release_source=f"{display_name} release documentation | Google",
        release_source_url=release_url,
        training_multimodal="yes",
        training_hardware_class_proxy="modern_hyperscale_gpu",
    )


def xai_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    parameter_source,
    notes,
    architecture_notes,
    *,
    reference_aliases="",
):
    docs_url = "https://docs.x.ai/developers/models"
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="xai",
        market_status="api",
        serving_mode="closed",
        server_country="United States or Ireland",
        server_country_status="documented_multi_region",
        server_country_source="xAI API regions and models documentation",
        server_country_source_url=docs_url,
        estimation_country_code="US",
        estimation_country_status="documented_region_proxy",
        estimation_country_source="xAI API documented regions include US and Ireland; US retained as reference",
        estimation_country_source_url="https://docs.x.ai/docs/overview#supported-regions",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="estimated",
        parameter_confidence="low",
        parameter_source=parameter_source,
        parameter_source_url=release_url or docs_url,
        notes=notes,
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens="2000000",
        max_output_tokens="",
        vision_support="yes",
        architecture_notes=architecture_notes,
        modalities_source="Models and Pricing | xAI",
        modalities_source_url=docs_url,
        context_source="Models and Pricing | xAI",
        context_source_url=docs_url,
        architecture_source="Models and Pricing | xAI",
        architecture_source_url=docs_url,
        release_date=release_date,
        release_source="xAI release documentation",
        release_source_url=release_url or docs_url,
        training_multimodal="yes",
        training_hardware_class_proxy="modern_hyperscale_gpu",
    )


def deepseek_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    notes,
    architecture_notes,
    *,
    max_output_tokens="384000",
    reference_aliases="",
):
    docs_url = "https://api-docs.deepseek.com/quick_start/pricing"
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="deepseek",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        server_country="China (provider proxy)",
        server_country_status="provider_country_proxy",
        server_country_source="DeepSeek API documentation",
        server_country_source_url=docs_url,
        estimation_country_code="CN",
        estimation_country_status="provider_country_proxy",
        estimation_country_source="DeepSeek provider country retained as screening proxy",
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="observed",
        parameter_confidence="high",
        parameter_source=f"{display_name} official release note",
        parameter_source_url=release_url,
        notes=notes,
        input_modalities="text",
        output_modalities="text",
        context_window_tokens="1000000",
        max_output_tokens=max_output_tokens,
        vision_support="no",
        architecture_notes=architecture_notes,
        modalities_source=f"{display_name} documentation | DeepSeek",
        modalities_source_url=docs_url,
        context_source=f"{display_name} documentation | DeepSeek",
        context_source_url=release_url,
        architecture_source=f"{display_name} official release note",
        architecture_source_url=release_url,
        release_date=release_date,
        release_source=f"{display_name} | DeepSeek",
        release_source_url=release_url,
        training_multimodal="no",
        training_hardware_class_proxy="mixed_gpu_cluster",
    )


def mistral_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    parameter_source,
    parameter_source_url,
    notes,
    architecture_notes,
    *,
    market_status,
    serving_mode,
    input_modalities,
    output_modalities,
    context_window_tokens,
    vision_support,
    training_multimodal,
    training_hardware_class_proxy,
    server_country,
    server_country_status,
    estimation_country_status,
    reference_aliases="",
    max_output_tokens="",
    parameter_value_status="observed",
    parameter_confidence="high",
):
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="mistral",
        market_status=market_status,
        serving_mode=serving_mode,
        server_country=server_country,
        server_country_status=server_country_status,
        server_country_source="Mistral model documentation",
        server_country_source_url=release_url,
        estimation_country_code="FR",
        estimation_country_status=estimation_country_status,
        estimation_country_source="Mistral provider country retained as screening proxy",
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status=parameter_value_status,
        parameter_confidence=parameter_confidence,
        parameter_source=parameter_source,
        parameter_source_url=parameter_source_url,
        notes=notes,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        vision_support=vision_support,
        architecture_notes=architecture_notes,
        modalities_source=f"{display_name} model documentation | Mistral",
        modalities_source_url=release_url,
        context_source=f"{display_name} model documentation | Mistral",
        context_source_url=release_url,
        architecture_source=f"{display_name} model documentation | Mistral",
        architecture_source_url=release_url,
        release_date=release_date,
        release_source=f"{display_name} | Mistral",
        release_source_url=release_url,
        training_multimodal=training_multimodal,
        training_hardware_class_proxy=training_hardware_class_proxy,
    )


def meta_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    notes,
    architecture_notes,
    training_tokens_estimate_trillion,
    training_tokens_source,
    *,
    context_window_tokens,
    reference_aliases="",
):
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="meta",
        market_status="open_weight",
        serving_mode="open",
        server_country="Variable by host",
        server_country_status="self_hosted_variable",
        server_country_source=f"{display_name} model card",
        server_country_source_url=release_url,
        estimation_country_code="US",
        estimation_country_status="comparative_reference",
        estimation_country_source="Project comparative reference country for self-hosted open-weight scenarios",
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="observed",
        parameter_confidence="high",
        parameter_source=f"{display_name} model card",
        parameter_source_url=release_url,
        notes=notes,
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens=context_window_tokens,
        max_output_tokens="",
        vision_support="yes",
        architecture_notes=architecture_notes,
        modalities_source=f"{display_name} model card",
        modalities_source_url=release_url,
        context_source=f"{display_name} model card",
        context_source_url=release_url,
        architecture_source=f"{display_name} model card",
        architecture_source_url=release_url,
        release_date=release_date,
        release_source=f"{display_name} model card",
        release_source_url=release_url,
        training_tokens_estimate_trillion=training_tokens_estimate_trillion,
        training_tokens_status="documented",
        training_tokens_source=training_tokens_source,
        training_tokens_source_url=release_url,
        training_multimodal="yes",
        training_hardware_class_proxy="standard_gpu_cluster",
    )


def qwen_row(
    model_id,
    display_name,
    release_date,
    release_url,
    active_parameters_billion,
    total_parameters_billion,
    notes,
    architecture_notes,
    *,
    reference_aliases="",
):
    return market_row(
        model_id=model_id,
        display_name=display_name,
        reference_aliases=reference_aliases,
        provider="alibaba",
        market_status="open_weight",
        serving_mode="open",
        server_country="China (provider proxy)",
        server_country_status="provider_country_proxy",
        server_country_source="Qwen provider-country proxy",
        server_country_source_url=release_url,
        estimation_country_code="CN",
        estimation_country_status="provider_country_proxy",
        estimation_country_source="Qwen provider country retained as screening proxy",
        estimation_country_source_url="",
        active_parameters_billion=active_parameters_billion,
        total_parameters_billion=total_parameters_billion,
        parameter_value_status="observed",
        parameter_confidence="high",
        parameter_source=f"{display_name} release repository",
        parameter_source_url=release_url,
        notes=notes,
        input_modalities="text",
        output_modalities="text",
        context_window_tokens="131072",
        max_output_tokens="32768",
        vision_support="no",
        architecture_notes=architecture_notes,
        modalities_source=f"{display_name} release repository",
        modalities_source_url=release_url,
        context_source=f"{display_name} release repository",
        context_source_url=release_url,
        architecture_source=f"{display_name} release repository",
        architecture_source_url=release_url,
        release_date=release_date,
        release_source=f"{display_name} | Qwen",
        release_source_url=release_url,
        training_multimodal="no",
        training_hardware_class_proxy="standard_gpu_cluster",
    )


MARKET_MODEL_UPDATES = [
    openai_row(
        "gpt-5.2",
        "GPT-5.2",
        "2025-12-11",
        "https://openai.com/index/introducing-gpt-5-2/",
        "https://developers.openai.com/api/docs/models/gpt-5.2",
        "300",
        "300",
        "Project screening family proxy anchored to the GPT-5.2 release line",
        "Family-level proxy retained for GPT-5.2 because OpenAI documents capabilities but does not publish a separate parameter count for this revision.",
        reference_aliases="gpt52",
    ),
    openai_row(
        "gpt-5.2-pro",
        "GPT-5.2-pro",
        "2025-12-11",
        "https://openai.com/index/introducing-gpt-5-2/",
        "https://developers.openai.com/api/docs/models/gpt-5.2-pro",
        "300",
        "300",
        "Project screening family proxy anchored to the GPT-5.2 release line",
        "Family-level proxy retained for GPT-5.2-pro because OpenAI documents capabilities but does not publish a separate parameter count for this revision.",
        reference_aliases="gpt-5.2 pro|gpt52pro",
    ),
    openai_row(
        "gpt-5.4",
        "GPT-5.4",
        "2026-03-05",
        "https://openai.com/index/introducing-gpt-5-4/",
        "https://developers.openai.com/api/docs/models/gpt-5.4",
        "1800",
        "1800",
        "Project screening frontier-family proxy for GPT-5.x closed models",
        "Closed-model family proxy retained because OpenAI does not publish a parameter count for GPT-5.4.",
    ),
    openai_row(
        "gpt-5.4-pro",
        "GPT-5.4-pro",
        "2026-03-05",
        "https://openai.com/index/introducing-gpt-5-4/",
        "https://developers.openai.com/api/docs/models/gpt-5.4-pro",
        "1800",
        "1800",
        "Project screening frontier-family proxy for GPT-5.x closed models",
        "Same retained frontier-family proxy as GPT-5.4; OpenAI documents the product tier but not a distinct parameter count.",
        reference_aliases="gpt-5.4 pro|gpt54pro",
    ),
    openai_row(
        "gpt-5.4-mini",
        "GPT-5.4 mini",
        "2026-03-17",
        "https://openai.com/index/introducing-gpt-5-4-mini-and-nano/",
        "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        "95",
        "95",
        "Project screening medium-size proxy anchored to the GPT-5.4 mini release line",
        "Medium-size proxy retained for GPT-5.4 mini because OpenAI documents capabilities but does not publish a parameter count.",
        reference_aliases="gpt-5.4 mini|gpt54mini",
    ),
    openai_row(
        "gpt-5.4-nano",
        "GPT-5.4 nano",
        "2026-03-17",
        "https://openai.com/index/introducing-gpt-5-4-mini-and-nano/",
        "https://developers.openai.com/api/docs/models/gpt-5.4-nano",
        "95",
        "95",
        "Project screening medium-size proxy anchored to the GPT-5.4 nano release line",
        "Medium-size proxy retained for GPT-5.4 nano because OpenAI documents capabilities but does not publish a parameter count.",
        reference_aliases="gpt-5.4 nano|gpt54nano",
    ),
    openai_row(
        "gpt-5.5",
        "GPT-5.5",
        "2026-04-23",
        "https://openai.com/index/introducing-gpt-5-5/",
        "https://developers.openai.com/api/docs/models/gpt-5.5",
        "2000",
        "2000",
        "Project screening frontier-family proxy for the GPT-5.5 release line",
        "Frontier-family proxy retained because OpenAI documents GPT-5.5 capabilities but not the parameter count.",
        reference_aliases="gpt55",
    ),
    openai_row(
        "gpt-5.5-pro",
        "GPT-5.5-pro",
        "2026-04-23",
        "https://openai.com/index/introducing-gpt-5-5/",
        "https://developers.openai.com/api/docs/models/gpt-5.5-pro",
        "2000",
        "2000",
        "Project screening frontier-family proxy for the GPT-5.5 release line",
        "Retained the GPT-5.5 frontier proxy for the Pro tier because OpenAI does not publish a separate parameter count.",
        context_window_tokens="1050000",
        max_output_tokens="128000",
        reference_aliases="gpt-5.5 pro|gpt55pro",
    ),
    anthropic_row(
        "claude-sonnet-4.6",
        "Claude Sonnet 4.6",
        "2026-02-17",
        "https://www.anthropic.com/news/claude-sonnet-4-6",
        "400",
        "400",
        "Project screening family proxy for Claude Sonnet 4.x",
        "Family proxy retained because Anthropic does not publish a parameter count for Claude Sonnet 4.6.",
        reference_aliases="claude sonnet 4.6|sonnet4.6",
    ),
    anthropic_row(
        "claude-opus-4.6",
        "Claude Opus 4.6",
        "2026-02-05",
        "https://www.anthropic.com/news/claude-opus-4-6",
        "2000",
        "2000",
        "Project screening family proxy for Claude Opus 4.x",
        "Family proxy retained because Anthropic does not publish a parameter count for Claude Opus 4.6.",
        max_output_tokens="32000",
        reference_aliases="claude opus 4.6|opus4.6",
    ),
    anthropic_row(
        "claude-opus-4.7",
        "Claude Opus 4.7",
        "2026-04-16",
        "https://www.anthropic.com/news/claude-opus-4-7",
        "2000",
        "2000",
        "Project screening family proxy for Claude Opus 4.x",
        "Family proxy retained because Anthropic does not publish a parameter count for Claude Opus 4.7.",
        max_output_tokens="32000",
        reference_aliases="claude opus 4.7|opus4.7",
    ),
    anthropic_row(
        "claude-mythos-preview",
        "Claude Mythos Preview",
        "2026-04-07",
        "https://www.anthropic.com/project/glasswing",
        "2600",
        "2600",
        "Project frontier proxy for the Claude Mythos preview line",
        "Restricted-preview research row retained for frontier benchmarking; public documentation remains limited and Anthropic does not publish a parameter count.",
        market_status="research",
        serving_mode="research",
        reference_aliases="claude mythos|mythos|mythos preview",
    ),
    google_row(
        "gemini-3-pro",
        "Gemini 3 Pro",
        "2025-11-18",
        "https://blog.google/products-and-platforms/products/gemini/gemini-3/",
        "220",
        "220",
        "Project screening family proxy for Gemini 3 Pro",
        "Family proxy retained because Google documents Gemini 3 Pro capabilities but not the parameter count.",
        reference_aliases="gemini 3 pro|gemini3pro",
    ),
    google_row(
        "gemini-3-flash",
        "Gemini 3 Flash",
        "2025-12-17",
        "https://deepmind.google/models/gemini/flash/",
        "100",
        "100",
        "Project screening family proxy for Gemini 3 Flash",
        "Family proxy retained because Google documents Gemini 3 Flash capabilities but not the parameter count.",
        reference_aliases="gemini 3 flash|gemini3flash",
    ),
    google_row(
        "gemini-3.1-pro",
        "Gemini 3.1 Pro",
        "2026-02-19",
        "https://deepmind.google/models/gemini/pro/",
        "240",
        "240",
        "Project screening family proxy for Gemini 3.1 Pro",
        "Family proxy retained because Google documents Gemini 3.1 Pro capabilities but not the parameter count.",
        reference_aliases="gemini 3.1 pro|gemini31pro",
    ),
    google_row(
        "gemini-3.1-flash-lite",
        "Gemini 3.1 Flash-Lite",
        "2026-03-03",
        "https://deepmind.google/models/gemini/flash/",
        "40",
        "40",
        "Project screening family proxy for Gemini 3.1 Flash-Lite",
        "Family proxy retained because Google documents Gemini 3.1 Flash-Lite capabilities but not the parameter count.",
        reference_aliases="gemini 3.1 flash lite|gemini31flashlite",
    ),
    xai_row(
        "grok-4.20-reasoning",
        "Grok 4.20 Reasoning",
        "2026-03-10",
        "https://docs.x.ai/docs/release-notes",
        "650",
        "650",
        "Project screening family proxy for the Grok 4.20 reasoning tier",
        "Family proxy retained because xAI documents model availability and limits but not a parameter count for Grok 4.20 Reasoning.",
        "Closed reasoning model; xAI documents a 2M-token context window and image understanding support but not architecture details or parameter count.",
        reference_aliases="grok 4.20 reasoning|grok420reasoning",
    ),
    xai_row(
        "grok-4.20-non-reasoning",
        "Grok 4.20",
        "2026-03-10",
        "https://docs.x.ai/docs/release-notes",
        "650",
        "650",
        "Project screening family proxy for the Grok 4.20 general tier",
        "Family proxy retained because xAI documents model availability and limits but not a parameter count for Grok 4.20.",
        "Closed multimodal chat model; xAI documents a 2M-token context window and image understanding support but not architecture details or parameter count.",
        reference_aliases="grok 4.20|grok420",
    ),
    xai_row(
        "grok-4.3",
        "Grok 4.3",
        "2026-04-16",
        "https://docs.x.ai/developers/models",
        "650",
        "650",
        "Project screening family proxy for Grok 4.3",
        "Current xAI models overview exposes Grok 4.3 as the recommended flagship chat model; xAI does not publish a parameter count, so the Grok 4.20 family proxy is retained. The dated catalog entry uses the models-page update date.",
        "Closed multimodal chat model; xAI currently surfaces Grok 4.3 as the main flagship model but does not publish architecture details or parameter count.",
        reference_aliases="grok 4.3|grok43",
    ),
    xai_row(
        "grok-4.1-fast-reasoning",
        "Grok 4.1 Fast Reasoning",
        "2025-11-19",
        "https://docs.x.ai/docs/release-notes",
        "120",
        "120",
        "Project screening family proxy for Grok 4.1 Fast",
        "Smaller family proxy retained because xAI documents Grok 4.1 Fast capabilities but not the parameter count.",
        "Closed reasoning model optimized for faster responses; xAI documents a 2M-token context window and image understanding support.",
        reference_aliases="grok 4.1 fast reasoning|grok41fastreasoning",
    ),
    xai_row(
        "grok-4.1-fast-non-reasoning",
        "Grok 4.1 Fast",
        "2025-11-19",
        "https://docs.x.ai/docs/release-notes",
        "120",
        "120",
        "Project screening family proxy for Grok 4.1 Fast",
        "Smaller family proxy retained because xAI documents Grok 4.1 Fast capabilities but not the parameter count.",
        "Closed multimodal chat model optimized for faster responses; xAI documents a 2M-token context window and image understanding support.",
        reference_aliases="grok 4.1 fast|grok41fast",
    ),
    deepseek_row(
        "deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "2026-04-24",
        "https://api-docs.deepseek.com/news/news260424",
        "49",
        "1600",
        "Official release note reports 1.6T total parameters and 49B active parameters.",
        "Closed MoE reasoning model; official docs report 1.6T total parameters, 49B active parameters, 1M-token context, and 384k output.",
        reference_aliases="deepseek v4 pro|deepseekv4pro",
    ),
    deepseek_row(
        "deepseek-v4-flash",
        "DeepSeek V4 Flash",
        "2026-04-24",
        "https://api-docs.deepseek.com/news/news260424",
        "13",
        "284",
        "Official release note reports 284B total parameters and 13B active parameters.",
        "Closed MoE chat model; official docs report 284B total parameters, 13B active parameters, 1M-token context, and 384k output.",
        reference_aliases="deepseek v4 flash|deepseekv4flash",
    ),
    mistral_row(
        "mistral-large-3",
        "Mistral Large 3",
        "2025-12-02",
        "https://docs.mistral.ai/models/mistral-large-3-25-12",
        "41",
        "675",
        "Mistral Large 3 model documentation",
        "https://docs.mistral.ai/models/mistral-large-3-25-12",
        "Official documentation reports 675B total parameters and 41B active parameters.",
        "Open-weight multimodal MoE model; official docs report 675B total parameters, 41B active parameters, and a 256k context window.",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens="256000",
        vision_support="yes",
        training_multimodal="yes",
        training_hardware_class_proxy="mixed_gpu_cluster",
        server_country="France (provider proxy)",
        server_country_status="provider_country_proxy",
        estimation_country_status="provider_country_proxy",
        reference_aliases="mistral large 3|mistrallarge3",
    ),
    mistral_row(
        "mistral-medium-3.5",
        "Mistral Medium 3.5",
        "",
        "https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04/",
        "128",
        "128",
        "Artificial Analysis estimate for Mistral Medium 3.5 (~128B)",
        "https://artificialanalysis.ai/models",
        "Third-party proxy retained because Mistral documents the product tier and open-weight release, but not a parameter count for Medium 3.5.",
        "Open-weight multimodal model; official docs expose a 256k context window but not architecture details or parameter count.",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens="256000",
        vision_support="yes",
        training_multimodal="yes",
        training_hardware_class_proxy="mixed_gpu_cluster",
        server_country="France (provider proxy)",
        server_country_status="provider_country_proxy",
        estimation_country_status="provider_country_proxy",
        parameter_value_status="estimated",
        parameter_confidence="low",
        reference_aliases="mistral medium 3.5|mistralmedium35",
    ),
    mistral_row(
        "mistral-small-4",
        "Mistral Small 4",
        "2026-03-16",
        "https://docs.mistral.ai/models/mistral-small-4-0-26-03",
        "6.5",
        "119",
        "Mistral Small 4 model documentation",
        "https://docs.mistral.ai/models/mistral-small-4-0-26-03",
        "Official documentation reports 119B total parameters and 6.5B active parameters.",
        "Open or hybrid multimodal MoE model; official docs report 119B total parameters, 6.5B active parameters, and a 256k context window.",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens="256000",
        vision_support="yes",
        training_multimodal="yes",
        training_hardware_class_proxy="mixed_gpu_cluster",
        server_country="France (provider proxy)",
        server_country_status="provider_country_proxy",
        estimation_country_status="provider_country_proxy",
        reference_aliases="mistral small 4|mistralsmall4",
    ),
    mistral_row(
        "devstral-2",
        "Devstral 2",
        "2025-12-09",
        "https://docs.mistral.ai/models/devstral-2-25-12",
        "123",
        "123",
        "Devstral 2 model documentation",
        "https://docs.mistral.ai/models/devstral-2-25-12",
        "Official documentation reports a 123B-parameter coding model.",
        "Hybrid coding model; official docs report a 123B parameter count and 256k context window.",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        input_modalities="text",
        output_modalities="text,code",
        context_window_tokens="256000",
        vision_support="no",
        training_multimodal="no",
        training_hardware_class_proxy="mixed_gpu_cluster",
        server_country="France (provider proxy)",
        server_country_status="provider_country_proxy",
        estimation_country_status="provider_country_proxy",
        reference_aliases="devstral 2|devstral2",
    ),
    mistral_row(
        "ministral-3-14b",
        "Ministral 3 14B",
        "2025-12-02",
        "https://docs.mistral.ai/models/ministral-3-14b-25-12",
        "14",
        "14",
        "Ministral 3 14B model documentation",
        "https://docs.mistral.ai/models/ministral-3-14b-25-12",
        "Official documentation reports a 14B multimodal edge model.",
        "Hybrid edge model with image understanding; official docs report a 14B parameter count and 256k context window.",
        market_status="api_and_open_weight",
        serving_mode="hybrid",
        input_modalities="text,image",
        output_modalities="text",
        context_window_tokens="256000",
        vision_support="yes",
        training_multimodal="yes",
        training_hardware_class_proxy="mixed_gpu_cluster",
        server_country="France (provider proxy)",
        server_country_status="provider_country_proxy",
        estimation_country_status="provider_country_proxy",
        reference_aliases="ministral 3 14b|ministral314b",
    ),
    meta_row(
        "llama-4-scout",
        "Llama 4 Scout",
        "2025-04-05",
        "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "17",
        "109",
        "Open-weight multimodal model; deployment impact depends on hosting location.",
        "Open multimodal MoE model; official card reports 109B total parameters, 17B active parameters, and a 10M-token context window.",
        "40.0000",
        "Official model card training-token disclosure (40T tokens)",
        context_window_tokens="10000000",
        reference_aliases="llama 4 scout|llama4scout",
    ),
    meta_row(
        "llama-4-maverick",
        "Llama 4 Maverick",
        "2025-04-05",
        "https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "17",
        "400",
        "Open-weight multimodal model; deployment impact depends on hosting location.",
        "Open multimodal MoE model; official card reports 400B total parameters, 17B active parameters, and a 1M-token context window.",
        "22.0000",
        "Official model card training-token disclosure (22T tokens)",
        context_window_tokens="1000000",
        reference_aliases="llama 4 maverick|llama4maverick",
    ),
    qwen_row(
        "qwen3-32b",
        "Qwen3 32B",
        "2025-04-29",
        "https://huggingface.co/Qwen/Qwen3-32B",
        "32.8",
        "32.8",
        "Open-weight model; deployment impact depends on hosting location.",
        "Open dense transformer from the Qwen3 release line; the official model card documents 32.8B parameters and 131k context support.",
        reference_aliases="qwen 3 32b|qwen332b",
    ),
    qwen_row(
        "qwen3-235b-a22b",
        "Qwen3 235B A22B",
        "2025-04-29",
        "https://huggingface.co/Qwen/Qwen3-235B-A22B",
        "22",
        "235",
        "Open-weight model; deployment impact depends on hosting location.",
        "Open MoE transformer from the Qwen3 release line; the official model card documents 235B total parameters, 22B active parameters, and 131k context support.",
        reference_aliases="qwen 3 235b a22b|qwen3235ba22b",
    ),
]


REFERENCE_MODEL_UPDATES = [
    {
        "model_id": row["model_id"],
        "provider": row["provider"],
        "aliases": build_alias_string(row),
        "active_parameters_billion": row["active_parameters_billion"],
        "total_parameters_billion": row["total_parameters_billion"],
        "parameter_value_status": row["parameter_value_status"],
        "parameter_confidence": row["parameter_confidence"],
        "parameter_source": row["parameter_source"],
        "parameter_source_url": row["parameter_source_url"],
        "notes": row["notes"],
    }
    for row in MARKET_MODEL_UPDATES
]


def normalize_row(row, header):
    return {column: stringify(row.get(column, "")) for column in header}


def load_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise SystemExit(f"Missing header in {path.name}")
        rows = [dict(row) for row in reader]
    return header, rows


def upsert_rows(path, updates):
    header, rows = load_csv_rows(path)
    rows_by_id = {row.get("model_id"): row for row in rows}

    for update in updates:
        model_id = update["model_id"]
        if model_id in rows_by_id:
            target = rows_by_id[model_id]
        else:
            target = {column: "" for column in header}
            target["model_id"] = model_id
            rows.append(target)
            rows_by_id[model_id] = target

        for key, value in update.items():
            if key in header:
                target[key] = stringify(value)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row, header))


def main():
    upsert_rows(MARKET_MODELS_PATH, MARKET_MODEL_UPDATES)
    upsert_rows(MODELS_PATH, REFERENCE_MODEL_UPDATES)
    print(
        "Updated "
        f"{len(MARKET_MODEL_UPDATES)} market-model profiles and "
        f"{len(REFERENCE_MODEL_UPDATES)} calculator reference profiles."
    )


if __name__ == "__main__":
    main()
