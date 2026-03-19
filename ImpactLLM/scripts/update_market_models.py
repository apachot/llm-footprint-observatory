#!/usr/bin/env python3
"""Add or update baseline entries in data/market_models.csv."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKET_MODELS_PATH = ROOT / "data" / "market_models.csv"

NEW_MARKET_MODELS = [
    {
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o mini",
        "provider": "openai",
        "market_status": "api",
        "serving_mode": "closed",
        "server_country": "United States",
        "server_country_status": "multi_region",
        "server_country_source": "Introducing GPT-4o mini | OpenAI",
        "server_country_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "estimation_country_code": "US",
        "estimation_country_status": "screening_proxy",
        "estimation_country_source": "Introducing GPT-4o mini | OpenAI",
        "estimation_country_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "active_parameters_billion": "8",
        "total_parameters_billion": "8",
        "parameter_value_status": "estimated",
        "parameter_confidence": "low",
        "parameter_source": "Internal approximate profile for extrapolation",
        "parameter_source_url": "",
        "notes": "OpenAI does not publish a parameter count for GPT-4o mini; the project retains an 8B screening-level proxy for comparative extrapolation.",
        "input_modalities": "text,image",
        "output_modalities": "text",
        "context_window_tokens": "128000",
        "max_output_tokens": "16384",
        "vision_support": "yes",
        "architecture_notes": "Cost-efficient multimodal small model in the GPT-4o family, exposed through the OpenAI API.",
        "modalities_source": "Introducing GPT-4o mini | OpenAI",
        "modalities_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "context_source": "GPT-4o mini model docs | OpenAI API",
        "context_source_url": "https://platform.openai.com/docs/models/gpt-4o-mini",
        "architecture_source": "Introducing GPT-4o mini | OpenAI",
        "architecture_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "release_date": "2024-07-18",
        "release_source": "Introducing GPT-4o mini | OpenAI",
        "release_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "screening_method_id": "market_multifactor_prompt_proxy_v1",
        "training_regime": "pretraining",
        "training_regime_status": "screening_prior",
        "training_regime_source": "Introducing GPT-4o mini | OpenAI",
        "training_regime_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "training_multimodal": "yes",
        "training_multimodal_source": "Introducing GPT-4o mini | OpenAI",
        "training_multimodal_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "training_hardware_class_proxy": "modern_hyperscale_gpu",
        "training_hardware_source": "Introducing GPT-4o mini | OpenAI",
        "training_hardware_source_url": "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "training_multifactor_method_id": "training_multifactor_proxy_v1",
        "training_multifactor_anchor": "Transformer (big) (strubell2019_co2_transformer_big); BLOOM 176B (luccioni2023_bloom_dynamic); BLOOM 176B (luccioni2023_bloom_extended)",
    },
    {
        "model_id": "gpt-3.5-turbo",
        "display_name": "GPT-3.5 Turbo",
        "provider": "openai",
        "market_status": "api",
        "serving_mode": "closed",
        "server_country": "United States",
        "server_country_status": "multi_region",
        "server_country_source": "ChatGPT launch blog",
        "server_country_source_url": "https://openai.com/blog/chatgpt",
        "estimation_country_code": "US",
        "estimation_country_status": "screening_proxy",
        "estimation_country_source": "ChatGPT launch blog",
        "estimation_country_source_url": "https://openai.com/blog/chatgpt",
        "active_parameters_billion": "175",
        "total_parameters_billion": "175",
        "parameter_value_status": "estimated",
        "parameter_confidence": "low",
        "parameter_source": "Nexos ChatGPT version history",
        "parameter_source_url": "https://nexos.ai/blog/chatgpt-version-history/",
        "notes": "GPT-3.5 Turbo powers ChatGPT since November 2022 and relies on the GPT-3 architecture at ~175B parameters.",
        "input_modalities": "text",
        "output_modalities": "text",
        "context_window_tokens": "4096",
        "max_output_tokens": "4096",
        "vision_support": "no",
        "architecture_notes": "Decoder transformer with RLHF tuned on conversational data per ChatGPT release notes.",
        "modalities_source": "OpenAI ChatGPT blog",
        "modalities_source_url": "https://openai.com/blog/chatgpt",
        "context_source": "OpenAI ChatGPT blog",
        "context_source_url": "https://openai.com/blog/chatgpt",
        "architecture_source": "OpenAI ChatGPT blog",
        "architecture_source_url": "https://openai.com/blog/chatgpt",
        "release_date": "2022-11-30",
        "release_source": "ChatGPT launch blog",
        "release_source_url": "https://openai.com/blog/chatgpt",
        "screening_method_id": "market_multifactor_prompt_proxy_v1",
        "training_regime": "pretraining",
        "training_regime_status": "documented",
        "training_regime_source": "OpenAI ChatGPT blog",
        "training_regime_source_url": "https://openai.com/blog/chatgpt",
        "training_multimodal": "no",
        "training_multimodal_source": "OpenAI ChatGPT blog",
        "training_multimodal_source_url": "https://openai.com/blog/chatgpt",
        "training_hardware_class_proxy": "standard_gpu_cluster",
        "training_hardware_source": "OpenAI ChatGPT blog",
        "training_hardware_source_url": "https://openai.com/blog/chatgpt",
        "training_multifactor_method_id": "training_multifactor_proxy_v1",
        "training_multifactor_anchor": "Transformer (big) (strubell2019_co2_transformer_big); BLOOM 176B (luccioni2023_bloom_dynamic); BLOOM 176B (luccioni2023_bloom_extended)",
    },
    {
        "model_id": "gpt-4",
        "display_name": "GPT-4",
        "provider": "openai",
        "market_status": "api",
        "serving_mode": "closed",
        "server_country": "United States",
        "server_country_status": "multi_region",
        "server_country_source": "OpenAI GPT-4 research release",
        "server_country_source_url": "https://openai.com/research/gpt-4",
        "estimation_country_code": "US",
        "estimation_country_status": "screening_proxy",
        "estimation_country_source": "OpenAI GPT-4 research release",
        "estimation_country_source_url": "https://openai.com/research/gpt-4",
        "active_parameters_billion": "440",
        "total_parameters_billion": "1760",
        "parameter_value_status": "estimated",
        "parameter_confidence": "low",
        "parameter_source": "Exploding Topics GPT-4 parameters report",
        "parameter_source_url": "https://explodingtopics.com/blog/gpt-parameters",
        "notes": "GPT-4 (Mar 2023) is treated as a multimodal MoE-style transformer with ~1.76T total parameters and ~440B active parameters in the central screening profile, based on widely cited public estimates.",
        "input_modalities": "text,image",
        "output_modalities": "text",
        "context_window_tokens": "8192",
        "max_output_tokens": "8192",
        "vision_support": "yes",
        "architecture_notes": "An Azure-hosted multimodal transformer with MoE-style routing and RLHF described in the GPT-4 research announcement.",
        "modalities_source": "OpenAI GPT-4 research release",
        "modalities_source_url": "https://openai.com/research/gpt-4",
        "context_source": "OpenAI GPT-4 research release",
        "context_source_url": "https://openai.com/research/gpt-4",
        "architecture_source": "OpenAI GPT-4 research release",
        "architecture_source_url": "https://openai.com/research/gpt-4",
        "release_date": "2023-03-14",
        "release_source": "OpenAI GPT-4 research release",
        "release_source_url": "https://openai.com/research/gpt-4",
        "screening_method_id": "market_multifactor_prompt_proxy_v1",
        "training_regime": "pretraining",
        "training_regime_status": "documented",
        "training_regime_source": "OpenAI GPT-4 technical report",
        "training_regime_source_url": "https://cdn.openai.com/papers/gpt-4.pdf",
        "training_multimodal": "yes",
        "training_multimodal_source": "OpenAI GPT-4 research release",
        "training_multimodal_source_url": "https://openai.com/research/gpt-4",
        "training_hardware_class_proxy": "modern_hyperscale_gpu",
        "training_hardware_source": "OpenAI GPT-4 research release",
        "training_hardware_source_url": "https://openai.com/research/gpt-4",
        "training_multifactor_method_id": "training_multifactor_proxy_v1",
        "training_multifactor_anchor": "Transformer (big) (strubell2019_co2_transformer_big); BLOOM 176B (luccioni2023_bloom_dynamic); BLOOM 176B (luccioni2023_bloom_extended)",
    },
]


def _normalize_row(row):
    return {key: (value or "") for key, value in row.items()}


def main():
    with MARKET_MODELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        assert header, "Missing header in market_models.csv"
        rows = [dict(row) for row in reader]

    rows_by_id = {row.get("model_id"): row for row in rows}

    for new_model in NEW_MARKET_MODELS:
        model_id = new_model["model_id"]
        if model_id in rows_by_id:
            target = rows_by_id[model_id]
        else:
            target = {col: "" for col in header}
            target["model_id"] = model_id
            rows.append(target)
            rows_by_id[model_id] = target

        for key, value in new_model.items():
            if key not in header:
                continue
            target[key] = value or ""

    with MARKET_MODELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))

    print("Market models dataset updated with GPT-4o mini, GPT-3.5 Turbo, and GPT-4 entries.")


if __name__ == "__main__":
    main()
