#!/usr/bin/env python3
"""Add or update baseline entries in data/market_models.csv."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKET_MODELS_PATH = ROOT / "data" / "market_models.csv"

NEW_MARKET_MODELS = [
    {
        "model_id": "gpt-3",
        "display_name": "GPT-3 (Davinci)",
        "provider": "openai",
        "market_status": "api",
        "serving_mode": "closed",
        "server_country": "United States",
        "server_country_status": "multi_region",
        "server_country_source": "OpenAI API launch blog",
        "server_country_source_url": "https://openai.com/blog/openai-api/",
        "estimation_country_code": "US",
        "estimation_country_status": "screening_proxy",
        "estimation_country_source": "OpenAI API launch blog",
        "estimation_country_source_url": "https://openai.com/blog/openai-api/",
        "active_parameters_billion": "175",
        "total_parameters_billion": "175",
        "parameter_value_status": "observed",
        "parameter_confidence": "medium",
        "parameter_source": "Brown et al. (2020)",
        "parameter_source_url": "https://arxiv.org/abs/2005.14165",
        "notes": "Original GPT-3 API release describing the 175B decoder-only transformer.",
        "input_modalities": "text",
        "output_modalities": "text",
        "context_window_tokens": "2048",
        "max_output_tokens": "2048",
        "vision_support": "no",
        "architecture_notes": "Dense decoder-only transformer with 96 layers as documented by Brown et al. (2020).",
        "modalities_source": "Brown et al. (2020)",
        "modalities_source_url": "https://arxiv.org/abs/2005.14165",
        "context_source": "Brown et al. (2020)",
        "context_source_url": "https://arxiv.org/abs/2005.14165",
        "architecture_source": "Brown et al. (2020)",
        "architecture_source_url": "https://arxiv.org/abs/2005.14165",
        "release_date": "2020-06-11",
        "release_source": "OpenAI API launch blog",
        "release_source_url": "https://openai.com/blog/openai-api/",
        "screening_method_id": "market_multifactor_prompt_proxy_v1",
        "training_tokens_estimate_trillion": "0.3",
        "training_tokens_status": "observed",
        "training_tokens_source": "Brown et al. (2020)",
        "training_tokens_source_url": "https://arxiv.org/abs/2005.14165",
        "training_regime": "pretraining",
        "training_regime_status": "documented",
        "training_regime_source": "Brown et al. (2020)",
        "training_regime_source_url": "https://arxiv.org/abs/2005.14165",
        "training_multimodal": "no",
        "training_multimodal_source": "Brown et al. (2020)",
        "training_multimodal_source_url": "https://arxiv.org/abs/2005.14165",
        "training_hardware_class_proxy": "standard_gpu_cluster",
        "training_hardware_source": "Brown et al. (2020)",
        "training_hardware_source_url": "https://arxiv.org/abs/2005.14165",
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

    print("Market models dataset updated with GPT-3, GPT-3.5 Turbo, and GPT-4 entries.")


if __name__ == "__main__":
    main()
