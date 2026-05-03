#!/usr/bin/env python3
"""Audit the source quality of the market-model training screening table."""

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
MARKET_MODELS_PATH = PROJECT_ROOT / "data" / "market_models.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "audits"
OUTPUT_CSV = OUTPUT_DIR / "market_training_sources_audit.csv"
OUTPUT_MD = OUTPUT_DIR / "market_training_sources_audit.md"

THIRD_PARTY_ESTIMATE_DOMAINS = {
    "artificialanalysis.ai",
    "explodingtopics.com",
    "lifearchitect.ai",
    "nexos.ai",
}


def domain(url):
    if not url:
        return ""
    return urlparse(url).netloc.replace("www.", "")


def classify_parameter(row):
    model_id = row["model_id"]
    status = row.get("parameter_value_status", "")
    source = row.get("parameter_source", "")
    url = row.get("parameter_source_url", "")
    notes = (row.get("notes") or "").lower()
    url_domain = domain(url)

    if model_id == "grok-1":
        return (
            "derivable_from_cited_source",
            "The cited xAI post publishes 314B total parameters and says 25% of weights are active per token; the retained 78.5B active value is derived from that ratio.",
        )

    if status in {"observed", "documented"}:
        if model_id in {"qwen3-32b", "qwen3-235b-a22b"} and "github.com/QwenLM/Qwen3" in url:
            return (
                "needs_source_tightening",
                "The retained value is exact, but the cited page is a generic release repository rather than the exact model card that prints the parameter count.",
            )
        return (
            "exact_in_cited_source",
            "The catalog marks this parameter value as observed/documented and points to a source expected to publish it directly.",
        )

    if url_domain in THIRD_PARTY_ESTIMATE_DOMAINS:
        return (
            "third_party_estimate",
            "The retained parameter value is estimated from a third-party taxonomy or report, not from the provider's own model page.",
        )

    if source.startswith("Project screening") or source.startswith("Project frontier"):
        return (
            "project_proxy_anchored_to_release",
            "The cited provider page anchors the model family or release line, but not a literal published parameter count.",
        )

    if "does not publish" in notes:
        return (
            "official_anchor_not_literal",
            "The provider page anchors the release line, while the numeric parameter count is retained by the project because no exact provider figure is published.",
        )

    return (
        "estimated_other",
        "The retained parameter value is estimated and should be treated as non-literal unless independently re-verified.",
    )


def classify_training_tokens(row):
    status = row.get("training_tokens_status", "")
    url = row.get("training_tokens_source_url", "")

    if status == "documented":
        return (
            "exact_in_cited_source",
            "The training-token count is marked as documented and should appear explicitly in the cited source.",
        )

    if status == "screening_prior":
        if url == "https://arxiv.org/abs/2203.15556":
            return (
                "project_prior_chinchilla",
                "The value is a project screening prior aligned with Chinchilla-style compute-optimal scaling, not a provider-published token count.",
            )
        if url:
            return (
                "project_prior_with_anchor",
                "The value is marked as a screening prior; the cited page serves as contextual anchoring rather than a literal token-count source.",
            )
        return (
            "project_prior_no_url",
            "The value is a project screening prior with no external source URL attached.",
        )

    return (
        "unclassified",
        "Unexpected training-token status; review manually.",
    )


def classify_training_regime(row):
    model_id = row["model_id"]
    status = row.get("training_regime_status", "")
    url = row.get("training_regime_source_url", "")

    if status == "documented":
        if model_id == "gpt-3.5-turbo":
            return (
                "potential_mismatch",
                "The cited ChatGPT post documents fine-tuning and RLHF around GPT-3.5; the retained table value simplifies the regime to pretraining.",
            )
        return (
            "directly_supported",
            "The cited source should directly support the retained training-regime label.",
        )

    if status == "screening_prior":
        if url:
            return (
                "project_prior_with_anchor",
                "The retained training regime is a project prior; the cited page only anchors the model family or related release context.",
            )
        return (
            "project_prior_no_url",
            "The retained training regime is a project prior with no attached URL.",
        )

    return (
        "unclassified",
        "Unexpected training-regime status; review manually.",
    )


def classify_training_multimodal(row):
    model_id = row["model_id"]
    value = row.get("training_multimodal", "")
    url = row.get("training_multimodal_source_url", "")

    if not url:
        return (
            "missing_source",
            "The multimodal training flag has no source URL attached.",
        )

    if model_id == "gpt-4o-mini" and value == "yes":
        return (
            "directly_supported",
            "The cited OpenAI page explicitly states that GPT-4o mini supports text and vision in the API.",
        )

    if model_id in {"qwen3-32b", "qwen3-235b-a22b"} and "github.com/QwenLM/Qwen3" in url:
        return (
            "needs_source_tightening",
            "The generic Qwen3 release repository is a weaker citation than the exact Hugging Face model card for a text-only capability claim.",
        )

    return (
        "capability_anchor",
        "The cited model page is used as the capability anchor for the retained multimodal yes/no flag.",
    )


def classify_training_hardware(row):
    model_id = row["model_id"]
    url = row.get("training_hardware_source_url", "")

    if not url:
        return (
            "project_proxy_no_url",
            "The hardware class is a project proxy retained from market status and serving mode, with no model-specific source URL.",
        )

    if model_id == "gpt-4o-mini":
        return (
            "proxy_anchor_not_explicit",
            "The retained hardware class is still a project proxy, and the cited GPT-4o mini release page does not explicitly publish training hardware.",
        )

    return (
        "project_proxy_with_anchor",
        "The retained hardware class is a project proxy; the cited source provides provider infrastructure context, not a literal hardware-class label from the provider.",
    )


def overall_status(field_classes):
    if any(value in {"needs_source_tightening", "potential_mismatch", "missing_source", "proxy_anchor_not_explicit"} for value in field_classes):
        return "needs_cleanup"
    if any(
        value
        in {
            "third_party_estimate",
            "project_proxy_anchored_to_release",
            "official_anchor_not_literal",
            "project_prior_chinchilla",
            "project_prior_with_anchor",
            "project_prior_no_url",
            "project_proxy_no_url",
            "project_proxy_with_anchor",
            "derivable_from_cited_source",
            "capability_anchor",
        }
        for value in field_classes
    ):
        return "mixed_documented_and_proxy"
    return "directly_documented"


def build_audit_rows(rows):
    audited = []
    for row in rows:
        parameter_class, parameter_note = classify_parameter(row)
        tokens_class, tokens_note = classify_training_tokens(row)
        regime_class, regime_note = classify_training_regime(row)
        multimodal_class, multimodal_note = classify_training_multimodal(row)
        hardware_class, hardware_note = classify_training_hardware(row)
        classes = [parameter_class, tokens_class, regime_class, multimodal_class, hardware_class]
        issues = []
        for label, value in (
            ("parameters", parameter_class),
            ("training_tokens", tokens_class),
            ("training_regime", regime_class),
            ("multimodal", multimodal_class),
            ("hardware", hardware_class),
        ):
            if value in {"needs_source_tightening", "potential_mismatch", "missing_source", "proxy_anchor_not_explicit"}:
                issues.append(label)

        audited.append(
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "provider": row["provider"],
                "training_parameters_billion": row.get("total_parameters_billion") or row.get("active_parameters_billion") or "",
                "training_parameters_audit": parameter_class,
                "training_parameters_note": parameter_note,
                "training_parameters_source_url": row.get("parameter_source_url", ""),
                "training_tokens_trillion": row.get("training_tokens_estimate_trillion", ""),
                "training_tokens_audit": tokens_class,
                "training_tokens_note": tokens_note,
                "training_tokens_source_url": row.get("training_tokens_source_url", ""),
                "training_regime": row.get("training_regime", ""),
                "training_regime_audit": regime_class,
                "training_regime_note": regime_note,
                "training_regime_source_url": row.get("training_regime_source_url", ""),
                "training_multimodal": row.get("training_multimodal", ""),
                "training_multimodal_audit": multimodal_class,
                "training_multimodal_note": multimodal_note,
                "training_multimodal_source_url": row.get("training_multimodal_source_url", ""),
                "training_hardware_class_proxy": row.get("training_hardware_class_proxy", ""),
                "training_hardware_audit": hardware_class,
                "training_hardware_note": hardware_note,
                "training_hardware_source_url": row.get("training_hardware_source_url", ""),
                "overall_audit_status": overall_status(classes),
                "issue_fields": ", ".join(issues),
                "issue_count": str(len(issues)),
            }
        )
    return audited


def write_csv(rows):
    fieldnames = [
        "model_id",
        "display_name",
        "provider",
        "training_parameters_billion",
        "training_parameters_audit",
        "training_parameters_note",
        "training_parameters_source_url",
        "training_tokens_trillion",
        "training_tokens_audit",
        "training_tokens_note",
        "training_tokens_source_url",
        "training_regime",
        "training_regime_audit",
        "training_regime_note",
        "training_regime_source_url",
        "training_multimodal",
        "training_multimodal_audit",
        "training_multimodal_note",
        "training_multimodal_source_url",
        "training_hardware_class_proxy",
        "training_hardware_audit",
        "training_hardware_note",
        "training_hardware_source_url",
        "overall_audit_status",
        "issue_fields",
        "issue_count",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows):
    overall_counts = Counter(row["overall_audit_status"] for row in rows)
    parameter_counts = Counter(row["training_parameters_audit"] for row in rows)
    token_counts = Counter(row["training_tokens_audit"] for row in rows)
    regime_counts = Counter(row["training_regime_audit"] for row in rows)
    multimodal_counts = Counter(row["training_multimodal_audit"] for row in rows)
    hardware_counts = Counter(row["training_hardware_audit"] for row in rows)
    flagged = [row for row in rows if row["overall_audit_status"] == "needs_cleanup"]

    lines = [
        "# Audit du tableau de screening d'entraînement",
        "",
        f"- Nombre de lignes auditées : {len(rows)}",
        f"- `directly_documented` : {overall_counts['directly_documented']}",
        f"- `mixed_documented_and_proxy` : {overall_counts['mixed_documented_and_proxy']}",
        f"- `needs_cleanup` : {overall_counts['needs_cleanup']}",
        "",
        "## Répartition par champ",
        "",
        "### Paramètres d'entraînement",
        "",
    ]
    for key, value in sorted(parameter_counts.items()):
        lines.append(f"- `{key}` : {value}")

    lines.extend(["", "### Tokens d'entraînement", ""])
    for key, value in sorted(token_counts.items()):
        lines.append(f"- `{key}` : {value}")

    lines.extend(["", "### Régime d'entraînement", ""])
    for key, value in sorted(regime_counts.items()):
        lines.append(f"- `{key}` : {value}")

    lines.extend(["", "### Multimodal", ""])
    for key, value in sorted(multimodal_counts.items()):
        lines.append(f"- `{key}` : {value}")

    lines.extend(["", "### Classe matérielle", ""])
    for key, value in sorted(hardware_counts.items()):
        lines.append(f"- `{key}` : {value}")

    lines.extend(
        [
            "",
            "## Lignes à nettoyer en priorité",
            "",
            "| Modèle | Champs | Motif |",
            "| --- | --- | --- |",
        ]
    )
    for row in flagged:
        notes = []
        for field_name, note_name in (
            ("training_parameters_audit", "training_parameters_note"),
            ("training_regime_audit", "training_regime_note"),
            ("training_multimodal_audit", "training_multimodal_note"),
            ("training_hardware_audit", "training_hardware_note"),
        ):
            if row[field_name] in {"needs_source_tightening", "potential_mismatch", "missing_source", "proxy_anchor_not_explicit"}:
                notes.append(row[note_name])
        lines.append(
            f"| {row['display_name']} | {row['issue_fields'] or '—'} | {' '.join(notes) or 'Review manually.'} |"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MARKET_MODELS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audited_rows = build_audit_rows(rows)
    write_csv(audited_rows)
    write_markdown(audited_rows)
    overall_counts = Counter(row["overall_audit_status"] for row in audited_rows)
    print(
        "Audit written to "
        f"{OUTPUT_CSV} and {OUTPUT_MD} "
        f"({len(audited_rows)} rows; needs_cleanup={overall_counts['needs_cleanup']})."
    )


if __name__ == "__main__":
    main()
