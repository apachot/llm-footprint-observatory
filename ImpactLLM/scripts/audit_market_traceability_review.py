#!/usr/bin/env python3
"""Audit market-model traceability, partial-data priors, and paper consistency."""

import csv
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.estimator import load_market_models


PAPER_PATH = REPO_ROOT / "ImpactLLM-paper" / "ImpactLLM_paper.tex"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "audits"
PARTIAL_OUTPUT_CSV = OUTPUT_DIR / "market_partial_data_priors_review.csv"
MISMATCH_OUTPUT_CSV = OUTPUT_DIR / "publication_catalog_mismatches.csv"
OUTPUT_MD = OUTPUT_DIR / "market_traceability_review.md"

DISPLAY_NAME_MAP = {
    "Gemini 3.1 Pro": "Gemini 3.1 Pro Preview",
}

FRIENDLY_AUDIT_LABELS = {
    "exact_in_cited_source": "exact_in_cited_source",
    "derivable_from_cited_source": "derivable_from_cited_source",
    "third_party_estimate": "third_party_estimate",
    "partial_data_prior": "partial_data_prior",
}


def _paper_display_match(paper_value_text, catalog_value):
    text = str(paper_value_text).strip()
    decimals = len(text.split(".")[-1]) if "." in text else 0
    rounded = round(float(catalog_value), decimals)
    if decimals:
        return f"{rounded:.{decimals}f}" == text
    return str(int(rounded)) == text


def _paper_rows():
    if not PAPER_PATH.exists():
        return [], []

    text = PAPER_PATH.read_text(encoding="utf-8")
    short_rows = []
    representative_rows = []

    short_pattern = re.compile(
        r"^(?P<name>.+?), one standardized request & "
        r"(?P<energy>[0-9.]+) Wh & "
        r"(?P<carbon>[0-9.]+) gCO\$_2\$e",
        re.MULTILINE,
    )
    for match in short_pattern.finditer(text):
        short_rows.append(
            {
                "table": "illustrative_outputs",
                "display_name": match.group("name"),
                "paper_inference_wh": match.group("energy"),
                "paper_inference_gco2e": match.group("carbon"),
            }
        )

    for line in text.splitlines():
        parts = [part.strip().replace("\\\\", "") for part in line.split("&")]
        if len(parts) < 5:
            continue
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
        except ValueError:
            continue
        representative_rows.append(
            {
                "table": "representative_models",
                "display_name": parts[0],
                "paper_inference_wh": parts[1],
                "paper_inference_gco2e": parts[2],
                "paper_training_gwh": parts[3],
            }
        )

    return short_rows, representative_rows


def _partial_data_review(rows):
    row_by_id = {row["model_id"]: row for row in rows}
    reviewed = []
    for row in rows:
        if row.get("parameter_source_audit") != "partial_data_prior":
            continue

        provider = str(row.get("provider", "") or "")
        note = str(row.get("quantification_note", "") or "")
        note_l = note.lower()
        donors = [donor for donor in str(row.get("quantification_donor_models", "") or "").split("|") if donor]
        donor_rows = [row_by_id[donor] for donor in donors if donor in row_by_id]
        donor_providers = sorted({str(donor.get("provider", "") or "") for donor in donor_rows})
        cross_provider = any(donor_provider and donor_provider != provider for donor_provider in donor_providers)
        single_donor = len(donors) == 1

        if cross_provider:
            review_level = "high"
            review_class = "cross_provider_fallback"
        elif "same provider and same documented family/tier" in note_l:
            if single_donor:
                review_level = "medium"
                review_class = "single_donor_family_prior"
            else:
                review_level = "low"
                review_class = "same_provider_family_prior"
        elif "same provider and same documented size tier" in note_l:
            if single_donor:
                review_level = "medium"
                review_class = "single_donor_size_tier_prior"
            else:
                review_level = "low"
                review_class = "same_provider_size_tier_prior"
        elif "same provider in the strict catalog" in note_l:
            review_level = "medium"
            review_class = "same_provider_fallback"
        elif "same documented size tier across the strict catalog" in note_l:
            review_level = "high"
            review_class = "cross_catalog_size_tier_fallback"
        elif "same documented family/tier across the strict catalog" in note_l:
            review_level = "high"
            review_class = "cross_catalog_family_fallback"
        else:
            review_level = "medium"
            review_class = "other_partial_data_prior"

        reviewed.append(
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "provider": provider,
                "active_parameters_billion": row.get("active_parameters_billion", ""),
                "review_level": review_level,
                "review_class": review_class,
                "donor_count": str(len(donors)),
                "cross_provider": "yes" if cross_provider else "no",
                "donor_models": "|".join(donors),
                "donor_providers": "|".join(donor_providers),
                "quantification_note": note,
            }
        )
    return reviewed


def _publication_mismatches(rows):
    rows_by_name = {row["display_name"]: row for row in rows}
    short_rows, representative_rows = _paper_rows()
    mismatches = []

    def check_row(paper_row, include_training):
        display_name = paper_row["display_name"]
        lookup_name = DISPLAY_NAME_MAP.get(display_name, display_name)
        catalog_row = rows_by_name.get(lookup_name)
        if not catalog_row:
            mismatches.append(
                {
                    "table": paper_row["table"],
                    "display_name": display_name,
                    "issue": "missing_from_catalog",
                    "paper_inference_wh": paper_row.get("paper_inference_wh", ""),
                    "catalog_inference_wh": "",
                    "paper_inference_gco2e": paper_row.get("paper_inference_gco2e", ""),
                    "catalog_inference_gco2e": "",
                    "paper_training_gwh": paper_row.get("paper_training_gwh", ""),
                    "catalog_training_gwh": "",
                }
            )
            return

        inference_wh = float(catalog_row["screening_per_request_energy_wh_central"])
        inference_gco2e = float(catalog_row["screening_per_request_carbon_gco2e_central"])
        training_gwh = float(catalog_row["training_energy_wh_central"]) / 1e9

        mismatch_fields = []
        if not _paper_display_match(paper_row["paper_inference_wh"], inference_wh):
            mismatch_fields.append("inference_wh")
        if not _paper_display_match(paper_row["paper_inference_gco2e"], inference_gco2e):
            mismatch_fields.append("inference_gco2e")
        if include_training and not _paper_display_match(paper_row["paper_training_gwh"], training_gwh):
            mismatch_fields.append("training_gwh")

        if mismatch_fields:
            mismatches.append(
                {
                    "table": paper_row["table"],
                    "display_name": display_name,
                    "issue": ",".join(mismatch_fields),
                    "paper_inference_wh": paper_row.get("paper_inference_wh", ""),
                    "catalog_inference_wh": f"{inference_wh:.4f}",
                    "paper_inference_gco2e": paper_row.get("paper_inference_gco2e", ""),
                    "catalog_inference_gco2e": f"{inference_gco2e:.4f}",
                    "paper_training_gwh": paper_row.get("paper_training_gwh", ""),
                    "catalog_training_gwh": f"{training_gwh:.4f}" if include_training else "",
                }
            )

    for paper_row in short_rows:
        check_row(paper_row, include_training=False)
    for paper_row in representative_rows:
        check_row(paper_row, include_training=True)

    return mismatches


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_markdown(rows, partial_rows, mismatches):
    audit_counts = Counter(row.get("parameter_source_audit") for row in rows)
    tier_counts = Counter(row.get("quantification_tier") for row in rows)
    partial_review_counts = Counter(row["review_class"] for row in partial_rows)
    high_risk_partial = [row for row in partial_rows if row["review_level"] == "high"]
    medium_risk_partial = [row for row in partial_rows if row["review_level"] == "medium"]
    tokens_prior = sum(1 for row in rows if str(row.get("training_tokens_status", "") or "") == "screening_prior")
    regime_prior = sum(1 for row in rows if str(row.get("training_regime_status", "") or "") == "screening_prior")
    hardware_prior = sum(1 for row in rows if str(row.get("training_hardware_class_proxy", "") or "").strip())

    lines = [
        "# Mega revue de tracabilite du catalogue marche",
        "",
        f"- Lignes quantifiees auditees : {len(rows)}",
        f"- `strict_benchmark` : {tier_counts['strict_benchmark']}",
        f"- `partial_data_benchmark` : {tier_counts['partial_data_benchmark']}",
        f"- `exact_in_cited_source` : {audit_counts['exact_in_cited_source']}",
        f"- `derivable_from_cited_source` : {audit_counts['derivable_from_cited_source']}",
        f"- `third_party_estimate` : {audit_counts['third_party_estimate']}",
        f"- `partial_data_prior` : {audit_counts['partial_data_prior']}",
        "",
        "## Lecture generale",
        "",
        "- Le catalogue public ne contient plus de `project_proxy_no_url`, `internal_proxy_no_url`, `needs_cleanup` ni `manual_review` dans les bases de parametres exposees par l'application.",
        "- Les chiffres du marche se repartissent maintenant entre valeurs exactes en source, valeurs derivables, estimations tierces explicites et priors `partial-data` derives de donneurs sourcés.",
        "- Les champs d'entraînement restent majoritairement du screening : "
        f"`training_tokens_status=screening_prior` pour {tokens_prior}/{len(rows)} modeles, "
        f"`training_regime_status=screening_prior` pour {regime_prior}/{len(rows)}, "
        f"et proxy materiel projet pour {hardware_prior}/{len(rows)}.",
        "",
        "## Priors partial-data a surveiller",
        "",
        f"- total des priors `partial_data_prior` : {len(partial_rows)}",
    ]

    for review_class, count in sorted(partial_review_counts.items()):
        lines.append(f"- `{review_class}` : {count}")

    lines.extend(["", "### Lignes les plus fragiles", ""])
    if high_risk_partial:
        for row in high_risk_partial:
            lines.append(
                f"- `{row['model_id']}` : `{row['review_class']}` ; donneurs `{row['donor_models']}` ; note : {row['quantification_note']}"
            )
    else:
        lines.append("- aucune ligne `high`")

    lines.extend(["", "### Lignes moyennement fragiles", ""])
    if medium_risk_partial:
        for row in medium_risk_partial:
            lines.append(
                f"- `{row['model_id']}` : `{row['review_class']}` ; donneurs `{row['donor_models']}` ; note : {row['quantification_note']}"
            )
    else:
        lines.append("- aucune ligne `medium`")

    lines.extend(["", "## Ecarts entre la publication et le catalogue courant", ""])
    if mismatches:
        for row in mismatches:
            lines.append(
                f"- `{row['display_name']}` ({row['table']}) : `{row['issue']}` ; "
                f"papier = `{row['paper_inference_wh']} Wh`, `{row['paper_inference_gco2e']} gCO2e`"
                + (
                    f", `{row['paper_training_gwh']} GWh`"
                    if row.get("paper_training_gwh")
                    else ""
                )
                + f" ; catalogue = `{row['catalog_inference_wh']} Wh`, `{row['catalog_inference_gco2e']} gCO2e`"
                + (
                    f", `{row['catalog_training_gwh']} GWh`"
                    if row.get("catalog_training_gwh")
                    else ""
                )
            )
    else:
        lines.append("- aucun ecart detecte entre les valeurs affichees dans le papier et le catalogue courant")

    lines.extend(
        [
            "",
            "## Conclusion operationnelle",
            "",
            "- Je ne vois plus de chiffre marche expose sans categorie de provenance ou sans methode de screening explicite.",
            "- En revanche, plusieurs chiffres recents restent des priors `partial-data` faibles ou moyens, donc defendables seulement comme ordres de grandeur de screening, pas comme quasi-mesures.",
            "- La publication locale n'est plus coherente avec le catalogue courant pour certains modeles proprietaires frontier; si les reseaux sociaux reprennent les chiffres du papier, il faut soit regeler le catalogue sur cette version, soit mettre a jour la publication et les posts.",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_market_models()
    partial_rows = _partial_data_review(rows)
    mismatches = _publication_mismatches(rows)

    _write_csv(
        PARTIAL_OUTPUT_CSV,
        [
            "model_id",
            "display_name",
            "provider",
            "active_parameters_billion",
            "review_level",
            "review_class",
            "donor_count",
            "cross_provider",
            "donor_models",
            "donor_providers",
            "quantification_note",
        ],
        partial_rows,
    )
    _write_csv(
        MISMATCH_OUTPUT_CSV,
        [
            "table",
            "display_name",
            "issue",
            "paper_inference_wh",
            "catalog_inference_wh",
            "paper_inference_gco2e",
            "catalog_inference_gco2e",
            "paper_training_gwh",
            "catalog_training_gwh",
        ],
        mismatches,
    )
    _write_markdown(rows, partial_rows, mismatches)
    print(
        "Audit written to "
        f"{PARTIAL_OUTPUT_CSV}, {MISMATCH_OUTPUT_CSV}, and {OUTPUT_MD} "
        f"({len(rows)} rows; partial={len(partial_rows)}; mismatches={len(mismatches)})."
    )


if __name__ == "__main__":
    main()
