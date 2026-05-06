#!/usr/bin/env python3
"""Audit the parameter-source quality of the market-model catalog."""

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.market_catalog import classify_market_parameter_source

MARKET_MODELS_PATH = PROJECT_ROOT / "data" / "market_models.csv"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "audits"
OUTPUT_CSV = OUTPUT_DIR / "market_parameter_sources_audit.csv"
OUTPUT_MD = OUTPUT_DIR / "market_parameter_sources_audit.md"


def load_rows():
    with MARKET_MODELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "model_id",
        "display_name",
        "provider",
        "parameter_value_billion",
        "parameter_value_status",
        "parameter_source_audit",
        "parameter_source_audit_note",
        "parameter_source",
        "parameter_source_url",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def write_md(rows, counts):
    flagged = [row for row in rows if row["parameter_source_audit"] in {"needs_cleanup", "manual_review"}]
    lines = [
        "# Audit des sources de parametres des modeles du marche",
        "",
        f"- lignes auditees : {len(rows)}",
        f"- `exact_in_cited_source` : {counts['exact_in_cited_source']}",
        f"- `derivable_from_cited_source` : {counts['derivable_from_cited_source']}",
        f"- `third_party_estimate` : {counts['third_party_estimate']}",
        f"- `partial_data_prior` : {counts['partial_data_prior']}",
        f"- `project_proxy_no_url` : {counts['project_proxy_no_url']}",
        f"- `internal_proxy_no_url` : {counts['internal_proxy_no_url']}",
        f"- `needs_cleanup` : {counts['needs_cleanup']}",
        f"- `manual_review` : {counts['manual_review']}",
        "",
    ]

    if flagged:
        lines.extend(
            [
                "## Lignes a revoir",
                "",
            ]
        )
        for row in flagged:
            lines.append(
                f"- `{row['model_id']}` : `{row['parameter_source_audit']}` - {row['parameter_source_audit_note']}"
            )
    else:
        lines.extend(
            [
                "## Lignes a revoir",
                "",
                "- aucune",
            ]
        )

    lines.extend(
        [
            "",
            "## Regles d'audit",
            "",
            "- `exact_in_cited_source` : la valeur retenue devrait apparaitre explicitement dans la source citee.",
            "- `derivable_from_cited_source` : la valeur retenue est derivee directement d'une information explicite de la source citee.",
            "- `third_party_estimate` : la valeur retenue vient d'une estimation tierce, pas d'une divulgation du fournisseur.",
            "- `partial_data_prior` : la valeur retenue est derivee du catalogue strict a partir de donneurs sourcés partageant des metadonnees documentees avec le modele cible.",
            "- `project_proxy_no_url` : la valeur retenue est un proxy de screening interne et le champ `parameter_source_url` est volontairement vide.",
            "- `internal_proxy_no_url` : la valeur retenue est une approximation interne conservee sans URL de source de parametres.",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = []
    counts = Counter()
    for row in load_rows():
        parameter_value = row.get("total_parameters_billion") or row.get("active_parameters_billion") or ""
        audit_status, audit_note = classify_market_parameter_source(row)
        counts[audit_status] += 1
        rows.append(
            {
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "provider": row["provider"],
                "parameter_value_billion": parameter_value,
                "parameter_value_status": row.get("parameter_value_status", ""),
                "parameter_source_audit": audit_status,
                "parameter_source_audit_note": audit_note,
                "parameter_source": row.get("parameter_source", ""),
                "parameter_source_url": row.get("parameter_source_url", ""),
            }
        )

    write_csv(rows)
    write_md(rows, counts)
    print(
        f"Wrote {len(rows)} audited rows to {OUTPUT_CSV} and {OUTPUT_MD} "
        f"(needs_cleanup={counts['needs_cleanup']}, manual_review={counts['manual_review']})."
    )


if __name__ == "__main__":
    main()
