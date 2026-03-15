#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "records.csv"
METADATA_PATH = ROOT / "data" / "record_metadata.csv"

REQUIRED_FIELDS = [
    "record_id",
    "study_key",
    "citation",
    "publication_year",
    "phase",
    "impact_category",
    "metric_name",
    "metric_value",
    "metric_unit",
    "model_or_scope",
    "geography",
    "time_scope",
    "system_boundary",
    "data_type",
    "source_locator",
    "source_url",
    "llm_normalized",
    "model_parameters_normalized",
    "country_normalized",
]


def main():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    if not records:
        raise SystemExit("Dataset is empty")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in records[0]]
    if missing_fields:
        raise SystemExit(f"Missing columns: {missing_fields}")

    record_ids = set()
    for index, record in enumerate(records, start=2):
        for field in REQUIRED_FIELDS:
            if record[field] == "":
                raise SystemExit(f"Empty required field '{field}' at CSV line {index}")
        if record["record_id"] in record_ids:
            raise SystemExit(f"Duplicate record_id '{record['record_id']}' at CSV line {index}")
        record_ids.add(record["record_id"])

    metadata_ids = {row["record_id"] for row in metadata_rows}
    if metadata_ids != record_ids:
        missing_in_metadata = sorted(record_ids - metadata_ids)
        extra_in_metadata = sorted(metadata_ids - record_ids)
        raise SystemExit(
            f"Metadata mismatch. Missing in metadata: {missing_in_metadata}. Extra in metadata: {extra_in_metadata}"
        )

    print(f"Validated {len(records)} records and {len(metadata_rows)} metadata rows")


if __name__ == "__main__":
    main()
