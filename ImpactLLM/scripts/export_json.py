#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "records.csv"
JSON_PATH = ROOT / "data" / "records.json"


def main():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Exported {len(records)} records to {JSON_PATH}")


if __name__ == "__main__":
    main()
