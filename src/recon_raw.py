"""Create a reproducible, non-destructive inventory of ``data/raw``.

Run from the repository root with ``python src/recon_raw.py``. This script
does not choose modelling data, modify raw files, or perform any modelling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "reconnaissance"
TIMESTAMP_FORMATS = (
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M %p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(32_768)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def clean_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()


def parse_timestamp(value: str) -> tuple[datetime | None, str]:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        return None, ""
    try:
        return datetime.fromisoformat(cleaned.removesuffix("Z")), "isoformat"
    except ValueError:
        pass
    for date_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(cleaned, date_format), date_format
        except ValueError:
            continue
    return None, ""


def separate_datetime(row: list[str], indexes: dict[str, int]) -> tuple[datetime | None, str]:
    required = ("year", "month", "day", "hour")
    if not all(key in indexes for key in required):
        return None, ""
    try:
        year = int(float(row[indexes["year"]]))
        month = int(float(row[indexes["month"]]))
        day = int(float(row[indexes["day"]]))
        hour = int(float(row[indexes["hour"]]))
        minute = int(float(row[indexes["minute"]])) if "minute" in indexes else 0
        return datetime(year, month, day, hour, minute), "separate date parts"
    except (IndexError, TypeError, ValueError):
        return None, ""


def timestamp_indexes(headers: list[str]) -> tuple[list[int], dict[str, int]]:
    timestamp_columns = [
        index
        for index, header in enumerate(headers)
        if re.search(r"date|time|timestamp", header, flags=re.IGNORECASE)
    ]
    parts: dict[str, int] = {}
    for index, header in enumerate(headers):
        normalized = re.sub(r"[^a-z]", "", header.lower())
        for part in ("year", "month", "day", "hour", "minute"):
            if normalized == part or normalized.endswith(part):
                parts.setdefault(part, index)
    return timestamp_columns, parts


def interval_summary(dates: list[datetime]) -> tuple[str, str]:
    if len(dates) < 2:
        return "", "INSUFFICIENT_TIMESTAMPS"
    differences = [
        (later - earlier).total_seconds() / 60
        for earlier, later in zip(dates, dates[1:])
        if later > earlier
    ]
    if not differences:
        return "", "NON_MONOTONIC_TIMESTAMPS"
    typical = median(differences)
    interval = f"{typical:g} minutes"
    span_minutes = (dates[-1] - dates[0]).total_seconds() / 60
    expected_rows = round(span_minutes / typical) + 1 if typical else 0
    ratio = len(dates) / expected_rows if expected_rows else 0
    if ratio < 0.95 or ratio > 1.05 or len(dates) != len(set(dates)):
        duplicate_count = len(dates) - len(set(dates))
        return interval, (
            f"ROW_DATE_RANGE_CHECK expected_about_{expected_rows}_parsed_{len(dates)}"
            f"_duplicate_timestamps_{duplicate_count}"
        )
    return interval, "OK"


def inspect_csv(path: Path) -> dict[str, Any]:
    dialect = csv_dialect(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, dialect)
        headers = [clean_header(header) for header in next(reader, [])]
        date_columns, date_parts = timestamp_indexes(headers)
        dates: list[datetime] = []
        parse_methods: Counter[str] = Counter()
        row_count = 0
        parsed_count = 0
        for row_count, row in enumerate(reader, start=1):
            parsed = None
            method = ""
            for index in date_columns:
                if index < len(row):
                    parsed, method = parse_timestamp(row[index])
                    if parsed is not None:
                        break
            if parsed is None:
                parsed, method = separate_datetime(row, date_parts)
            if parsed is not None:
                dates.append(parsed)
                parsed_count += 1
                parse_methods[method] += 1
    interval, consistency = interval_summary(dates)
    return {
        "column_headers": " | ".join(headers),
        "row_count": row_count,
        "parsed_timestamp_count": parsed_count,
        "unparsed_timestamp_count": row_count - parsed_count,
        "date_parse_method": "; ".join(
            f"{method}:{count}" for method, count in sorted(parse_methods.items())
        ),
        "date_start": min(dates).isoformat(sep=" ") if dates else "",
        "date_end": max(dates).isoformat(sep=" ") if dates else "",
        "typical_interval": interval,
        "row_date_consistency": consistency,
        "read_note": "",
    }


def classify(relative_path: str, headers: str) -> tuple[str, str]:
    text = f"{relative_path} {headers}".lower()
    has_pv = bool(
        re.search(
            r"production|inverter|\binv\s*\d|ac[ _-]*production|power \(w\)|energy \(wh\)|kwh",
            text,
        )
    )
    has_weather = bool(
        re.search(r"irradiance|temperature|humidity|wind|weather|all sky|insolation|clearness", text)
    )
    is_nasa = bool(re.search(r"nasa|power[_ ]?point|all sky surface", text))
    is_template = bool(re.search(r"template|example|sample|\bbook\d*\b", text))
    is_split_or_model = bool(
        re.search(r"training|train|validat|\btest\b|prediction|sarimax|xgboost|prophet|model", text)
    )
    is_combined = bool(re.search(r"combined|dataset", text))
    evidence = []
    if has_pv:
        evidence.append("PV/production markers")
    if has_weather:
        evidence.append("weather markers")
    if is_nasa:
        evidence.append("NASA POWER markers")
    if is_split_or_model:
        evidence.append("split/model marker")
    if is_combined:
        evidence.append("combined/dataset marker")
    if is_template:
        evidence.append("template/example marker")
    if is_template:
        category = "template/other"
    elif (has_pv and has_weather) or is_split_or_model or is_combined:
        category = "combined/model dataset"
    elif is_nasa:
        category = "NASA POWER"
    elif has_pv:
        category = "PV/production"
    elif has_weather:
        category = "weather"
    else:
        category = "template/other"
    return category, "; ".join(evidence) or "no recognised source markers"


def safety_flags(relative_path: str, date_start: str, date_end: str) -> str:
    name = relative_path.lower()
    flags: list[str] = []
    if re.search(r"training|train|validat|\btest\b", name):
        flags.append("LABELLED_SPLIT_DO_NOT_USE_FOR_TRAINING")
    if date_start and date_end:
        start = datetime.fromisoformat(date_start)
        end = datetime.fromisoformat(date_end)
        if start <= datetime(2025, 3, 31, 23, 59, 59) and end >= datetime(2025, 3, 1):
            flags.append("CONTAINS_MARCH_2025_DO_NOT_USE_FOR_TRAINING")
    if re.search(r"march[^/]*2025", name):
        if "CONTAINS_MARCH_2025_DO_NOT_USE_FOR_TRAINING" not in flags:
            flags.append("MARCH_2025_FILENAME_DO_NOT_USE_FOR_TRAINING")
    elif "march" in name:
        flags.append("MARCH_IN_FILENAME_DATE_UNVERIFIED")
    return "; ".join(flags)


def inspect_file(path: Path) -> dict[str, Any]:
    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
    record: dict[str, Any] = {
        "file": relative_path,
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "sha256": sha256(path),
        "column_headers": "",
        "row_count": "",
        "parsed_timestamp_count": "",
        "unparsed_timestamp_count": "",
        "date_parse_method": "",
        "date_start": "",
        "date_end": "",
        "typical_interval": "",
        "row_date_consistency": "",
        "read_note": "",
    }
    if path.suffix.lower() == ".csv":
        try:
            record.update(inspect_csv(path))
        except (OSError, UnicodeError, csv.Error) as error:
            record["read_note"] = f"CSV inspection failed: {error}"
    else:
        record["read_note"] = "Not inspected: non-CSV file (metadata retained)."
    category, evidence = classify(relative_path, record["column_headers"])
    record["classification"] = category
    record["classification_evidence"] = evidence
    record["safety_flags"] = safety_flags(relative_path, record["date_start"], record["date_end"])
    return record


def add_duplicate_flags(records: list[dict[str, Any]]) -> None:
    hashes = Counter(record["sha256"] for record in records)
    canonical: dict[str, str] = {}
    for record in records:
        digest = record["sha256"]
        if hashes[digest] > 1:
            if digest not in canonical:
                canonical[digest] = record["file"]
                continue
            duplicate_note = f"EXACT_DUPLICATE_OF={canonical[digest]}"
            record["safety_flags"] = "; ".join(
                filter(None, [record["safety_flags"], duplicate_note])
            )


def write_inventory(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with (output_dir / "raw_file_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "raw_file_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.raw_root.is_dir():
        raise SystemExit(f"Raw-data directory does not exist: {args.raw_root}")
    records = [
        inspect_file(path)
        for path in sorted(args.raw_root.rglob("*"))
        if path.is_file()
    ]
    add_duplicate_flags(records)
    write_inventory(records, args.output_dir)
    print(f"Inventoried {len(records)} files.")
    print(f"CSV output: {args.output_dir / 'raw_file_inventory.csv'}")
    print(f"JSON output: {args.output_dir / 'raw_file_inventory.json'}")


if __name__ == "__main__":
    main()
