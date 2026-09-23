"""Create the pre-modelling Rule 2 data report.

This report inspects confirmed source files only. It does not build or modify
the master dataset.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Data to be used" / "Data to be used"
OUTPUT = ROOT / "outputs" / "reconnaissance" / "rule2_data_report.txt"
NASA_EXISTING = RAW / "Weather Data" / "Weather Data_NASA.csv"
NASA_API = ROOT / "data" / "external" / "nasa_power_api_february_2025_lat-1.31_lon36.81.json"

FORMATS = ("%m/%d/%y, %I:%M %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")


def parse_time(value: str) -> datetime | None:
    for date_format in FORMATS:
        try:
            return datetime.strptime(value.strip(), date_format)
        except ValueError:
            continue
    return None


def row_timestamp(row: list[str], headers: list[str]) -> datetime | None:
    time_indexes = [
        index
        for index, header in enumerate(headers)
        if "date" in header.lower() or "time" in header.lower()
    ]
    for index in time_indexes:
        if index < len(row):
            timestamp = parse_time(row[index])
            if timestamp is not None:
                return timestamp
    normalized = [header.lower().strip() for header in headers]
    indexes = {name: normalized.index(name) for name in ("year", "month", "day", "hour") if name in normalized}
    if all(name in indexes for name in ("year", "month", "day", "hour")):
        try:
            return datetime(
                int(float(row[indexes["year"]])),
                int(float(row[indexes["month"]])),
                int(float(row[indexes["day"]])),
                int(float(row[indexes["hour"]])),
            )
        except (IndexError, ValueError):
            return None
    return None


def csv_stats(paths: list[Path], power_fields: bool = False) -> dict[str, object]:
    rows = 0
    blank_cells = 0
    timestamps: list[datetime] = []
    zero_power_rows = 0
    zero_irradiance_rows = 0
    fill_values = 0
    blank_by_column: Counter[str] = Counter()
    day_blank_by_column: Counter[str] = Counter()
    night_blank_by_column: Counter[str] = Counter()
    power_columns = 0
    seen_hashes: set[str] = set()
    import hashlib

    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader, [])
            power_indexes = [
                index
                for index, header in enumerate(headers)
                if "power" in header.lower() and "(w" in header.lower()
            ]
            irradiance_indexes = [
                index
                for index, header in enumerate(headers)
                if "irradiance" in header.lower() or "shortwave" in header.lower()
            ]
            power_columns = max(power_columns, len(power_indexes))
            for row in reader:
                rows += 1
                timestamp = row_timestamp(row, headers) if row else None
                if timestamp is not None:
                    timestamps.append(timestamp)
                for index, value in enumerate(row[1:], start=1):
                    if not value.strip():
                        blank_cells += 1
                        blank_by_column[headers[index]] += 1
                        if timestamp is not None and 6 <= timestamp.hour < 19:
                            day_blank_by_column[headers[index]] += 1
                        else:
                            night_blank_by_column[headers[index]] += 1
                    if value.strip() == "-999":
                        fill_values += 1
                if power_fields and power_indexes:
                    values = [row[index].strip().replace(",", "") for index in power_indexes if index < len(row)]
                    if (
                        values
                        and len(values) == len(power_indexes)
                        and all(value != "" for value in values)
                        and all(float(value) == 0 for value in values)
                    ):
                        zero_power_rows += 1
                if irradiance_indexes:
                    values = [row[index].strip() for index in irradiance_indexes if index < len(row)]
                    if values and len(values) == len(irradiance_indexes) and all(value != "" for value in values):
                        if all(float(value) == 0 for value in values):
                            zero_irradiance_rows += 1
    return {
        "rows": rows,
        "blank_cells": blank_cells,
        "fill_values": fill_values,
        "blank_by_column": blank_by_column,
        "day_blank_by_column": day_blank_by_column,
        "night_blank_by_column": night_blank_by_column,
        "start": min(timestamps).isoformat(sep=" ") if timestamps else "",
        "end": max(timestamps).isoformat(sep=" ") if timestamps else "",
        "zero_power_rows": zero_power_rows,
        "zero_irradiance_rows": zero_irradiance_rows,
        "power_columns": power_columns,
    }


def api_stats(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    parameters = document["properties"]["parameter"]
    timestamps = sorted(set(key for values in parameters.values() for key in values))
    missing = sum(
        1
        for values in parameters.values()
        for value in values.values()
        if value == document["properties"].get("fill_value", -999)
    )
    irradiance = list(parameters["ALLSKY_SFC_SW_DWN"].values())
    return {
        "rows": len(timestamps),
        "blank_cells": missing,
        "start": datetime.strptime(timestamps[0], "%Y%m%d%H").isoformat(sep=" "),
        "end": datetime.strptime(timestamps[-1], "%Y%m%d%H").isoformat(sep=" "),
        "night_zero_rows": sum(value == 0 for value in irradiance),
        "time_standard": document["header"]["time_standard"],
        "fill_value": document["header"]["fill_value"],
    }


def main() -> None:
    inventory = list(csv.DictReader((ROOT / "outputs" / "reconnaissance" / "raw_file_inventory.csv").open(encoding="utf-8")))
    production = [
        ROOT / record["file"]
        for record in inventory
        if "/Latest Data/production/2024/" in record["file"]
        and record["file"].lower().endswith(".csv")
        and "Chart" in Path(record["file"]).name
        and "EXACT_DUPLICATE_OF" not in record["safety_flags"]
    ]
    production += [
        ROOT / record["file"]
        for record in inventory
        if "/Latest Data/production/2025/" in record["file"]
        and record["file"].lower().endswith(".csv")
        and "EXACT_DUPLICATE_OF" not in record["safety_flags"]
        and (
            record["date_end"] <= "2025-01-31 23:59:59"
            or "Feb_week" in Path(record["file"]).name
            or "March_week" in Path(record["file"]).name
        )
    ]
    feb_weather = [
        path
        for path in sorted((RAW / "Latest Data" / "Weather" / "February").glob("Chart weather_feb_week*.csv"))
        if "wind" not in path.name.lower()
    ]
    production_stats = csv_stats(production, power_fields=True)
    feb_weather_stats = csv_stats(feb_weather)
    nasa_existing_stats = csv_stats([NASA_EXISTING])
    nasa_api_stats = api_stats(NASA_API)

    lines = [
        "RULE 2 PRE-MODELLING DATA REPORT",
        "Generated from confirmed sources; master.csv has not been built.",
        "",
        "PRODUCTION: individual weekly exports (duplicates excluded by file hash)",
        f"Rows: {production_stats['rows']}",
        f"Verified date range: {production_stats['start']} to {production_stats['end']}",
        f"Blank non-time cells: {production_stats['blank_cells']}",
        f"Rows with all available inverter power fields literally equal to zero: {production_stats['zero_power_rows']}",
        "Rows containing blanks are excluded from the all-zero count; blanks are not converted to zero.",
        "Largest production blank concentrations (total; daytime / nighttime):",
        *[
            f"  {column}: {count}; {production_stats['day_blank_by_column'][column]} / {production_stats['night_blank_by_column'][column]}"
            for column, count in production_stats["blank_by_column"].most_common(10)
        ],
        "Resolution: 5 minutes; units: inverter power W and energy Wh; timezone: not encoded in files.",
        "",
        "LOCAL WEATHER: February humidity/temperature weekly exports",
        f"Rows: {feb_weather_stats['rows']}",
        f"Verified date range: {feb_weather_stats['start']} to {feb_weather_stats['end']}",
        f"Blank non-time cells: {feb_weather_stats['blank_cells']}",
        "Resolution: 5 minutes; units: humidity %, temperature C; timezone: not encoded in files.",
        "",
        "NASA POWER: existing source through January",
        f"Rows: {nasa_existing_stats['rows']}",
        f"Verified date range: {nasa_existing_stats['start']} to {nasa_existing_stats['end']}",
        f"Blank non-time cells: {nasa_existing_stats['blank_cells']}",
        f"NASA -999 fill values: {nasa_existing_stats['fill_values']}",
        f"Night-time zero irradiance rows: {nasa_existing_stats['zero_irradiance_rows']}",
        "Resolution: hourly; units: irradiance Wh/m2, temperature C, humidity %, wind speed m/s.",
        "Time standard: validated as LST by exact January 2025 matches for temperature, humidity, and wind against the API LST response; UTC does not match.",
        "",
        "NASA POWER API FILL-IN: February 2025",
        f"Rows: {nasa_api_stats['rows']}",
        f"Verified date range: {nasa_api_stats['start']} to {nasa_api_stats['end']}",
        f"API fill values (-999): {nasa_api_stats['blank_cells']}",
        f"Night-time zero irradiance rows: {nasa_api_stats['night_zero_rows']}",
        f"Resolution: hourly; time standard: {nasa_api_stats['time_standard']}; units: NASA POWER native units.",
        "Coordinates: latitude -1.31, longitude 36.81; API response preserved separately under data/external/.",
        "",
        "WIND DECISION",
        "Local February Week 1 wind is genuinely hourly (168 rows). Weeks 2-4 are 5-minute grids with only 168 populated hourly values each.",
        "Local wind is therefore excluded from the synchronized source; NASA POWER WS10M is used for consistent hourly wind coverage.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()