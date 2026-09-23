"""Build the hourly synchronized master dataset before modelling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Data to be used" / "Data to be used"
INVENTORY = ROOT / "outputs" / "reconnaissance" / "raw_file_inventory.csv"
NASA_EXISTING = RAW / "Weather Data" / "Weather Data_NASA.csv"
NASA_API = ROOT / "data" / "external" / "nasa_power_api_february_2025_lat-1.31_lon36.81.json"
OUTPUT = ROOT / "data" / "processed" / "master.csv"


def parse_time(value: str) -> datetime | None:
    for date_format in ("%m/%d/%y, %I:%M %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), date_format)
        except ValueError:
            continue
    return None


def production_paths() -> list[Path]:
    records = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    paths = [
        ROOT / record["file"]
        for record in records
        if "/Latest Data/production/2024/" in record["file"]
        and record["file"].lower().endswith(".csv")
        and "EXACT_DUPLICATE_OF" not in record["safety_flags"]
    ]
    paths += [
        ROOT / record["file"]
        for record in records
        if "/Latest Data/production/2025/" in record["file"]
        and record["file"].lower().endswith(".csv")
        and "EXACT_DUPLICATE_OF" not in record["safety_flags"]
        and (
            record["date_end"] <= "2025-01-31 23:59:59"
            or "Feb_week" in Path(record["file"]).name
            or "March_week" in Path(record["file"]).name
        )
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique


def number(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if not value or value == "-999":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_production(paths: list[Path] | None = None) -> dict[datetime, dict[str, float | None]]:
    buckets: defaultdict[datetime, dict[str, list[float]]] = defaultdict(lambda: {"power": [], "energy": []})
    for path in paths or production_paths():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader, [])
            power_index = next(
                (i for i, value in enumerate(headers) if value.strip() == "Strathmore University- Production - Power (W)"),
                None,
            )
            energy_index = next(
                (i for i, value in enumerate(headers) if value.strip() == "Strathmore University- Production - Energy (Wh)"),
                None,
            )
            for row in reader:
                timestamp = parse_time(row[0]) if row else None
                if timestamp is None:
                    continue
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                for name, index in (("power", power_index), ("energy", energy_index)):
                    if index is None:
                        continue
                    value = number(row[index]) if index < len(row) else None
                    if value is None and timestamp.hour < 6 or value is None and timestamp.hour >= 19:
                        value = 0.0
                    if value is not None:
                        buckets[hour][name].append(value)
    return {
        timestamp: {
            "plant_power_w": sum(values["power"]) / len(values["power"]) if values["power"] else None,
            "plant_energy_wh": sum(values["energy"]) if values["energy"] else None,
        }
        for timestamp, values in buckets.items()
    }


def add_weather_row(weather: dict[datetime, dict[str, float | str]], timestamp: datetime, values: dict[str, float | None], source: str) -> None:
    weather[timestamp] = {**values, "weather_source": source}


def build_weather(include_api: bool = True) -> dict[datetime, dict[str, float | str]]:
    weather: dict[datetime, dict[str, float | str]] = {}
    with NASA_EXISTING.open(encoding="cp1252", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
        indexes = {header.lower().replace("\n", " "): index for index, header in enumerate(headers)}
        for row in reader:
            if len(row) < 14:
                continue
            timestamp = datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            add_weather_row(
                weather,
                timestamp,
                {
                    "ghi_wh_m2": number(row[5]),
                    "dni_wh_m2": number(row[6]),
                    "dhi_wh_m2": number(row[7]),
                    "temperature_c": number(row[8]),
                    "relative_humidity_pct": number(row[12]),
                    "wind_speed_m_s": number(row[13]),
                },
                "existing_nasa_power_lst",
            )
    if not include_api:
        return weather
    document = json.loads(NASA_API.read_text(encoding="utf-8"))
    parameters = document["properties"]["parameter"]
    for key in parameters["T2M"]:
        timestamp = datetime.strptime(key, "%Y%m%d%H")
        add_weather_row(
            weather,
            timestamp,
            {
                "ghi_wh_m2": number(str(parameters["ALLSKY_SFC_SW_DWN"][key])),
                "dni_wh_m2": number(str(parameters["ALLSKY_SFC_SW_DNI"][key])),
                "dhi_wh_m2": number(str(parameters["ALLSKY_SFC_SW_DIFF"][key])),
                "temperature_c": number(str(parameters["T2M"][key])),
                "relative_humidity_pct": number(str(parameters["RH2M"][key])),
                "wind_speed_m_s": number(str(parameters["WS10M"][key])),
            },
            "nasa_power_api_february_2025_lst",
        )
    return weather


def interpolate_short_gaps(weather: dict[datetime, dict[str, float | str]]) -> int:
    filled = 0
    fields = ("ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2")
    timestamps = sorted(weather)
    for field in fields:
        for index, timestamp in enumerate(timestamps):
            if weather[timestamp].get(field) is not None:
                continue
            gap: list[datetime] = []
            cursor = index
            while cursor < len(timestamps) and weather[timestamps[cursor]].get(field) is None and len(gap) <= 2:
                gap.append(timestamps[cursor])
                cursor += 1
            if len(gap) > 2 or not gap or cursor >= len(timestamps):
                continue
            left = timestamps[index - 1] if index else None
            right = timestamps[cursor]
            if left is None or left.date() != right.date() or timestamp.date() != right.date():
                continue
            left_value = weather[left].get(field)
            right_value = weather[right].get(field)
            if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
                continue
            step = (right_value - left_value) / (len(gap) + 1)
            for offset, missing_timestamp in enumerate(gap, start=1):
                weather[missing_timestamp][field] = left_value + step * offset
                filled += 1
    return filled


def main() -> None:
    production = build_production()
    weather = build_weather()
    interpolated = interpolate_short_gaps(weather)
    rows: list[dict[str, object]] = []
    for timestamp in sorted(set(production) & set(weather)):
        row = {"timestamp": timestamp.isoformat(sep=" ")}
        row.update(production[timestamp])
        row.update(weather[timestamp])
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "plant_power_w", "plant_energy_wh", "ghi_wh_m2", "dni_wh_m2",
        "dhi_wh_m2", "temperature_c", "relative_humidity_pct", "wind_speed_m_s", "weather_source",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(fields)}")
    print(f"Date range: {rows[0]['timestamp']} to {rows[-1]['timestamp']}")
    print(f"Short irradiance gaps interpolated: {interpolated}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()