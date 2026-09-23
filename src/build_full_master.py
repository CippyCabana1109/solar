"""Build the transparent full master with verified and derived source flags."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_master import NASA_EXISTING, ROOT, build_production, build_weather


RAW = ROOT / "data" / "raw" / "Data to be used" / "Data to be used"
API_FEB = ROOT / "data" / "external" / "nasa_power_api_february_2025_lat-1.31_lon36.81.json"
API_MARCH = ROOT / "data" / "external" / "nasa_power_api_march_2025_lat-1.31_lon36.81.json"
OUTPUT = ROOT / "data" / "processed" / "master.csv"
INVENTORY = ROOT / "outputs" / "reconnaissance" / "raw_file_inventory.csv"


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%m/%d/%y, %I:%M %p")
    except ValueError:
        return None


def number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned == "-999":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def derived_paths() -> list[Path]:
    return sorted(
        list((RAW / "Latest Data" / "production" / "2025").glob("Chart Feb_week*_prod*.csv"))
        + list((RAW / "Latest Data" / "production" / "2025").glob("Chart March_week_*_prod*.csv"))
    )


def build_partial_production() -> pd.DataFrame:
    buckets: dict[datetime, dict[str, list[float] | list[int]]] = defaultdict(
        lambda: {"power": [], "energy": [], "coverage": []}
    )
    seen: set[str] = set()
    for path in derived_paths():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader, [])
            power_indexes = [i for i, value in enumerate(headers) if "AC Production - Power (W)" in value]
            energy_indexes = [i for i, value in enumerate(headers) if "AC Production - Energy (Wh)" in value]
            for row in reader:
                timestamp = parse_time(row[0]) if row else None
                if timestamp is None:
                    continue
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                power_values = [number(row[i]) for i in power_indexes if i < len(row)]
                energy_values = [number(row[i]) for i in energy_indexes if i < len(row)]
                power_values = [value for value in power_values if value is not None]
                energy_values = [value for value in energy_values if value is not None]
                buckets[hour]["power"].append(sum(power_values))
                buckets[hour]["energy"].append(sum(energy_values))
                buckets[hour]["coverage"].append(len(power_values))
    rows = []
    for timestamp, values in sorted(buckets.items()):
        rows.append(
            {
                "timestamp": timestamp,
                "plant_power_w": sum(values["power"]) / len(values["power"]),
                "plant_energy_wh": sum(values["energy"]) / len(values["energy"]) if values["energy"] else None,
                "inverter_coverage_count": sum(values["coverage"]) / len(values["coverage"]),
                "inverter_coverage_min": min(values["coverage"]),
                "inverter_coverage_max": max(values["coverage"]),
                "inverter_coverage_pct": 100 * sum(values["coverage"]) / (26 * len(values["coverage"])),
                "data_source": "derived_partial_sum",
            }
        )
    return pd.DataFrame(rows)


def api_weather(path: Path) -> pd.DataFrame:
    document = json.loads(path.read_text(encoding="utf-8"))
    parameter = document["properties"]["parameter"]
    rows = []
    for key in parameter["T2M"]:
        rows.append(
            {
                "timestamp": datetime.strptime(key, "%Y%m%d%H"),
                "ghi_wh_m2": parameter["ALLSKY_SFC_SW_DWN"][key],
                "dni_wh_m2": parameter["ALLSKY_SFC_SW_DNI"][key],
                "dhi_wh_m2": parameter["ALLSKY_SFC_SW_DIFF"][key],
                "temperature_c": parameter["T2M"][key],
                "relative_humidity_pct": parameter["RH2M"][key],
                "wind_speed_m_s": parameter["WS10M"][key],
            }
        )
    return pd.DataFrame(rows)


def build_weather_full() -> pd.DataFrame:
    existing = build_weather(include_api=False)
    rows = [{"timestamp": timestamp, **values} for timestamp, values in existing.items()]
    weather = pd.DataFrame(rows)
    weather = pd.concat([weather, api_weather(API_FEB), api_weather(API_MARCH)], ignore_index=True)
    weather = weather.drop_duplicates("timestamp", keep="last")
    for column in ["ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2", "temperature_c", "relative_humidity_pct", "wind_speed_m_s"]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce").replace(-999, pd.NA)
    return weather


def main() -> None:
    verified = pd.read_csv(ROOT / "data" / "processed" / "master_provisional_jan2024_jan2025.csv", parse_dates=["timestamp"])
    verified["data_source"] = "verified_aggregate"
    verified = verified.drop(
        columns=[
            "ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2", "temperature_c",
            "relative_humidity_pct", "wind_speed_m_s", "weather_source",
        ]
    )
    verified["inverter_coverage_count"] = pd.NA
    verified["inverter_coverage_min"] = pd.NA
    verified["inverter_coverage_max"] = pd.NA
    verified["inverter_coverage_pct"] = pd.NA
    partial = build_partial_production()
    weather = build_weather_full()
    production = pd.concat([verified, partial], ignore_index=True, sort=False)
    production = production.drop_duplicates("timestamp", keep="first")
    full = production.merge(weather, on="timestamp", how="left", suffixes=("", "_weather"))
    full = full.sort_values("timestamp")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT, index=False)
    print(f"Rows: {len(full)}")
    print(f"Columns: {len(full.columns)}")
    print(f"Date range: {full.timestamp.min()} to {full.timestamp.max()}")
    print(f"Data sources: {full.data_source.value_counts().to_dict()}")
    for label, start, end in [("February", "2025-02-01", "2025-03-01"), ("March Week 1", "2025-03-01", "2025-03-08"), ("March Week 2", "2025-03-08", "2025-03-15"), ("March Week 3", "2025-03-15", "2025-03-22")]:
        subset = full[(full.timestamp >= start) & (full.timestamp < end)]
        print(f"{label} coverage_pct_mean: {subset.inverter_coverage_pct.mean()}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()