"""Build the plant-level provisional master for January 2024-January 2025."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

from build_master import (
    INVENTORY,
    OUTPUT,
    ROOT,
    build_production,
    build_weather,
    interpolate_short_gaps,
)


PROVISIONAL_OUTPUT = ROOT / "data" / "processed" / "master_provisional_jan2024_jan2025.csv"
START = datetime(2024, 1, 1)
END = datetime(2025, 1, 31, 23, 0)


def has_plant_columns(path: Path) -> bool:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        headers = next(csv.reader(handle), [])
    return any(value.strip() == "Strathmore University- Production - Power (W)" for value in headers)


def paths() -> list[Path]:
    records = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    candidates = [
        ROOT / record["file"]
        for record in records
        if "/Latest Data/production/" in record["file"]
        and record["extension"] == ".csv"
        and record["date_end"] >= "2024-01-01 00:00:00"
        and record["date_start"] <= "2025-01-31 23:59:59"
        and "EXACT_DUPLICATE_OF" not in record["safety_flags"]
    ]
    selected: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not has_plant_columns(path):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            selected.append(path)
            seen.add(digest)
    return selected


def main() -> None:
    production = build_production(paths())
    weather = build_weather(include_api=False)
    interpolated = interpolate_short_gaps(weather)
    timestamps = sorted(
        timestamp
        for timestamp in set(production) & set(weather)
        if START <= timestamp <= END
    )
    fields = [
        "timestamp", "plant_power_w", "plant_energy_wh", "ghi_wh_m2", "dni_wh_m2",
        "dhi_wh_m2", "temperature_c", "relative_humidity_pct", "wind_speed_m_s", "weather_source",
    ]
    PROVISIONAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PROVISIONAL_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timestamp in timestamps:
            row = {"timestamp": timestamp.isoformat(sep=" ")}
            row.update(production[timestamp])
            row.update(weather[timestamp])
            writer.writerow(row)
    print(f"Rows: {len(timestamps)}")
    print(f"Columns: {len(fields)}")
    print(f"Date range: {timestamps[0].isoformat(sep=' ')} to {timestamps[-1].isoformat(sep=' ')}")
    print(f"Plant-level source files: {len(paths())}")
    print(f"Short irradiance gaps interpolated: {interpolated}")
    print(f"Wrote {PROVISIONAL_OUTPUT}")


if __name__ == "__main__":
    main()