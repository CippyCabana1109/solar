"""Objective 3 Newsvendor commitment analysis for the 12:00-13:00 interval."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "obj3"
PREDICTIONS = ROOT / "outputs" / "obj1" / "march_week1_test_predictions.csv"
ERRORS = ROOT / "outputs" / "obj2" / "errors.csv"
MASTER = ROOT / "data" / "processed" / "master.csv"

PROVISIONAL_SOFT_CAP_W = 350_872.0
ILLUSTRATIVE_MARKET_PRICE_KSH_PER_KWH = 15.0
SCENARIOS = {
    "Conservative": {"Cu_KSh_per_kWh": 5.0, "Co_KSh_per_kWh": 15.0},
    "Balanced": {"Cu_KSh_per_kWh": 10.0, "Co_KSh_per_kWh": 10.0},
    "Aggressive": {"Cu_KSh_per_kWh": 15.0, "Co_KSh_per_kWh": 5.0},
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["timestamp"])
    errors = pd.read_csv(ERRORS, parse_dates=["timestamp"])
    master = pd.read_csv(MASTER, parse_dates=["timestamp"])
    predictions["date"] = predictions.timestamp.dt.date
    errors["date"] = errors.timestamp.dt.date
    master["date"] = master.timestamp.dt.date
    return predictions, errors, master


def quantile(error_kwh: np.ndarray, q: float) -> float:
    return float(np.quantile(error_kwh, q, method="linear"))


def calculate_row(date, forecast_power_w: float, actual_power_w: float, error_kwh: np.ndarray, scenario_name: str) -> dict[str, object]:
    costs = SCENARIOS[scenario_name]
    q_star = costs["Cu_KSh_per_kWh"] / (costs["Cu_KSh_per_kWh"] + costs["Co_KSh_per_kWh"])
    forecast_kwh = forecast_power_w / 1000.0
    actual_kwh = actual_power_w / 1000.0
    error_quantile_kwh = quantile(error_kwh, q_star)
    raw_commitment = forecast_kwh + error_quantile_kwh
    cap_kwh = PROVISIONAL_SOFT_CAP_W / 1000.0
    commitment_kwh = max(0.0, min(raw_commitment, cap_kwh))
    shortfall_kwh = max(commitment_kwh - actual_kwh, 0.0)
    surplus_kwh = max(actual_kwh - commitment_kwh, 0.0)
    imbalance_cost = costs["Co_KSh_per_kWh"] * shortfall_kwh + costs["Cu_KSh_per_kWh"] * surplus_kwh
    revenue = commitment_kwh * ILLUSTRATIVE_MARKET_PRICE_KSH_PER_KWH
    return {
        "date": date,
        "scenario": scenario_name,
        "forecast_energy_kWh": forecast_kwh,
        "relevant_error_quantile_kWh": error_quantile_kwh,
        "critical_fractile_q_star": q_star,
        "raw_commitment_kWh": raw_commitment,
        "committed_energy_kWh": commitment_kwh,
        "commitment_as_pct_forecast": 100 * commitment_kwh / forecast_kwh if forecast_kwh else np.nan,
        "actual_energy_kWh_ex_post": actual_kwh,
        "shortfall_kWh_ex_post": shortfall_kwh,
        "surplus_kWh_ex_post": surplus_kwh,
        "imbalance_cost_KSh": imbalance_cost,
        "illustrative_energy_revenue_KSh": revenue,
        "net_economic_outcome_KSh": revenue - imbalance_cost,
        "capacity_cap_used": "PROVISIONAL_SOFT_CAP_NOT_TRUE_RATED_CAPACITY",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions, errors, master = load_inputs()
    midday_errors = errors[(errors.timestamp.dt.hour == 12) & (errors.timestamp.dt.minute == 0)].copy()
    error_kwh = midday_errors.error_W.to_numpy(dtype=float) / 1000.0
    if len(error_kwh) != 7:
        raise ValueError(f"Expected 7 midday error observations, found {len(error_kwh)}")

    week_rows = []
    for date in sorted(predictions.date.unique()):
        forecast = predictions[predictions.date == date].loc[lambda x: x.timestamp.dt.hour == 12]
        actual = master[(master.date == date) & (master.timestamp.dt.hour == 12)]
        if len(forecast) != 1 or len(actual) != 1:
            raise ValueError(f"Expected one 12:00 row for {date}")
        for scenario in SCENARIOS:
            week_rows.append(calculate_row(date, float(forecast.iloc[0].XGBoost), float(actual.iloc[0].plant_power_w), error_kwh, scenario))
    appendix = pd.DataFrame(week_rows)
    appendix.to_csv(OUTPUT / "week1_daily_appendix.csv", index=False)

    headline_date = pd.Timestamp("2025-03-03").date()
    headline = appendix[appendix.date == headline_date].copy()
    headline.to_csv(OUTPUT / "headline_march3_results.csv", index=False)
    consolidated = headline.set_index("scenario").T.reset_index().rename(columns={"index": "Parameter"})
    consolidated.to_csv(OUTPUT / "consolidated_results.csv", index=False)

    chart = appendix.drop_duplicates("scenario")
    plt.figure(figsize=(8, 5))
    for scenario, group in appendix.groupby("scenario"):
        ordered = group.sort_values("critical_fractile_q_star")
        plt.plot(ordered["critical_fractile_q_star"], ordered["committed_energy_kWh"], marker="o", label=scenario)
    plt.xlabel("Newsvendor critical fractile q*")
    plt.ylabel("Committed energy (kWh)")
    plt.title("March Week 1 daily commitment versus critical fractile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "committed_energy_vs_critical_fractile.png", dpi=150)
    plt.close()

    summary = f"""Objective 3 determined market commitments for the 12:00-13:00 interval using the confirmed XGBoost forecasts and the empirical seven-observation March Week 1 midday error distribution from Objective 2. March 3, 2025 was selected as the headline day because it had the highest daytime inverter coverage in the week (78.30% mean); all seven days are provided in week1_daily_appendix.csv.

The scenarios are risk-ratio scenarios, not increasing total stakes: Conservative uses Cu={SCENARIOS['Conservative']['Cu_KSh_per_kWh']:.0f} and Co={SCENARIOS['Conservative']['Co_KSh_per_kWh']:.0f} KSh/kWh (q*=0.25), Balanced uses Cu={SCENARIOS['Balanced']['Cu_KSh_per_kWh']:.0f} and Co={SCENARIOS['Balanced']['Co_KSh_per_kWh']:.0f} KSh/kWh (q*=0.50), and Aggressive uses Cu={SCENARIOS['Aggressive']['Cu_KSh_per_kWh']:.0f} and Co={SCENARIOS['Aggressive']['Co_KSh_per_kWh']:.0f} KSh/kWh (q*=0.75). In every scenario Cu+Co=20 KSh/kWh.

Commitment was calculated as B*=forecast energy plus the q*-quantile of the midday forecast-error distribution, floored at zero and capped at {PROVISIONAL_SOFT_CAP_W:.0f} W / {PROVISIONAL_SOFT_CAP_W/1000:.3f} kWh. The capacity cap is a provisional soft cap based on the observed maximum, not the true rated plant capacity, and must be replaced when the authoritative rating is supplied.

The energy price of KSh {ILLUSTRATIVE_MARKET_PRICE_KSH_PER_KWH:.0f}/kWh is an illustrative assumption, not a verified Strathmore tariff, feed-in rate, or settlement price. As of 2026, Kenya Power base commercial/domestic rates plus fuel and forex adjustments typically push effective delivered electricity prices somewhat above KSh 15/kWh; this is therefore a conservative round placeholder, not an upper-bound estimate. Actual Strathmore settlement or feed-in pricing may differ and should replace it when confirmed.

Shortfall=max(commitment-actual,0) and surplus=max(actual-commitment,0) were used only for ex-post evaluation. Imbalance cost=Co*shortfall+Cu*surplus. Net economic outcome=illustrative energy revenue minus imbalance cost. Actual generation did not determine the commitment.

The midday error distribution contains only seven observations, so the commitments are scenario calculations with substantial quantile uncertainty. March production totals are derived_partial_sum values with approximately 77% average daytime inverter coverage, rather than certified full-plant readings. The economic results should therefore be interpreted as provisional, scenario-based results.
"""
    (OUTPUT / "objective3_summary.txt").write_text(summary, encoding="utf-8")
    print(headline.to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()