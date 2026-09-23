"""Objective 2 forecast-error and uncertainty analysis for XGBoost."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import DAYTIME_END_HOUR_EXCLUSIVE, DAYTIME_START_HOUR


ROOT = Path(__file__).resolve().parents[1]
OBJ1 = ROOT / "outputs" / "obj1"
OUTPUT = ROOT / "outputs" / "obj2"
MASTER = ROOT / "data" / "processed" / "master.csv"
QUANTILES = {"80%": (0.10, 0.90), "90%": (0.05, 0.95), "95%": (0.025, 0.975)}


def daytime(frame: pd.DataFrame) -> pd.DataFrame:
    hours = frame["timestamp"].dt.hour
    return frame[(hours >= DAYTIME_START_HOUR) & (hours < DAYTIME_END_HOUR_EXCLUSIVE)].copy()


def error_summary(frame: pd.DataFrame, label: str) -> tuple[dict[str, object], pd.DataFrame]:
    errors = frame["error_W"].to_numpy(dtype=float)
    forecast = frame["forecast_W"].to_numpy(dtype=float)
    relative = frame.loc[frame["forecast_W"].abs() > 1e-9, "relative_error_pct"].dropna()
    row: dict[str, object] = {
        "analysis": label,
        "observations": len(frame),
        "MBE_W": float(np.mean(errors)),
        "error_std_W": float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan,
        "forecast_mean_W": float(np.mean(forecast)),
        "relative_error_mean_pct": float(relative.mean()) if len(relative) else np.nan,
        "relative_error_median_pct": float(relative.median()) if len(relative) else np.nan,
    }
    for probability in (0.025, 0.05, 0.10, 0.90, 0.95, 0.975):
        row[f"error_q{probability:g}"] = float(np.quantile(errors, probability))
    for name, (lower_probability, upper_probability) in QUANTILES.items():
        row[f"empirical_{name}_lower_W"] = row[f"error_q{lower_probability:g}"]
        row[f"empirical_{name}_upper_W"] = row[f"error_q{upper_probability:g}"]
        z = {"80%": 1.2815516, "90%": 1.6448536, "95%": 1.959964}[name]
        row[f"normal_{name}_lower_W"] = row["MBE_W"] - z * row["error_std_W"]
        row[f"normal_{name}_upper_W"] = row["MBE_W"] + z * row["error_std_W"]
    return row, relative.to_frame(name="relative_error_pct")


def build_primary_errors() -> pd.DataFrame:
    predictions = pd.read_csv(OBJ1 / "march_week1_test_predictions.csv", parse_dates=["timestamp"])
    master = pd.read_csv(MASTER, parse_dates=["timestamp"])
    actual = master[["timestamp", "plant_power_w", "data_source", "inverter_coverage_pct"]]
    frame = predictions[["timestamp", "actual_W", "XGBoost"]].rename(columns={"XGBoost": "forecast_W"})
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    actual = actual[actual["timestamp"].isin(frame["timestamp"])].sort_values("timestamp").reset_index(drop=True)
    if len(frame) != len(actual) or not frame["timestamp"].equals(actual["timestamp"]):
        raise ValueError("XGBoost prediction and master timestamps do not match exactly.")
    if not np.allclose(frame["actual_W"].to_numpy(), actual["plant_power_w"].to_numpy(), equal_nan=False):
        raise ValueError("Saved XGBoost actual values do not match master plant_power_w values.")
    frame = frame.merge(actual.drop(columns="plant_power_w"), on="timestamp", how="left", validate="one_to_one")
    frame["error_W"] = frame["actual_W"] - frame["forecast_W"]
    frame["relative_error_pct"] = np.where(
        frame["forecast_W"].abs() > 1e-9,
        100 * frame["error_W"] / frame["forecast_W"].abs(),
        np.nan,
    )
    frame["daytime"] = frame["timestamp"].dt.hour.between(DAYTIME_START_HOUR, DAYTIME_END_HOUR_EXCLUSIVE - 1)
    frame["evaluation_source"] = "genuine_march_week1_test"
    return frame


def build_supplementary_errors() -> pd.DataFrame:
    predictions = pd.read_csv(OBJ1 / "validation_only_predictions.csv", parse_dates=["timestamp"])
    frame = predictions[predictions["model"] == "XGBoost"].copy()
    frame = frame.rename(columns={"actual_W": "actual_W", "predicted_W": "forecast_W"})
    frame["error_W"] = frame["actual_W"] - frame["forecast_W"]
    frame["relative_error_pct"] = np.where(
        frame["forecast_W"].abs() > 1e-9,
        100 * frame["error_W"] / frame["forecast_W"].abs(),
        np.nan,
    )
    frame["daytime"] = frame["timestamp"].dt.hour.between(DAYTIME_START_HOUR, DAYTIME_END_HOUR_EXCLUSIVE - 1)
    frame["evaluation_source"] = "supplementary_9_origin_7_day_oof"
    return frame[["timestamp", "actual_W", "forecast_W", "error_W", "relative_error_pct", "daytime", "evaluation_source"]]


def save_histogram(primary_day: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(primary_day["error_W"], bins=16, color="steelblue", edgecolor="white")
    plt.axvline(primary_day["error_W"].mean(), color="darkred", linestyle="--", label="MBE")
    plt.xlabel("Forecast error: actual - forecast (W)")
    plt.ylabel("Count")
    plt.title("XGBoost forecast-error distribution: March Week 1 daytime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "error_histogram.png", dpi=150)
    plt.close()


def save_bounds_plot(primary_day: pd.DataFrame, summary: dict[str, object]) -> None:
    plot = primary_day.sort_values("timestamp").copy()
    plt.figure(figsize=(11, 5))
    plt.plot(plot["timestamp"], plot["actual_W"], color="black", label="Actual")
    plt.plot(plot["timestamp"], plot["forecast_W"], color="tab:blue", label="XGBoost forecast")
    colors = {"80%": "#a7d8f0", "90%": "#70b7d6", "95%": "#3d8fb3"}
    for name in ("95%", "90%", "80%"):
        lower = plot["forecast_W"] + summary[f"empirical_{name}_lower_W"]
        upper = plot["forecast_W"] + summary[f"empirical_{name}_upper_W"]
        plt.fill_between(plot["timestamp"], lower, upper, color=colors[name], alpha=0.25, label=f"Empirical {name} interval")
    plt.ylabel("Plant power (W)")
    plt.title("XGBoost actual versus forecast with empirical uncertainty bounds")
    plt.legend(ncol=2)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT / "actual_forecast_uncertainty_bounds.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = build_primary_errors()
    supplementary = build_supplementary_errors()
    primary.to_csv(OUTPUT / "errors.csv", index=False)
    primary_day = daytime(primary)
    supplementary_day = daytime(supplementary)
    primary_summary, _ = error_summary(primary_day, "primary_genuine_march_week1_daytime")
    supplementary_summary, _ = error_summary(supplementary_day, "supplementary_9_origin_oof_daytime")
    stats = pd.DataFrame([primary_summary, supplementary_summary])
    stats.to_csv(OUTPUT / "summary_statistics.csv", index=False)

    midday = primary[(primary["timestamp"].dt.hour >= 12) & (primary["timestamp"].dt.hour < 13)].copy()
    midday_day = daytime(midday)
    midday_summary, _ = error_summary(midday_day, "primary_march_week1_12_to_13")
    pd.DataFrame([midday_summary]).to_csv(OUTPUT / "midday_12_to_13_summary.csv", index=False)

    final_rows = []
    for label, result, interpretation in [
        ("Mean Error / Bias", primary_summary["MBE_W"], "Average actual-minus-forecast error; positive means under-forecasting."),
        ("Error Standard Deviation", primary_summary["error_std_W"], "Typical dispersion of daytime errors around the bias."),
        ("Relative Uncertainty", primary_summary["relative_error_median_pct"], "Median forecast-relative error; capacity-relative value pending rated capacity."),
        ("80% uncertainty interval", f"{primary_summary['empirical_80%_lower_W']:.2f} to {primary_summary['empirical_80%_upper_W']:.2f} W", "Empirical daytime error range containing the central 80% of March test errors."),
        ("90% uncertainty interval", f"{primary_summary['empirical_90%_lower_W']:.2f} to {primary_summary['empirical_90%_upper_W']:.2f} W", "Empirical daytime error range containing the central 90% of March test errors."),
        ("95% uncertainty interval", f"{primary_summary['empirical_95%_lower_W']:.2f} to {primary_summary['empirical_95%_upper_W']:.2f} W", "Empirical daytime error range containing the central 95% of March test errors."),
        ("Capacity-relative uncertainty", "PENDING_RATED_CAPACITY", "Cannot calculate until the plant rated capacity in kW or kWp is provided."),
        ("12:00-13:00 error sample", len(midday_day), "Small primary test-week sample; one observation per day, so quantiles are unstable."),
    ]:
        final_rows.append({"Uncertainty Measure": label, "Result": result, "Interpretation": interpretation})
    pd.DataFrame(final_rows).to_csv(OUTPUT / "uncertainty_table.csv", index=False)
    save_histogram(primary_day)
    save_bounds_plot(primary_day, primary_summary)

    summary_text = f"""Objective 2 quantified uncertainty for the confirmed XGBoost model.

Primary result: genuine March Week 1 out-of-sample forecasts, daytime window
{DAYTIME_START_HOUR:02d}:00-{DAYTIME_END_HOUR_EXCLUSIVE - 1:02d}:59, {len(primary_day)} daytime observations.
Supplementary result: nine-origin, seven-day rolling out-of-fold XGBoost
forecasts from Objective 1, {len(supplementary_day)} daytime observations.

Forecast error was defined as e_t = actual - forecast. Relative uncertainty
was calculated against the absolute forecast value. Capacity-relative
uncertainty is intentionally pending because the plant rated capacity has not
yet been verified from an authoritative source. The placeholder is recorded
in uncertainty_table.csv and must be filled without changing the other
results once rated capacity is supplied.

The primary March Week 1 daytime MBE was {primary_summary['MBE_W']:.2f} W and
the error standard deviation was {primary_summary['error_std_W']:.2f} W.
The empirical 80%, 90%, and 95% error intervals are saved in the summary
tables. Normal-approximation intervals are included for comparison.

The separate 12:00-13:00 analysis contains {len(midday_day)} daytime test
observations. This is a small sample and its empirical quantiles should be
treated as descriptive rather than stable population estimates. The
supplementary rolling backtest is the stronger sample-size robustness check,
but it is not a replacement for the genuine March test-week result.

The primary March Week 1 bias was negative (-4,268 W), whereas the larger
9-origin backtest bias was positive (+8,860 W), and the seven-observation
12:00-13:00 slice was also positive (+11,805 W). This disagreement is not
enough to claim a definite seasonal change: the March-week estimate is noisy,
and its sampling uncertainty is large relative to its small bias. However, the
larger backtest provides evidence that the model more commonly underpredicted
in the historical validation periods, while March Week 1 showed a small
opposite-direction bias. The evidence therefore supports reporting a possible
period-specific difference, or sampling/data-condition variation, without
attributing it specifically to changing seasonal irradiance patterns. The
partial inverter coverage in March is an additional limitation when comparing
these bias estimates.

March Week 1 plant totals are derived_partial_sum values with approximately
77% average daytime inverter coverage. Therefore, the measured errors are
relative to best-available partial-coverage actuals, not certified full-plant
readings.
"""
    (OUTPUT / "objective2_summary.txt").write_text(summary_text, encoding="utf-8")
    print(stats[["analysis", "observations", "MBE_W", "error_std_W", "empirical_80%_lower_W", "empirical_80%_upper_W"]].to_string(index=False))
    print(f"Midday observations: {len(midday_day)}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()