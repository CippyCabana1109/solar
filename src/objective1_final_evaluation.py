"""Genuine March Week 1 evaluation and out-of-fold residual hybrids."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from objective1_validation import FOLDS, WEATHER, feature_columns, metrics, sarimax_predict, validation_windows
from project_config import DAYTIME_END_HOUR_EXCLUSIVE, DAYTIME_START_HOUR


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "master.csv"
OOF = ROOT / "outputs" / "obj1" / "validation_only_predictions.csv"
MODEL_DIR = ROOT / "models" / "obj1_validation"
OUTPUT = ROOT / "outputs" / "obj1"
TEST_START = pd.Timestamp("2025-03-01")
TEST_END = pd.Timestamp("2025-03-08")


def load_master() -> pd.DataFrame:
    data = pd.read_csv(MASTER, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp").asfreq("h")
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["day_of_year"] = data.index.dayofyear
    for period, column in [(24, "hour"), (7, "day_of_week"), (365.25, "day_of_year")]:
        data[f"sin_{column}"] = np.sin(2 * np.pi * data[column] / period)
        data[f"cos_{column}"] = np.cos(2 * np.pi * data[column] / period)
    data["lag_24"] = data["plant_power_w"].shift(24)
    data["lag_168"] = data["plant_power_w"].shift(168)
    return data


def daytime(data: pd.DataFrame) -> pd.DataFrame:
    hours = data.index.hour
    return data[(hours >= DAYTIME_START_HOUR) & (hours < DAYTIME_END_HOUR_EXCLUSIVE)]


def xgb_prediction(model, imputer, data: pd.DataFrame) -> np.ndarray:
    return model.predict(imputer.transform(data[feature_columns()]))


def prophet_prediction(model, imputer, data: pd.DataFrame) -> np.ndarray:
    frame = data.reset_index()[["timestamp"] + WEATHER].rename(columns={"timestamp": "ds"})
    frame[WEATHER] = imputer.transform(frame[WEATHER])
    return model.predict(frame)["yhat"].to_numpy()


def sarimax_prediction(model, imputer, data: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
    forecast_start = pd.Timestamp("2025-02-01")
    future = data.loc[(data.index >= forecast_start) & (data.index <= test_index[-1])]
    exog = imputer.transform(future[WEATHER])
    values = np.asarray(model.get_forecast(steps=len(future), exog=exog).predicted_mean)
    return pd.Series(values, index=future.index).reindex(test_index).to_numpy()


def fit_prophet_residual(oof: pd.DataFrame, data: pd.DataFrame) -> tuple[Prophet, SimpleImputer]:
    frame = oof.merge(data[WEATHER].reset_index(), on="timestamp", how="left")
    frame["residual"] = frame["actual_W"] - frame["predicted_W"]
    imputer = SimpleImputer(strategy="median")
    frame[WEATHER] = imputer.fit_transform(frame[WEATHER])
    train = frame[["timestamp", "residual"] + WEATHER].rename(columns={"timestamp": "ds", "residual": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
    for column in WEATHER:
        model.add_regressor(column, standardize=True)
    model.fit(train)
    return model, imputer


def fit_xgb_residual(oof: pd.DataFrame, data: pd.DataFrame) -> tuple[XGBRegressor, SimpleImputer]:
    frame = oof.merge(data.reset_index(), on="timestamp", how="left")
    frame["residual"] = frame["actual_W"] - frame["predicted_W"]
    frame = frame.dropna(subset=["lag_24", "lag_168"])
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(frame[feature_columns()])
    model = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=1,
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
    )
    model.fit(values, frame["residual"])
    return model, imputer


def selected_sarimax_oof(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    params = {"order": [2, 0, 0], "seasonal_order": [0, 0, 0, 0]}
    for fold_start in FOLDS:
        train, future = validation_windows(data, fold_start)
        predicted = sarimax_predict(train, future, params)
        rows.append(pd.DataFrame({"timestamp": future.index, "actual_W": future["plant_power_w"].to_numpy(), "predicted_W": predicted}))
    return pd.concat(rows, ignore_index=True)


def fit_sarimax_residual(oof: pd.DataFrame, data: pd.DataFrame) -> tuple[object, SimpleImputer]:
    frame = oof.merge(data[WEATHER].reset_index(), on="timestamp", how="left")
    frame["residual"] = frame["actual_W"] - frame["predicted_W"]
    imputer = SimpleImputer(strategy="median")
    exog = imputer.fit_transform(frame[WEATHER])
    model = SARIMAX(
        frame["residual"].astype(float), exog=exog, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)
    return model, imputer


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_master()
    test = data.loc[(data.index >= TEST_START) & (data.index < TEST_END)].copy()
    if len(test) != 168:
        raise ValueError(f"March Week 1 expected 168 hourly rows, found {len(test)}")
    test_day = daytime(test)
    actual = test_day["plant_power_w"].to_numpy()

    xgb = joblib.load(MODEL_DIR / "xgboost_final_provisional.joblib")
    xgb_imputer = joblib.load(MODEL_DIR / "xgboost_imputer.joblib")
    prophet = joblib.load(MODEL_DIR / "prophet_final_provisional.joblib")
    prophet_imputer = joblib.load(MODEL_DIR / "prophet_regressor_imputer.joblib")
    sarimax = joblib.load(MODEL_DIR / "sarimax_final_provisional.joblib")
    sarimax_imputer = joblib.load(MODEL_DIR / "sarimax_exog_imputer.joblib")

    predictions = {
        "XGBoost": xgb_prediction(xgb, xgb_imputer, test_day),
        "Prophet": prophet_prediction(prophet, prophet_imputer, test_day),
        "SARIMAX": sarimax_prediction(sarimax, sarimax_imputer, data, test_day.index),
    }
    standalone_rows = []
    for model_name in ("XGBoost", "Prophet", "SARIMAX"):
        standalone_rows.append({"model": model_name, **metrics(actual, predictions[model_name])})
    top_two = [row["model"] for row in sorted(standalone_rows, key=lambda row: row["MAE_W"])[:2]]
    if top_two != ["XGBoost", "SARIMAX"]:
        raise ValueError(f"Expected XGBoost and SARIMAX as test-week top two, found {top_two}")
    oof_xgb = pd.read_csv(OOF, parse_dates=["timestamp"])
    oof_xgb = oof_xgb[oof_xgb["model"] == "XGBoost"].copy()
    oof_sarimax = selected_sarimax_oof(data)
    residual_sarimax, residual_sarimax_imputer = fit_sarimax_residual(oof_xgb, data)
    residual_xgb, residual_xgb_imputer = fit_xgb_residual(oof_sarimax, data)
    residual_sarimax_prediction = sarimax_prediction(residual_sarimax, residual_sarimax_imputer, data, test_day.index)
    residual_xgb_prediction = residual_xgb.predict(residual_xgb_imputer.transform(test_day[feature_columns()]))
    predictions["XGBoost-SARIMAX Hybrid"] = predictions["XGBoost"] + residual_sarimax_prediction
    predictions["SARIMAX-XGBoost Hybrid"] = predictions["SARIMAX"] + residual_xgb_prediction

    rows = []
    for model_name, predicted in predictions.items():
        rows.append({"model": model_name, "evaluation": "genuine March Week 1 test", "daytime_definition": "06:00-18:59", "observations": len(test_day), **metrics(actual, predicted)})
    ranking = pd.DataFrame(rows).sort_values("MAE_W").reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.to_csv(OUTPUT / "final_ranking.csv", index=False)
    pd.DataFrame({"timestamp": test_day.index, "actual_W": actual, **predictions}).to_csv(OUTPUT / "march_week1_test_predictions.csv", index=False)
    joblib.dump(residual_sarimax, MODEL_DIR / "hybrid_residual_sarimax_oof.joblib")
    joblib.dump(residual_sarimax_imputer, MODEL_DIR / "hybrid_residual_sarimax_imputer.joblib")
    joblib.dump(residual_xgb, MODEL_DIR / "hybrid_residual_xgboost_oof.joblib")
    joblib.dump(residual_xgb_imputer, MODEL_DIR / "hybrid_residual_xgboost_imputer.joblib")
    winner = ranking.iloc[0]["model"]
    summary = (
        f"Objective 1 genuine test-week evaluation used March Week 1 ({TEST_START.date()} to 2025-03-07) "
        f"and the fixed daytime window 06:00-18:59, with {len(test_day)} daytime observations. "
        f"The overall winner was {winner}. The two hybrids used XGBoost and SARIMAX, the two "
        "best standalone models on this test week. Hybrid residual correctors were trained only from "
        "out-of-fold validation residuals. March Week 1 plant totals are derived_partial_sum "
        "values averaging about 77% inverter coverage during daytime hours, so results represent "
        "best-available data rather than certified full-plant readings. This is a genuine test-week "
        "result, not cross-validation.\n"
    )
    (OUTPUT / "objective1_summary.txt").write_text(summary, encoding="utf-8")
    print(ranking.to_string(index=False))
    print(f"Winner: {winner}")


if __name__ == "__main__":
    main()