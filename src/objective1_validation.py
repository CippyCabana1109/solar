"""Validation-only Objective 1 pipeline for the provisional master dataset."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from project_config import (
    DAYTIME_END_HOUR_EXCLUSIVE,
    DAYTIME_START_HOUR,
    VALIDATION_ORIGINS,
    VALIDATION_WINDOW_DAYS,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "master_provisional_jan2024_jan2025.csv"
OUTPUT = ROOT / "outputs" / "obj1"
MODEL_DIR = ROOT / "models" / "obj1_validation"
TARGET = "plant_power_w"
WEATHER = ["ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2", "temperature_c", "relative_humidity_pct", "wind_speed_m_s"]
FOLDS = [pd.Timestamp(value) for value in VALIDATION_ORIGINS]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, parse_dates=["timestamp"]).sort_values("timestamp")
    data = data.set_index("timestamp").asfreq("h")
    data.index.name = "timestamp"
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["day_of_year"] = data.index.dayofyear
    for period, column in [(24, "hour"), (7, "day_of_week"), (365.25, "day_of_year")]:
        data[f"sin_{column}"] = np.sin(2 * np.pi * data[column] / period)
        data[f"cos_{column}"] = np.cos(2 * np.pi * data[column] / period)
    data["lag_24"] = data[TARGET].shift(24)
    data["lag_168"] = data[TARGET].shift(168)
    return data


def feature_columns() -> list[str]:
    return WEATHER + [
        "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
        "sin_day_of_year", "cos_day_of_year", "lag_24", "lag_168",
    ]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    valid_smape = denominator > 1e-9
    smape = np.mean(np.abs(actual[valid_smape] - predicted[valid_smape]) / denominator[valid_smape]) * 100
    return {
        "MAE_W": float(mean_absolute_error(actual, predicted)),
        "RMSE_W": float(np.sqrt(mean_squared_error(actual, predicted))),
        "sMAPE_pct": float(smape),
        "R2": float(r2_score(actual, predicted)),
    }


def prepare_features(train: pd.DataFrame, future: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    columns = feature_columns()
    train = train.dropna(subset=[TARGET] + ["lag_24", "lag_168"])
    future = future.dropna(subset=["lag_24", "lag_168"])
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train[columns])
    future_values = imputer.transform(future[columns])
    return train_values, future_values, imputer


def xgb_predict(train: pd.DataFrame, future: pd.DataFrame, params: dict[str, object]) -> tuple[np.ndarray, SimpleImputer]:
    train_values, future_values, imputer = prepare_features(train, future)
    model = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=1,
        n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]), subsample=0.9, colsample_bytree=0.9,
    )
    model.fit(train_values, train.loc[train.dropna(subset=[TARGET, "lag_24", "lag_168"]).index, TARGET])
    return model.predict(future_values), imputer


def sarimax_predict(train: pd.DataFrame, future: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    train = train[[TARGET] + WEATHER].copy()
    future = future[[TARGET] + WEATHER].copy()
    imputer = SimpleImputer(strategy="median")
    train_exog = imputer.fit_transform(train[WEATHER])
    future_exog = imputer.transform(future[WEATHER])
    model = SARIMAX(
        train[TARGET].astype(float), exog=train_exog,
        order=tuple(params["order"]), seasonal_order=tuple(params["seasonal_order"]),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50)
    return np.asarray(fitted.forecast(steps=len(future), exog=future_exog))


def prophet_predict(train: pd.DataFrame, future: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train_weather = pd.DataFrame(imputer.fit_transform(train[WEATHER]), columns=WEATHER, index=train.index)
    future_weather = pd.DataFrame(imputer.transform(future[WEATHER]), columns=WEATHER, index=future.index)
    train_frame = train.reset_index()[["timestamp", TARGET] + WEATHER].rename(columns={"timestamp": "ds", TARGET: "y"})
    future_frame = future.reset_index()[["timestamp"] + WEATHER].rename(columns={"timestamp": "ds"})
    train_frame[WEATHER] = train_weather.reset_index(drop=True)
    future_frame[WEATHER] = future_weather.reset_index(drop=True)
    model = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
        changepoint_prior_scale=float(params["changepoint_prior_scale"]),
        seasonality_mode="additive", interval_width=0.95,
    )
    for column in WEATHER:
        model.add_regressor(column, standardize=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_frame)
    return model.predict(future_frame)["yhat"].to_numpy()


def validation_windows(data: pd.DataFrame, fold_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.index < fold_start].copy()
    future = data.loc[
        (data.index >= fold_start)
        & (data.index < fold_start + pd.Timedelta(days=VALIDATION_WINDOW_DAYS))
    ].copy()
    expected = 24 * VALIDATION_WINDOW_DAYS
    if len(future) != expected:
        raise ValueError(f"Fold {fold_start.date()} does not contain {expected} hourly observations.")
    return train, future


def tune_and_validate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    grids = {
        "SARIMAX": [
            {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0]},
            {"order": [2, 0, 0], "seasonal_order": [0, 0, 0, 0]},
        ],
        "XGBoost": [
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
            {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
        ],
        "Prophet": [{"changepoint_prior_scale": 0.05}],
    }
    scores: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    best: dict[str, dict[str, object]] = {}
    for model_name, candidates in grids.items():
        candidate_scores = []
        for candidate_index, params in enumerate(candidates):
            fold_metrics = []
            for fold_start in FOLDS:
                train, future = validation_windows(data, fold_start)
                if model_name == "SARIMAX":
                    predicted = sarimax_predict(train, future, params)
                elif model_name == "XGBoost":
                    predicted, _ = xgb_predict(train, future, params)
                else:
                    predicted = prophet_predict(train, future, params)
                actual = future[TARGET].to_numpy()
                fold_metrics.append(metrics(actual, predicted))
                if candidate_index == 0:
                    predictions.append(pd.DataFrame({
                        "timestamp": future.index, "actual_W": actual,
                        "predicted_W": predicted, "model": model_name,
                        "fold_start": fold_start.date().isoformat(),
                    }))
            average = {key: float(np.mean([item[key] for item in fold_metrics])) for key in fold_metrics[0]}
            candidate_scores.append(average)
            scores.append({"model": model_name, "candidate": candidate_index, **params, **average, "validation_only": True})
        winner = min(candidate_scores, key=lambda item: item["MAE_W"])
        winner_index = candidate_scores.index(winner)
        best[model_name] = {**candidates[winner_index], "candidate": winner_index, **winner}
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True), best


def fit_final_models(data: pd.DataFrame, best: dict[str, dict[str, object]]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train = data.dropna(subset=[TARGET]).copy()
    for model_name, params in best.items():
        if model_name == "XGBoost":
            fit_train = train.dropna(subset=[TARGET, "lag_24", "lag_168"])
            imputer = SimpleImputer(strategy="median")
            train_values = imputer.fit_transform(fit_train[feature_columns()])
            model = XGBRegressor(
                objective="reg:squarederror", random_state=42, n_jobs=1,
                n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]), subsample=0.9, colsample_bytree=0.9,
            )
            model.fit(train_values, fit_train[TARGET])
            joblib.dump(model, MODEL_DIR / "xgboost_final_provisional.joblib")
            joblib.dump(imputer, MODEL_DIR / "xgboost_imputer.joblib")
        elif model_name == "SARIMAX":
            exog_imputer = SimpleImputer(strategy="median")
            exog = exog_imputer.fit_transform(train[WEATHER])
            fitted = SARIMAX(train[TARGET], exog=exog, order=tuple(params["order"]), seasonal_order=tuple(params["seasonal_order"]), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
            joblib.dump(fitted, MODEL_DIR / "sarimax_final_provisional.joblib")
            joblib.dump(exog_imputer, MODEL_DIR / "sarimax_exog_imputer.joblib")
        else:
            prophet_imputer = SimpleImputer(strategy="median")
            frame = train.reset_index()[["timestamp", TARGET] + WEATHER].rename(columns={"timestamp": "ds", TARGET: "y"})
            frame[WEATHER] = prophet_imputer.fit_transform(frame[WEATHER])
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=float(params["changepoint_prior_scale"]), seasonality_mode="additive")
            for column in WEATHER:
                model.add_regressor(column, standardize=True)
            model.fit(frame.rename(columns={"timestamp": "ds"}))
            joblib.dump(model, MODEL_DIR / "prophet_final_provisional.joblib")
            joblib.dump(prophet_imputer, MODEL_DIR / "prophet_regressor_imputer.joblib")
    (MODEL_DIR / "selected_configs.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    scores, predictions, best = tune_and_validate(data)
    scores.to_csv(OUTPUT / "validation_only_model_scores.csv", index=False)
    predictions.to_csv(OUTPUT / "validation_only_predictions.csv", index=False)
    prediction_master = data[[TARGET, "ghi_wh_m2"]].reset_index()
    predictions = predictions.merge(prediction_master, on="timestamp", how="left")
    predictions["daytime"] = predictions["timestamp"].dt.hour.between(
        DAYTIME_START_HOUR, DAYTIME_END_HOUR_EXCLUSIVE - 1
    )
    daytime_scores = []
    for model_name, group in predictions.groupby("model"):
        subset = group[group["daytime"]].copy()
        daytime_scores.append({"model": model_name, "observations": len(subset), **metrics(subset["actual_W"].to_numpy(), subset["predicted_W"].to_numpy()), "validation_only": True, "daytime_definition": f"{DAYTIME_START_HOUR:02d}:00-{DAYTIME_END_HOUR_EXCLUSIVE - 1:02d}:59"})
    pd.DataFrame(daytime_scores).sort_values("MAE_W").to_csv(OUTPUT / "validation_only_daytime_scores.csv", index=False)
    ranking = scores.loc[scores.groupby("model")["MAE_W"].idxmin()].sort_values("MAE_W")
    ranking.to_csv(OUTPUT / "validation_only_model_ranking.csv", index=False)
    for model_name, group in predictions.groupby("model"):
        figure, axis = plt.subplots(figsize=(10, 4))
        for fold_start, fold in group.groupby("fold_start"):
            axis.plot(fold["timestamp"], fold["actual_W"], color="black", alpha=0.5)
            axis.plot(fold["timestamp"], fold["predicted_W"], label=fold_start)
        axis.set_title(f"{model_name}: validation-only next-day predictions")
        axis.set_ylabel("Plant power (W)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT / f"{model_name.lower()}_validation_only.png", dpi=150)
        plt.close(figure)
    fit_final_models(data, best)
    (OUTPUT / "validation_only_summary.txt").write_text(
        "Objective 1 standalone model validation only. No March holdout was used. "
        "Hybrids were intentionally not built. Final provisional model objects are "
        "saved for reuse when a genuine March test dataset is available.\n",
        encoding="utf-8",
    )
    print(scores.to_string(index=False))
    print("Saved validation outputs and provisional final model artifacts.")


if __name__ == "__main__":
    main()