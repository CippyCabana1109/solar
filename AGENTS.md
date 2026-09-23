FILENAME NOTE: the actual docs filenames use spaces and a single underscore, not double underscores. Use exactly this pattern: "Disertation Help_ Objective 1.docx", "Disertation Help_ Objective 2.docx", "Disertation Help_ Objective 3.docx", and their .txt versions "Disertation Help_ Objective 1.txt", "Disertation Help_ Objective 2.txt", "Disertation Help_ Objective 3.txt".

PROJECT: Dissertation analysis on next-day solar PV forecasting and market energy commitment. Three objective documents are in docs/. For the current objective, read its document in full before doing anything. A .txt version of each doc already exists in docs/, use that directly, no need to reconvert.

DEFINITIONS (some equations in the docs come out garbled, so use these):
Forecast error: e_t = P_actual_t - P_forecast_t
Newsvendor critical fractile: q* = Cu / (Cu + Co)
Commitment: B* = forecast + the q*-quantile of the forecast-error distribution (same as the q*-quantile of the forecast distribution of actual energy).

RULES FOR EVERY STEP:
(1) Python only, reproducible, fixed random seeds. Code in src/, list dependencies in requirements.txt (tell me before installing anything), results in outputs/obj1, outputs/obj2, outputs/obj3, and one run_all.py at the end.
(2) Before any modelling, inspect data/raw and print a short data report: time resolution, date range, missing values, night-time zeros, timezone, units (W vs kWh). Show it to me and wait for my OK.
(3) One synchronized master dataset (PV joined with NASA POWER irradiance, temperature, wind speed, humidity), saved to data/processed/master.csv and used by all models.
(4) Split: train Jan 2024 to Feb 2025, test on one unseen week in March 2025, rolling next-day forecasting. No data leakage: no test data in any fitting, scaling, feature-building or tuning step, and no lagged PV from the target day.
(5) Never invent numbers or assumptions. If something is missing or ambiguous, stop and ask me.
(6) After each objective, write a plain-language summary .txt in that objective's outputs folder that I can paste into my dissertation, including assumptions and limitations.
(7) Token discipline: never print whole dataframes or long logs. Use .head() and .describe(), and save big outputs to files. Plan first, then implement.
(8) Each objective must start from files saved by the previous one (not from chat history), so always save what the next objective needs.