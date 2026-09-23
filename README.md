

# Solar PV Next-Day Forecasting and Market Commitment

Dissertation project analyzing next-day solar PV output forecasting and energy market commitment strategy for the Strathmore University solar plant, Nairobi, Kenya.

## What this project does

Three objectives, run in order:

1. Compares five forecasting models (SARIMAX, XGBoost, Prophet, and two hybrids) to find the best next-day solar power forecaster.
2. Measures how much that winning model's forecasts are typically wrong by, to quantify uncertainty.
3. Uses that forecast plus its uncertainty to decide how much energy to commit to the market for a given hour, under different cost scenarios.

## Requirements

- Python 3.10 or newer
- pip
- Git
- Dependencies listed in requirements.txt (install with: pip install -r requirements.txt)

This project does not require GPU access. All models run on CPU.

## Data

Raw PV inverter and NASA POWER weather data are NOT included in this repository due to file size and because some of it is not public. To reproduce this analysis, you need:

- Historical PV inverter production data for the target plant
- NASA POWER hourly weather data for the same location and period (free, downloadable at power.larc.nasa.gov)

Place raw files under data/raw/ before running anything. See src/recon_raw.py for how raw files are inventoried and classified.

## Project structure

- docs/ — the three objective specification documents
- src/ — all analysis and pipeline code
- data/raw/ — raw input data (not included, see Data section above)
- data/processed/ — cleaned, synchronized master dataset (generated, not included)
- outputs/obj1/ — model comparison results
- outputs/obj2/ — forecast uncertainty analysis
- outputs/obj3/ — market commitment results
- outputs/dissertation_assumptions_and_limitations.txt — consolidated list of every assumption and limitation across all three objectives
- run_all.py — runs the full pipeline end to end
- AGENTS.md — project rules followed by AI coding assistants (Codex) during development

## How to run

Run the full pipeline from the project root:

python run_all.py

This runs all eight stages in order: raw-data inventory, the pre-modelling data report, the provisional master dataset, Objective 1 model validation and fitting, the full synchronized master dataset, Objective 1's final test-week evaluation, Objective 2's uncertainty analysis, and Objective 3's market commitment calculation. The script stops immediately if any stage fails, and prints a summary of every file produced in outputs/obj1, outputs/obj2, and outputs/obj3 once complete.

Note: this regenerates all model artifacts and results from scratch, including model retraining, so it can take a while to complete.

## Known limitations

See outputs/dissertation_assumptions_and_limitations.txt for the full list, including data coverage gaps, the provisional plant capacity cap, and illustrative market pricing used in place of confirmed figures.

## Author

Cyprian Kabana 

Once saved, push it:

git add .
git commit -m "Update README with run_all.py instructions"
git push