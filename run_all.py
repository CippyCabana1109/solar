"""Run the dissertation analysis pipeline from the repository root.

This runner stops at the first failed stage. It intentionally regenerates
pipeline outputs, including model artifacts and files in outputs/obj1-obj3.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# The provisional-master and validation stages are dependencies of the final
# Objective 1 evaluator: it loads the model artifacts they create.
STAGES = (
    ("Stage 1 of 8: Raw-data inventory", "recon_raw.py"),
    ("Stage 2 of 8: Pre-modelling data report", "data_report.py"),
    ("Stage 3 of 8: Provisional master dataset", "build_provisional_master.py"),
    ("Stage 4 of 8: Objective 1 validation and model fitting", "objective1_validation.py"),
    ("Stage 5 of 8: Full synchronized master dataset", "build_full_master.py"),
    ("Stage 6 of 8: Objective 1 final evaluation", "objective1_final_evaluation.py"),
    ("Stage 7 of 8: Objective 2 uncertainty analysis", "objective2_uncertainty.py"),
    ("Stage 8 of 8: Objective 3 market commitment", "objective3_commitment.py"),
)

OUTPUT_DIRECTORIES = ("obj1", "obj2", "obj3")


def run_stage(title: str, script_name: str) -> None:
    script = SRC / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Required stage script is missing: {script}")

    print(f"\n{'=' * 72}\n{title}\nScript: {script.relative_to(ROOT)}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def print_output_summary() -> None:
    print(f"\n{'=' * 72}\nPIPELINE COMPLETE: OUTPUT SUMMARY\n{'=' * 72}")
    for directory_name in OUTPUT_DIRECTORIES:
        directory = ROOT / "outputs" / directory_name
        print(f"\noutputs/{directory_name}")
        if not directory.is_dir():
            print("  No directory produced.")
            continue
        files = sorted(path for path in directory.iterdir() if path.is_file())
        if not files:
            print("  No files produced.")
            continue
        for path in files:
            print(f"  {path.name}")


def main() -> None:
    try:
        for title, script_name in STAGES:
            run_stage(title, script_name)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"\nPIPELINE STOPPED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print_output_summary()


if __name__ == "__main__":
    main()
