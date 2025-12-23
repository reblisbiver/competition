#!/usr/bin/env python3
"""Collect real_results CSVs that contain a completed 100-trial run (row starting with 'total').

- Scans the repository-level `real_results` folder.
- If a CSV has any row whose first column is exactly `total`, it is treated as complete.
- Copies qualifying files into `experiment/complete_results/`, creating the folder if needed.
"""
import os
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS = ROOT / "real_results"
DEST = ROOT / "complete_results"


def is_complete_csv(path: Path) -> bool:
    """Return True if the CSV contains a row whose first column is 'total'."""
    try:
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().lower() == "total":
                    return True
    except Exception:
        return False
    return False


def collect():
    if not REAL_RESULTS.is_dir():
        print(f"real_results folder not found at {REAL_RESULTS}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for root, _, files in os.walk(REAL_RESULTS):
        for name in files:
            if not name.lower().endswith('.csv'):
                continue
            src = Path(root) / name
            if is_complete_csv(src):
                dst = DEST / name
                shutil.copy2(src, dst)
                copied += 1
    print(f"Copied {copied} complete CSV files to {DEST}")


if __name__ == "__main__":
    collect()
