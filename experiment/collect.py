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
    summary = []
    for root, _, files in os.walk(REAL_RESULTS):
        for name in files:
            if not name.lower().endswith('.csv'):
                continue
            src = Path(root) / name
            if not is_complete_csv(src):
                continue

            # Keep original behavior: copy complete files
            dst = DEST / name
            shutil.copy2(src, dst)
            copied += 1

            # Extract trial 99 observed_reward and total row second column
            trial99_flag = 0
            total_second = ''
            try:
                with src.open(newline='', encoding='utf-8') as f:
                    reader = list(csv.reader(f))
            except Exception:
                reader = []

            if reader:
                # detect header
                first = [c.strip().lower() for c in reader[0]]
                has_header = any('trial' in h for h in first) and any('observ' in h for h in first)
                if has_header:
                    # find indices
                    idx_trial = next((i for i, h in enumerate(first) if 'trial' in h), 0)
                    idx_obs = next((i for i, h in enumerate(first) if 'observ' in h), 1)
                    for row in reader[1:]:
                        if not row:
                            continue
                        # total row
                        if row[0].strip().lower() == 'total':
                            if len(row) > 1:
                                total_second = row[1].strip()
                        # trial 99
                        try:
                            if len(row) > idx_trial and int(row[idx_trial].strip()) == 99:
                                if len(row) > idx_obs and row[idx_obs].strip() == '1':
                                    trial99_flag = 1
                                else:
                                    trial99_flag = 0
                        except Exception:
                            pass
                else:
                    # no header: assume trial_number in col0, observed_reward in col1
                    for row in reader:
                        if not row:
                            continue
                        if row[0].strip().lower() == 'total':
                            if len(row) > 1:
                                total_second = row[1].strip()
                            continue
                        try:
                            if int(row[0].strip()) == 99:
                                if len(row) > 1 and row[1].strip() == '1':
                                    trial99_flag = 1
                                else:
                                    trial99_flag = 0
                        except Exception:
                            pass

            summary.append([name, str(trial99_flag), total_second])

    # write summary CSV
    summary_path = DEST / 'complete_results_summary.csv'
    try:
        with summary_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'trial99_observed', 'total_second'])
            for r in summary:
                writer.writerow(r)
    except Exception as e:
        print('Failed writing summary:', e)

    print(f"Copied {copied} complete CSV files to {DEST}; summary written to {summary_path}")


if __name__ == "__main__":
    collect()
