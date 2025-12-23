"""Merge 'total' rows from CSV files under complete_results.

需求
- complete_results 位于 competition 文件夹下（默认：./complete_results）。
- 遍历该文件夹中的每个 CSV 文件。
- 找到“第一列为 total 的那一行”（每个文件仅一行）。
- 将这些行整合为一个 CSV，保存到相同文件夹下。

用法
  python merge_complete_results_totals.py
    python merge_complete_results_totals.py --dir complete_results --out totals_summary.csv

说明
- 默认会遍历 .csv。
- 默认把输出 CSV 用 UTF-8 BOM（utf-8-sig）写出，方便 Excel 直接打开。
- 支持两种 CSV 形式：
    - 有表头（header）：如果第二行第一列是 total，则第一行视为 header。
    - 无表头：第一行第一列就是 total。
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractResult:
    file: str
    sheet: str
    found: bool
    message: str


def _normalize_first_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _iter_csv_files(folder: Path) -> list[Path]:
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def _read_csv_rows(path: Path) -> tuple[list[list[str]], str]:
    """Read CSV rows with a couple of common encodings.

    Returns: (rows, encoding_used)
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "gb18030"]
    last_err: Exception | None = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader]
            return rows, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法解码文件 {path.name}，已尝试 {encodings_to_try}。最后错误：{last_err}",
    )


def extract_total_row_from_csv(path: Path):
    """Return (row_dict, ExtractResult).

    - Matches a row where first column value == 'total' (case-insensitive, trimmed).
    - Supports CSV with or without header.
    """
    try:
        rows, enc = _read_csv_rows(path)
    except Exception as e:
        return None, ExtractResult(file=path.name,
                                   sheet="",
                                   found=False,
                                   message=f"无法读取 CSV：{e}")

    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return None, ExtractResult(file=path.name,
                                   sheet="",
                                   found=False,
                                   message="CSV 为空")

    # Detect header: if second row exists and its first cell is 'total', treat first row as header.
    header: list[str] | None = None
    start_idx = 0
    if len(rows) >= 2 and _normalize_first_cell(
            rows[1][0] if rows[1] else "") == "total":
        header = [str(h).strip() for h in rows[0]]
        start_idx = 1

    hits: list[list[str]] = []
    for r in rows[start_idx:]:
        if not r:
            continue
        if _normalize_first_cell(r[0]) == "total":
            hits.append(r)

    if not hits:
        return None, ExtractResult(file=path.name,
                                   sheet="",
                                   found=False,
                                   message="未找到第一列为 total 的行")

    row = hits[0]
    if header is not None:
        # Pad header/row to same length
        if len(row) > len(header):
            header = header + [
                f"extra_{i}" for i in range(len(header), len(row))
            ]
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        row_dict = {header[i]: row[i] for i in range(len(header))}
    else:
        row_dict = {
            f"col_{i}": (row[i] if i < len(row) else "")
            for i in range(len(row))
        }

    msg = "OK" if len(hits) == 1 else f"找到多行 total（{len(hits)}），已取第一行"
    return row_dict, ExtractResult(file=path.name,
                                   sheet="",
                                   found=True,
                                   message=msg)


def main():
    parser = argparse.ArgumentParser(
        description="Merge total rows from CSV files in complete_results")
    parser.add_argument(
        "--dir",
        default="complete_results",
        help="Folder containing CSV files (default: ./complete_results)",
    )
    parser.add_argument(
        "--out",
        default="complete_results_totals_merged.csv",
        help="Output CSV filename (saved inside --dir by default)",
    )
    parser.add_argument(
        "--audit",
        default=None,
        help=
        "Audit CSV filename (saved inside --dir by default). Default: <out>_audit.csv",
    )
    args = parser.parse_args()

    folder = Path(args.dir).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"目录不存在：{folder}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = folder / out_path

    audit_path = Path(
        args.audit) if args.audit else out_path.with_name(out_path.stem +
                                                          "_audit.csv")
    if not audit_path.is_absolute():
        audit_path = folder / audit_path

    files = _iter_csv_files(folder)
    if not files:
        raise SystemExit(f"目录下没有 CSV 文件：{folder}")

    rows: list[dict[str, str]] = []
    audit: list[ExtractResult] = []
    field_order: list[str] = ["source_file"]

    for f in files:
        row_dict, result = extract_total_row_from_csv(f)
        audit.append(result)
        if row_dict is None:
            continue

        merged: dict[str, str] = {"source_file": result.file}
        for k, v in row_dict.items():
            key = str(k).strip() if str(k).strip() else "(blank)"
            merged[key] = "" if v is None else str(v)
            if key not in field_order and key != "source_file":
                field_order.append(key)

        rows.append(merged)

    if not rows:
        os.makedirs(audit_path.parent, exist_ok=True)
        with open(audit_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "found", "message"])
            w.writeheader()
            for a in audit:
                w.writerow({
                    "file": a.file,
                    "found": a.found,
                    "message": a.message
                })
        raise SystemExit("没有从任何文件中提取到 total 行；已输出审计文件：" + str(audit_path))

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # fill missing keys
            for k in field_order:
                r.setdefault(k, "")
            w.writerow(r)

    with open(audit_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "found", "message"])
        w.writeheader()
        for a in audit:
            w.writerow({
                "file": a.file,
                "found": a.found,
                "message": a.message
            })

    print(f"已写入：{out_path}")
    print(f"审计文件：{audit_path}")
    print(f"合并行数：{len(rows)} / 文件数：{len(files)}")


if __name__ == "__main__":
    main()
