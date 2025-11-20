#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd

"""
count_gnss_month.py

Read a CSV (default: ./2023/converted.csv), detect a date/time column, and count rows per month.
Outputs counts to stdout and (optionally) to a CSV file.
"""

try:
    import pandas as pd
except Exception:
    print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)


def try_read_csv(path: Path):
    # Try utf-8, then cp932 (Shift_JIS) for Japanese environments
    for enc in ("utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def detect_date_column(df: pd.DataFrame, hint: str = None):
    if hint and hint in df.columns:
        return hint

    lname_map = {c: c.lower() for c in df.columns}
    candidates = []

    for key in ("time", "date", "datetime", "timestamp", "obs_time"):
        for col, low in lname_map.items():
            if key in low:
                candidates.append(col)

    if candidates:
        return candidates[0]

    return df.columns[0]


def main():
    parser = argparse.ArgumentParser(description="Count CSV rows per month from a date/time column.")
    parser.add_argument(
        "-i", "--input", default="2025/converted.csv",
        help="Input CSV file (default: 2025/converted.csv)"
    )
    parser.add_argument(
        "-c", "--date-column", default=None,
        help="Name of the date/time column (if not provided, detected automatically)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Optional output CSV file for monthly counts"
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(2)

    df = try_read_csv(path)
    if df.empty:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(3)

    date_col = detect_date_column(df, args.date_column)

    # ---- 日付＋時刻が分かれているケースに対応 ----
    low_cols = [c.lower() for c in df.columns]
    has_separate_time = "sdate" in low_cols and "stime" in low_cols

    if has_separate_time:
        # sdate + stime → datetime
        df["__datetime"] = pd.to_datetime(
            df["sdate"].astype(str) + " " + df["stime"].astype(str),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
        dt = df["__datetime"]
    else:
        dt = pd.to_datetime(df[date_col], errors="coerce")

    if dt.isna().all():
        print(f"Could not parse any dates from column '{date_col}'.", file=sys.stderr)
        sys.exit(4)

    # ==== 月を日本語で表記 ("1月", "2月", ...) ====
    month_labels = dt.dt.month.astype("Int64").apply(lambda m: f"{m}月")
    counts = month_labels.value_counts().sort_index()
    # ==================================================

    print("month,count")
    for month, cnt in counts.items():
        print(f"{month},{cnt}")

    if args.output:
        out_df = counts.reset_index()
        out_df.columns = ["month", "count"]
        out_df.to_csv(args.output, index=False)
        print(f"Saved monthly counts to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
