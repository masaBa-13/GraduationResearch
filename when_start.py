import argparse
import sys
import re
import pandas as pd
from pathlib import Path


def parse_date_column(series: pd.Series):
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace("／", "/").str.replace("－", "-")

    def normalize_date(s):
        m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
        if not m:
            return None
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    normalized = cleaned.apply(normalize_date)
    return pd.to_datetime(normalized, errors="coerce")


def when_start(df, min_count=120):
    if "sdate" not in df.columns or "vessel" not in df.columns:
        raise ValueError("必要な列（sdate, vessel）がありません。")

    df["date"] = parse_date_column(df["sdate"])
    df = df.dropna(subset=["date"])

    results = []

    # 船ごと
    for vessel, group in df.groupby("vessel"):

        # 日付ごとの件数
        daily_counts = group.groupby("date").size()

        # 1日に min_count 件以上ある日だけ抽出
        active_days = daily_counts[daily_counts >= min_count]

        if len(active_days) > 0:
            earliest_day = active_days.index.min()
            results.append([vessel, earliest_day])

    return pd.DataFrame(results, columns=["vessel", "operation_start"])


def main():
    parser = argparse.ArgumentParser(description="Detect operation start date per vessel.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"入力ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    result = when_start(df)

    result.to_csv(args.output, index=False)
    print(f"結果を保存しました: {args.output}")

    print("\n=== 操業開始日 ===")
    print(result)


if __name__ == "__main__":
    main()
