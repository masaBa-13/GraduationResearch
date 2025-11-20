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

def ships_by_month(df):
    if "sdate" not in df.columns or "vessel" not in df.columns:
        raise ValueError("必要な列（sdate, vessel）がありません。")

    df["date"] = parse_date_column(df["sdate"])
    df = df.dropna(subset=["date"])

    # “年月”列を作る
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # 月ごとに出船している船を取得
    grouped = df.groupby("year_month")["vessel"].unique().reset_index()
    grouped["vessels"] = grouped["vessel"].apply(lambda arr: ",".join(arr))
    return grouped[["year_month","vessels"]]

def main():
    parser = argparse.ArgumentParser(description="List vessels by month.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", required=False, help="Optional output CSV file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"入力ファイルが見つかりません: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    result = ships_by_month(df)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"結果を保存しました: {args.output}")

    print("\n=== 月別出船リスト ===")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
