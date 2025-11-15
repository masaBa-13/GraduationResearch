from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# パラメータ
# ------------------------------------------------------------
MIN_GAP_SECONDS = 3600          # 1時間 (秒)
MIN_OPERATION_DISTANCE_KM = 10  # 操業とみなす最小移動距離


# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """haversine距離 (km)。NaNはそのまま返す。"""
    r = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def normalize_name(value: str | float) -> str:
    if isinstance(value, str):
        return value.strip().replace("　", "")
    return ""


def determine_op_date(ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.hour >= 18:
        ts = ts + pd.Timedelta(days=1)
    return ts.normalize()


# ------------------------------------------------------------
# マグログ処理
# ------------------------------------------------------------
def prepare_maglog(maglog_path: Path, vessel_lookup_path: Path) -> pd.DataFrame:
    if not maglog_path.exists():
        raise FileNotFoundError(f"マグログが見つかりません: {maglog_path}")
    if not vessel_lookup_path.exists():
        raise FileNotFoundError(f"船名対応表が見つかりません: {vessel_lookup_path}")

    mag = pd.read_csv(maglog_path)
    required_cols = {"name", "total", "time1", "time2"}
    missing = required_cols - set(mag.columns)
    if missing:
        raise ValueError(f"マグログデータに必要なカラムが不足しています: {missing}")

    mag["ship_name"] = mag["name"].map(normalize_name)
    mag["time1_dt"] = pd.to_datetime(mag["time1"], errors="coerce")
    mag["date"] = mag["time1_dt"].dt.normalize()
    mag["time2_dt"] = pd.to_datetime(mag["time2"], errors="coerce")
    mag["time2_sort"] = mag["time2_dt"].fillna(pd.Timestamp("1900-01-01"))

    mag = (
        mag.dropna(subset=["date"])
        .sort_values(["ship_name", "date", "time2_sort"])
        .groupby(["ship_name", "date"], as_index=False)
        .tail(1)
    )

    mag["tuna_kg"] = pd.to_numeric(mag["total"], errors="coerce")

    lookup = pd.read_csv(vessel_lookup_path)
    if "vessel_id" not in lookup.columns or "vessel_name" not in lookup.columns:
        raise ValueError("船名対応表に vessel_id, vessel_name が必要です")
    lookup["ship_name"] = lookup["vessel_name"].map(normalize_name)
    mag = mag.merge(lookup[["vessel_id", "ship_name"]], on="ship_name", how="left", suffixes=("", "_lookup"))

    return mag[["ship_name", "vessel_id", "date", "tuna_kg"]]


# ------------------------------------------------------------
# GNSS → 操業分割
# ------------------------------------------------------------
def read_gnss(gnss_path: Path) -> pd.DataFrame:
    if not gnss_path.exists():
        raise FileNotFoundError(f"GNSSファイルが見つかりません: {gnss_path}")

    df = pd.read_csv(gnss_path)
    required_cols = {"vessel", "sdate", "stime", "lat_dd", "long_dd"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"converted.csv に必要なカラムがありません: {missing}")

    df["ts"] = pd.to_datetime(df["sdate"].astype(str) + " " + df["stime"].astype(str), errors="coerce")
    df["vessel_id"] = df["vessel"].astype(str)
    df["lat_dd"] = pd.to_numeric(df["lat_dd"], errors="coerce")
    df["long_dd"] = pd.to_numeric(df["long_dd"], errors="coerce")
    return df


def split_operations(gnss: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gnss = gnss.sort_values(["vessel_id", "ts"]).reset_index(drop=True)
    gnss["delta_sec"] = gnss.groupby("vessel_id")["ts"].diff().dt.total_seconds()
    is_new = gnss["delta_sec"].isna() | (gnss["delta_sec"] >= MIN_GAP_SECONDS)
    gnss["op_seq"] = is_new.groupby(gnss["vessel_id"]).cumsum().astype("Int64")

    gnss["lat_prev"] = gnss.groupby(["vessel_id", "op_seq"])["lat_dd"].shift(1)
    gnss["lon_prev"] = gnss.groupby(["vessel_id", "op_seq"])["long_dd"].shift(1)

    seg = np.zeros(len(gnss))
    mask = gnss["lat_prev"].notna() & gnss["lon_prev"].notna()
    seg[mask] = haversine_np(
        gnss.loc[mask, "lat_prev"].to_numpy(),
        gnss.loc[mask, "lon_prev"].to_numpy(),
        gnss.loc[mask, "lat_dd"].to_numpy(),
        gnss.loc[mask, "long_dd"].to_numpy(),
    )
    gnss["seg_km"] = seg

    op_summary = (
        gnss.groupby(["vessel_id", "op_seq"], as_index=False)
        .agg(
            op_start=("ts", "min"),
            op_end=("ts", "max"),
            points=("ts", "size"),
            total_dist_km=("seg_km", "sum"),
        )
        .assign(op_date=lambda d: d["op_start"].map(determine_op_date))
    )

    op_summary["op_id"] = (
        op_summary["vessel_id"].astype(str)
        + "_"
        + op_summary["op_date"].dt.strftime("%Y%m%d").fillna("unknown")
        + "_"
        + op_summary["op_seq"].astype(str)
    )
    op_summary["valid_operation"] = op_summary["total_dist_km"] >= MIN_OPERATION_DISTANCE_KM

    gnss = gnss.drop(columns=["delta_sec", "lat_prev", "lon_prev"])
    return gnss, op_summary


# ------------------------------------------------------------
# 年ごとの処理
# ------------------------------------------------------------
def process_year(
    year: int,
    root: Path,
    maglog: pd.DataFrame,
) -> None:
    gnss_path = root / str(year) / "converted.csv"
    if not gnss_path.exists():
        print(f"[WARN] {gnss_path} が見つからないためスキップします")
        return

    gnss = read_gnss(gnss_path)
    gnss_original_cols = [c for c in gnss.columns]

    gnss, op_summary = split_operations(gnss)

    if op_summary.empty:
        print(f"[WARN] {year} の操業が見つかりませんでした")
        return

    op_summary = op_summary.merge(
        maglog[["vessel_id", "date", "tuna_kg"]],
        left_on=["vessel_id", "op_date"],
        right_on=["vessel_id", "date"],
        how="left",
    )
    op_summary = op_summary.drop(columns=["date"])

    valid_ops = op_summary[op_summary["valid_operation"]].copy()
    merge_cols = ["vessel_id", "op_seq", "op_id", "op_date", "tuna_kg"]
    gnss = gnss.merge(valid_ops[merge_cols], on=["vessel_id", "op_seq"], how="inner")

    output_cols = gnss_original_cols + ["ts", "vessel_id", "op_id", "op_date", "tuna_kg"]
    output = gnss[output_cols].copy()
    output["op_date"] = pd.to_datetime(output["op_date"]).dt.date

    out_path = root / f"gnss_maglog{year}.csv"
    output.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] {out_path.name} を出力しました ({len(output)} 行)")

    ops_out_path = root / f"operations_summary_{year}.csv"
    valid_ops_to_save = valid_ops.copy()
    valid_ops_to_save["op_date"] = pd.to_datetime(valid_ops_to_save["op_date"]).dt.date
    valid_ops_to_save.to_csv(ops_out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] {ops_out_path.name} を出力しました ({len(valid_ops_to_save)} 操業)")


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main(years: Iterable[int] = (2024, 2025)) -> None:
    root = Path(__file__).resolve().parent
    maglog_path = root / "マグログデータ.csv"
    lookup_path = root / "船名対応表.csv"
    maglog = prepare_maglog(maglog_path, lookup_path)

    for year in years:
        process_year(year, root, maglog)


if __name__ == "__main__":
    main()
