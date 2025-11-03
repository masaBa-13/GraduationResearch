# -*- coding: utf-8 -*-
"""
build_trips_and_tuna_seasons.py

目的
----
1) GNSS（converted.csv）から港座標を使って入出港を検知し、航海(trip)を作る
2) 「帰港日＝水揚げ日」かつ“まぐろ”のみをシーズン（7–1月）ごとに抽出する

入力（ファイル名は手元のものに合わせる）
- 2023/converted.csv, 2024/converted.csv, 2025/converted.csv
- ports_coords.csv
- 漁獲データ 2023.csv, 漁獲データ 2024.csv
- 船名対応表.csv

出力
- out_trips/<YEAR>/gnss_with_trip.csv
- out_trips/<YEAR>/trips.csv
- out_trips/trips_all_years.csv           … 3年分連結
- out_trips/trips_tuna_season.csv         … 同日マッチ済み・7–1月のみ・シーズン付
"""

from pathlib import Path
import pandas as pd
import numpy as np
import math
import re

# -----------------------------
# パラメータ（必要に応じて調整）
# -----------------------------
BUFFER_KM    = 2.0   # 港バッファ半径
LOW_SPEED_KN = 3.0   # 低速の目安（操業/待機の粗判定）
SEASON_START = 7     # シーズン開始月（7=7月）
SEASON_END   = 1     # シーズン終了月（1=1月）

# -----------------------------
# ユーティリティ
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """haversine 距離(km)"""
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def normalize_name(s):
    """荷主名・船名のゆらぎ吸収（全角空白・旧字体など）"""
    if pd.isna(s): return None
    s = str(s).replace("　", "").replace(" ", "")
    for k, v in {"寶":"宝", "榮":"栄", "辨":"弁"}.items():
        s = s.replace(k, v)
    return s

def label_season(date_ts, season_start=7, season_end=1):
    """日付→シーズン年とラベル（7–12月は当年，1月は前年）"""
    y, m = date_ts.year, date_ts.month
    if m >= season_start:
        sy = y
    elif m <= season_end:
        sy = y - 1
    else:
        return None, None
    return sy, f"{sy}-{sy+1}"

def nearest_port_km(lat, lon, ports_df):
    # まずユークリッド近傍でインデックス候補をとる→そのポートとhaversine
    idx = ((ports_df["lat"]-lat)**2 + (ports_df["lon"]-lon)**2).idxmin()
    plat, plon = ports_df.loc[idx, "lat"], ports_df.loc[idx, "lon"]
    return haversine(lat, lon, plat, plon)

# -----------------------------
# 1) GNSS→trip生成（年ごと）
# -----------------------------
def build_trips_for_year(gnss_path: Path, ports_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # 港座標
    ports = pd.read_csv(ports_path)
    assert {"lat","lon"}.issubset(ports.columns), "ports_coords.csv に lat, lon が必要"

    # GNSS
    gnss = pd.read_csv(gnss_path)
    # タイムスタンプ（mdate+mtime 優先、無ければ sdate+stime）
    def mk_ts(row):
        d = str(row.get("mdate") or row.get("sdate"))
        t = str(row.get("mtime") or row.get("stime"))
        d = d.replace(".", "/").replace("-", "/")
        try:
            y, m, d2 = [int(x) for x in d.split("/")]
        except Exception:
            y, m, d2 = int(d[:4]), int(d[4:6]), int(d[6:8])
        hh, mm, ss = [int(x) for x in t.split(":")]
        return pd.Timestamp(year=y, month=m, day=d2, hour=hh, minute=mm, second=ss)

    gnss["ts"] = gnss.apply(mk_ts, axis=1)
    # vessel_id はそのまま 'vessel' を採用（既存convertedに合わせる）
    gnss["vessel_id"] = gnss["vessel"].astype(str)

    # 度分 → 度 の列が既にある想定（lat_dd / long_dd）
    if "lat_dd" in gnss.columns and "long_dd" in gnss.columns:
        gnss["lat_dd"] = pd.to_numeric(gnss["lat_dd"], errors="coerce")
        gnss["lon_dd"] = pd.to_numeric(gnss["long_dd"], errors="coerce")
    else:
        raise ValueError("converted.csv に lat_dd, long_dd が必要です。")

    # 速度
    if "sokudo" in gnss.columns:
        gnss["speed_kn"] = pd.to_numeric(gnss["sokudo"], errors="coerce")
    else:
        gnss["speed_kn"] = np.nan

    gnss = gnss.sort_values(["vessel_id", "ts"]).reset_index(drop=True)

    # 最近港距離・港内判定
    gnss["dist_port_km"] = gnss.apply(lambda r: nearest_port_km(r["lat_dd"], r["lon_dd"], ports), axis=1)
    gnss["in_port"] = (gnss["dist_port_km"] <= BUFFER_KM).astype(int)

    # 出港/入港イベント（港内→沖 = depart, 沖→港内 = arrive）
    gnss["prev_in_port"] = gnss.groupby("vessel_id")["in_port"].shift(1).fillna(gnss["in_port"])
    gnss["event"] = np.select(
        [
            (gnss["prev_in_port"]==1) & (gnss["in_port"]==0),
            (gnss["prev_in_port"]==0) & (gnss["in_port"]==1),
        ],
        ["depart", "arrive"], default=""
    )

    # trip連番（出港イベントをカウント）
    gnss["trip_seq"] = (gnss["event"]=="depart").groupby(gnss["vessel_id"]).cumsum()
    gnss["trip_id"] = np.where(
        gnss["in_port"]==0,
        gnss["vessel_id"] + "_T" + gnss["trip_seq"].astype(int).astype(str),
        np.nan
    )

    # 区間距離・時間
    gnss["lat_prev"] = gnss.groupby(["vessel_id","trip_seq"])["lat_dd"].shift(1)
    gnss["lon_prev"] = gnss.groupby(["vessel_id","trip_seq"])["lon_dd"].shift(1)
    gnss["ts_prev"]  = gnss.groupby(["vessel_id","trip_seq"])["ts"].shift(1)
    gnss["seg_km"]   = gnss.apply(lambda r: haversine(r["lat_prev"], r["lon_prev"], r["lat_dd"], r["lon_dd"]) if pd.notna(r["lat_prev"]) else 0.0, axis=1)
    gnss["seg_h"]    = (gnss["ts"] - gnss["ts_prev"]).dt.total_seconds()/3600.0

    # 低速/高速時間
    low = gnss["speed_kn"] <= LOW_SPEED_KN
    gnss["seg_h_low"]  = np.where(low, gnss["seg_h"], 0.0)
    gnss["seg_h_high"] = np.where(~low, gnss["seg_h"], 0.0)

    # tripサマリー
    trips = (
        gnss.dropna(subset=["trip_id"])
            .groupby("trip_id", as_index=False)
            .agg(vessel_id=("vessel_id","first"),
                 dep_ts=("ts","min"),
                 arr_ts=("ts","max"),
                 duration_h=("seg_h","sum"),
                 fishing_h=("seg_h_low","sum"),
                 steaming_h=("seg_h_high","sum"),
                 distance_km=("seg_km","sum"),
                 dep_lat=("lat_dd","first"),
                 dep_lon=("lon_dd","first"),
                 arr_lat=("lat_dd","last"),
                 arr_lon=("lon_dd","last"))
    )
    trips["dep_date"] = pd.to_datetime(trips["dep_ts"]).dt.date
    trips["arr_date"] = pd.to_datetime(trips["arr_ts"]).dt.date

    # 保存
    gnss.to_csv(outdir / "gnss_with_trip.csv", index=False, encoding="utf-8-sig")
    trips.to_csv(outdir / "trips.csv", index=False, encoding="utf-8-sig")
    
    # フィルタリング済み航海も保存
    trips_filtered = filter_valid_fishing_trips(trips, min_duration_h=1.0, min_distance_km=10.0, 
                                               min_fishing_h=0.1, max_port_distance_km=100.0)
    trips_filtered.to_csv(outdir / "trips_filtered.csv", index=False, encoding="utf-8-sig")
    
    return trips

# -----------------------------
# 2) 同日マッチ（まぐろ）→シーズン抽出
# -----------------------------
def extract_tuna_seasons(trips_all: pd.DataFrame, lookup_path: Path, land_files: list[Path], out_csv: Path):
    # 荷主名→vessel_id
    lu = pd.read_csv(lookup_path)
    for c in ["vessel_name","user_name1","user_name2"]:
        if c in lu.columns:
            lu[c] = lu[c].astype(str).map(normalize_name)
    owner2vid = {}
    for _, r in lu.iterrows():
        if isinstance(r.get("user_name1"), str) and r["user_name1"]:
            owner2vid[r["user_name1"]] = r["vessel_id"]
        if isinstance(r.get("user_name2"), str) and r["user_name2"]:
            owner2vid[r["user_name2"]] = r["vessel_id"]


    # 水揚げ（まぐろのみ）連結
    land_list = []
    for p in land_files:
        L = pd.read_csv(p)
        if "水揚日" not in L.columns:
            raise ValueError(f"水揚日 列が見つかりません: {p}")
        L["date"] = pd.to_datetime(L["水揚日"].astype(str), format="%Y%m%d", errors="coerce")
        tuna = L["魚種名"].astype(str).str.contains("まぐろ", na=False)
        L = L[tuna].copy()
        if "荷主名" in L.columns:
            L["荷主名_norm"] = L["荷主名"].map(normalize_name)
            L["vessel_id"] = L["荷主名_norm"].map(owner2vid)
        if "仕切数量" in L.columns:
            L["kg"] = pd.to_numeric(L["仕切数量"], errors="coerce")
        land_list.append(L[["vessel_id","date","kg"]])

    land = pd.concat(land_list, ignore_index=True).dropna(subset=["vessel_id","date"])
    land_daily = (land.groupby(["vessel_id","date"], as_index=False)
                    .agg(landed_kg=("kg","sum"),
                         n_records=("kg","count"))
                    .rename(columns={"date":"arr_date"}))

    # trips 側の帰港日
    t = trips_all.copy()
    t["arr_date"] = pd.to_datetime(t["arr_ts"]).dt.date
    
    # land_daily の arr_date も date型に変換
    land_daily["arr_date"] = pd.to_datetime(land_daily["arr_date"]).dt.date

    # 厳密同日ジョイン（船×日）
    tm = t.merge(land_daily, on=["vessel_id","arr_date"], how="inner")

    # 7–12月 & 1月のみ
    tm["arr_date_ts"] = pd.to_datetime(tm["arr_date"])
    tm = tm[tm["arr_date_ts"].dt.month.isin([7,8,9,10,11,12,1])].copy()
    
    # 有効な漁業航海のみフィルタリング
    tm = filter_valid_fishing_trips(tm, min_duration_h=1.0, min_distance_km=10.0, 
                                   min_fishing_h=0.1, max_port_distance_km=100.0)

    # シーズン付与
    seasons = tm["arr_date_ts"].apply(lambda d: label_season(d, SEASON_START, SEASON_END))
    tm["season_year"]  = seasons.apply(lambda x: x[0])
    tm["season_label"] = seasons.apply(lambda x: x[1])

    cols = ["season_year","season_label","vessel_id","trip_id",
            "dep_ts","arr_ts","arr_date","landed_kg","n_records",
            "fishing_h","distance_km"]
    cols = [c for c in cols if c in tm.columns]
    tm = tm[cols].sort_values(["season_year","vessel_id","arr_date"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tm.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return tm

# -----------------------------
# 3) まぐろ漁獲船舶の全航跡抽出（年ごと）
# -----------------------------
def extract_tuna_vessels_tracks(year: int, gnss_path: Path, land_files: list[Path], lookup_path: Path, outdir: Path):
    """まぐろ漁獲があった船舶の全航跡データを抽出（1分間隔）"""
    
    # 荷主名→vessel_id マッピング
    lu = pd.read_csv(lookup_path)
    for c in ["vessel_name","user_name1","user_name2"]:
        if c in lu.columns:
            lu[c] = lu[c].astype(str).map(normalize_name)
    owner2vid = {}
    for _, r in lu.iterrows():
        if isinstance(r.get("user_name1"), str) and r["user_name1"]:
            owner2vid[r["user_name1"]] = r["vessel_id"]
        if isinstance(r.get("user_name2"), str) and r["user_name2"]:
            owner2vid[r["user_name2"]] = r["vessel_id"]
    
    # 指定年のまぐろ漁獲があった船舶を特定
    tuna_vessels = set()
    for p in land_files:
        L = pd.read_csv(p)
        if "水揚日" not in L.columns:
            continue
        L["date"] = pd.to_datetime(L["水揚日"].astype(str), format="%Y%m%d", errors="coerce")
        # 指定年のデータのみ
        L = L[L["date"].dt.year == year].copy()
        # まぐろのみ
        tuna = L["魚種名"].astype(str).str.contains("まぐろ", na=False)
        L = L[tuna].copy()
        if "荷主名" in L.columns:
            L["荷主名_norm"] = L["荷主名"].map(normalize_name)
            L["vessel_id"] = L["荷主名_norm"].map(owner2vid)
            L = L.dropna(subset=["vessel_id"])
            
            # まぐろ漁獲があった船舶IDを記録
            tuna_vessels.update(L["vessel_id"].unique())
    
    if not tuna_vessels:
        print(f"[INFO] {year}年: まぐろ漁獲船舶が見つかりませんでした")
        return
    
    print(f"[INFO] {year}年: まぐろ漁獲船舶 {len(tuna_vessels)}隻 - {sorted(tuna_vessels)}")
    
    # 指定年のGNSSデータを読み込み、まぐろ漁獲船舶の全航跡を抽出
    if not gnss_path.exists():
        print(f"[WARNING] {gnss_path} が見つかりません")
        return
        
    gnss = pd.read_csv(gnss_path)
    
    # タイムスタンプ作成の関数（既存のmk_ts関数と同じ）
    def mk_ts(row):
        d = str(row.get("mdate") or row.get("sdate"))
        t = str(row.get("mtime") or row.get("stime"))
        d = d.replace(".", "/").replace("-", "/")
        try:
            y, m, d2 = [int(x) for x in d.split("/")]
        except Exception:
            y, m, d2 = int(d[:4]), int(d[4:6]), int(d[6:8])
        hh, mm, ss = [int(x) for x in t.split(":")]
        return pd.Timestamp(year=y, month=m, day=d2, hour=hh, minute=mm, second=ss)
    
    gnss["ts"] = gnss.apply(mk_ts, axis=1)
    gnss["vessel_id"] = gnss["vessel"].astype(str)
    gnss["date"] = gnss["ts"].dt.date
    
    # まぐろ漁獲船舶の航跡のみ抽出
    filtered_gnss = gnss[gnss["vessel_id"].isin(tuna_vessels)].copy()
    
    if filtered_gnss.empty:
        print(f"[INFO] {year}年: まぐろ漁獲船舶のGNSSデータが見つかりませんでした")
        return
    
    print(f"[INFO] {year}年: {len(filtered_gnss)}件のGNSSデータを抽出")
    print(f"[INFO] 対象船舶: {sorted(filtered_gnss['vessel_id'].unique())}")
    
    # 出力
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"tuna_vessels_tracks.csv"
    filtered_gnss.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 出力完了: {out_path}")

# -----------------------------
# 4) 有効漁業航海フィルタリング
# -----------------------------
def filter_valid_fishing_trips(trips_df: pd.DataFrame, min_duration_h: float = 2.0, min_distance_km: float = 5.0, 
                               min_fishing_h: float = 0.5, max_port_distance_km: float = 50.0) -> pd.DataFrame:
    """
    港間移動や無効航海を除外して、実際の漁業航海のみをフィルタリング
    
    Parameters:
    - min_duration_h: 最小航海時間（時間）
    - min_distance_km: 最小移動距離（km）
    - min_fishing_h: 最小漁労時間（時間）
    - max_port_distance_km: 出発地-到着地の最大距離（これを超えると長距離移動とみなす）
    """
    before_count = len(trips_df)
    
    # 1. 基本的な時間・距離フィルタ
    filtered = trips_df[
        (trips_df['duration_h'] >= min_duration_h) &
        (trips_df['distance_km'] >= min_distance_km)
    ].copy()
    
    # 2. 漁労時間フィルタは撤廃（低速時間は指標保持のみ）
    
    # 3. 港間距離チェック（出発地と到着地が近すぎる場合を除外）
    if all(col in filtered.columns for col in ['dep_lat', 'dep_lon', 'arr_lat', 'arr_lon']):
        port_distances = filtered.apply(
            lambda row: haversine(row['dep_lat'], row['dep_lon'], row['arr_lat'], row['arr_lon']), 
            axis=1
        )
        filtered = filtered[port_distances <= max_port_distance_km].copy()
    
    after_count = len(filtered)
    print(f"[INFO] 航海フィルタリング: {before_count}件 → {after_count}件 ({after_count/before_count*100:.1f}%)")
    
    return filtered

# -----------------------------
# 5) まぐろ水揚げ日のGNSS抽出（年ごと）
# -----------------------------
def extract_gnss_tuna_days(year: int, gnss_path: Path, land_files: list[Path], lookup_path: Path, outdir: Path, trips_filtered_path: Path = None):
    """指定年のGNSSデータから、まぐろ水揚げがあった船舶のその日のGNSSデータのみを抽出
    
    例: matsumae11が9/11にまぐろ漁獲 → 9/11のmatsumae11のGNSSデータのみ抽出
    """
    
    # 荷主名→vessel_id マッピング
    lu = pd.read_csv(lookup_path)
    for c in ["vessel_name","user_name1","user_name2"]:
        if c in lu.columns:
            lu[c] = lu[c].astype(str).map(normalize_name)
    owner2vid = {}
    for _, r in lu.iterrows():
        if isinstance(r.get("user_name1"), str) and r["user_name1"]:
            owner2vid[r["user_name1"]] = r["vessel_id"]
        if isinstance(r.get("user_name2"), str) and r["user_name2"]:
            owner2vid[r["user_name2"]] = r["vessel_id"]
    
    # 水揚げデータからまぐろ水揚げ日を特定
    tuna_dates_by_vessel = {}
    for p in land_files:
        L = pd.read_csv(p)
        if "水揚日" not in L.columns:
            continue
        L["date"] = pd.to_datetime(L["水揚日"].astype(str), format="%Y%m%d", errors="coerce")
        # 指定年のデータのみ
        L = L[L["date"].dt.year == year].copy()
        # まぐろのみ
        tuna = L["魚種名"].astype(str).str.contains("まぐろ", na=False)
        L = L[tuna].copy()
        if "荷主名" in L.columns:
            L["荷主名_norm"] = L["荷主名"].map(normalize_name)
            L["vessel_id"] = L["荷主名_norm"].map(owner2vid)
            L = L.dropna(subset=["vessel_id"])
            
            # 船別・日別でまぐろ水揚げ日を記録
            for _, row in L.iterrows():
                vessel_id = row["vessel_id"]
                date = row["date"].date()
                if vessel_id not in tuna_dates_by_vessel:
                    tuna_dates_by_vessel[vessel_id] = set()
                tuna_dates_by_vessel[vessel_id].add(date)
    
    # フィルタリング済み航海データがある場合は、有効な航海の日付のみを使用
    if trips_filtered_path and trips_filtered_path.exists():
        trips_filtered = pd.read_csv(trips_filtered_path)
        valid_dates_by_vessel = {}
        
        for _, trip in trips_filtered.iterrows():
            vessel_id = trip["vessel_id"]
            arr_date = pd.to_datetime(trip["arr_ts"]).date()
            
            if vessel_id not in valid_dates_by_vessel:
                valid_dates_by_vessel[vessel_id] = set()
            valid_dates_by_vessel[vessel_id].add(arr_date)
        
        # まぐろ水揚げ日と有効航海日の交集合を取る
        filtered_tuna_dates = {}
        total_tuna_dates_before = sum(len(dates) for dates in tuna_dates_by_vessel.values())
        
        for vessel_id, tuna_dates in tuna_dates_by_vessel.items():
            if vessel_id in valid_dates_by_vessel:
                filtered_tuna_dates[vessel_id] = tuna_dates & valid_dates_by_vessel[vessel_id]
        
        total_tuna_dates_after = sum(len(dates) for dates in filtered_tuna_dates.values())
        tuna_dates_by_vessel = filtered_tuna_dates
        print(f"[INFO] {year}年: フィルタリング済み航海の水揚げ日のみを使用")
        print(f"[DEBUG] {year}年: 水揚げ日数 {total_tuna_dates_before} → {total_tuna_dates_after} 件")
    
    if not tuna_dates_by_vessel:
        print(f"[INFO] {year}年: まぐろ水揚げデータが見つかりませんでした")
        return
    
    print(f"[INFO] {year}年: {len(tuna_dates_by_vessel)}隻のまぐろ水揚げを確認")
    
    # GNSSデータを読み込み、まぐろ水揚げ日のみ抽出
    if not gnss_path.exists():
        print(f"[WARNING] {gnss_path} が見つかりません")
        return
        
    gnss = pd.read_csv(gnss_path)
    
    # タイムスタンプ作成
    def mk_ts(row):
        d = str(row.get("mdate") or row.get("sdate"))
        t = str(row.get("mtime") or row.get("stime"))
        d = d.replace(".", "/").replace("-", "/")
        try:
            y, m, d2 = [int(x) for x in d.split("/")]
        except Exception:
            y, m, d2 = int(d[:4]), int(d[4:6]), int(d[6:8])
        hh, mm, ss = [int(x) for x in t.split(":")]
        return pd.Timestamp(year=y, month=m, day=d2, hour=hh, minute=mm, second=ss)
    
    gnss["ts"] = gnss.apply(mk_ts, axis=1)
    gnss["vessel_id"] = gnss["vessel"].astype(str)
    gnss["date"] = gnss["ts"].dt.date
    
    # まぐろ水揚げがあった日のGNSSデータのみ抽出
    filtered_gnss = []
    total_gnss = len(gnss)
    matched_count = 0
    
    for _, row in gnss.iterrows():
        vessel_id = row["vessel_id"]
        date = row["date"]
        if vessel_id in tuna_dates_by_vessel and date in tuna_dates_by_vessel[vessel_id]:
            filtered_gnss.append(row)
            matched_count += 1
    
    print(f"[DEBUG] {year}年: GNSS総数={total_gnss}, まぐろ水揚げ船・日マッチ={matched_count}, 対象船舶数={len(tuna_dates_by_vessel)}")
    
    if not filtered_gnss:
        print(f"[INFO] {year}年: まぐろ水揚げ船舶のその日のGNSSデータが見つかりませんでした")
        return
    
    filtered_df = pd.DataFrame(filtered_gnss)
    print(f"[INFO] {year}年: {len(filtered_df)}件のGNSSデータを抽出（まぐろ水揚げ船舶のその日のみ）")
    
    # 抽出されたデータの内訳を表示
    vessel_counts = filtered_df.groupby('vessel_id')['date'].nunique().to_dict()
    print(f"[DEBUG] 船舶別抽出日数: {dict(sorted(vessel_counts.items()))}")
    
    # 出力
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"gnss_in_tuna_season.csv"
    filtered_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 出力完了: {out_path}")
    print(f"[INFO] 内容: まぐろ水揚げ船舶のその日のGNSSデータのみ（例: matsumae11の9/11水揚げ → 9/11のmatsumae11のGNSSのみ）")

# -----------------------------
# メインフロー
# -----------------------------
if __name__ == "__main__":
    root = Path(".")
    ports_path = root / "ports_coords.csv"
    lookup_path = root / "船名対応表.csv"

    # 年ごとに trips を作成
    trips_all = []
    for y in [2023, 2024, 2025]:
        gnss_path = root / f"{y}/converted.csv"
        if gnss_path.exists():
            outdir = root / "out_trips" / str(y)
            trips = build_trips_for_year(gnss_path, ports_path, outdir)
            # 年情報（デバッグ/集計用）
            trips["year"] = y
            trips_all.append(trips)
    if not trips_all:
        raise SystemExit("converted.csv が見つかりません（2023/2024/2025）")

    trips_all = pd.concat(trips_all, ignore_index=True)
    # 連結 trips を保存（後工程で使う）
    out_all = root / "out_trips" / "trips_all_years.csv"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    trips_all.to_csv(out_all, index=False, encoding="utf-8-sig")

    # 水揚げファイル（手元にある年のみ使用）
    land_files = [root / "漁獲データ2023.csv", root / "漁獲データ2024.csv"]
    land_files = [p for p in land_files if p.exists()]
    if not land_files:
        raise SystemExit("水揚げCSV（漁獲データ20XX.csv）が見つかりません")

    # 同日まぐろのみ → シーズン抽出
    out_season = root / "out_trips" / "trips_tuna_season.csv"
    extract_tuna_seasons(trips_all, lookup_path, land_files, out_season)

    # 年ごとにまぐろ漁獲船舶の全航跡データを抽出
    print("\n[INFO] まぐろ漁獲船舶の全航跡データを年ごとに抽出中...")
    for y in [2023, 2024, 2025]:
        gnss_path = root / f"{y}/converted.csv"
        if gnss_path.exists():
            outdir = root / "out_trips" / str(y)
            extract_tuna_vessels_tracks(y, gnss_path, land_files, lookup_path, outdir)

    # 年ごとにまぐろ水揚げ日のGNSSデータを抽出（フィルタリング済み航海に基づく）
    print("\n[INFO] まぐろ水揚げ日のGNSSデータを年ごとに抽出中（フィルタリング済み）...")
    for y in [2023, 2024, 2025]:
        gnss_path = root / f"{y}/converted.csv"
        if gnss_path.exists():
            outdir = root / "out_trips" / str(y)
            trips_filtered_path = outdir / "trips_filtered.csv"
            extract_gnss_tuna_days(y, gnss_path, land_files, lookup_path, outdir, trips_filtered_path)

    print("[DONE] trips, seasons, GNSS まぐろデータの出力を完了しました。")
