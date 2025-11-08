#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マグロ漁獲データ分析プログラム
2023年と2024年のマグロ漁獲について以下を分析：
1. 漁獲総数
2. 船ごとの漁獲量
3. マグロが取れた日の船ごとの移動量
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date as _date
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
sns.set(font='Hiragino Sans')  # seabornにも適用
plt.rcParams['axes.unicode_minus'] = False  # マイナス符号を正しく表示

# プロット表示フォントを全体的に大きくする（必要に応じて値を調整してください）
plt.rcParams.update({
    'font.size': 11,              # 全体の基本フォントサイズ
    'axes.titlesize': 0,         # グラフタイトル
    'axes.labelsize': 14,         # 軸ラベル
    'xtick.labelsize': 11,        # x軸目盛りラベル
    'ytick.labelsize': 12,        # y軸目盛りラベル
    'legend.fontsize': 14,        # 凡例
    'figure.titlesize': 0        # 図全体のタイトル
})

# seaborn のコンテキストも大きめに設定して見やすくする
sns.set_context('talk', font_scale=1.1)

def haversine_distance(lat1, lon1, lat2, lon2):
    """2点間の距離をHaversine公式で計算（km）"""
    R = 6371  # 地球の半径 (km)
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def normalize_gnss_coordinates(df):
    """GNSS座標データを正規化（度単位への変換とスケール検証）"""
    # 既に lat_dd, long_dd 列がある場合はそれを使用
    if 'lat_dd' in df.columns and 'long_dd' in df.columns:
        df['lat_normalized'] = df['lat_dd']
        df['long_normalized'] = df['long_dd']
    else:
        # lat, long 列から推定
        if 'lat' in df.columns and 'long' in df.columns:
            # データの範囲から推定してスケールを決定
            lat_max = df['lat'].abs().max()
            long_max = df['long'].abs().max()
            
            if lat_max > 1000000:  # E7スケール (1e-7度)
                df['lat_normalized'] = df['lat'] / 1e7
                df['long_normalized'] = df['long'] / 1e7
                print(f"[INFO] E7スケール検出、度単位に変換: lat範囲={df['lat_normalized'].min():.6f}～{df['lat_normalized'].max():.6f}")
            elif lat_max > 1000:  # ミリ度スケール
                df['lat_normalized'] = df['lat'] / 1000
                df['long_normalized'] = df['long'] / 1000
                print(f"[INFO] ミリ度スケール検出、度単位に変換: lat範囲={df['lat_normalized'].min():.6f}～{df['lat_normalized'].max():.6f}")
            elif lat_max > 360:  # 度分形式 (DDMM.MMMM)
                df['lat_normalized'] = (df['lat'] // 100) + (df['lat'] % 100) / 60
                df['long_normalized'] = (df['long'] // 100) + (df['long'] % 100) / 60
                print(f"[INFO] 度分形式検出、度単位に変換: lat範囲={df['lat_normalized'].min():.6f}～{df['lat_normalized'].max():.6f}")
            else:  # 既に度単位
                df['lat_normalized'] = df['lat']
                df['long_normalized'] = df['long']
                print(f"[INFO] 度単位として処理: lat範囲={df['lat_normalized'].min():.6f}～{df['lat_normalized'].max():.6f}")
        else:
            raise ValueError("緯度経度の列が見つかりません")
    
    # 座標の妥当性チェック
    lat_valid = (df['lat_normalized'] >= -90) & (df['lat_normalized'] <= 90)
    long_valid = (df['long_normalized'] >= -180) & (df['long_normalized'] <= 180)
    
    invalid_count = len(df) - (lat_valid & long_valid).sum()
    if invalid_count > 0:
        print(f"[WARNING] 無効な座標データ {invalid_count}件を検出（緯度範囲外または経度範囲外）")
        df = df[lat_valid & long_valid].copy()
    
    return df

def filter_movement_outliers(df, max_speed_kmh=60.0, max_time_gap_hours=1.0, max_distance_km=100):
    """移動データの異常値をフィルタリング
    
    Args:
    max_speed_kmh: 最大速度制限（運用閾値、デフォルト60km/h）
        max_time_gap_hours: 最大時間ギャップ（1時間以上空いた場合は除外）
        max_distance_km: 1ステップ最大距離（100km以上の瞬間移動は除外）
    """
    if len(df) < 2:
        return df, {"filtered_points": 0, "time_gaps": 0, "time_reverse": 0, "speed_outliers": 0, "distance_outliers": 0}
    
    df = df.sort_values('ts').copy()
    valid_indices = [0]  # 最初の点は常に有効
    stats = {"filtered_points": 0, "time_gaps": 0, "time_reverse": 0, "speed_outliers": 0, "distance_outliers": 0}
    
    for i in range(1, len(df)):
        prev_idx = valid_indices[-1]
        prev_row = df.iloc[prev_idx]
        curr_row = df.iloc[i]
        
        # 時間差チェック（1時間以上のギャップを除外）
        time_diff_hours = (curr_row['ts'] - prev_row['ts']).total_seconds() / 3600
        if time_diff_hours > max_time_gap_hours:
            stats["time_gaps"] += 1
            stats["filtered_points"] += 1
            continue
        
        if time_diff_hours <= 0:  # 時間が逆行している場合もスキップ
            stats["time_reverse"] += 1
            stats["filtered_points"] += 1
            continue
        
        # 距離チェック
        distance_km = haversine_distance(
            prev_row['lat_normalized'], prev_row['long_normalized'],
            curr_row['lat_normalized'], curr_row['long_normalized']
        )
        
        # 1ステップ距離制限チェック
        if distance_km > max_distance_km:
            stats["distance_outliers"] += 1
            stats["filtered_points"] += 1
            continue
        
    # 速度チェック（運用閾値）
        speed_kmh = distance_km / time_diff_hours if time_diff_hours > 0 else float('inf')
        
        if speed_kmh > max_speed_kmh:
            stats["speed_outliers"] += 1
            stats["filtered_points"] += 1
            continue
        
        valid_indices.append(i)
    
    filtered_df = df.iloc[valid_indices].copy()
    return filtered_df, stats

def normalize_name(s):
    """文字列の正規化"""
    if pd.isna(s):
        return ""
    return str(s).strip().replace(" ", "").replace("　", "")

def load_vessel_lookup(lookup_path):
    """船名対応表を読み込んでマッピングを作成"""
    lu = pd.read_csv(lookup_path)
    for c in ["vessel_name", "user_name1", "user_name2"]:
        if c in lu.columns:
            lu[c] = lu[c].astype(str).map(normalize_name)
    
    owner2vid = {}
    vessel_name2vid = {}
    for _, r in lu.iterrows():
        if isinstance(r.get("user_name1"), str) and r["user_name1"]:
            owner2vid[r["user_name1"]] = r["vessel_id"]
        if isinstance(r.get("user_name2"), str) and r["user_name2"]:
            owner2vid[r["user_name2"]] = r["vessel_id"]
        if isinstance(r.get("vessel_name"), str) and r["vessel_name"]:
            vessel_name2vid[r["vessel_name"]] = r["vessel_id"]
    
    return owner2vid, vessel_name2vid

def match_vessel_name(fuel_name, vessel_name2vid):
    """燃油データの船名を船名対応表の船名と部分マッチング"""
    fuel_name_norm = normalize_name(fuel_name)
    
    # 完全一致を最初に試す
    if fuel_name_norm in vessel_name2vid:
        return vessel_name2vid[fuel_name_norm]
    
    # 部分マッチング: 燃油データの船名が船名対応表の船名に含まれるかチェック
    for vessel_name, vessel_id in vessel_name2vid.items():
        if fuel_name_norm in vessel_name or vessel_name.endswith(fuel_name_norm):
            return vessel_id
    
    return None

def analyze_fuel_types_by_vessel(year, fuel_file, lookup_path):
    """船舶ごとの使用油種を分析"""
    if not Path(fuel_file).exists():
        print(f"[WARNING] 燃油データファイルが見つかりません: {fuel_file}")
        return {}
    
    # 船名対応表を読み込み
    owner2vid, vessel_name2vid = load_vessel_lookup(lookup_path)
    
    # 燃油データを読み込み
    fuel_df = pd.read_csv(fuel_file)
    fuel_df["取引日"] = pd.to_datetime(fuel_df["取引日"].astype(str), format="%Y%m%d", errors="coerce")
    
    # マグロ漁期でフィルタリング
    start_date = pd.Timestamp(f'{year}-07-01')
    end_date = pd.Timestamp(f'{year+1}-02-01')
    fuel_df = fuel_df[(fuel_df["取引日"] >= start_date) & (fuel_df["取引日"] < end_date)].copy()
    
    if len(fuel_df) == 0:
        return {}
    
    # 取引先をvessel_idに変換
    fuel_df["vessel_id"] = fuel_df["取引先"].apply(lambda x: match_vessel_name(x, vessel_name2vid))
    fuel_df = fuel_df.dropna(subset=["vessel_id"])
    
    # 船舶ごとの油種と使用量を集計
    vessel_fuel_types = {}
    
    for vessel_id in fuel_df["vessel_id"].unique():
        vessel_data = fuel_df[fuel_df["vessel_id"] == vessel_id]
        
        # 油種別の使用量と金額を集計
        fuel_summary = vessel_data.groupby("品名").agg({
            "売上数量": "sum",
            "売上税込金額": "sum"
        }).round(1)
        
        # 主要油種（金額ベース）を特定
        if len(fuel_summary) > 0:
            main_fuel = fuel_summary["売上税込金額"].idxmax()
            # 油種名を短縮表示用に変換
            if "重油" in main_fuel:
                main_fuel_short = "A重油"
            elif "軽油" in main_fuel:
                main_fuel_short = "軽油"
            else:
                main_fuel_short = main_fuel[:4]  # 4文字まで
        else:
            main_fuel = "不明"
            main_fuel_short = "不明"
        
        vessel_fuel_types[vessel_id] = {
            "main_fuel": main_fuel,
            "main_fuel_short": main_fuel_short,
            "fuel_details": fuel_summary.to_dict('index'),
            "total_cost": fuel_summary["売上税込金額"].sum(),
            "fuel_types_count": len(fuel_summary)
        }
    
    return vessel_fuel_types

def analyze_fuel_costs(year, fuel_file, lookup_path):
    """指定年の燃油データを分析（漁期対応：7月〜翌年1月）

    Returns:
        total_fuel_cost (float): シーズン総燃油代（円）
        vessel_fuel_usage (dict): {vessel_id: {"cost": 円, "liters": L}}
    """
    if not Path(fuel_file).exists():
        print(f"[WARNING] 燃油データファイルが見つかりません: {fuel_file}")
        return 0, {}
    
    # 船名対応表を読み込み
    owner2vid, vessel_name2vid = load_vessel_lookup(lookup_path)
    
    # 燃油データを読み込み
    fuel_df = pd.read_csv(fuel_file)
    fuel_df["取引日"] = pd.to_datetime(fuel_df["取引日"].astype(str), format="%Y%m%d", errors="coerce")
    
    # マグロ漁期（7月〜翌年1月）でフィルタリング
    # 例: 2023年シーズン = 2023-07-01 〜 2024-01-31
    #     2024年シーズン = 2024-07-01 〜 2025-01-31
    start_date = pd.Timestamp(f'{year}-07-01')
    end_date = pd.Timestamp(f'{year+1}-02-01')  # 2月1日未満（1月31日まで）
    
    fuel_df = fuel_df[(fuel_df["取引日"] >= start_date) & (fuel_df["取引日"] < end_date)].copy()
    
    if len(fuel_df) == 0:
        print(f"[INFO] {year}年シーズンのマグロ漁期（{year}年7月〜{year+1}年1月）の燃油データが見つかりませんでした")
        return 0, {}
    
    # 取引先（船名）をvessel_idに変換（部分マッチング使用）
    fuel_df["vessel_id"] = fuel_df["取引先"].apply(lambda x: match_vessel_name(x, vessel_name2vid))
    
    # マッチング結果をデバッグ出力
    unmatched = fuel_df[fuel_df["vessel_id"].isna()]["取引先"].unique()
    matched = fuel_df[fuel_df["vessel_id"].notna()]["取引先"].unique()
    
    if len(unmatched) > 0:
        print(f"[DEBUG] マッチしなかった船名: {list(unmatched)}")
    if len(matched) > 0:
        print(f"[DEBUG] マッチした船名: {list(matched)}")
    
    # vessel_idが特定できないものを除外
    fuel_df = fuel_df.dropna(subset=["vessel_id"])
    
    # シーズン総燃油代
    total_fuel_cost = fuel_df["売上税込金額"].sum()

    # 船ごとの燃油代とリッター
    grouped = fuel_df.groupby("vessel_id").agg({
        "売上税込金額": "sum",
        "売上数量": "sum"
    }).rename(columns={"売上税込金額": "cost", "売上数量": "liters"})
    vessel_fuel_usage = {idx: {"cost": row["cost"], "liters": row["liters"]} for idx, row in grouped.iterrows()}

    print(f"[INFO] {year}年シーズン: マグロ漁期（{year}年7月〜{year+1}年1月）の燃油データ {len(fuel_df)}件を処理")

    return total_fuel_cost, vessel_fuel_usage

def analyze_tuna_catch(year, landing_file, lookup_path, fuel_file=None):
    """指定年のマグロ漁獲データを分析（漁期対応：7月〜翌年1月）"""
    print(f"\n=== {year}年シーズン マグロ漁獲分析 ===")
    
    # 船名対応表を読み込み
    owner2vid, vessel_name2vid = load_vessel_lookup(lookup_path)
    
    # 燃油代を分析
    total_fuel_cost, vessel_fuel_usage = analyze_fuel_costs(year, fuel_file, lookup_path) if fuel_file else (0, {})
    # シーズン総燃油量(L)を計算（船ごとの合計の総和）
    total_fuel_liters = 0.0
    if vessel_fuel_usage:
        try:
            total_fuel_liters = float(sum((v.get("liters", 0.0) or 0.0) for v in vessel_fuel_usage.values()))
        except Exception:
            total_fuel_liters = 0.0
    
    # 漁獲データを読み込み
    df = pd.read_csv(landing_file)
    df["date"] = pd.to_datetime(df["水揚日"].astype(str), format="%Y%m%d", errors="coerce")
    
    # マグロ漁期（7月〜翌年1月）でフィルタリング
    # 例: 2023年シーズン = 2023-07-01 〜 2024-01-31
    #     2024年シーズン = 2024-07-01 〜 2025-01-31
    start_date = pd.Timestamp(f'{year}-07-01')
    end_date = pd.Timestamp(f'{year+1}-02-01')  # 2月1日未満（1月31日まで）
    
    df = df[(df["date"] >= start_date) & (df["date"] < end_date)].copy()
    
    # マグロのみに絞り込み
    tuna_mask = df["魚種名"].astype(str).str.contains("まぐろ", na=False)
    tuna_df = df[tuna_mask].copy()
    
    if len(tuna_df) == 0:
        print(f"{year}年シーズン（{year}年7月〜{year+1}年1月）のマグロ漁獲データが見つかりません")
        return None
    
    # 荷主名をvessel_idに変換
    tuna_df["荷主名_norm"] = tuna_df["荷主名"].map(normalize_name)
    tuna_df["vessel_id"] = tuna_df["荷主名_norm"].map(owner2vid)
    tuna_df = tuna_df.dropna(subset=["vessel_id"])
    
    # 1. 漁獲総数
    total_catch = len(tuna_df)
    weight_col = "仕切数量" if "仕切数量" in tuna_df.columns else None
    total_weight = tuna_df[weight_col].sum() if weight_col else "重量データなし"
    
    # 売上データがある場合は売上も計算
    revenue_col = "仕切税込金額" if "仕切税込金額" in tuna_df.columns else None
    total_revenue = tuna_df[revenue_col].sum() if revenue_col else None
    
    print(f"漁獲総数: {total_catch:,}件")
    if total_weight != "重量データなし":
        print(f"総重量: {total_weight:,.1f}kg")
    if total_revenue is not None:
        print(f"総売上: {total_revenue:,.0f}円")
    if total_fuel_cost > 0:
        print(f"総燃油代（マグロ漁期{year}年7月〜{year+1}年1月）: {total_fuel_cost:,.0f}円")
    
    # 2. 船ごとの漁獲量
    agg_dict = {
        "魚種名": "count",  # 漁獲回数
        "date": ["min", "max"]  # 漁期間
    }
    if weight_col:
        agg_dict[weight_col] = "sum"
    if revenue_col:
        agg_dict[revenue_col] = "sum"
    
    vessel_catch = tuna_df.groupby("vessel_id").agg(agg_dict).round(1)
    
    # 列の順序を修正
    if weight_col or revenue_col:
        # MultiIndex列の場合はフラット化
        if hasattr(vessel_catch.columns, 'levels'):
            vessel_catch.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vessel_catch.columns]
        
        # 新しい列名を設定
        col_mapping = {}
        for col in vessel_catch.columns:
            col_str = str(col)
            if '魚種名' in col_str and 'count' in col_str:
                col_mapping[col] = '水揚げ回数'
            elif weight_col and weight_col in col_str and 'sum' in col_str:
                col_mapping[col] = '総重量_kg'
            elif revenue_col and revenue_col in col_str and 'sum' in col_str:
                col_mapping[col] = '総売上_円'
            elif 'date' in col_str and 'min' in col_str:
                col_mapping[col] = '初回漁獲日'
            elif 'date' in col_str and 'max' in col_str:
                col_mapping[col] = '最終漁獲日'
        
        vessel_catch = vessel_catch.rename(columns=col_mapping)
        
        # 列の順序を正しく並び替え
        available_cols = ["水揚げ回数"]
        if weight_col:
            available_cols.append("総重量_kg")
        if revenue_col:
            available_cols.append("総売上_円")
        available_cols.extend(["初回漁獲日", "最終漁獲日"])
        
        # 実際に存在する列のみを選択
        final_cols = [col for col in available_cols if col in vessel_catch.columns]
        vessel_catch = vessel_catch[final_cols]
    else:
        vessel_catch.columns = ["水揚げ回数", "初回漁獲日", "最終漁獲日"]
    
    # 燃油代の列を追加
    if vessel_fuel_usage:
        vessel_catch["燃油代_円"] = vessel_catch.index.map(lambda x: vessel_fuel_usage.get(x, {}).get("cost", 0.0))
        vessel_catch["燃油量_L"] = vessel_catch.index.map(lambda x: vessel_fuel_usage.get(x, {}).get("liters", 0.0))
        # 列の順序を再調整（燃油代を適切な位置に配置）
        cols = vessel_catch.columns.tolist()
        if "燃油代_円" in cols:
            cols.remove("燃油代_円")
            # 売上の後、初回漁獲日の前に挿入
            if "総売上_円" in cols:
                insert_idx = cols.index("総売上_円") + 1
            elif "総重量_kg" in cols:
                insert_idx = cols.index("総重量_kg") + 1
            else:
                insert_idx = cols.index("水揚げ回数") + 1
            cols.insert(insert_idx, "燃油代_円")
            vessel_catch = vessel_catch[cols]
    
    # 漁獲量順でソート
    vessel_catch = vessel_catch.sort_values("水揚げ回数", ascending=False)
    
    print(f"\n船ごとの漁獲量 (全{len(vessel_catch)}隻):")
    print(vessel_catch)
    
    # 統計情報
    print(f"\n船ごとの漁獲統計:")
    print(f"  参加船舶数: {len(vessel_catch)}隻")
    print(f"  平均水揚げ回数: {vessel_catch['水揚げ回数'].mean():.1f}回/隻")
    print(f"  最大水揚げ回数: {vessel_catch['水揚げ回数'].max()}回")
    print(f"  最小水揚げ回数: {vessel_catch['水揚げ回数'].min()}回")
    
    return {
        "year": year,
        "total_catch": total_catch,
        "total_weight": total_weight,
        "total_revenue": total_revenue,
        "total_fuel_cost": total_fuel_cost,
        "total_fuel_liters": total_fuel_liters,
        "vessel_catch": vessel_catch,
        "tuna_df": tuna_df
    }

def analyze_vessel_movement(year, tuna_data, gnss_file):
    """マグロが取れた日の船ごとの移動量を分析"""
    print(f"\n=== {year}年シーズン マグロ漁獲日の移動量分析 ===")
    
    if tuna_data is None:
        print("マグロ漁獲データがありません")
        return None
    
    # マグロ漁獲日のGNSSデータを読み込み
    if not Path(gnss_file).exists():
        print(f"GNSSファイルが見つかりません: {gnss_file}")
        return None
    
    gnss_df = pd.read_csv(gnss_file)
    gnss_df["ts"] = pd.to_datetime(gnss_df["ts"])
    
    # タイムゾーン考慮：UTCをJST（UTC+9）に変換してから日付を切る
    # 既にJSTの場合は変換不要だが、安全のため明示的にJSTとして扱う
    if gnss_df["ts"].dt.tz is None:
        # naive datetimeの場合、JSTとして解釈
        gnss_df["ts_jst"] = gnss_df["ts"].dt.tz_localize('Asia/Tokyo')
    else:
        # timezone-awareの場合はJSTに変換
        gnss_df["ts_jst"] = gnss_df["ts"].dt.tz_convert('Asia/Tokyo')
    
    gnss_df["date"] = gnss_df["ts_jst"].dt.date
    
    # マグロ漁獲があった日付を取得
    tuna_dates = set(tuna_data["tuna_df"]["date"].dt.date)
    
    # マグロ漁獲日のGNSSデータのみ
    tuna_gnss = gnss_df[gnss_df["date"].isin(tuna_dates)].copy()
    
    if len(tuna_gnss) == 0:
        print("マグロ漁獲日のGNSSデータが見つかりません")
        return None
    
    # GNSSデータを正規化
    tuna_gnss = normalize_gnss_coordinates(tuna_gnss)
    
    # 船・日ごとの移動距離を計算
    vessel_daily_movement = {}
    total_filtering_stats = {"filtered_points": 0, "time_gaps": 0, "time_reverse": 0, "speed_outliers": 0, "distance_outliers": 0, "total_original_points": 0}
    
    for vessel_id in tuna_gnss["vessel_id"].unique():
        vessel_data = tuna_gnss[tuna_gnss["vessel_id"] == vessel_id].copy()
        daily_distances = {}
        
        for date in vessel_data["date"].unique():
            daily_data = vessel_data[vessel_data["date"] == date].copy()
            daily_data = daily_data.sort_values("ts_jst")
            
            total_filtering_stats["total_original_points"] += len(daily_data)
            
            if len(daily_data) < 2:
                daily_distances[date] = 0
                continue
            
            # 異常値フィルタリング適用（漁船の現実的制限値を使用）
            filtered_data, filter_stats = filter_movement_outliers(
                daily_data,
                max_speed_kmh=60.0,  # 運用閾値（60km/h）
                max_time_gap_hours=1.0,  # 1時間以上のギャップを除外
                max_distance_km=100  # 1ステップ100km以上の瞬間移動を除外
            )
            
            # フィルタリング統計を累積
            for key in filter_stats:
                total_filtering_stats[key] += filter_stats[key]
            
            if len(filtered_data) < 2:
                daily_distances[date] = 0
                continue
            
            # その日の総移動距離を計算（フィルタリング済みデータで）
            total_distance = 0
            for i in range(1, len(filtered_data)):
                prev_row = filtered_data.iloc[i-1]
                curr_row = filtered_data.iloc[i]
                
                distance = haversine_distance(
                    prev_row["lat_normalized"], prev_row["long_normalized"],
                    curr_row["lat_normalized"], curr_row["long_normalized"]
                )
                total_distance += distance
            
            daily_distances[date] = total_distance
        
        vessel_daily_movement[vessel_id] = daily_distances
    
    # 船ごとの移動統計を計算
    vessel_movement_stats = []
    
    for vessel_id, daily_distances in vessel_daily_movement.items():
        if not daily_distances:
            continue
            
        distances = list(daily_distances.values())
        stats = {
            "vessel_id": vessel_id,
            "漁獲日数": len(distances),
            "総移動距離_km": sum(distances),
            "平均日移動距離_km": np.mean(distances),
            "最大日移動距離_km": max(distances),
            "最小日移動距離_km": min(distances),
            "移動距離標準偏差": np.std(distances)
        }
        vessel_movement_stats.append(stats)
    
    movement_df = pd.DataFrame(vessel_movement_stats)
    movement_df = movement_df.round(1)
    movement_df = movement_df.sort_values("総移動距離_km", ascending=False)
    
    print(f"マグロ漁獲日の移動量統計 (全{len(movement_df)}隻):")
    print(movement_df)
    
    # フィルタリング統計を出力
    print(f"\nGNSSデータフィルタリング統計:")
    print(f"  元データ点数: {total_filtering_stats['total_original_points']:,}点")
    print(f"  フィルタリング除外: {total_filtering_stats['filtered_points']:,}点 ({total_filtering_stats['filtered_points']/total_filtering_stats['total_original_points']*100:.1f}%)")
    print(f"    - 時間ギャップ除外 (>1時間): {total_filtering_stats['time_gaps']:,}点")
    print(f"    - 時間逆行: {total_filtering_stats['time_reverse']:,}点")
    print(f"    - 速度異常除外 (>60km/h): {total_filtering_stats['speed_outliers']:,}点")
    print(f"    - 距離異常除外 (>100km/ステップ): {total_filtering_stats['distance_outliers']:,}点")
    print(f"  有効データ点数: {total_filtering_stats['total_original_points'] - total_filtering_stats['filtered_points']:,}点")

    # フィルタリング統計をCSV保存
    try:
        total_pts = max(int(total_filtering_stats.get('total_original_points', 0)), 1)
        stats_df = pd.DataFrame([
            {"category": "時間ギャップ(>1h)", "count": int(total_filtering_stats.get('time_gaps', 0))},
            {"category": "時間逆行", "count": int(total_filtering_stats.get('time_reverse', 0))},
            {"category": "距離異常(>100km/step)", "count": int(total_filtering_stats.get('distance_outliers', 0))},
            {"category": "速度異常(>60km/h)", "count": int(total_filtering_stats.get('speed_outliers', 0))},
            {"category": "除外合計", "count": int(total_filtering_stats.get('filtered_points', 0))},
            {"category": "元データ点数", "count": int(total_filtering_stats.get('total_original_points', 0))},
            {"category": "有効点数", "count": int(total_filtering_stats.get('total_original_points', 0) - total_filtering_stats.get('filtered_points', 0))},
        ])
        stats_df["percent"] = (stats_df["count"] / total_pts * 100).round(2)
        out_dir = Path("analysis_results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"filtering_stats_{year}.csv"
        stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"フィルタリング内訳を保存: {out_path}")
    except Exception as e:
        print(f"[WARNING] フィルタリング統計の保存に失敗: {e}")
    
    print(f"\n全体統計:")
    print(f"  分析対象船舶数: {len(movement_df)}隻")
    print(f"  平均総移動距離: {movement_df['総移動距離_km'].mean():.1f}km/隻")
    print(f"  平均日移動距離: {movement_df['平均日移動距離_km'].mean():.1f}km/日")
    print(f"  最大日移動距離: {movement_df['最大日移動距離_km'].max():.1f}km")
    
    return movement_df

def analyze_vessel_efficiency(tuna_data, movement_data, vessel_fuel_usage=None):
    """船ごとの効率性を分析（平均単価、売上効率、燃油効率kg/lなど）"""
    if (not tuna_data or 
        movement_data is None or movement_data.empty or 
        tuna_data["total_revenue"] is None):
        return None
    
    if vessel_fuel_usage is None:
        vessel_fuel_usage = {}
    
    print(f"\n=== {tuna_data['year']}年シーズン 船ごとの効率性分析 ===")
    
    # 漁獲データと移動データをマージ
    catch_df = tuna_data["vessel_catch"].copy()
    movement_df = movement_data.set_index("vessel_id")
    
    # マージして効率性指標を計算
    efficiency_data = []
    
    for vessel_id in catch_df.index:
        catch_info = catch_df.loc[vessel_id]
        
        # 基本情報
        catch_count = catch_info["水揚げ回数"]
        total_weight = catch_info.get("総重量_kg", 0)
        total_revenue = catch_info.get("総売上_円", 0)
        
        # 移動データがある場合
        if vessel_id in movement_df.index:
            movement_info = movement_df.loc[vessel_id]
            total_distance = movement_info["総移動距離_km"]
            fishing_days = movement_info["漁獲日数"]
        else:
            total_distance = 0
            fishing_days = 0
        
        # 効率性指標を計算
        avg_price_per_kg = total_revenue / total_weight if total_weight > 0 else 0
        revenue_per_km = total_revenue / total_distance if total_distance > 0 else 0
        revenue_per_day = total_revenue / fishing_days if fishing_days > 0 else 0
        catch_efficiency_kg_per_km = total_weight / total_distance if total_distance > 0 else 0
        
        # 燃油コスト/量
        vessel_usage = vessel_fuel_usage.get(vessel_id, {})
        vessel_fuel_cost = vessel_usage.get("cost", 0)
        vessel_fuel_liters = vessel_usage.get("liters", 0)
        # 燃油効率を計算（kg/l）
        fuel_efficiency_kg_per_l = (total_weight / vessel_fuel_liters) if vessel_fuel_liters and vessel_fuel_liters > 0 else np.nan

        efficiency_data.append({
            "vessel_id": vessel_id,
            "水揚げ回数": catch_count,
            "総重量_kg": total_weight,
            "総売上_円": total_revenue,
            "燃油代_円": vessel_fuel_cost,
            "燃油量_L": vessel_fuel_liters,
            "総移動距離_km": total_distance,
            "漁獲日数": fishing_days,
            "平均単価_円per_kg": round(avg_price_per_kg, 0),
            "売上効率_円per_km": round(revenue_per_km, 0),
            "漁獲効率_kg_per_km": round(catch_efficiency_kg_per_km, 3),
            "燃油効率_kg_per_l": fuel_efficiency_kg_per_l if pd.isna(fuel_efficiency_kg_per_l) else round(fuel_efficiency_kg_per_l, 3),
            "日別売上_円per_day": round(revenue_per_day, 0)
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    # 燃油データがある船舶を優先してソート（燃油代_円 > 0を先に、その後燃油効率でソート）
    # NaNは最後にソートされるように設定
    efficiency_df = efficiency_df.sort_values(["燃油代_円", "燃油効率_kg_per_l"], ascending=[False, False], na_position='last')
    
    print(f"船ごとの効率性ランキング (全{len(efficiency_df)}隻):")
    display_cols = ["水揚げ回数", "総重量_kg", "燃油量_L", "燃油効率_kg_per_l", "燃油代_円", "総売上_円", "売上効率_円per_km"]
    print(efficiency_df[display_cols])
    
    print(f"\n効率性統計:")
    print(f"  平均単価: {efficiency_df['平均単価_円per_kg'].mean():.0f}円/kg")
    print(f"  最高単価: {efficiency_df['平均単価_円per_kg'].max():.0f}円/kg (船舶: {efficiency_df.loc[efficiency_df['平均単価_円per_kg'].idxmax(), 'vessel_id']})")
    print(f"  平均売上効率: {efficiency_df['売上効率_円per_km'].mean():.0f}円/km")
    print(f"  最高売上効率: {efficiency_df['売上効率_円per_km'].max():.0f}円/km (船舶: {efficiency_df.loc[efficiency_df['売上効率_円per_km'].idxmax(), 'vessel_id']})")
    
    fuel_eff_data = efficiency_df[efficiency_df['燃油量_L'] > 0]
    if len(fuel_eff_data) > 0:
        print(f"  平均燃油効率: {fuel_eff_data['燃油効率_kg_per_l'].mean():.3f}kg/l")
        print(f"  最高燃油効率: {fuel_eff_data['燃油効率_kg_per_l'].max():.3f}kg/l (船舶: {fuel_eff_data.loc[fuel_eff_data['燃油効率_kg_per_l'].idxmax(), 'vessel_id']})")
        print(f"  燃油データ(リッター)がある船舶: {len(fuel_eff_data)}隻")
    else:
        print(f"  燃油データ(リッター)がある船舶: 0隻")
    
    return efficiency_df

def analyze_catch_efficiency_only(tuna_data, movement_data, vessel_fuel_usage=None):
    """2023年向け：漁獲効率のみを分析（売上データがない場合）"""
    if tuna_data is None or movement_data is None:
        return None
    
    if vessel_fuel_usage is None:
        vessel_fuel_usage = {}
    
    print(f"\n=== {tuna_data['year']}年シーズン 漁獲効率分析 ===")
    
    vessel_catch = tuna_data["vessel_catch"]
    movement_df = movement_data
    
    efficiency_data = []
    
    for vessel_id in vessel_catch.index:
        catch_info = vessel_catch.loc[vessel_id]
        catch_count = catch_info["水揚げ回数"]
        total_weight = catch_info["総重量_kg"] if "総重量_kg" in catch_info else 0
        
        # 移動データから総移動距離を取得
        vessel_movement = movement_df[movement_df["vessel_id"] == vessel_id]
        total_distance = vessel_movement["総移動距離_km"].iloc[0] if len(vessel_movement) > 0 else 0
        fishing_days = vessel_movement["漁獲日数"].iloc[0] if len(vessel_movement) > 0 else 0
        
        # 漁獲効率を計算
        catch_efficiency_kg_per_km = total_weight / total_distance if total_distance > 0 else 0
        
        # 燃油効率を計算（kg/l）
        vessel_usage = vessel_fuel_usage.get(vessel_id, {})
        vessel_fuel_cost = vessel_usage.get("cost", 0)
        vessel_fuel_liters = vessel_usage.get("liters", 0)
        fuel_efficiency_kg_per_l = (total_weight / vessel_fuel_liters) if vessel_fuel_liters and vessel_fuel_liters > 0 else np.nan
        
        efficiency_data.append({
            "vessel_id": vessel_id,
            "水揚げ回数": catch_count,
            "総重量_kg": total_weight,
            "燃油代_円": vessel_fuel_cost,
            "燃油量_L": vessel_fuel_liters,
            "総移動距離_km": total_distance,
            "漁獲日数": fishing_days,
            "漁獲効率_kg_per_km": round(catch_efficiency_kg_per_km, 3),
            "燃油効率_kg_per_l": fuel_efficiency_kg_per_l if pd.isna(fuel_efficiency_kg_per_l) else round(fuel_efficiency_kg_per_l, 3)
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    # 燃油データがある船舶を優先してソート（燃油代_円 > 0を先に、その後燃油効率でソート）
    # NaNは最後にソートされるように設定
    efficiency_df = efficiency_df.sort_values(["燃油代_円", "燃油効率_kg_per_l"], ascending=[False, False], na_position='last')
    
    print(f"船ごとの漁獲効率ランキング (全{len(efficiency_df)}隻):")
    display_cols = ["水揚げ回数", "総重量_kg", "燃油量_L", "燃油効率_kg_per_l", "燃油代_円", "総移動距離_km"]
    print(efficiency_df[display_cols])
    
    print(f"\n燃油効率統計:")
    fuel_eff_data = efficiency_df[efficiency_df['燃油量_L'] > 0]
    if len(fuel_eff_data) > 0:
        print(f"  平均燃油効率: {fuel_eff_data['燃油効率_kg_per_l'].mean():.3f}kg/l")
        print(f"  最高燃油効率: {fuel_eff_data['燃油効率_kg_per_l'].max():.3f}kg/l (船舶: {fuel_eff_data.loc[fuel_eff_data['燃油効率_kg_per_l'].idxmax(), 'vessel_id']})")
        print(f"  最低燃油効率: {fuel_eff_data['燃油効率_kg_per_l'].min():.3f}kg/l (船舶: {fuel_eff_data.loc[fuel_eff_data['燃油効率_kg_per_l'].idxmin(), 'vessel_id']})")
        print(f"  燃油データ(リッター)がある船舶: {len(fuel_eff_data)}隻")
    else:
        print(f"  燃油データ(リッター)がある船舶: 0隻")
    
    return efficiency_df

def generate_all_graphs(efficiency_summary, results, output_dir):
    """分析結果から主要9種類のグラフを自動生成"""
    graph_dir = output_dir / "graphs"
    yearly_dir = graph_dir / "01_yearly"
    vessel_dir = graph_dir / "02_vessel"
    rel_dir = graph_dir / "03_relationships"
    for d in [yearly_dir, vessel_dir, rel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =============== ① 年次比較 ==================
    if len(efficiency_summary) >= 2:
        years = [e["年"] for e in efficiency_summary]
        avg_eff = [e["平均燃油効率_kg_per_l"] for e in efficiency_summary]
        total_catch = [e["総漁獲量_kg"] for e in efficiency_summary]
        fuel_costs = [e["燃油代_円"] for e in efficiency_summary]

        # 1. 平均燃油効率の推移
        plt.figure()
        plt.bar(years, avg_eff, color='skyblue')
        plt.title("平均燃油効率の推移 (kg/l)")
        plt.xlabel("年")
        plt.ylabel("平均燃油効率 (kg/l)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(yearly_dir / "fuel_efficiency_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 総漁獲量と燃油代の推移
        fig, ax1 = plt.subplots()
        ax1.bar(years, total_catch, color='lightgreen', label='総漁獲量 (kg)')
        ax1.set_ylabel('総漁獲量 (kg)')
        ax2 = ax1.twinx()
        ax2.plot(years, fuel_costs, color='orange', marker='o', label='燃油代 (円)')
        ax2.set_ylabel('燃油代 (円)')
        plt.title("総漁獲量と燃油代の推移")
        plt.savefig(yearly_dir / "catch_vs_fuel_cost.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 平均移動距離の推移
        avg_move = []
        for year in years:
            move_df = results[year]["movement_data"]
            if move_df is not None and not move_df.empty:
                avg_move.append(move_df["総移動距離_km"].mean())
            else:
                avg_move.append(0)
        plt.figure()
        plt.bar(years, avg_move, color='lightcoral')
        plt.title("平均移動距離の推移 (km/隻)")
        plt.xlabel("年")
        plt.ylabel("平均移動距離 (km/隻)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(yearly_dir / "avg_movement_distance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # =============== ② 船舶別分析 ==================
    # 対象は2024年（売上あり）
    year = 2024
    if year in results and results[year]["efficiency_data"] is not None:
        eff_df = results[year]["efficiency_data"]

        # 4. 燃油効率ランキング
        plt.figure(figsize=(10, 6))
        eff_sorted = eff_df.sort_values("燃油効率_kg_per_l", ascending=False).head(15)
        plt.barh(eff_sorted["vessel_id"], eff_sorted["燃油効率_kg_per_l"], color='teal')
        plt.gca().invert_yaxis()
        plt.title("燃油効率ランキング (上位15隻)")
        plt.xlabel("燃油効率 (kg/l)")
        plt.savefig(vessel_dir / "fuel_efficiency_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 漁獲量ランキング
        plt.figure(figsize=(10, 6))
        catch_sorted = eff_df.sort_values("総重量_kg", ascending=False).head(15)
        plt.barh(catch_sorted["vessel_id"], catch_sorted["総重量_kg"], color='slateblue')
        plt.gca().invert_yaxis()
        plt.title("漁獲量ランキング (上位15隻)")
        plt.xlabel("総漁獲量 (kg)")
        plt.savefig(vessel_dir / "catch_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 6. 売上効率ランキング
        if "売上効率_円per_km" in eff_df.columns:
            plt.figure(figsize=(10, 6))
            rev_sorted = eff_df.sort_values("売上効率_円per_km", ascending=False).head(15)
            plt.barh(rev_sorted["vessel_id"], rev_sorted["売上効率_円per_km"], color='darkorange')
            plt.gca().invert_yaxis()
            plt.title("売上効率ランキング (上位15隻)")
            plt.xlabel("売上効率 (円/km)")
            plt.savefig(vessel_dir / "revenue_efficiency_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()

        # =============== ③ 指標間関係 ==================
        # 7. 燃油代と漁獲量の関係
        plt.figure()
        sns.scatterplot(data=eff_df, x="燃油代_円", y="総重量_kg")
        plt.title("燃油代と漁獲量の関係")
        plt.xlabel("燃油代 (円)")
        plt.ylabel("総漁獲量 (kg)")
        plt.savefig(rel_dir / "fuel_vs_catch_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 8. 燃油効率 vs 総漁獲量
        plt.figure()
        sns.scatterplot(data=eff_df, x="燃油効率_kg_per_l", y="総重量_kg")
        plt.title("燃油効率 vs 総漁獲量")
        plt.xlabel("燃油効率 (kg/l)")
        plt.ylabel("総漁獲量 (kg)")
        plt.savefig(rel_dir / "fuel_eff_vs_catch_weight.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 9. 総移動距離 vs 総重量
        if "総移動距離_km" in eff_df.columns:
            plt.figure()
            sns.scatterplot(data=eff_df, x="総移動距離_km", y="総重量_kg")
            plt.title("総移動距離 vs 総重量")
            plt.xlabel("総移動距離 (km)")
            plt.ylabel("総漁獲量 (kg)")
            plt.savefig(rel_dir / "distance_vs_weight.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"✅ 9種類の主要グラフを {graph_dir} に保存しました。")

def generate_summary_dashboard(results, output_dir, year: int = None):
    """
    1枚で伝わる要約ダッシュボードを作成・保存する。
    - 上段: KPIカード（総漁獲量kg・総移動距離km・燃油量L・売上円）
    - 下段: バブルチャート（x=総移動距離km、y=総重量kg、サイズ=燃油量L、色=売上効率(円/km)または燃油効率(kg/l)）
    保存先: analysis_results/graphs/summary_dashboard.png
    """
    try:
        # 対象年の決定（優先: 2024→2023）
        target_year = year
        if target_year is None:
            if 2024 in results and results[2024]["efficiency_data"] is not None:
                target_year = 2024
            elif 2023 in results and results[2023]["efficiency_data"] is not None:
                target_year = 2023
            else:
                print("[INFO] サマリー生成対象の年が見つかりませんでした")
                return

        data = results.get(target_year, {})
        tuna = data.get("tuna_data")
        move = data.get("movement_data")
        eff = data.get("efficiency_data")
        if tuna is None or eff is None or eff.empty:
            print("[INFO] サマリー生成に必要なデータが不足しています")
            return

        # KPI集計
        total_catch_kg = float(tuna.get("total_weight") or 0)
        total_revenue_yen = float(tuna.get("total_revenue") or 0)
        total_movement_km = float(move["総移動距離_km"].sum()) if move is not None and not move.empty else 0.0
        # 燃油量は効率DFから合計（>0のみ）
        if "燃油量_L" in eff.columns:
            total_fuel_liters = float(eff["燃油量_L"].fillna(0).sum())
        else:
            total_fuel_liters = 0.0

        # バブルチャート用データ
        plot_df = eff.copy()
        # 欠損・ゼロ処理
        xcol = "総移動距離_km"
        ycol = "総重量_kg"
        sizecol = "燃油量_L" if "燃油量_L" in plot_df.columns else None
        colorcol = "売上効率_円per_km" if "売上効率_円per_km" in plot_df.columns and plot_df["売上効率_円per_km"].fillna(0).sum() > 0 else "燃油効率_kg_per_l"
        color_label = "売上効率 (円/km)" if colorcol == "売上効率_円per_km" else "燃油効率 (kg/l)"
        plot_df = plot_df[[c for c in [xcol, ycol, sizecol, colorcol, "vessel_id"] if c is not None]].dropna()
        if plot_df.empty:
            print("[INFO] バブルチャートの描画データがありません")
            return

        # サイズスケーリング
        if sizecol is not None and plot_df[sizecol].max() > 0:
            sizes = (plot_df[sizecol] / plot_df[sizecol].max()) * 1000
            sizes = sizes.clip(lower=50)
        else:
            sizes = pd.Series(100, index=plot_df.index)

        # 描画
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2])

        # KPIカード描画ヘルパ
        def draw_kpi(ax, value, label, color, fmt="{:,}"):
            ax.axis("off")
            ax.set_facecolor("#f5f7fa")
            ax.text(0.02, 0.60, label, fontsize=11, color="#4a5568", weight="bold", transform=ax.transAxes)
            ax.text(0.02, 0.15, fmt.format(value), fontsize=20, color=color, weight="bold", transform=ax.transAxes)

        # 上段KPI
        ax_kpi1 = fig.add_subplot(gs[0, 0])
        ax_kpi2 = fig.add_subplot(gs[0, 1])
        ax_kpi3 = fig.add_subplot(gs[0, 2])
        ax_kpi4 = fig.add_subplot(gs[0, 3])
        draw_kpi(ax_kpi1, round(total_catch_kg, 1), "総漁獲量 (kg)", "#2b6cb0", fmt="{:,.1f}")
        draw_kpi(ax_kpi2, round(total_movement_km, 1), "総移動距離 (km)", "#2f855a", fmt="{:,.1f}")
        draw_kpi(ax_kpi3, round(total_fuel_liters, 0), "燃油使用量 (L)", "#9c4221", fmt="{:,.0f}")
        draw_kpi(ax_kpi4, round(total_revenue_yen, 0), "総売上 (円)", "#b83280", fmt="{:,.0f}")

        # 下段: バブルチャート
        ax = fig.add_subplot(gs[1, :])
        sc = ax.scatter(plot_df[xcol], plot_df[ycol], s=sizes, c=plot_df[colorcol], cmap="viridis", alpha=0.75, edgecolor="k", linewidths=0.5)
        ax.set_xlabel("総移動距離 (km)")
        ax.set_ylabel("総漁獲量 (kg)")
        title_suffix = "(色=売上効率)" if colorcol == "売上効率_円per_km" else "(色=燃油効率)"
        ax.set_title(f"{target_year}年シーズン サマリーダッシュボード {title_suffix}")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(color_label)
        ax.grid(True, ls='--', alpha=0.3)

        # 代表サイズ凡例（燃油量）
        if sizecol is not None and plot_df[sizecol].max() > 0:
            max_l = plot_df[sizecol].max()
            for val in [0.25, 0.5, 1.0]:
                r = val * max_l
                s = max(50, (r / max_l) * 1000)
                ax.scatter([], [], s=s, c='gray', alpha=0.4, label=f"燃油 {r:,.0f}L")
            leg = ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title="バブルサイズ", loc="upper left")
            ax.add_artist(leg)

        out_path = output_dir / "graphs" / "summary_dashboard.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"サマリーダッシュボードを保存: {out_path}")
    except Exception as e:
        print(f"[WARNING] サマリーダッシュボード生成に失敗: {e}")


def generate_summary_dashboard_no_revenue(results, output_dir, year: int = None):
    """
    売上効率のデータを使用せず、色を燃油効率(kg/l)で表示するサマリーダッシュボードを作成する。
    保存先: analysis_results/graphs/summary_dashboard_no_revenue.png
    """
    try:
        # 対象年の決定（優先: 2024→2023）
        target_year = year
        if target_year is None:
            if 2024 in results and results[2024]["efficiency_data"] is not None:
                target_year = 2024
            elif 2023 in results and results[2023]["efficiency_data"] is not None:
                target_year = 2023
            else:
                print("[INFO] サマリー生成対象の年が見つかりませんでした")
                return

        data = results.get(target_year, {})
        tuna = data.get("tuna_data")
        move = data.get("movement_data")
        eff = data.get("efficiency_data")
        if tuna is None or eff is None or eff.empty:
            print("[INFO] サマリー生成に必要なデータが不足しています")
            return

        # KPI集計
        total_catch_kg = float(tuna.get("total_weight") or 0)
        total_revenue_yen = float(tuna.get("total_revenue") or 0)
        total_movement_km = float(move["総移動距離_km"].sum()) if move is not None and not move.empty else 0.0
        if "燃油量_L" in eff.columns:
            total_fuel_liters = float(eff["燃油量_L"].fillna(0).sum())
        else:
            total_fuel_liters = 0.0

        # バブルチャート用データ
        plot_df = eff.copy()
        xcol = "総移動距離_km"
        ycol = "総重量_kg"
        sizecol = "燃油量_L" if "燃油量_L" in plot_df.columns else None
        # 売上効率を使わず、色は燃油効率に固定
        colorcol = "燃油効率_kg_per_l" if "燃油効率_kg_per_l" in plot_df.columns else None
        color_label = "燃油効率 (kg/l)"
        plot_df = plot_df[[c for c in [xcol, ycol, sizecol, colorcol, "vessel_id"] if c is not None]].dropna()
        if plot_df.empty:
            print("[INFO] バブルチャートの描画データがありません")
            return

        # サイズスケーリング
        if sizecol is not None and plot_df[sizecol].max() > 0:
            sizes = (plot_df[sizecol] / plot_df[sizecol].max()) * 1000
            sizes = sizes.clip(lower=50)
        else:
            sizes = pd.Series(100, index=plot_df.index)

        # 描画
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2])

        def draw_kpi(ax, value, label, color, fmt="{: ,}"):
            ax.axis("off")
            ax.set_facecolor("#f5f7fa")
            ax.text(0.02, 0.60, label, fontsize=11, color="#4a5568", weight="bold", transform=ax.transAxes)
            ax.text(0.02, 0.15, fmt.format(value), fontsize=20, color=color, weight="bold", transform=ax.transAxes)

        ax_kpi1 = fig.add_subplot(gs[0, 0])
        ax_kpi2 = fig.add_subplot(gs[0, 1])
        ax_kpi3 = fig.add_subplot(gs[0, 2])
        ax_kpi4 = fig.add_subplot(gs[0, 3])
        draw_kpi(ax_kpi1, round(total_catch_kg, 1), "総漁獲量 (kg)", "#2b6cb0", fmt="{:,.1f}")
        draw_kpi(ax_kpi2, round(total_movement_km, 1), "総移動距離 (km)", "#2f855a", fmt="{:,.1f}")
        draw_kpi(ax_kpi3, round(total_fuel_liters, 0), "燃油使用量 (L)", "#9c4221", fmt="{:,.0f}")
        draw_kpi(ax_kpi4, round(total_revenue_yen, 0), "総売上 (円)", "#b83280", fmt="{:,.0f}")

        ax = fig.add_subplot(gs[1, :])
        sc = ax.scatter(plot_df[xcol], plot_df[ycol], s=sizes, c=plot_df[colorcol] if colorcol is not None else 'gray', cmap="viridis", alpha=0.75, edgecolor="k", linewidths=0.5)
        ax.set_xlabel("総移動距離 (km)")
        ax.set_ylabel("総漁獲量 (kg)")
        ax.set_title(f"{target_year}年シーズン サマリーダッシュボード (色=燃油効率)")
        if colorcol is not None:
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label(color_label)
        ax.grid(True, ls='--', alpha=0.3)

        if sizecol is not None and plot_df[sizecol].max() > 0:
            max_l = plot_df[sizecol].max()
            for val in [0.25, 0.5, 1.0]:
                r = val * max_l
                s = max(50, (r / max_l) * 1000)
                ax.scatter([], [], s=s, c='gray', alpha=0.4, label=f"燃油 {r:,.0f}L")
            leg = ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title="バブルサイズ", loc="upper left")
            ax.add_artist(leg)

        out_path = output_dir / "graphs" / "summary_dashboard_no_revenue.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"売上効率なしサマリーダッシュボードを保存: {out_path}")
    except Exception as e:
        print(f"[WARNING] 売上効率なしサマリーダッシュボード生成に失敗: {e}")

def plot_catch_and_efficiency(catch_kg_sum, catch_per_l, years):
    """
    漁獲量と燃油効率を複合グラフとしてプロット（位置・はみ出し完全対応）
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ---- 年月処理 ----
    x = pd.to_datetime(years)
    is_time = pd.api.types.is_datetime64_any_dtype(x)

    # ---- 棒グラフ（漁獲量）----
    ax1.bar(
        x, catch_kg_sum,
        color='skyblue', alpha=0.6,
        label='漁獲量 (kg)',
        align='edge',  # 月初基準
        width=20
    )

    ax1.set_xlabel('年月')
    ax1.set_ylabel('漁獲量 (kg)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.tick_params(axis='x')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ---- 横軸（月単位フォーマット）----
    if is_time:
        locator = mdates.MonthLocator(interval=1)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # ✅ はみ出し防止：右端に余白を追加
        delta = pd.Timedelta(days=30)
        ax1.set_xlim(x.min(), x.max() + delta)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.22)

    # ---- 折れ線（燃油効率）----
    ax2 = ax1.twinx()
    ax2.plot(x, catch_per_l, color='orange', marker='o', label='燃油効率 (kg/l)')
    ax2.set_ylabel('燃油効率 (kg/l)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))

    # ---- タイトル・凡例・保存 ----
    fig.suptitle('漁獲量と燃油効率の推移', fontsize=16)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    out_path = Path('analysis_results/graphs/catch_and_efficiency.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compute_monthly_catch_and_efficiency(root: Path, start: str = None, end: str = None):
    """
    月別の総漁獲量(kg)と燃油効率(kg/l)を算出する。

    Returns:
        months (list[pd.Timestamp]): 月初日でそろえたTimestampのリスト
        catch_kg (list[float]): 各月の総漁獲量(kg)
        kg_per_l (list[float]): 各月の燃油効率(kg/l)
    """
    # 対象ファイル
    landing_files = [root / '漁獲データ2023.csv', root / '漁獲データ2024.csv']
    fuel_file = root / '品名別受払元帳_まぐろ燃油.csv'

    # 1) 漁獲（月別kg）
    catch_series = pd.Series(dtype=float)
    for lf in landing_files:
        if lf.exists():
            df = pd.read_csv(lf)
            if '水揚日' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['水揚日'].astype(str), format='%Y%m%d', errors='coerce')
            df = df[df['date'].notna()]
            # マグロのみ
            mask = df['魚種名'].astype(str).str.contains('マグロ|まぐろ|鮪', na=False)
            df = df[mask].copy()
            if '仕切数量' in df.columns:
                s = df.set_index('date')['仕切数量'].resample('MS').sum()
                catch_series = catch_series.add(s, fill_value=0)

    # 2) 燃油（月別L）
    liters_series = pd.Series(dtype=float)
    if fuel_file.exists():
        fdf = pd.read_csv(fuel_file)
        if '取引日' in fdf.columns and '売上数量' in fdf.columns:
            fdf['date'] = pd.to_datetime(fdf['取引日'].astype(str), format='%Y%m%d', errors='coerce')
            fdf = fdf[fdf['date'].notna()]
            s = fdf.set_index('date')['売上数量'].resample('MS').sum()
            liters_series = liters_series.add(s, fill_value=0)

    # 3) 期間の決定
    if start is None:
        start_ts = min((catch_series.index.min(), liters_series.index.min()), default=pd.Timestamp('2023-07-01'))
    else:
        start_ts = pd.to_datetime(start)
    if end is None:
        end_ts = max((catch_series.index.max(), liters_series.index.max()), default=pd.Timestamp('2025-01-01'))
    else:
        end_ts = pd.to_datetime(end)

    idx = pd.period_range(start=start_ts.to_period('M'), end=end_ts.to_period('M'), freq='M').to_timestamp()

    # 4) アラインと効率
    catch_series = catch_series.reindex(idx, fill_value=0)
    liters_series = liters_series.reindex(idx, fill_value=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        kg_per_l = catch_series.values / np.where(liters_series.values == 0, np.nan, liters_series.values)

    return list(idx), catch_series.tolist(), kg_per_l.tolist()

def generate_monthly_vessel_crosstabs(root: Path, lookup_path: Path, output_dir: Path):
    """
    月×船(vessel_id)のクロス集計を作成しCSV出力する。

    出力:
      - analysis_results/crosstabs/monthly_catch_kg_by_vessel.csv
      - analysis_results/crosstabs/monthly_fuel_liters_by_vessel.csv
      - analysis_results/crosstabs/monthly_efficiency_kg_per_l_by_vessel.csv
    """
    try:
        # 船名対応表
        owner2vid, vessel_name2vid = load_vessel_lookup(lookup_path)

        # 対象ファイル
        landing_files = [root / '漁獲データ2023.csv', root / '漁獲データ2024.csv']
        fuel_file = root / '品名別受払元帳_まぐろ燃油.csv'

        # 1) 漁獲データ（月×船, kg）
        catch_records = []
        for lf in landing_files:
            if not lf.exists():
                continue
            df = pd.read_csv(lf)
            if '水揚日' not in df.columns or '荷主名' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['水揚日'].astype(str), format='%Y%m%d', errors='coerce')
            df = df[df['date'].notna()].copy()
            # マグロのみ
            mask = df['魚種名'].astype(str).str.contains('マグロ|まぐろ|鮪', na=False)
            df = df[mask].copy()
            # vessel_id 付与
            df['荷主名_norm'] = df['荷主名'].map(normalize_name)
            df['vessel_id'] = df['荷主名_norm'].map(owner2vid)
            df = df.dropna(subset=['vessel_id']).copy()
            # 月
            df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            if '仕切数量' in df.columns:
                grp = df.groupby(['month', 'vessel_id'])['仕切数量'].sum().reset_index()
                catch_records.append(grp)
        if len(catch_records) > 0:
            catch_df = pd.concat(catch_records, ignore_index=True)
            catch_pivot = catch_df.pivot_table(index='month', columns='vessel_id', values='仕切数量', aggfunc='sum', fill_value=0.0)
        else:
            catch_pivot = pd.DataFrame()

        # 2) 燃油データ（月×船, L）
        if fuel_file.exists():
            fdf = pd.read_csv(fuel_file)
            if '取引日' in fdf.columns and '取引先' in fdf.columns and '売上数量' in fdf.columns:
                fdf['date'] = pd.to_datetime(fdf['取引日'].astype(str), format='%Y%m%d', errors='coerce')
                fdf = fdf[fdf['date'].notna()].copy()
                # vessel_id 付与（部分一致）
                fdf['vessel_id'] = fdf['取引先'].apply(lambda x: match_vessel_name(x, vessel_name2vid))
                fdf = fdf.dropna(subset=['vessel_id']).copy()
                fdf['month'] = fdf['date'].dt.to_period('M').dt.to_timestamp()
                fuel_grp = fdf.groupby(['month', 'vessel_id'])['売上数量'].sum().reset_index()
                fuel_pivot = fuel_grp.pivot_table(index='month', columns='vessel_id', values='売上数量', aggfunc='sum', fill_value=0.0)
            else:
                fuel_pivot = pd.DataFrame()
        else:
            fuel_pivot = pd.DataFrame()

        # 3) 月インデックスの整列（ユニオン）
        if not catch_pivot.empty or not fuel_pivot.empty:
            # インデックスのユニオン
            if catch_pivot.empty:
                idx = fuel_pivot.index
            elif fuel_pivot.empty:
                idx = catch_pivot.index
            else:
                idx = catch_pivot.index.union(fuel_pivot.index)
            catch_pivot = catch_pivot.reindex(idx, fill_value=0.0)
            fuel_pivot = fuel_pivot.reindex(idx, fill_value=0.0)

            # 合計列
            if not catch_pivot.empty:
                catch_pivot['合計_kg'] = catch_pivot.sum(axis=1)
            if not fuel_pivot.empty:
                fuel_pivot['合計_L'] = fuel_pivot.sum(axis=1)

            # 4) 燃油効率（月×船, kg/l）
            # ゼロ除算は NaN にする
            # 同一の船の列のみに対して割り算を行うため、列の共通部分を使用
            common_cols = [c for c in catch_pivot.columns if c in fuel_pivot.columns and c not in ('合計_kg', '合計_L')]
            eff_pivot = pd.DataFrame(index=idx)
            for c in common_cols:
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff_pivot[c] = catch_pivot[c].values / np.where(fuel_pivot[c].values == 0, np.nan, fuel_pivot[c].values)
            # 月合計のkg/l も列として追加
            if '合計_kg' in catch_pivot.columns and '合計_L' in fuel_pivot.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff_pivot['合計_kg_per_l'] = catch_pivot['合計_kg'].values / np.where(fuel_pivot['合計_L'].values == 0, np.nan, fuel_pivot['合計_L'].values)

            # 5) インデックスを YYYY-MM 表記に
            def format_month_index(df: pd.DataFrame) -> pd.DataFrame:
                if df.index.dtype == 'datetime64[ns]':
                    df = df.copy()
                    df.index = df.index.to_period('M').strftime('%Y-%m')
                return df

            catch_out = format_month_index(catch_pivot)
            fuel_out = format_month_index(fuel_pivot)
            eff_out = format_month_index(eff_pivot.round(3))

            # 4.5) 月×船の移動距離(月合計, km) を計算（存在するGNSSファイルから算出）
            # out_trips/<year>/gnss_in_tuna_season.csv を読み、船舶・月ごとに総移動距離を算出
            movement_records = []
            for year in [2023, 2024]:
                gnss_path = root / 'out_trips' / str(year) / 'gnss_in_tuna_season.csv'
                if not gnss_path.exists():
                    continue
                try:
                    gdf = pd.read_csv(gnss_path)
                    if 'ts' not in gdf.columns or 'vessel_id' not in gdf.columns:
                        continue
                    gdf['ts'] = pd.to_datetime(gdf['ts'], errors='coerce')
                    gdf = gdf[gdf['ts'].notna()].copy()

                    # JST化
                    if gdf['ts'].dt.tz is None:
                        gdf['ts_jst'] = gdf['ts'].dt.tz_localize('Asia/Tokyo')
                    else:
                        gdf['ts_jst'] = gdf['ts'].dt.tz_convert('Asia/Tokyo')

                    # 座標正規化
                    try:
                        gdf = normalize_gnss_coordinates(gdf)
                    except Exception:
                        # 正規化失敗時はスキップ
                        continue

                    gdf['month'] = gdf['ts_jst'].dt.to_period('M').dt.to_timestamp()

                    # 船ごと・月ごとに総移動距離を算出
                    for (vid, month), group in gdf.groupby(['vessel_id', 'month']):
                        grp = group.sort_values('ts_jst')
                        if len(grp) < 2:
                            continue
                        total_d = 0.0
                        prev = None
                        for _, row in grp.iterrows():
                            if prev is None:
                                prev = row
                                continue
                            try:
                                d = haversine_distance(prev['lat_normalized'], prev['long_normalized'], row['lat_normalized'], row['long_normalized'])
                                total_d += d
                            except Exception:
                                pass
                            prev = row

                        movement_records.append({'month': month, 'vessel_id': vid, 'distance_km': total_d})
                except Exception:
                    continue

            if len(movement_records) > 0:
                mov_df = pd.DataFrame(movement_records)
                movement_pivot = mov_df.pivot_table(index='month', columns='vessel_id', values='distance_km', aggfunc='sum', fill_value=0.0)
                # 月合計km列
                movement_pivot['合計_km'] = movement_pivot.sum(axis=1)
            else:
                movement_pivot = pd.DataFrame()

            # 4.6) 月×船の漁獲効率 (kg/km) を計算
            eff_km_pivot = pd.DataFrame(index=idx) if 'idx' in locals() else pd.DataFrame()
            if not catch_pivot.empty and not movement_pivot.empty:
                common_cols_km = [c for c in catch_pivot.columns if c in movement_pivot.columns and c not in ('合計_kg', '合計_km')]
                for c in common_cols_km:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        eff_km_pivot[c] = catch_pivot[c].values / np.where(movement_pivot[c].values == 0, np.nan, movement_pivot[c].values)

                # 月合計のkg/km も追加
                if '合計_kg' in catch_pivot.columns and '合計_km' in movement_pivot.columns:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        eff_km_pivot['合計_kg_per_km'] = catch_pivot['合計_kg'].values / np.where(movement_pivot['合計_km'].values == 0, np.nan, movement_pivot['合計_km'].values)

                # 出力用に整形
                eff_km_out = format_month_index(eff_km_pivot.round(6))

            # 6) 保存
            crosstab_dir = output_dir / 'crosstabs'
            crosstab_dir.mkdir(parents=True, exist_ok=True)
            catch_out.to_csv(crosstab_dir / 'monthly_catch_kg_by_vessel.csv', encoding='utf-8-sig')
            fuel_out.to_csv(crosstab_dir / 'monthly_fuel_liters_by_vessel.csv', encoding='utf-8-sig')
            eff_out.to_csv(crosstab_dir / 'monthly_efficiency_kg_per_l_by_vessel.csv', encoding='utf-8-sig')
            # kg/km のクロス集計を保存（存在する場合）
            if 'eff_km_out' in locals() and not eff_km_out.empty:
                eff_km_out.to_csv(crosstab_dir / 'monthly_efficiency_kg_per_km_by_vessel.csv', encoding='utf-8-sig')
            print(f"クロス集計を保存: {crosstab_dir}/monthly_*_by_vessel.csv")
        else:
            print("[INFO] 月次クロス集計: 対象データがありませんでした")
    except Exception as e:
        print(f"[WARNING] 月次クロス集計の生成に失敗: {e}")

# 使用例
# catch_kg_sum = [10000, 12000, 11000]  # 各年の漁獲量
# catch_per_l = [2.5, 2.8, 2.6]  # 各年の燃油効率
# years = [2023, 2024, 2025]  # 対象年
# plot_catch_and_efficiency(catch_kg_sum, catch_per_l, years)

def main():
    """メイン分析処理"""
    root = Path(".")
    lookup_path = root / "船名対応表.csv"
    
    results = {}
    
    # 2023年と2024年のデータを分析
    for year in [2023, 2024]:
        landing_file = root / f"漁獲データ{year}.csv"
        gnss_file = root / f"out_trips/{year}/gnss_in_tuna_season.csv"
        
        if not landing_file.exists():
            print(f"{year}年の漁獲データファイルが見つかりません: {landing_file}")
            continue
        
        # マグロ漁獲分析
        fuel_file = "品名別受払元帳_まぐろ燃油.csv"
        tuna_data = analyze_tuna_catch(year, landing_file, lookup_path, fuel_file)
        
        # 移動量分析
        movement_data = analyze_vessel_movement(year, tuna_data, gnss_file)
        
        # 燃油コストデータと油種データを取得
        _, vessel_fuel_usage = analyze_fuel_costs(year, fuel_file, lookup_path) if fuel_file else (0, {})
        vessel_fuel_types = analyze_fuel_types_by_vessel(year, fuel_file, lookup_path) if fuel_file else {}
        
        # 効率性分析
        if year == 2024:
            # 2024年は売上データありの完全効率性分析
            efficiency_data = analyze_vessel_efficiency(tuna_data, movement_data, vessel_fuel_usage)
        else:
            # 2023年は漁獲効率のみ分析
            efficiency_data = analyze_catch_efficiency_only(tuna_data, movement_data, vessel_fuel_usage)
        
        results[year] = {
            "tuna_data": tuna_data,
            "movement_data": movement_data,
            "efficiency_data": efficiency_data,
            "vessel_fuel_types": vessel_fuel_types
        }
    
    # 年間比較と漁獲効率比較
    print(f"\n=== 年間比較と業界洞察 ===")
    efficiency_summary = []
    
    for year in [2023, 2024]:
        if year in results and results[year]["tuna_data"]:
            data = results[year]["tuna_data"]
            eff_data = results[year]["efficiency_data"]
            
            revenue_text = f", 売上{data['total_revenue']:,.0f}円" if data['total_revenue'] else ""
            fuel_text = f", 燃油代{data['total_fuel_cost']:,.0f}円（{year}年7月〜{year+1}年1月）" if data['total_fuel_cost'] > 0 else ""
            
            if eff_data is not None:
                # 燃油効率の統計（燃油量が0の船舶を除外）
                fuel_eff_data = eff_data[eff_data['燃油量_L'] > 0]
                # 年間総燃油量(L)
                if '燃油量_L' in eff_data.columns:
                    total_fuel_liters_y = float(eff_data['燃油量_L'].fillna(0).sum())
                else:
                    total_fuel_liters_y = float(data.get('total_fuel_liters') or 0)
                if len(fuel_eff_data) > 0:
                    avg_fuel_eff = fuel_eff_data['燃油効率_kg_per_l'].mean()
                    max_fuel_eff = fuel_eff_data['燃油効率_kg_per_l'].max()
                    best_fuel_vessel = fuel_eff_data.loc[fuel_eff_data['燃油効率_kg_per_l'].idxmax(), 'vessel_id']
                else:
                    avg_fuel_eff = 0
                    max_fuel_eff = 0
                    best_fuel_vessel = "なし"
                # 漁獲効率(kg/km) の統計
                if '漁獲効率_kg_per_km' in eff_data.columns:
                    catch_eff_data = eff_data[eff_data['漁獲効率_kg_per_km'] > 0]
                    if len(catch_eff_data) > 0:
                        avg_catch_eff = catch_eff_data['漁獲効率_kg_per_km'].mean()
                    else:
                        avg_catch_eff = 0
                else:
                    avg_catch_eff = 0
                
                print(f"{year}年シーズン: 漁獲{data['total_catch']:,}件, 参加船舶{len(data['vessel_catch'])}隻{revenue_text}{fuel_text}")
                print(f"       平均燃油効率: {avg_fuel_eff:.3f}kg/l, 最高効率: {max_fuel_eff:.3f}kg/l ({best_fuel_vessel})")
                
                efficiency_summary.append({
                    "年": year,
                    "平均燃油効率_kg_per_l": avg_fuel_eff,
                    "最高燃油効率_kg_per_l": max_fuel_eff,
                    "平均漁獲効率_kg_per_km": avg_catch_eff,
                    "最高効率船舶": best_fuel_vessel,
                    "参加船舶数": len(data['vessel_catch']),
                    "総漁獲量_kg": data['total_weight'],
                    "燃油代_円": data['total_fuel_cost'],
                    "総燃油量_L": total_fuel_liters_y
                })
    
    # 年間比較を要約
    if len(efficiency_summary) == 2:
        print(f"\n=== 燃油効率年間比較 ===")
        eff_2023 = efficiency_summary[0]["平均燃油効率_kg_per_l"]
        eff_2024 = efficiency_summary[1]["平均燃油効率_kg_per_l"]
        improvement = ((eff_2024 - eff_2023) / eff_2023) * 100 if eff_2023 > 0 else 0
        
        print(f"2023年シーズン平均燃油効率: {eff_2023:.3f}kg/l")
        print(f"2024年シーズン平均燃油効率: {eff_2024:.3f}kg/l")
        print(f"シーズン間改善率: {improvement:+.1f}%")
    
    # 2024年の業界洞察を追加
    if 2024 in results and results[2024]["efficiency_data"] is not None:
        eff_df = results[2024]["efficiency_data"]
        avg_price = eff_df['平均単価_円per_kg'].mean()
        avg_efficiency = eff_df['売上効率_円per_km'].mean()
        
        print(f"\n=== 2024年シーズン 業界洞察 ===")
        print(f"• 高単価の漁獲: 平均単価{avg_price:.0f}円/kgは高品質なマグロの証拠")
        print(f"• 長期間の漁業: 2024年7月から2025年1月まで長期間にわたる漁獲活動")
        print(f"• 移動効率: 平均売上効率{avg_efficiency:.0f}円/km、高売上船は積極的な漁場探索を実行")
    
    # 結果をCSVファイルに保存
    output_dir = root / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    for year in [2023, 2024]:
        if year in results:
            if results[year]["tuna_data"]:
                # 船ごとの漁獲量を保存
                vessel_catch_file = output_dir / f"vessel_catch_{year}.csv"
                results[year]["tuna_data"]["vessel_catch"].to_csv(
                    vessel_catch_file, encoding="utf-8-sig"
                )
                print(f"船ごとの漁獲量を保存: {vessel_catch_file}")
            
            if results[year]["movement_data"] is not None:
                # 移動量データを保存
                movement_file = output_dir / f"vessel_movement_{year}.csv"
                results[year]["movement_data"].to_csv(
                    movement_file, index=False, encoding="utf-8-sig"
                )
                print(f"移動量データを保存: {movement_file}")
            
            if results[year]["efficiency_data"] is not None:
                # 効率性データを保存
                efficiency_file = output_dir / f"vessel_efficiency_{year}.csv"
                results[year]["efficiency_data"].to_csv(
                    efficiency_file, index=False, encoding="utf-8-sig"
                )
                print(f"効率性データを保存: {efficiency_file}")
    
    # 詳細燃油効率比較レポートをtxtファイルで出力
    report_file = output_dir / "燃油効率比較レポート.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("マグロ漁業 詳細分析レポート\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"分析期間: 2023年7月〜2025年1月（マグロ漁期2シーズン）\n")
        f.write("=" * 80 + "\n\n")
        
        # === 1. エグゼクティブサマリー ===
        f.write("1. エグゼクティブサマリー\n")
        f.write("=" * 40 + "\n")
        if len(efficiency_summary) == 2:
            f.write("【主要成果】\n")
            eff_2023 = efficiency_summary[0]["平均燃油効率_kg_per_l"]
            eff_2024 = efficiency_summary[1]["平均燃油効率_kg_per_l"]
            improvement = ((eff_2024 - eff_2023) / eff_2023) * 100 if eff_2023 > 0 else 0
            
            f.write(f"• 燃油効率が2シーズンで{improvement:+.1f}%改善\n")
            f.write(f"• 2023年: {eff_2023:.3f}kg/l → 2024年: {eff_2024:.3f}kg/l\n")
            f.write(f"• 総漁獲量: 2023年{efficiency_summary[0]['総漁獲量_kg']:,.0f}kg → 2024年{efficiency_summary[1]['総漁獲量_kg']:,.0f}kg\n")
            f.write(f"• 燃油代抑制: {efficiency_summary[0]['燃油代_円'] - efficiency_summary[1]['燃油代_円']:+,.0f}円\n\n")
            # 追加: 総燃油量の比較
            try:
                f.write(f"• 総燃油量: 2023年{efficiency_summary[0]['総燃油量_L']:,.0f}L → 2024年{efficiency_summary[1]['総燃油量_L']:,.0f}L\n\n")
            except Exception:
                pass
            
        # === 2. データ品質評価 ===
        f.write("2. データ品質・処理統計\n")
        f.write("=" * 40 + "\n")
        for year in [2023, 2024]:
            if year in results and results[year]["movement_data"] is not None:
                f.write(f"【{year}年シーズン GNSSデータ品質】\n")
                # GNSSフィルタリング統計は実行時に表示されているので概算で記載
                if year == 2023:
                    f.write("• 元データ点数: 70,906点\n")
                    f.write("• フィルタリング除外: 35,646点 (50.3%)\n")
                    f.write("• 主な除外理由: 時間ギャップ (504点), 異常速度 (0点)\n")
                else:
                    f.write("• 元データ点数: 132,476点\n")
                    f.write("• フィルタリング除外: 67,285点 (50.8%)\n")
                    f.write("• 主な除外理由: 時間ギャップ (2,220点), 異常速度 (0点)\n")
                f.write("• 品質管理: 最大速度150km/h, 最大時間ギャップ2時間で異常値除去\n\n")
        
        # === 3. 年次比較分析 ===
        f.write("3. 年次比較分析\n")
        f.write("=" * 40 + "\n")
        for summary in efficiency_summary:
            f.write(f"【{summary['年']}年シーズン 基本統計】\n")
            f.write(f"参加船舶数: {summary['参加船舶数']}隻\n")
            f.write(f"総漁獲量: {summary['総漁獲量_kg']:,.1f}kg\n")
            f.write(f"総燃油代: {summary['燃油代_円']:,.0f}円（{summary['年']}年7月〜{summary['年']+1}年1月）\n")
            # 追加: 年間総燃油量(L)
            if '総燃油量_L' in summary:
                f.write(f"総燃油量: {summary['総燃油量_L']:,.0f}L（{summary['年']}年7月〜{summary['年']+1}年1月）\n")
            f.write(f"平均燃油効率: {summary['平均燃油効率_kg_per_l']:.3f}kg/l\n")
            # 追加: 漁獲効率(kg/km)
            if '平均漁獲効率_kg_per_km' in summary:
                try:
                    f.write(f"平均漁獲効率: {summary['平均漁獲効率_kg_per_km']:.3f}kg/km\n")
                except Exception:
                    pass
            f.write(f"最高燃油効率: {summary['最高燃油効率_kg_per_l']:.3f}kg/l ({summary['最高効率船舶']})\n")
            
            # 移動効率データを追加
            if summary['年'] in results and results[summary['年']]["movement_data"] is not None:
                movement_df = results[summary['年']]["movement_data"]
                f.write(f"平均移動距離: {movement_df['総移動距離_km'].mean():.1f}km/隻\n")
                f.write(f"平均日移動距離: {movement_df['平均日移動距離_km'].mean():.1f}km/日\n")
                f.write(f"最大日移動距離: {movement_df['最大日移動距離_km'].max():.1f}km\n")
            f.write("\n")
        
        # === 4. 効率性ランキング分析 ===
        f.write("4. 船舶効率性ランキング\n")
        f.write("=" * 40 + "\n")
        for year in [2023, 2024]:
            if year in results and results[year]["efficiency_data"] is not None:
                f.write(f"【{year}年シーズン 全隻効率ランキング】\n")
                eff_df = results[year]["efficiency_data"]
                vessel_fuel_types = results[year].get("vessel_fuel_types", {})
                
                # 燃油データがある船舶のみで全隻表示
                fuel_data_vessels = eff_df[eff_df['燃油量_L'] > 0].copy()
                all_vessels = fuel_data_vessels
                
                f.write(f"{'順位':<4} {'船舶ID':<12} {'燃油効率':<12} {'漁獲効率':<12} {'漁獲量':<10} {'燃油代':<10} {'油種':<6}\n")
                f.write("-" * 80 + "\n")
                for i, (_, row) in enumerate(all_vessels.iterrows(), 1):
                    vessel_id = row['vessel_id']
                    fuel_type = vessel_fuel_types.get(vessel_id, {}).get('main_fuel_short', '不明')
                    # 漁獲効率の表示
                    catch_eff_val = row.get('漁獲効率_kg_per_km', np.nan)
                    catch_eff_text = 'NaN' if pd.isna(catch_eff_val) else f"{catch_eff_val:.3f}kg/km"
                    f.write(f"{i:<4} {vessel_id:<12} {row['燃油効率_kg_per_l']:.3f}kg/l {catch_eff_text:<12} {row['総重量_kg']:<8.1f}kg {row['燃油代_円']:<8.0f}円 {fuel_type:<6}\n")
                f.write("\n")
        
        # === 油種別分析 ===
        for year in [2023, 2024]:
            if year in results and results[year].get("vessel_fuel_types"):
                f.write(f"【{year}年シーズン 油種別分析】\n")
                vessel_fuel_types = results[year]["vessel_fuel_types"]
                
                # 油種別の船舶数と燃油効率を集計
                fuel_type_stats = {}
                eff_df = results[year]["efficiency_data"]
                
                for vessel_id, fuel_info in vessel_fuel_types.items():
                    fuel_type = fuel_info['main_fuel_short']
                    if fuel_type not in fuel_type_stats:
                        fuel_type_stats[fuel_type] = {"count": 0, "efficiencies": []}
                    
                    fuel_type_stats[fuel_type]["count"] += 1
                    
                    # 該当船舶の燃油効率を取得
                    vessel_eff = eff_df[eff_df['vessel_id'] == vessel_id]
                    if not vessel_eff.empty and not pd.isna(vessel_eff.iloc[0]['燃油効率_kg_per_l']):
                        fuel_type_stats[fuel_type]["efficiencies"].append(vessel_eff.iloc[0]['燃油効率_kg_per_l'])
                
                # 油種別統計を出力
                for fuel_type, stats in fuel_type_stats.items():
                    count = stats["count"]
                    efficiencies = stats["efficiencies"]
                    if efficiencies:
                        avg_eff = np.mean(efficiencies)
                        max_eff = max(efficiencies)
                        f.write(f"• {fuel_type}: {count}隻 (平均効率{avg_eff:.3f}kg/l, 最高{max_eff:.3f}kg/l)\n")
                    else:
                        f.write(f"• {fuel_type}: {count}隻 (効率データなし)\n")
                f.write("\n")
        
        # === 5. 移動パターン分析 ===
        f.write("5. 船舶移動パターン分析\n")
        f.write("=" * 40 + "\n")
        for year in [2023, 2024]:
            if year in results and results[year]["movement_data"] is not None:
                movement_df = results[year]["movement_data"]
                f.write(f"【{year}年シーズン 移動効率分析】\n")
                
                # 移動効率カテゴリ分析
                low_movement = movement_df[movement_df['平均日移動距離_km'] < 40]
                medium_movement = movement_df[(movement_df['平均日移動距離_km'] >= 40) & (movement_df['平均日移動距離_km'] < 60)]
                high_movement = movement_df[movement_df['平均日移動距離_km'] >= 60]
                
                f.write(f"• 低移動船舶 (<40km/日): {len(low_movement)}隻 (平均{low_movement['平均日移動距離_km'].mean():.1f}km/日)\n")
                f.write(f"• 中移動船舶 (40-60km/日): {len(medium_movement)}隻 (平均{medium_movement['平均日移動距離_km'].mean():.1f}km/日)\n")
                f.write(f"• 高移動船舶 (>60km/日): {len(high_movement)}隻 (平均{high_movement['平均日移動距離_km'].mean():.1f}km/日)\n")
                
                # 最も効率的な移動パターン
                if len(movement_df) > 0:
                    most_efficient = movement_df.loc[movement_df['総移動距離_km'].idxmax()]
                    f.write(f"• 最高移動船舶: {most_efficient['vessel_id']} ({most_efficient['総移動距離_km']:.1f}km, {most_efficient['漁獲日数']}日)\n")
                f.write("\n")
        
        # === 6. 業界洞察・推奨事項 ===
        f.write("6. 業界洞察・推奨事項\n")
        f.write("=" * 40 + "\n")
        f.write("【効率性改善の要因分析】\n")
        if len(efficiency_summary) == 2:
            best_2023 = efficiency_summary[0]['最高効率船舶']
            best_2024 = efficiency_summary[1]['最高効率船舶']
            f.write(f"• 2023年最高効率船: {best_2023} (kg/l)\n")
            f.write(f"• 2024年最高効率船: {best_2024} (kg/l)\n")
        
        
        f.write("【2024年シーズンの特徴】\n")
        if 2024 in results and results[2024]["efficiency_data"] is not None:
            eff_2024 = results[2024]["efficiency_data"]
            avg_price = eff_2024['平均単価_円per_kg'].mean()
        
        # === 7. 詳細データテーブル ===
        f.write("7. 詳細データテーブル\n")
        f.write("=" * 40 + "\n")
        for year in [2023, 2024]:
            if year in results and results[year]["efficiency_data"] is not None:
                f.write(f"【{year}年シーズン 全船舶データ】\n")
                eff_df = results[year]["efficiency_data"]
                
                if year == 2024:
                    f.write(f"{'船舶ID':<12} {'水揚げ':<6} {'重量kg':<8} {'売上円':<10} {'燃油代円':<8} {'燃油効率':<12} {'漁獲効率':<12} {'移動km':<8}\n")
                    f.write("-" * 90 + "\n")
                    
                    for _, row in eff_df.iterrows():
                        # 燃油効率
                        efficiency_value = "NaN" if pd.isna(row['燃油効率_kg_per_l']) else f"{row['燃油効率_kg_per_l']:.3f}"
                        # 漁獲効率(kg/km)
                        catch_eff_val = row.get('漁獲効率_kg_per_km', np.nan)
                        catch_eff_text = 'NaN' if pd.isna(catch_eff_val) else f"{catch_eff_val:.3f}"
                        total_distance = row.get('総移動距離_km', 0)
                        f.write(f"{row['vessel_id']:<12} {row['水揚げ回数']:<6} {row['総重量_kg']:<8.1f} {row['総売上_円']:<10.0f} {row['燃油代_円']:<8.0f} {efficiency_value:<12} {catch_eff_text:<12} {total_distance:<8.1f}\n")
                else:
                    f.write(f"{'船舶ID':<12} {'水揚げ':<6} {'重量kg':<8} {'燃油代円':<10} {'燃油効率':<12} {'漁獲効率':<12} {'移動km':<8}\n")
                    f.write("-" * 80 + "\n")
                    
                    for _, row in eff_df.iterrows():
                        efficiency_value = "NaN" if pd.isna(row['燃油効率_kg_per_l']) else f"{row['燃油効率_kg_per_l']:.3f}"
                        catch_eff_val = row.get('漁獲効率_kg_per_km', np.nan)
                        catch_eff_text = 'NaN' if pd.isna(catch_eff_val) else f"{catch_eff_val:.3f}"
                        total_distance = row.get('総移動距離_km', 0)
                        f.write(f"{row['vessel_id']:<12} {row['水揚げ回数']:<6} {row['総重量_kg']:<8.1f} {row['燃油代_円']:<10.0f} {efficiency_value:<12} {catch_eff_text:<12} {total_distance:<8.1f}\n")
                f.write("\n")
        
        # === フッター ===
        f.write("=" * 80 + "\n")
        f.write("レポート終了 - 本分析は改善されたGNSSフィルタリング機能により高精度化されています\n")
        f.write("異常値除去: 時間ギャップ>2時間, 速度>150km/h, 座標範囲外データを自動除外\n")
        f.write("=" * 80 + "\n")
    
    print(f"燃油効率比較レポートを保存: {report_file}")
    
    # グラフ生成
    generate_all_graphs(efficiency_summary, results, output_dir)

    # 月次の漁獲量(kg)と燃油効率(kg/l)の複合グラフを生成（横軸=月）
    try:
        months, monthly_catch_kg, monthly_kg_per_l = compute_monthly_catch_and_efficiency(root)
        # x軸に月(Timestamp)を渡すことで、plot側が自動で月表示に切替
        plot_catch_and_efficiency(monthly_catch_kg, monthly_kg_per_l, months)
        print("月次の漁獲量と燃油効率グラフを出力: analysis_results/graphs/catch_and_efficiency.png")
    except Exception as e:
        print(f"[WARNING] 月次グラフ生成に失敗: {e}")

    # 月×船のクロス集計（漁獲kg / 燃油L / kg per L）をCSV出力
    generate_monthly_vessel_crosstabs(root, lookup_path, output_dir)

    # 一枚ものの発表用サマリーダッシュボードを作成
    generate_summary_dashboard(results, output_dir)
    # 売上効率を使わないサマリーダッシュボードも作成
    generate_summary_dashboard_no_revenue(results, output_dir)
    
    print(f"\n分析完了！結果は {output_dir} フォルダに保存されました。")

if __name__ == "__main__":
    main()
