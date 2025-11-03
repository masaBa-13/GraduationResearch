import os
import importlib.util
import matplotlib

# GUI不要のバックエンドを使用
matplotlib.use('Agg')

WORKSPACE = "/Users/codepro2020mbp02/Library/CloudStorage/GoogleDrive-masaharu2000.01.04@gmail.com/マイドライブ/研究データ/原デｰﾀ".replace("ﾃ", "テ")
WORKSPACE = "/Users/codepro2020mbp02/Library/CloudStorage/GoogleDrive-masaharu2000.01.04@gmail.com/マイドライブ/研究データ/原データ"

# analysis.py を動かす前にカレントディレクトリをプロジェクトルートへ
os.chdir(WORKSPACE)

# analysis.py の関数をロード
analysis_path = os.path.join(WORKSPACE, "analysis.py")
spec = importlib.util.spec_from_file_location("analysis", analysis_path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

# テストデータ
catch_kg_sum = [10000, 12000, 11000]
catch_per_l = [2.5, 2.8, 2.6]
years = [2023, 2024, 2025]

# 出力ディレクトリの存在確認（なければ作成）
out_dir = os.path.join(WORKSPACE, "analysis_results", "graphs")
os.makedirs(out_dir, exist_ok=True)

# 実行
analysis.plot_catch_and_efficiency(catch_kg_sum, catch_per_l, years)

# 検証
out_path = os.path.join(out_dir, "catch_and_efficiency.png")
exists = os.path.exists(out_path)
size = os.path.getsize(out_path) if exists else -1
print(f"RESULT exists={exists} size={size} path={out_path}")
