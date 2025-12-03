import os
import requests
import sqlite3
from time import sleep
from datetime import datetime

# --- 設定 ---
# 環境変数から APIキー取得。環境変数に設定していない場合は直接キーを記述
API_KEY = os.getenv("APISPORTS_KEY") 
HEADERS = {"x-apisports-key": API_KEY}

# SQLite DB 設定
# スクリプト自体のディレクトリパスを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルート（srcの1つ上）
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

# データベースファイルへのパス
DB_PATH = os.path.join(PROJECT_ROOT, "db", "matches.db")
LEAGUE_ID = 39  # プレミアリーグ
# 取得したいシーズンを明示的に指定
SEASONS = [2021, 2022, 2023, 2024, 2025] 
# ----------------

if not API_KEY:
    print("❌ エラー: APIキー (APISPORTS_KEY) が設定されていません。")
    exit()

# DB接続とディレクトリ作成
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# --- データベーススキーマ作成 ---
# matches テーブル
c.execute('''
CREATE TABLE IF NOT EXISTS matches (
    fixture_id INTEGER PRIMARY KEY,
    date TEXT,
    season INTEGER,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT
)
''')
# match_statistics テーブル
c.execute('''
CREATE TABLE IF NOT EXISTS match_statistics (
    fixture_id INTEGER,
    team_id INTEGER,
    team_name TEXT,
    shots_on_goal INTEGER,
    shots_off_goal INTEGER,
    possession REAL,
    passes INTEGER,
    passes_accuracy REAL,
    fouls INTEGER,
    corners INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    PRIMARY KEY (fixture_id, team_id)
)
''')
conn.commit()
print("DBスキーマの準備が完了しました。")

# --- 統計値取得ユーティリティ関数 ---
def get_stat(statistics_list, stat_name, is_percent=False):
    """API応答の統計リストから指定された値を抽出する"""
    for s in statistics_list:
        if s['type'] == stat_name:
            value = s['value']
            if value is None:
                return None
            if is_percent and isinstance(value, str) and '%' in value:
                # パーセンテージ記号を削除して float に変換
                return float(value.strip('%'))
            # 値が数値として取得できることを確認
            try:
                return int(value) if stat_name not in ["Ball Possession", "Passes accurate"] else value
            except (ValueError, TypeError):
                return None
    return None

# --- データ取得とDB保存 ---
for season in SEASONS:
    print(f"\n=== Fetching season {season} ===")
    
    # 試合一覧取得 URL
    url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={season}"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ APIリクエスト中にエラーが発生しました ({season}): {e}")
        continue

    if data.get('errors'):
        print(f"⚠️ API returned errors for season {season}: {data['errors']}")
        continue

    matches = data.get('response', [])
    if not matches:
        print(f"No matches found for season {season}.")
        continue

    matches_to_insert = []
    
    # --- 1. 試合情報 (matches) DBに保存 ---
    for match in matches:
        fixture = match['fixture']
        teams = match['teams']
        scores = match['score']['fulltime']
        
        
        matches_to_insert.append((
            fixture['id'],
            fixture['date'],
            season,
            teams['home']['name'],
            teams['away']['name'],
            scores['home'],
            scores['away'],
            fixture['status']['short']
        ))

    c.executemany('''
    INSERT OR REPLACE INTO matches (
        fixture_id, date, season, home_team, away_team, home_score, away_score, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', matches_to_insert)
    conn.commit()

    # match_statistics テーブルに shots_off_goal カラムを追加する（既に存在する場合はスキップ）
    try:
        c.execute('ALTER TABLE match_statistics ADD COLUMN shots_off_goal INTEGER')
        conn.commit()
        print("✅ スキーマ修正: shots_off_goal カラムを追加しました。")
    except sqlite3.OperationalError as e:
        # カラムが既にある場合（duplicate column name）や他のエラーを捕捉
        if "duplicate column name" in str(e):
            pass # 既にカラムが存在するのでOK
        else:
            # その他の重要なエラーであれば再スロー
            raise e
    
    
    
    # ----------------------------------------------------------------------
    print(f"✔️ {len(matches_to_insert)} matches processed for season {season}")

    # --- 2. 各試合の statistics 取得 ---
    # FT (Full Time) または PEN (Penalty) で終了した試合のみ統計情報を取得
    completed_matches = [m for m in matches if m['fixture']['status']['short'] in ('FT', 'PEN')]
    print(f"   - Fetching statistics for {len(completed_matches)} completed matches...")

    for match in completed_matches:
        fixture_id = match['fixture']['id']

        # 既に統計情報がDBに存在するかチェック
        c.execute("SELECT 1 FROM match_statistics WHERE fixture_id = ?", (fixture_id,))
        if c.fetchone():
            # print(f"   - Statistics already exist for fixture {fixture_id}. Skipping.")
            continue

        stats_url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
        stats_response = requests.get(stats_url, headers=HEADERS)
        stats_data = stats_response.json()

        if stats_data.get('errors'):
            print(f"⚠️ Error fetching statistics for fixture {fixture_id}: {stats_data['errors']}")
            sleep(1)
            continue

        stats_list = stats_data.get('response', [])
        stats_to_insert = []

        for team_stats in stats_list:
            team = team_stats['team']
            statistics = team_stats.get('statistics', [])

            stats_to_insert.append((
                fixture_id,
                team['id'],
                team['name'],
                # shots_on_goal, shots_off_goalはAPIの統計名に合わせる
                get_stat(statistics, "Shots on Goal"),
                get_stat(statistics, "Shots off Goal"),
                get_stat(statistics, "Ball Possession", is_percent=True),
                get_stat(statistics, "Total passes"),
                get_stat(statistics, "Passes accurate", is_percent=True),
                get_stat(statistics, "Fouls"),
                get_stat(statistics, "Corner Kicks"),
                get_stat(statistics, "Yellow Cards"),
                get_stat(statistics, "Red Cards")
            ))

        if stats_to_insert:
            c.executemany('''
            INSERT OR IGNORE INTO match_statistics (
                fixture_id, team_id, team_name, shots_on_goal, shots_off_goal, possession,
                passes, passes_accuracy, fouls, corners, yellow_cards, red_cards
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', stats_to_insert)
            conn.commit()
            # print(f"   - Statistics inserted for fixture {fixture_id}")
        
        sleep(1) # API制限回避 (1分あたり30リクエストの制限を考慮)

conn.close()
print("\n=======================================================")
print("✅ 全てのシーズン (2021年〜2025年) のデータ取得と保存が完了しました。")
print("データは 'db/matches.db' に格納されています。")
print("=======================================================")