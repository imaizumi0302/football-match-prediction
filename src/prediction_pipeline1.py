import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, log_loss, f1_score, classification_report
from IPython.display import display

import pickle
import gc
import os
import datetime as dt
from datetime import datetime
import json 

# --------------------------------------------------------
# ★★★ 修正点: 絶対パスの定義 ★★★
# スクリプト自体のディレクトリパスを取得し、すべての相対パスを絶対パスに変換します。
# これにより、タスクスケジューラの実行開始ディレクトリに依存しなくなります。
# ★★★ 絶対パスの定義 ★★★
# スクリプト自体のディレクトリパスを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルート（srcの1つ上）
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

# データベースファイルへのパス
DB_PATH = os.path.join(PROJECT_ROOT, "db", "matches.db")

# 過去シーズンデータファイルへのパス
SEASON_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "premier_league.csv")

# モデル保存ディレクトリへのパス
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# --------------------------------------------------------

#モデル学習に使用する特徴量の選択
# FEATURES = ["home_team","away_team",'home_season_wins_ave_overall',
#        'away_season_wins_ave_overall','home_last_points','away_last_points']

FEATURES = ["home_team","away_team",'home_season_wins_ave_overall',
            'away_season_wins_ave_overall','home_last_points','away_last_points',
            'home_last_gd','away_last_gd','home_recent_10_goal_diff',
            'away_recent_10_goal_diff','points_difference']

TARGET = "target"

#ハイパーパラメータの設定
params = {
    "n_estimators":1000,
    "learning_rate":0.05,
    "num_leaves":32
}


def feature_engineering(matches_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    matchesデータとstatisticsデータを結合し、前処理と特徴量計算を実行する。
    """
    # ホームチーム用統計データのカラム名を変更
    home_stats = stats_df.copy()
    home_stats = home_stats.rename(columns={
        "team_name": "home_team", # 結合キーとして利用
        "shots": "home_shots",
        "shots_on_goal": "home_shots_on_goal",
        "possession": "home_possession",
        "passes": "home_passes",
        "passes_accuracy": "home_passes_accuracy",
        "corners": "home_corners",
        "offsides": "home_offsides",
        "fouls": "home_fouls",
        "yellow_cards": "home_yellow_cards",
        "red_cards": "home_red_cards"
    })

    # アウェイチーム用統計データのカラム名を変更
    away_stats = stats_df.copy()
    away_stats = away_stats.rename(columns={
        "team_name": "away_team", # 結合キーとして利用
        "shots": "away_shots",
        "shots_on_goal": "away_shots_on_goal",
        "possession": "away_possession",
        "passes": "away_passes",
        "passes_accuracy": "away_passes_accuracy",
        "corners": "away_corners",
        "offsides": "away_offsides",
        "fouls": "away_fouls",
        "yellow_cards": "away_yellow_cards",
        "red_cards": "away_red_cards"
    })
    
    # 統計データを試合データに結合
    df = pd.merge(matches_df, home_stats, on=['fixture_id', 'home_team'], how='left')
    df = pd.merge(df, away_stats, on=['fixture_id', 'away_team'], how='left')

    # ----------------------------------------------------
    # 3. 欠損値処理 (FT データのみを対象)
    # ----------------------------------------------------
    
    # FTのデータのみ欠損値処理 & 再結合
    merged_df_FT = df[df["status"] == "FT"].copy()
    merged_df_NS = df[df["status"] != "FT"].copy() # FT以外（NSなど）のデータ

    # 欠損値を0で埋める (FTデータのみ)
    merged_df_FT = merged_df_FT.fillna(0)

    # statusがFTのデータとNSのデータを再結合
    df = pd.concat([merged_df_FT, merged_df_NS], ignore_index=True)



    # ----------------------------------------------------
    # 4. 過去シーズンのデータを結合（昇格チームの欠損値処理も含む）
    # ----------------------------------------------------
    
    try:
        # 過去シーズンのデータを読み込み (★修正: 絶対パスを使用)
        season_df = pd.read_csv(SEASON_DATA_PATH)
        season_df = season_df.drop(columns = ["played","notes"],axis = 1)
    except FileNotFoundError:
        # スクリプトパスが間違っているか、ファイルが存在しない場合のエラー表示
        print(f"エラー: 過去シーズンデータ '{SEASON_DATA_PATH}' が見つかりません。過去シーズン成績の結合をスキップします。")
        return df # ファイルがない場合は結合せずにそのまま返す

    # 2つのデータフレームのチーム名の表記の仕方をそろえる
    mapping = {'Manchester City':'Manchester City','Manchester Utd':'Manchester United','Liverpool':'Liverpool','Chelsea':'Chelsea',
              'Leicester City':'Leicester','West Ham':'West Ham','Tottenham':'Tottenham','Arsenal':'Arsenal','Leeds United':'Leeds',
              'Everton':'Everton','Aston Villa':'Aston Villa','Newcastle Utd':'Newcastle','Wolves':'Wolves','Crystal Palace':'Crystal Palace',
              'Southampton':'Southampton','Brighton':'Brighton','Burnley':'Burnley','Fulham':'Fulham','Sheffield Utd':'Sheffield Utd',
              'Brentford':'Brentford','Watford':'Watford','Norwich City':'Norwich','Bournemouth':'Bournemouth','Nottingham Forest':'Nottingham Forest','Luton Town':'Luton','Ipswich':'Ipswich'}    

     # チーム名の表記を統一
    season_df["team"] = season_df["team"].replace(mapping) 


    # --- 結合後の新しいカラム名の定義 ---
    # premier_league.csv の 'points' のリネーム
    season_col_map = {
        "points":"last_points", 
        "position":"last_position", "won":"last_won", "drawn":"last_drawn", 
        "lost":"last_lost", "gf":"last_gf", "ga":"last_ga", "gd":"last_gd"
    }

    # 1. ホームチームとして結合するためのデータ準備
    season_home_df = season_df.rename(columns = season_col_map)
    season_home_df = season_home_df.rename(columns = {
        "season_end_year":"season", "team":"home_team",
    })
    
    # 結合後の列名に 'home_' 接頭辞を付加
    new_home_cols = {v: f"home_{v}" for v in season_col_map.values()}
    season_home_df = season_home_df.rename(columns = new_home_cols)

    # 2. アウェイチームとして結合するためのデータ準備
    season_away_df = season_df.rename(columns = season_col_map)
    season_away_df = season_away_df.rename(columns = {
        "season_end_year":"season", "team":"away_team",
    })

    # 結合後の列名に 'away_' 接頭辞を付加
    new_away_cols = {v: f"away_{v}" for v in season_col_map.values()}
    season_away_df = season_away_df.rename(columns = new_away_cols)

    # --- 結合ロジック ---
    df = pd.merge(df, season_home_df, on = ["season","home_team"], how = "left")
    df = pd.merge(df, season_away_df, on = ["season","away_team"], how = "left")
    # この時点で、dfには 'home_last_points' などが追加され、昇格組は NaN

    # ----------------------------------------------------
    # 5. 昇格組の欠損値処理（各試合の【結合済みの前シーズン】の17位の値で埋める）
    # ----------------------------------------------------
    
   # 欠損値を埋める対象となる、season_df内の【元の列名】
    original_cols = list(season_col_map.keys()) # ['points', 'position', 'won', 'drawn', ...]
    
    # 試合のシーズンごとに欠損値の埋め立てを実行
    for target_season in df['season'].unique():
        # target_season は試合が行われるシーズンの終了年 (例: 2025年)
        # season_dfの target_season の行は、その試合の前シーズンの成績 (2025年終了のシーズン)
        
        # 17位チームの成績を取得
        relegation_avoidance_team = season_df[
            (season_df["season_end_year"] == target_season) & 
            (season_df["position"] == 17)
        ]
        
        # 代理値ルックアップテーブルを作成
        fill_values = {}
        if not relegation_avoidance_team.empty:
            for original_col in original_cols:
                # 17位の成績をそのまま取得
                val = relegation_avoidance_team[original_col].iloc[0] 
                
                # 結合後のカラム名（ home_last_points など）を生成
                new_suffix = season_col_map[original_col] 
                
                fill_values[f'home_{new_suffix}'] = val
                fill_values[f'away_{new_suffix}'] = val
        else:
             print(f"警告: シーズン {target_season} の17位のデータが見つかりませんでした。このシーズンの昇格組の処理をスキップします。")
             continue
        
        # このシーズン（target_season）の試合に限定して欠損値を埋める
        target_fill_cols = list(fill_values.keys())

        # 欠損値（昇格組）に対してのみ、17位の値で埋める
        for full_col_name in target_fill_cols:
            is_target_season = df['season'] == target_season
            is_nan = df[full_col_name].isna()
            
            # 論理インデックスを使用して、対象のセルのみに値を代入
            df.loc[is_target_season & is_nan, full_col_name] = fill_values[full_col_name]


    # ----------------------------------------------------
    # 6. データタイプの変換と列の削除
    # ----------------------------------------------------

    # dateカラムのデータタイプをdatetimeに変換
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # shotsカラムとshots on goalカラムの値が全く同じになっているため、shotsカラムををdropする
    df = df.drop(columns = ["home_shots","away_shots"], axis = 1, errors='ignore')




    
    
    # -------------------------------------------------特徴量エンジニアリング（ここからユーザー提供ロジック）----------------------------------------------------------------

    # カラムの型変換（Int64: 欠損値を許容する整数型）
    df[['home_passes_accuracy','home_yellow_cards', 'home_red_cards','away_passes_accuracy',
                'away_yellow_cards', 'away_red_cards','home_last_position', 'home_last_won', 'home_last_drawn',
                'home_last_lost', 'home_last_gf', 'home_last_ga', 'home_last_gd','home_last_points',
                'away_last_position', 'away_last_won','away_last_drawn', 'away_last_lost', 
                'away_last_gf', 'away_last_ga','away_last_gd', 'away_last_points']] = df[['home_passes_accuracy',
                                                                    'home_yellow_cards', 'home_red_cards','away_passes_accuracy',
                                                                    'away_yellow_cards', 'away_red_cards','home_last_position', 
                                                                    'home_last_won', 'home_last_drawn','home_last_lost', 
                                                                    'home_last_gf', 'home_last_ga', 'home_last_gd','home_last_points',
                                                                    'away_last_position', 'away_last_won','away_last_drawn', 
                                                                    'away_last_lost', 'away_last_gf', 'away_last_ga',
                                                                    'away_last_gd', 'away_last_points']].astype("Int64")


    # データを日付とfixture_idでソート (時系列順に並べるため)
    df = df.sort_values(by=['date', 'fixture_id']).reset_index(drop=True)    

    #targetカラム作成(試合の勝敗カラム)　H:home win, A:away win, D:draw
    def target_create(row):
        if row["home_score"] > row["away_score"]:
            return  "H"    
        
        elif row["home_score"] < row["away_score"]:
            return  "A"  
        
        else:
            return  "D"  
        
    
    df["target"] = df.apply(target_create,axis = 1)

    # objectからcategoryに変換したいカラム
    columns = ["home_team","away_team","status","target"]
    
    # タイプ変換関数
    def change_type(columns):
        for col in columns:
            df[col] = df[col].astype("category")
        return df
    
    df = change_type(columns) 

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # タイムゾーン情報を削除
    df['date'] = df['date'].dt.tz_localize(None)


    # ホーム勝利を示す一時的な列を作成
    # 'target'が 'H' の場合に1、それ以外は0
    df['is_home_win'] = (df['target'] == 'H').astype(int)
    # アウェイ勝利を示す一時的な列を作成
    # 'target'が 'Ａ' の場合に1、それ以外は0
    df['is_away_win'] = (df['target'] == 'A').astype(int)
    
    # 得失点差カラム作成
    df["home_goal_difference"] = df["home_score"] - df["away_score"]
    df["away_goal_difference"] = df["away_score"] - df["home_score"]


    # --------------------------------------------------------------------------------
    # 過去の試合結果に基づくローリング特徴量計算（.transform()で安全に置き換え）
    # --------------------------------------------------------------------------------

    # --- 直近N試合の勝利数/スコア/得失点差のローリング計算関数 ---
    def calculate_rolling_feature(df, group_col, target_col, window_size, new_col_name, agg_func='sum'):
        """
        グループごとにローリング集計を行い、1つシフトした結果を新しいカラムとして追加する。
        group_col: グループ化するカラム ('home_team' or 'away_team')
        target_col: 集計対象のカラム ('is_home_win', 'home_score', 'home_goal_difference' など)
        window_size: ローリングウィンドウサイズ (直近N試合なら N+1)
        new_col_name: 新しい特徴量のカラム名
        agg_func: 集計関数 ('sum' or 'mean')
        """
        # ローリング計算。window_sizeは現在の行を含むため、直近N試合を見る場合は N+1
        # shift(1)で現在の試合結果を除外し、fillna(0)で試合前のNaNを埋める
        if agg_func == 'sum':
            new_feature = df.groupby(group_col, observed=False)[target_col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).sum().shift(1).fillna(0)
            ).astype(int)
        elif agg_func == 'mean':
            # 勝率の場合、結果をパーセンテージにし、小数点第2位まで丸める
            new_feature = df.groupby(group_col, observed=False)[target_col].transform(
                lambda x: (x.rolling(window=window_size, min_periods=1).mean().shift(1) * 100).round(2).fillna(0)
            )
        else:
            raise ValueError("Unsupported agg_func")
            
        df[new_col_name] = new_feature
        return df

    # 直近5試合 (window=6) の計算
    df = calculate_rolling_feature(df, 'home_team', 'is_home_win', 6, 'home_team_recent_5_wins')
    df = calculate_rolling_feature(df, 'away_team', 'is_away_win', 6, 'away_team_recent_5_wins')
    
    df = calculate_rolling_feature(df, 'home_team', 'home_score', 6, 'home_recent_5_scores')
    df = calculate_rolling_feature(df, 'away_team', 'away_score', 6, 'away_recent_5_scores')
    
    df = calculate_rolling_feature(df, 'home_team', 'away_score', 6, 'home_recent_5_goal_against')
    df = calculate_rolling_feature(df, 'away_team', 'home_score', 6, 'away_recent_5_goal_against')
    
    df = calculate_rolling_feature(df, 'home_team', 'home_goal_difference', 6, 'home_recent_5_goal_diff')
    df = calculate_rolling_feature(df, 'away_team', 'away_goal_difference', 6, 'away_recent_5_goal_diff')


    # 直近10試合 (window=11) の計算
    df = calculate_rolling_feature(df, 'home_team', 'home_score', 11, 'home_recent_10_scores')
    df = calculate_rolling_feature(df, 'away_team', 'away_score', 11, 'away_recent_10_scores')
    
    df = calculate_rolling_feature(df, 'home_team', 'away_score', 11, 'home_recent_10_goal_against')
    df = calculate_rolling_feature(df, 'away_team', 'home_score', 11, 'away_recent_10_goal_against')
    
    df = calculate_rolling_feature(df, 'home_team', 'home_goal_difference', 11, 'home_recent_10_goal_diff')
    df = calculate_rolling_feature(df, 'away_team', 'away_goal_difference', 11, 'away_recent_10_goal_diff')

    
    # 直近20試合 (window=21) の計算
    df = calculate_rolling_feature(df, 'home_team', 'home_score', 21, 'home_recent_20_scores')
    df = calculate_rolling_feature(df, 'away_team', 'away_score', 21, 'away_recent_20_scores')
    
    df = calculate_rolling_feature(df, 'home_team', 'away_score', 21, 'home_recent_20_goal_against')
    df = calculate_rolling_feature(df, 'away_team', 'home_score', 21, 'away_recent_20_goal_against')
    
    df = calculate_rolling_feature(df, 'home_team', 'home_goal_difference', 21, 'home_recent_20_goal_diff')
    df = calculate_rolling_feature(df, 'away_team', 'away_goal_difference', 21, 'away_recent_20_goal_diff')


    #------------------勝ち点カラム作成 (ホーム/アウェイ区別なしの全体成績)--------------------
    # 
    # 各試合に一意のIDを付与 (後のマージのために利用)
    df['match_id'] = df.index 
    
    # 2. チーム視点でのデータ作成 (スタッキング)
    # -----------------------------------------------
    
    # ① ホームチーム視点のデータフレームを作成
    df_home = df[['match_id','season','date', 'home_team', 'target']].copy()
    df_home.rename(columns={'home_team': 'team'}, inplace=True)
    # 勝ち点計算
    df_home['points'] = df_home['target'].map({"H":3,"D":1,"A":0})
    # 勝利フラグ (勝率計算用)
    df_home['is_win'] = (df_home['target'] == 'H').astype(int)
    
    # # ② アウェイチーム視点のデータフレームを作成
    df_away = df[['match_id','season', 'date', 'away_team', 'target']].copy()
    df_away.rename(columns={'away_team': 'team'}, inplace=True)
    
    # 勝ち点計算
    df_away['points'] = df_away['target'].map({"H":0,"D":1,"A":3})
    
    # 勝利フラグ (勝率計算用)
    df_away['is_win'] = (df_away['target'] == 'A').astype(int)
    
    # # ③ 2つの視点のデータを結合し、時系列順にソート
    df_stacked = pd.concat([df_home, df_away],ignore_index=True)
    df_stacked = df_stacked.sort_values(by=['date', 'match_id']).reset_index(drop=True)


    # seasonとteamでグループ化し、各シーズン内での合計勝ち点 (試合前まで) を計算
    # transform を利用して累積勝ち点を計算
    df_stacked['total_points'] = df_stacked.groupby(["season","team"], observed=False)["points"].transform(
        # 勝ち点の累積和を計算し、1つシフト (現在の試合結果を除く)、NaNを0で埋める
        lambda x: x.rolling(window = 38,min_periods = 1).sum().shift(1).fillna(0)
    ).astype(int) 
    

    
    
    # ホームチームの総合成績フィーチャーを抽出 (match_id と team で紐づけ)
    df_home_points = df_stacked[['match_id','team','total_points']].copy()
    df_home_points.rename(columns={'team': 'home_team', 'total_points': 'home_total_points'}, inplace=True)
    
    # マージ
    df = pd.merge(
        df,
        df_home_points,
        on = ['match_id','home_team'],
        how = 'left'
    )
    # アウェイチームの総合成績フィーチャーを抽出
    df_away_points = df_stacked[['match_id','team','total_points']].copy()
    df_away_points.rename(columns={'team': 'away_team', 'total_points': 'away_total_points'}, inplace=True)
    
    # マージ
    df = pd.merge(
        df,
        df_away_points,
        on = ['match_id','away_team'],
        how = 'left'
    )
    
    df['home_total_points'] = df['home_total_points'].astype(int)
    df['away_total_points'] = df['away_total_points'].astype(int)
    
    
    # home teamとaway teamの勝ち点差カラムを作成
    df["points_difference"] = df['home_total_points'] - df['away_total_points']


    # --------------各チームの直近5試合での勝利数カラム作成(home、アウェイの区別なし)-----------------
    
    # 総合成績のローリング計算 (home, away区別なし)
    
    # チーム視点での直近5試合の勝率
    df_stacked['recent_5_wins_overall'] = df_stacked.groupby(['season','team'], observed=False)['is_win'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean().shift(1).fillna(0)
    ).astype(int)

    # チーム視点でのシーズン勝率
    df_stacked['season_wins_ave_overall_temp'] = df_stacked.groupby(['season','team'], observed=False)['is_win'].transform(
        lambda x: (x.rolling(window=38, min_periods=1).mean().shift(1) * 100).round(2).fillna(0)
    )

    
    # -------------------home teamへのマージ------------------------
    
    # ホームチームの総合成績フィーチャーを抽出
    df_home_feature = df_stacked[['match_id', 'team', 'recent_5_wins_overall', 'season_wins_ave_overall_temp']].copy()
    df_home_feature.rename(columns={
        'team': 'home_team',
        'recent_5_wins_overall': 'home_team_recent_5_wins_overall',
        'season_wins_ave_overall_temp': 'home_season_wins_ave_overall'
    }, inplace=True)
    
    # 元のデータフレームにマージ
    df = pd.merge(
        df, 
        df_home_feature, 
        on=['match_id', 'home_team'], 
        how='left'
    )
    
    
    # ------------------away teamへのマージ--------------------
    
    # アウェイチームの総合成績フィーチャーを抽出
    df_away_feature = df_stacked[['match_id', 'team', 'recent_5_wins_overall', 'season_wins_ave_overall_temp']].copy()
    df_away_feature.rename(columns={
        'team': 'away_team',
        'recent_5_wins_overall': 'away_team_recent_5_wins_overall',
        'season_wins_ave_overall_temp': 'away_season_wins_ave_overall'
    }, inplace=True)
    
    # 元のデータフレームにマージ
    df = pd.merge(
        df, 
        df_away_feature, 
        on=['match_id', 'away_team'], 
        how='left'
    )


    # ------------------ NS (Not Started) 試合の欠損値補完 ------------------
    # 2025年シーズンのまだ行われていない試合(status="NS")に対し、直前の "FT" (Full Time) の試合結果でデータを補完する。
    
    fill_features_home = ['home_team_recent_5_wins',
                          'home_recent_5_scores', 
                          'home_recent_5_goal_diff', 
                          'home_recent_5_goal_against', 
                          'home_recent_10_scores', 
                          'home_recent_10_goal_diff', 
                          'home_recent_10_goal_against', 
                          'home_recent_20_scores', 
                          'home_recent_20_goal_diff', 
                          'home_recent_20_goal_against']
    
    
    fill_features_away = ['away_team_recent_5_wins',
                          'away_recent_5_scores',
                          'away_recent_5_goal_diff',
                          'away_recent_5_goal_against',
                          'away_recent_10_scores',
                          'away_recent_10_goal_diff',
                          'away_recent_10_goal_against',
                          'away_recent_20_scores',
                          'away_recent_20_goal_diff',
                          'away_recent_20_goal_against']
    
    fill_features_home_overall = ['home_total_points','home_team_recent_5_wins_overall','home_season_wins_ave_overall']
    
    fill_features_away_overall = ['away_total_points','away_team_recent_5_wins_overall','away_season_wins_ave_overall']

    # 2025年シーズンのチームリストを取得
    teams = df[df["season"] == 2025]["home_team"].unique()

    # ホーム限定データ (homeでの試合のみでカウントしているデータ) の補充
    for team in teams:
        # 置き換え元（FTのそのチームの最終ホーム試合）の fill_features を取得
        source_df = df[(df["status"] == "FT") & (df["home_team"] == team)].sort_values("date")
        if not source_df.empty:
            source_vals = source_df.iloc[-1][fill_features_home]
        
            # 置き換え先（NSのそのチームのホーム試合）に代入
            df.loc[(df["status"] == "NS") & (df["home_team"] == team), fill_features_home] = source_vals.values
        
    # アウェイ限定データ (awayでの試合のみでカウントしているデータ) の補充
    for team in teams:
        # 置き換え元（FTのそのチームの最終アウェイ試合）の fill_features を取得
        source_df = df[(df["status"] == "FT") & (df["away_team"] == team)].sort_values("date")
        if not source_df.empty:
            source_vals = source_df.iloc[-1][fill_features_away]
        
            # 置き換え先（NSのそのチームのアウェイ試合）に代入
            df.loc[(df["status"] == "NS") & (df["away_team"] == team), fill_features_away] = source_vals.values
        
    # home,awayの区別なしでカウントしているデータ (overall) の補充
    for team in teams:
        # 1. 最新試合を日付順で取得
        source_df = df[(df["status"] == "FT") & ((df["home_team"] == team) | (df["away_team"] == team))].sort_values("date")
        if source_df.empty:
            continue
        
        last_row = source_df.iloc[-1]
        
        # 2. 対象チームが最新試合でホームだったかアウェイだったかを判別
        team_was_home = last_row["home_team"] == team
        
        # 3. 最新試合の行から、対象チーム自身の統計値のみを1セット取得
        if team_was_home:
            # チームがホームの場合、統計値は home_overall カラムにある
            latest_team_vals = last_row[fill_features_home_overall].values
        else:
            # チームがアウェイの場合、統計値は away_overall カラムにある
            latest_team_vals = last_row[fill_features_away_overall].values
            
        # 4. NS の行ごとに index を取得
        ns_home_idx = df[(df["status"] == "NS") & (df["home_team"] == team)].index
        ns_away_idx = df[(df["status"] == "NS") & (df["away_team"] == team)].index
        
        # 5. ホームチームとして NS の行に、対象チームの統計値で代入
        if len(ns_home_idx) > 0:
            # 未来のホーム試合の「ホーム側」カラムに、対象チームの統計値を埋める
            df.loc[ns_home_idx, fill_features_home_overall] = [latest_team_vals] * len(ns_home_idx)
        
        # 6. アウェイチームとして NS の行に、対象チームの統計値で代入
        if len(ns_away_idx) > 0:
            # 未来のアウェイ試合の「アウェイ側」カラムに、対象チームの統計値を埋める
            df.loc[ns_away_idx, fill_features_away_overall] = [latest_team_vals] * len(ns_away_idx)


    # 一時的なカラムを削除
    df.drop(columns=['home_goal_difference', 'away_goal_difference', 'is_home_win', 'is_away_win', 'match_id'], inplace=True, errors='ignore')

    # statusがFTの試合を除外して学習データを作成
    train_df = df[df["status"] == "FT"].copy().reset_index(drop=True)
    # 予測対象データ（statusがNSの試合）を抽出
    predict_df = df[df["status"] == "NS"].copy().reset_index(drop=True)
    
    return train_df, predict_df

# --------------------------------------------------------------------------------
# 評価関数 (evaluate_model)
# --------------------------------------------------------------------------------
def evaluate_model(model, X, y):
    """モデルを評価し、各種メトリクスを計算する"""
    y_pred_proba = model.predict_proba(X)
    pred_idx = np.argmax(y_pred_proba, axis=1)
    
    # 評価は factorize された数値ラベル (0, 1, 2) に対して行う
    y_pred = pred_idx 
    
    # ロス計算 (ラベルが factorize された数値であることを前提)
    try:
        # yのユニークな値の順序をlabelsとして指定
        unique_y = np.unique(y)
        ll = log_loss(y, y_pred_proba, labels=unique_y) 
    except ValueError:
        ll = np.nan
        
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")
    
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    
    return acc, ll, f1_macro, f1_weighted, report, y_pred, y_pred_proba




# --------------------------------------------------------------------------------
# 動的folds生成関数 
# --------------------------------------------------------------------------------
def generate_dynamic_folds(end_date_str, n_folds=3, val_period_days=30, gap_days=10):
    """
    現在の実行日を基準に、バックテスティング用のfoldsを動的に生成する
    """
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    folds = []
    
    for i in range(n_folds):
        val_end = end_date - dt.timedelta(days=i * gap_days)
        val_start = val_end - dt.timedelta(days=val_period_days)
        train_end = val_start - dt.timedelta(days=1)
        
        folds.append({
            "train_end": train_end.strftime('%Y-%m-%d'),
            "val_start": val_start.strftime('%Y-%m-%d'),
            "val_end": val_end.strftime('%Y-%m-%d'),
        })
        
    print("動的に生成されたfolds:")
    for fold in folds:
        print(f"  Train End: {fold['train_end']}, Val Period: {fold['val_start']} ~ {fold['val_end']}")
        
    return folds

#訓練データと検証データのindex作成
# foldsの中から、今回設定した範囲を取り出し、その範囲に入っているかどうかを判断し、その範囲内のデータのみを訓練データと検証データとしていく。
#これを3周する


def train_lgb(original_df,
              input_x,
              input_y,
              folds,
              params=params
              ):

    
    #評価値を入れる変数の作成
    metrics_val = [] #検証データ用
    

    # 'H', 'D', 'A' のラベルを数値 (0, 1, 2) に変換
    input_y_factorized, target_labels = pd.factorize(input_y)
    
    print(f"ターゲットラベルの順序: {target_labels}")

    for i, fold in enumerate(folds):
        nfold = i
        print("-" * 10, f"CV Fold {nfold}: Train End={fold['train_end']}, Val Start={fold['val_start']}", "-" * 10)
        
        # 学習用インデックス
        train_idx = original_df["date"] <= fold["train_end"]
        x_tr = input_x[train_idx]
        y_tr = input_y_factorized[train_idx] # 数値ラベルを使用
        
        # 検証用インデックス
        val_idx = (original_df["date"] >= fold["val_start"]) & (original_df["date"] <= fold["val_end"])
        x_val = input_x[val_idx]
        y_val = input_y_factorized[val_idx] # 数値ラベルを使用

        if len(x_val) == 0:
            print(f"警告: Fold {nfold} の検証データがありません。スキップします。")
            continue

        # LightGBM モデル
        model = lgb.LGBMClassifier(**params)
        
        
        # モデルの訓練
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
            early_stopping(stopping_rounds=50,verbose=False)  # 早期停止
            ]
            )
                
        
        # モデルの評価
        acc_val, ll_val, f1_macro_val, f1_weighted_val, _, _, _ = evaluate_model(model, x_val, y_val)
        
        print(f"Fold {nfold} ACC: {acc_val:.4f}, F1(weighted): {f1_weighted_val:.4f}")

        # 検証スコアを格納
        metrics_val.append({
            "nfold": nfold,
            "accuracy": acc_val,
            "log_loss": ll_val,
            "f1_weighted": f1_weighted_val,
        })

    # CV全体の平均メトリクスを計算
    df_metrics_val = pd.DataFrame(metrics_val)
    mean_accuracy = df_metrics_val['accuracy'].mean()
    mean_f1 = df_metrics_val['f1_weighted'].mean()
    
    print("-" * 10, "CV平均結果 (KPI)", "-" * 10)
    print(f"CV平均精度: {mean_accuracy:.4f}")
    print(f"CV平均F1 (Weighted): {mean_f1:.4f}")
    
    # KPI情報のみを返す
    return mean_accuracy, mean_f1, target_labels

# --------------------------------------------------------------------------------
# ★★★ NEW: 最終モデル学習関数 (全データ学習) ★★★
# --------------------------------------------------------------------------------
def train_final_model(X_all, y_all_factorized):
    """
    全ての学習データを使って最終予測モデルを訓練し、保存する
    """
    # CVと同じパラメータで設定
    model = lgb.LGBMClassifier(**params)
    
    print("-" * 10, "最終モデル学習 (全データ)", "-" * 10)
    # 全データでモデルを訓練 (検証セットなしで早期停止は行わない)
    # n_estimators は params で指定された1000回を使用
    model.fit(X_all, y_all_factorized)
    
    # モデルの保存
    final_model_path = os.path.join(MODEL_DIR, "final_model.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(final_model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"最終モデルを {final_model_path} に保存しました。")
    return final_model_path


# --------------------------------------------------------------------------------
# 予測実行とDB保存関数 
# --------------------------------------------------------------------------------
def predict_and_save(model_path, predict_df, target_labels):
    """
    最終モデルを使用して予測を実行し、結果をDBに保存する
    """
    if predict_df.empty:
        print("予測対象の試合データがありません。")
        return pd.DataFrame() 

    # モデルのロード
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"エラー: 最終モデルファイル {model_path} が見つかりません。")
        return pd.DataFrame()


    # 予測の実行
    X_predict = predict_df[FEATURES]
    # モデルにカテゴリ特徴量を与えるために、型を合わせる
    # X_predict["home_team"] = X_predict["home_team"].astype('category')
    # X_predict["away_team"] = X_predict["away_team"].astype('category')

    y_pred_proba = model.predict_proba(X_predict)
    
    # 予測結果（確率が最大のクラス）
    predicted_classes_idx = np.argmax(y_pred_proba, axis=1)
    
    # クラスラベルに再変換 (factorizeで得られたラベル順序を使用)
    predicted_result = target_labels[predicted_classes_idx]
    
    # 確率をDataFrameに追加 (ラベル順序に基づいて列名を割り当てる)
    proba_columns = {
        target_labels[i]: y_pred_proba[:, i] for i in range(len(target_labels))
    }
    proba_df = pd.DataFrame(proba_columns, index=predict_df.index)
    proba_df = proba_df.rename(columns={
        'H': 'proba_H', 'D': 'proba_D', 'A': 'proba_A'
    })
    
    predict_df = pd.concat([predict_df, proba_df], axis=1)

    # 予測結果を追加
    predict_df['predicted_result'] = predicted_result
    
    # 予測実行日時を追加
    predict_df['prediction_time'] = dt.datetime.now()

    # 予測結果を保存する DataFrame を整形
    df_results = predict_df[[\
        'fixture_id', 'date', 'home_team', 'away_team', \
        'predicted_result', 'proba_H', 'proba_D', 'proba_A', 'prediction_time'\
    ]]
    
    conn = sqlite3.connect(DB_PATH)
    df_results.to_sql('predictions', conn, if_exists='replace', index=False)
    conn.close()
    
    print("予測の実行とDBへの保存が完了しました。")
    return df_results 


# --------------------------------------------------------------------------------
# メイン処理 (CVと全データ学習を分離)
# --------------------------------------------------------------------------------
def main():
    try:
        # 1. データ取得
        conn = sqlite3.connect(DB_PATH)
        matches_df = pd.read_sql_query("SELECT * FROM matches", conn)
        stats_df = pd.read_sql_query("SELECT * FROM match_statistics", conn)
        conn.close()
        
        
        # 2. 特徴量エンジニアリング
        train_df, predict_df = feature_engineering(matches_df, stats_df)
        
        # 3. 学習用データの準備
        x_all = train_df[FEATURES]
        y_all = train_df[TARGET]
        
        # 4. 動的foldsの生成
        # 最新の結果が出ている試合の日付を基準にする
        latest_match_date = train_df['date'].max()
        folds = generate_dynamic_folds(
            end_date_str=latest_match_date.strftime('%Y-%m-%d'), 
            n_folds=3, 
            val_period_days=30,
            gap_days=10
        )
        
        # 5. モデル学習と評価 (CV) -> KPI算出のみ
        mean_accuracy, mean_f1, target_labels = train_lgb(
            original_df=train_df,
            input_x=x_all,
            input_y=y_all,
            folds=folds,
            params=params
        )

        # 6. 最終予測モデルを全データで学習し、保存
        y_all_factorized, _ = pd.factorize(y_all) # mainでもfactorizeが必要
        final_model_path = train_final_model(x_all, y_all_factorized)
        
        # 7. 予測の実行とDB保存
        # CVで算出したKPIではなく、全データで学習した final_model を使用
        df_results = predict_and_save(final_model_path, predict_df, target_labels)
        
        
        # 8. Streamlit アプリケーション向けに結果をJSONとして保存 (KPIはCV平均を使用)
        kpi_data = {
            "accuracy": f"{mean_accuracy * 100:.1f}%",
            "f1": f"{mean_f1:.2f}",
            "matches": len(train_df),
            "lastUpdate": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        }

        # 予測結果DataFrameを整形し、JSON形式に変換
        df_for_json = df_results[[
            'date', 'home_team', 'away_team', 'predicted_result', 'proba_H', 'proba_D', 'proba_A'
        ]].copy()

        # 'date' カラムを 'YYYY-MM-DD' 形式の文字列に変換します。
        df_for_json['date'] = df_for_json['date'].dt.strftime('%Y-%m-%d')
        
        # 信頼度(confidence)は、予測された結果の最大確率を使用
        df_for_json['confidence'] = df_for_json[['proba_H', 'proba_D', 'proba_A']].max(axis=1)
        
        # 最終的なデータ構造
        output_data = {
            "kpis": kpi_data,
            "predictions": df_for_json.rename(columns={'predicted_result': 'prediction'}).to_dict(orient='records')
        }

        # JSONファイルとして保存
        JSON_FILE_PATH = os.path.join(SCRIPT_DIR,"..","data", 'latest_predictions.json')
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"Streamlit向け予測結果とKPIを {JSON_FILE_PATH} に保存しました。")

    except Exception as e:
        print(f"メイン処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()