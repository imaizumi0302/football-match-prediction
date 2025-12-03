import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --------------------------------------------------------
# データの読み込み
# --------------------------------------------------------

# スクリプト自体のディレクトリパスを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルート（srcの1つ上）
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

#dataフォルダ内にあるJSONファイルのパス
JSON_FILE_PATH = os.path.join(PROJECT_ROOT,"data", "latest_predictions.json")


try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    kpis = data['kpis']
    df_predictions = pd.DataFrame(data['predictions'])
    
    # 確信度を小数点以下1桁のパーセンテージ表示に整形
    df_predictions['confidence_display'] = (df_predictions['confidence'] * 100).round(1).astype(str) + ' %'
    
    # 予測結果を分かりやすい日本語に変換
    result_map = {'H': 'ホーム勝', 'D': '引分け', 'A': 'アウェイ勝'}
    df_predictions['Prediction (日本語)'] = df_predictions['prediction'].map(result_map)

except FileNotFoundError:
    st.error("🚨 エラー: 予測データファイル `latest_predictions.json` が見つかりません。")
    st.error("先に `prediction_pipeline1.py` を実行して予測データを作成してください。")
    st.stop()
except json.JSONDecodeError:
    st.error("🚨 エラー: JSONファイルの読み込みに失敗しました。ファイルが壊れていないか確認してください。")
    st.stop()
    
# --------------------------------------------------------
# Streamlit アプリの構成
# --------------------------------------------------------

st.set_page_config(layout="wide")

st.title("⚽ サッカー試合結果予測ダッシュボード")
st.caption(f"最終パイプライン実行日時: {kpis['lastUpdate']}")

## 🏆 KPIサマリー
st.header("🏆 モデル性能指標 (CV平均)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CV平均精度", kpis['accuracy'])
with col2:
    st.metric("CV平均F1スコア", kpis['f1'])
with col3:
    st.metric("学習データ数", f"{kpis['matches']} 試合")
with col4:
    # 実際は予測対象の試合数を表示
    st.metric("予測対象試合数", f"{len(df_predictions)} 試合")


st.markdown("---")


## 🔮 サイドバーと予測テーブル
st.header("🔮 今後の試合の予測")

# サイドバーのフィルタ設定
st.sidebar.header("🔍 フィルターオプション")
all_teams = sorted(list(set(df_predictions['home_team'].unique()) | set(df_predictions['away_team'].unique())))
selected_team = st.sidebar.selectbox("チームで絞り込み:", ['全チーム'] + all_teams)
min_confidence = st.sidebar.slider("最小確信度 (%)", 0, 100, 50)


# データのフィルタリング
filtered_df = df_predictions.copy()

# チームフィルタ
if selected_team != '全チーム':
    filtered_df = filtered_df[
        (filtered_df['home_team'] == selected_team) | 
        (filtered_df['away_team'] == selected_team)
    ]

# 確信度フィルター
filtered_df = filtered_df[filtered_df['confidence'] * 100 >= min_confidence]


# 表示用DataFrameの整形
df_display = filtered_df[[
    'date',
    'home_team', 
    'away_team', 
    'Prediction (日本語)', 
    'confidence_display', 
    'proba_H', 
    'proba_D', 
    'proba_A',
]].rename(columns={
    'date': '日付',
    'home_team': 'ホーム',
    'away_team': 'アウェイ',
    'confidence_display': '確信度',
    'proba_H': 'H確率',
    'proba_D': 'D確率',
    'proba_A': 'A確率',
}).sort_values('確信度', ascending=False)


# Streamlitでのテーブル表示
if df_display.empty:
    st.warning("選択されたフィルター条件に一致する試合がありません。")
else:
    st.dataframe(df_display, use_container_width=True, hide_index=True)