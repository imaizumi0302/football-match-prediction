# サッカーマッチ予測パイプライン
## プロジェクト概要

このプロジェクトは、過去のサッカーデータを基に試合結果を予測するパイプラインです。
Python と機械学習を使用し、プレミアリーグの試合結果を予測するモデルを構築しています。
また、Streamlit による可視化や、JSON 形式での出力も可能です。

## 技術スタック

言語: Python 3.12

ライブラリ: pandas, numpy, scikit-learn, lightgbm, imbalanced-learn, Streamlit

データベース: SQLite

バージョン管理: Git, GitHub

## フォルダ・ファイル構成

soccer_pipeline/
├─ data/                     # データファイル格納
│  ├─ premier_league.csv
│  ├─ processed_data.csv
│  └─ latest_predictions.json
├─ db/                       # SQLite データベース
│  └─ matches.db
├─ logs/                     # 実行ログ
│  ├─ pipeline.log
│  └─ service_main.log
├─ models/                   # 学習済みモデル
│  ├─ final_model.pkl
│  └─ model_lgb_fold*.pickle
├─ notebooks/                # Jupyter Notebook
│  ├─ EDA_notebook.ipynb
│  ├─ processed_notebook.ipynb
│  └─ season_data_process.ipynb
├─ src/                      # スクリプト
│  ├─ app.py
│  ├─ data_fetcher2.py
│  └─ prediction_pipeline1.py
└─ .gitignore


## 実行手順
1. 必要なパッケージのインストール
```
pip install -r "requirements.txt"

```

2. データベース準備
```
db/matches.db が存在することを確認してください。
存在しない場合は、CSV データからインポートしてください。

```
3. スクリプト実行
```
python src/prediction_pipeline1.py

```

4. Jupyter Notebook 実行
```
jupyter notebook notebooks/processed_notebook.ipynb

```

## 出力結果

data/processed_data.csv : 前処理後のデータ

data/latest_predictions.json : Streamlit 用の予測結果 JSON

prediction : 予測結果 (H: ホーム勝利, D: 引き分け, A: アウェイ勝利)

confidence : 予測の信頼度

logs/ : 実行ログ

## 今後の改善点

モデルの精度向上（特徴量追加、ハイパーパラメータ調整）

Streamlit ダッシュボードの UI 改善

リアルタイムデータ取得への対応

## 注意点

DB_PATH や SEASON_DATA_PATH はスクリプト内で絶対パスを設定してください

GitHub にアップロードする際は .gitignore で不要ファイルを除外しています