# 07 時系列予測

## 目的

実データの時系列で, 窓で切り出した系列データを LSTM で学習し, チェックポイントからの学習の再開を確かめる.
時系列らしく, 未来のデータが学習に漏れない分割にする.

## ディレクトリ構成

```
07_weather_forecast/
├── configs/
│   ├── default/train.yaml          # 基準の config (nn.LSTM, 30 エポック)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── full60.yaml         # 実験 3: 通しで 60 エポック
│           └── custom_lstm.yaml    # 実験 4: 自作の LSTM
├── prepare_data.py                 # 気象庁の CSV を整形して data/jma/tokyo_daily.csv を作る
├── model.py                        # LSTMCell, LSTMLayer, CustomLSTM (自作の LSTM), WeatherForecaster
├── train.py                        # データ, step, 学習の実行
└── outputs/tokyo_temp/<日付>/<時刻>/ # ML-Found が実験ごとに作る (git の管理外)
```

## データ

東京の過去 14 日分の気象から, 翌日の最高気温と最低気温を予測する.

| 項目 | 内容 |
| --- | --- |
| データセット | 気象庁の東京の **日別値**, 2024-11-01 から 2026-10-31 の 730 日 |
| 取得方法 | **用意してもらったもの**. 気象庁の「過去の気象データ・ダウンロード」の CSV を `data/jma/raw/weather.csv` に置き, `prepare_data.py` で `data/jma/tokyo_daily.csv` に整形する |
| 入力 | 過去 14 日分の気象要素 11 個 (平均, 最高, 最低気温, 日照時間, 平均, 最大風速, 蒸気圧, 平均, 最小湿度, 海面気圧, 雲量) と, 降水量の対数, 最多風向と季節を表す `sin` と `cos` の, 計 16 個 |
| 出力 | 翌日の最高気温と最低気温 |
| 分割 | 予測する日で分け, 2026-06-30 までを学習用 (593 組), 2026-07-01 以降を検証用 (123 組). 検証用の入力には 6 月以前の日も含むが, 予測する値は学習に使わない |
| 前処理 | 特徴量と予測する値を, 学習用の期間の日だけで求めた平均と標準偏差で標準化 |

最初に README で想定した時別値ではなく日別値のデータになったので, 「日別値しか用意できない場合」の案 (過去 14 日分から翌日の最高気温と最低気温を予測) にした.
2 年分しかないので, 検証用の 4 か月 (7 月から 10 月) と同じ季節は, 学習用には 2025 年の 1 年分しか無い.

### データの整形 (`prepare_data.py`)

ダウンロードした CSV は Shift_JIS で, 見出しが複数行あり, 各要素の後ろに品質情報と均質番号の列が付く (降水量, 日照時間, 最低海面気圧は現象なし情報も. 最大風速と最大瞬間風速の風向は品質情報だけ).
`prepare_data.py` は次のように整形する.

- 文字コードを UTF-8 にし, 見出しを英語の列名の 1 行にする (`date`, `temp_mean`, `temp_max`, ...)
- 品質情報, 均質番号, 現象なし情報の列を除き, 値の列だけを残す
- 日付を `2024-11-01` の形にする
- 16 方位の風向 (`wind_max_dir`, `gust_dir`, `wind_mode_dir`) を北から時計回りの角度 (0 から 337.5) にする
- 天気概況 (`weather_day`, `weather_night`) は文章のまま残す (学習には使わない)

整形の結果, 730 日, 20 要素で空の値は無かった. 品質情報は 14,579 個が「8」(正常), 21 個が「5」(準正常, 一部の欠測あり) だった. 品質情報は数えるだけで, 「5」の値もそのまま使う.

`prepare_data.py` は `--input` と `--output` でパスを変えられる (既定は `data/jma/raw/weather.csv` と `data/jma/tokyo_daily.csv`). 相対パスはタスクのディレクトリから辿る. 既定のまま実行すると `data/jma/tokyo_daily.csv` を上書きする.

### データの取り直し

`data/` は git の管理外なので, 新しく clone した場合は気象庁の「過去の気象データ・ダウンロード」(https://www.data.jma.go.jp/risk/obsdl/) から次の条件で取得し, `data/jma/raw/weather.csv` に置く.

- 地点: 東京 (東京都)
- 項目: 日別値. 平均, 最高, 最低気温, 降水量の日合計, 日照時間, 平均風速, 最大風速 (風向も), 最大瞬間風速 (風向も), 最多風向, 平均蒸気圧, 平均湿度, 最小相対湿度, 平均現地気圧, 平均海面気圧, 最低海面気圧, 平均雲量, 天気概況 (昼, 夜)
- 期間: 2024-11-01 から 2026-10-31 (この README の結果と同じにする場合)
- CSV の形式は既定のまま (品質情報と均質番号を含む)

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `WeatherForecaster`: LSTM (2 層, 隠れ層 64) の最後の日の出力から, `nn.Linear` で翌日の 2 つの値を出す. パラメータ数 54,402
- `model.lstm_type` で LSTM を切り替える
    - `torch` (既定): `nn.LSTM` (`batch_first=True`)
    - `custom`: 自作の LSTM (`LSTMCell`, `LSTMLayer`, `CustomLSTM`). 時刻を Python のループでたどる. 入力が `(系列長, バッチ, 特徴量)` の順なので転置して渡す
- 比較用に, 「翌日も今日と同じ気温」とする単純な予測 (persistence) の誤差も求めて `train.log` に記録する

自作の LSTM の設計.

- `LSTMCell` は入力ゲート, 忘却ゲート, セルの候補, 出力ゲートの重みを, 入力用と隠れ状態用の行列にそれぞれまとめて計算する. 重みは `nn.LSTM` と同じく ±1/√hidden_dim の一様分布で初期化し, バイアスは 0 にする
- 双方向と残差接続は持たない
- `CustomLSTM` は全ての層で共通の初期の隠れ状態 `h_0` (形は `(バッチ, 隠れ層)`, 省略すると 0) を受け取り, 最終層の各時刻の出力だけを返す

## step の設計

- `batch` は `(x, y)` のタプル, `x` は `(バッチ, 14, 16)`, `y` は `(バッチ, 2)` (標準化した値)
- `nn.MSELoss()` で損失を求め, ℃ に戻した平均絶対誤差 `mae_max` と `mae_min` も返す
- 標準偏差を使うので, `make_step(y_std)` で関数を作って `Trainer` に渡す

## config

```yaml
# default
experiment:
  name: tokyo_temp
  seed: 42
  device: cpu            # 小さな LSTM なので cpu
optimizer:
  name: Adam
  params:
    lr: 0.001
scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 60            # 再開して 60 エポックまで学習するため epochs より大きくする
trainer:
  epochs: 30
  grad_clip_norm: 1.0
data:
  path: ../../data/jma/tokyo_daily.csv
  valid_from: "2026-07-01"
  input_days: 14
  batch_size: 32
model:
  lstm_type: torch       # torch または custom
  hidden_dim: 64
  num_layers: 2
```

実験は次の通り.

1. 差分なし, 30 エポック
2. 実験 1 の `last.pt` から, `trainer.epochs: 60` と `trainer.resume_from` を書いて再開する
3. 比較として, 最初から 60 エポック学習する (`full60.yaml`). 実験 2 と `history.csv` が一致するかを見る
4. 自作の LSTM (`custom_lstm.yaml`)

## 確かめる ML-Found の機能

- [x] `resume_from` で続きのエポックから学習が再開される (`train.log` に「再開元: ... (エポック 30)」)
- [x] 再開後の `history.csv` と学習曲線に, 再開前のエポックも含まれる (60 行)
- [x] Scheduler (`CosineAnnealingLR`) の状態が引き継がれ, 学習率が連続している (エポック 30 の 0.000526 からエポック 31 の 0.000500 へ)
- [x] 乱数の状態が戻り, 通しで学習した場合と同じ結果になる (DataLoader の shuffle を含む) → 実験 2 と 3 の `history.csv` が完全に一致した
- [x] 再開時に experiment で `trainer.epochs` を変えたことが `diff.yaml` に残る (`trainer.resume_from` も残る)
- [x] 再開元とは別のディレクトリに出力され, 再開元の `best.pt` がコピーされる

## 期待する結果

- 最高気温と最低気温の MAE が, 前日と同じとする予測より良い
- 再開した学習と通しの学習で, 指標がほぼ (CPU なら完全に) 一致する

(最初の README の「24 時間先までの MAE が 1.5 から 2.0 ℃」は時別値の想定だったので, 日別値に合わせて変えた)

## 予想される問題

- DataLoader の shuffle に使う乱数はエポックの先頭で決まるので, 再開しても一致するはず → 一致した. `num_workers > 0` の場合は試していない
- MPS では乱数の状態の保存と復元が環境によって不完全な可能性がある → cpu で学習したので確かめていない
- 再開時の `last.pt` の相対パスがプロジェクトのディレクトリ (このタスクのディレクトリ) から辿られるので, `outputs/...` と書く → その通りだった

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
データを整形してから学習する. `train.py` と `prepare_data.py` はどこから実行してもよく, 学習の出力先は常に `tasks/07_weather_forecast/outputs/tokyo_temp/<日付>/<時刻>/` になる.

```bash
# リポジトリのルートから
python tasks/07_weather_forecast/prepare_data.py     # data/jma/raw/weather.csv から data/jma/tokyo_daily.csv を作る

python tasks/07_weather_forecast/train.py                                                      # 実験 1 (約 2 秒)
python tasks/07_weather_forecast/train.py --experiment configs/experiment/examples/full60.yaml   # 実験 3
python tasks/07_weather_forecast/train.py --experiment configs/experiment/examples/custom_lstm.yaml   # 実験 4
```

実験 2 (再開) は, 実験 1 の出力先を指す experiment の config を作って実行する.

```yaml
# 例: resume60.yaml (<日付>/<時刻> は実験 1 の保存先)
# このファイルはタスクのディレクトリか実行する場所に置く (再開元が毎回違うので examples/ には置かない)
trainer:
  epochs: 60
  # 相対パスは実行した場所ではなく tasks/07_weather_forecast/ から辿る. 絶対パスでもよい
  resume_from: outputs/tokyo_temp/<日付>/<時刻>/last.pt
```

```bash
python tasks/07_weather_forecast/train.py --experiment resume60.yaml
```

- `resume_from` の相対パスは, ML-Found の規則でタスクのディレクトリから辿る. リポジトリのルートからのパス (`tasks/07_weather_forecast/outputs/...`) を書くとパスが二重になり, 出力先を作る前に `resume_from` が無いというエラーになる (ML-Found `3aca743` では torch の `FileNotFoundError` で止まり, 学習結果の無い出力先が残った)
- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 実験 2 と 3 は, 実験 1 と同じく cpu で学習すると完全に一致する

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, macOS (Apple Silicon), Python 3.13, cpu で実行.

| 実験 | LSTM | エポック数 | ベストのエポック | 検証用 MAE 最高気温 | 検証用 MAE 最低気温 | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- |
| 前日と同じとする予測 | - | - | - | 2.500 ℃ | 1.254 ℃ | - |
| 1 差分なし | `nn.LSTM` | 30 | 28 | 2.292 ℃ | 1.208 ℃ | 1.7 秒 |
| 2 再開 (30 → 60) | `nn.LSTM` | 60 | 28 | 2.292 ℃ | 1.208 ℃ | 1.6 秒 (再開後の 30 エポック) |
| 3 `full60` | `nn.LSTM` | 60 | 28 | 2.292 ℃ | 1.208 ℃ | 3.3 秒 |
| 4 `custom_lstm` | 自作の LSTM | 30 | 30 | 2.323 ℃ | 1.225 ℃ | 2.1 秒 (他の学習と並行した再実行では 3.1 秒) |

- ベストのエポック (`valid/loss` で選んだ) では, 前日と同じとする予測より最高気温で 0.21 ℃, 最低気温で 0.05 ℃ 良かった. ただしテスト用データは無く, 同じ検証用データでエポックを選んだ値. 最高気温は 60 エポック中 49 エポックで前日と同じとする予測に勝ったが, 最低気温は 21 エポックだけで, 最後のエポック (30: 1.271 ℃, 60: 1.261 ℃) は負けていた. 学習用が 593 組と少なく, 最低気温の改善はほとんど無い
- 実験 2 と 3 の `history.csv` は完全に一致した. 再開でモデル, Optimizer, Scheduler, 記録, 乱数 (DataLoader の並べ替えを含む) が全て正しく戻っている. 60 エポックまで学習しても, ベストはエポック 28 のままだった
- 実験 4 の自作の LSTM は `nn.LSTM` とほぼ同じ精度で, 学習時間は約 1.2 倍だった. 系列が 14 日と短いので, Python のループで時刻をたどっても差は小さい

### ML-Found について気付いたこと

- 学習の再開は, 乱数の状態まで含めて正しく働いた. 新しい FB は無い
- 再開には `resume_from` に再開元のパスを書いた experiment の config が要る. 再開元は実行ごとに違うので, `examples/` に置いておけない. コマンドラインから `resume_from` を渡せると良い (01 の「experiment の config の切り替え」の FB と同じ)

### 最新の ML-Found での再実行 (2026-11-13)

上の結果は ML-Found `3aca743` で得たもの. 2026-11-13 に, 全ての実験をその後の ML-Found (`24f18c2`) で流し直し, 学習時間は `e2df46e` でも測り直した. 4 つの実験の `history.csv` は全て一致し, 実験 2 (再開) と 3 (通し) の `history.csv` が完全に一致することも確かめた. 学習時間もほぼ同じ (実験 1 は 1.60 秒).
