# 07 時系列予測

## 目的

実データの時系列で, 窓で切り出した系列データを LSTM で学習し, チェックポイントからの学習の再開を確かめる.
時系列らしく, 未来のデータが学習に漏れない分割にする.

## データ

過去の気象データから, 東京の 24 時間先までの気温を予測する.

| 項目 | 内容 |
| --- | --- |
| データセット | 気象庁の東京の 1 時間ごとの観測データ |
| 取得方法 | **用意してもらう** (`data/jma/tokyo_hourly.csv` に置く) |
| 入力 | 過去 72 時間分の気象要素 (下記) と, 時刻と季節を表す特徴量 (`sin` と `cos` で表した時刻と日付) |
| 出力 | 次の 24 時間の気温 |
| 分割 | 時間順に分け, 最初の 10 か月を学習用, 最後の 2 か月を検証用. 境界をまたぐ窓は使わない |
| 前処理 | 学習用データの平均と標準偏差で標準化, 欠損は前の値で埋める |

### 用意してほしいデータ

[気象庁の過去の気象データ・ダウンロード](https://www.data.jma.go.jp/risk/obsdl/) から, 次の条件で取得したもの.

| 項目 | 条件 |
| --- | --- |
| 地点 | 東京 (東京都) |
| 期間 | 直近 1 年分, できれば 2 から 3 年分 (季節をまたいだ学習と検証のため) |
| 間隔 | **時別値** (1 時間ごと). 1 年で 8,760 行 |
| 要素 | 気温, 降水量, 相対湿度, 現地気圧または海面気圧, 風速, 風向, 日照時間 |
| 形式 | ダウンロードした CSV のままでよい (Shift_JIS, ヘッダが複数行, 品質情報と均質番号の列を含む形) |

日別値しか用意できない場合は, 1 年で 365 行と少ないので, 過去 14 日から翌日の最高気温と最低気温を予測する形に変える.
その場合は 5 年分以上あると良い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `nn.LSTM` (2 層, 隠れ層 64, `batch_first=True`) の最後の時刻の出力から `nn.Linear` で 24 時間分を一度に出力
- 比較用に, OddNN の `models/rnn_base.py` の自作 LSTM (`LSTMCell`, `LSTMLayer`, `LSTM`) も試す. 双方向と残差接続を config の `model.bidirectional` と `model.residual` で切り替えられる
    - 自作 LSTM は入力が `(系列長, バッチ, 特徴量)` の順なので, `step` かモデルの中で転置する
    - Python のループで時刻をたどるので, `nn.LSTM` より大幅に遅いことが予想される. 速度の差も記録する
- 比較用に, 「24 時間前と同じ気温」とする単純な予測 (persistence) の誤差も求めて README に記録する

## step の設計

- `batch` は `(x, y)` のタプル, `x` は `(B, 72, 特徴量の数)`, `y` は `(B, 24)`
- `nn.MSELoss()` で損失を求める (標準化した空間)
- MAE は元の単位 (°C) に戻して `mae_c` として返す

## config

```yaml
# default
experiment:
  name: tokyo_temp
optimizer:
  name: Adam
  params:
    lr: 0.001
scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 60   # 再開して 60 エポックまで学習するため epochs より大きくする
trainer:
  epochs: 30
  grad_clip_norm: 1.0
data:
  path: ../../data/jma/tokyo_hourly.csv
  input_hours: 72
  output_hours: 24
  valid_months: 2
  batch_size: 128
model:
  hidden_dim: 64
  num_layers: 2
```

学習の再開を次の手順で確かめる.

1. default のまま `epochs: 30` で学習する
2. experiment に `trainer.epochs: 60` と `trainer.resume_from: outputs/tokyo_temp/<日付>/<時刻>/last.pt` を書いて再開する
3. 比較として, 最初から `epochs: 60` で学習し, 2 の結果と `history.csv` が一致するかを見る

## 確かめる ML-Found の機能

- [ ] `resume_from` で続きのエポックから学習が再開される
- [ ] 再開後の `history.csv` と学習曲線に, 再開前のエポックも含まれる
- [ ] Scheduler (`CosineAnnealingLR`) の状態が引き継がれ, 学習率が連続している
- [ ] 乱数の状態が戻り, 通しで学習した場合と同じ結果になる (DataLoader の shuffle を含む)
- [ ] 再開時に experiment で `trainer.epochs` を変えたことが `diff.yaml` に残る
- [ ] 再開元とは別のディレクトリに出力され, 再開元の `best.pt` がコピーされる

## 期待する結果

- 24 時間先までの気温の MAE が 1.5 から 2.0 °C 程度で, persistence の予測より良い
- 再開した学習と通しの学習で, 指標がほぼ (CPU なら完全に) 一致する

## 予想される問題

- DataLoader の shuffle に使う乱数は, エポックの途中ではなく先頭で決まるので, 再開しても一致するはず. `num_workers > 0` の場合は一致しない可能性がある
- MPS では乱数の状態の保存と復元が環境によって不完全な可能性がある
- 再開時の `last.pt` の相対パスがプロジェクトのディレクトリ (このタスクのディレクトリ) から辿られるので, `outputs/...` と書く

## 実行方法

```bash
python tasks/07_weather_forecast/train.py
```

## 結果のメモ

(実行後に記入)
