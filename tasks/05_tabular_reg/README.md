# 05 表形式の回帰

## 目的

回帰タスクで, ReduceLROnPlateau と, `.` を含む名前で自作の Optimizer を読み込む機能を確かめる.
RMSE のように, バッチの平均をとると値が変わってしまう指標の扱いも確かめる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | California Housing (`sklearn.datasets.fetch_california_housing`) |
| 取得方法 | scikit-learn が自動でダウンロード (`data/sklearn/` に保存, 約 400 KB) |
| 規模 | 20,640 サンプル, 特徴量 8 (所得の中央値, 築年数, 部屋数, 緯度経度など) |
| 目的変数 | 地区の住宅価格の中央値 (単位: 10 万ドル) |
| 分割 | 学習用 80 % と検証用 20 % |
| 前処理 | 特徴量は `StandardScaler`, 目的変数も標準化して学習し, 指標は元の単位に戻して求める |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- 3 層の MLP (`Linear → ReLU` を 2 回と出力層), 出力は 1 次元
- 自作の Optimizer として, Lion (符号だけを使う Optimizer) を `custom_optim.py` に `optim.Optimizer` の子クラスで実装し, config で `optimizer.name: custom_optim.Lion` と指定する

## step と Trainer の設計

- `batch` は `(x, y)` のタプル
- `nn.MSELoss()` で損失を求める
- MAE はバッチの平均から求めても正しいので `step` で返す
- RMSE はバッチごとの RMSE の平均が全体の RMSE と一致しないので, `Trainer` を継承し `on_epoch_end` で二乗誤差の合計から求める. `step` で返した場合との違いも比べる

## config

```yaml
# default
experiment:
  name: california
optimizer:
  name: Adam
  params:
    lr: 0.001
scheduler:
  name: ReduceLROnPlateau
  params:
    factor: 0.5
    patience: 5
trainer:
  epochs: 200
  early_stopping:
    use: true
    patience: 20
data:
  valid_size: 0.2
  batch_size: 128
model:
  hidden_dim: 128
```

experiment で次の差分を試す.

1. `optimizer.name: custom_optim.Lion`, `params: {lr: 0.0001, weight_decay: 0.01}`
2. `scheduler.params.mode: max` (`trainer.mode: min` と食い違うのでエラーになることの確認)
3. `scheduler.interval: step` (ReduceLROnPlateau ではエラーになることの確認)

## 確かめる ML-Found の機能

- [ ] ReduceLROnPlateau が `valid/loss` を見て学習率を下げ, `lr_curve.png` に階段状に出る
- [ ] ReduceLROnPlateau の `mode` を省略すると `trainer.mode` に合わせられる
- [ ] `mode` や `interval` が食い違うと分かりやすいエラーになる
- [ ] `custom_optim.Lion` のように `.` を含む名前で自作の Optimizer を読み込める (train.py のあるディレクトリが `sys.path` に入っていれば読める想定)
- [ ] `on_epoch_end` で求めた RMSE と, `step` で求めた RMSE の違い

## 期待する結果

- 検証用データの RMSE が 0.55 (5.5 万ドル) 以下
- Lion でも Adam と同程度まで下がる

## 予想される問題

- 指標の平均の扱い (RMSE, R² など) は利用者が気付きにくいので, ドキュメントでの注意か, 合計と件数を返す仕組みがあると良さそう
- `train.py` を別のディレクトリから実行すると, `custom_optim` が `sys.path` に無くて読み込めない可能性がある

## 実行方法

```bash
python tasks/05_tabular_reg/train.py
```

## 結果のメモ

(実行後に記入)
