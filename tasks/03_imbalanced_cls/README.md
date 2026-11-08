# 03 不均衡な二値分類

## 目的

正解率では評価できない不均衡なデータで, バッチの平均では求められない指標 (ROC-AUC, PR-AUC) を `on_epoch_end` で記録できるかを確かめる.
`Trainer` を継承する使い方の確認がこのタスクの中心になる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成データ (`sklearn.datasets.make_classification`) |
| 取得方法 | その場で生成, ダウンロード不要 |
| 規模 | 20,000 サンプル, 特徴量 20 (有効 8, 冗長 4), 正例の割合 5 % |
| 分割 | 学習用 80 % と検証用 20 %, `stratify` で正例の割合を保つ |
| 前処理 | `StandardScaler` |

正例の割合やサンプル数は config の `data` で変えられるようにする.
実データで試したい場合は, クレジットカードの不正検知 (Kaggle の Credit Card Fraud Detection, 正例 0.17 %) に差し替えられる.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- 3 層の MLP (`Linear → ReLU → Dropout` を 2 回と出力層), 出力は 1 次元のロジット
- `nn.BCEWithLogitsLoss(pos_weight=...)` で正例に重みを付ける

## step と Trainer の設計

`Trainer` を継承して次のメソッドを上書きする.

- `on_epoch_start(phase)`: 予測確率と正解ラベルを集めるリストを用意
- `step(batch, phase)`: 損失を求め, 予測確率を `detach().cpu()` してリストに追加, `{"loss": loss}` を返す
- `on_epoch_end(phase)`: 集めた予測から `roc_auc_score` と `average_precision_score` を求めて `{"auc": ..., "ap": ...}` を返す

あわせて, `batch_size` を戻り値に入れる書き方も確かめる.

## config

```yaml
# default
experiment:
  name: imbalanced
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 50
  monitor: valid/ap
  mode: max
  early_stopping:
    use: true
    patience: 10
data:
  n_samples: 20000
  pos_ratio: 0.05
  valid_size: 0.2
  batch_size: 256
model:
  hidden_dim: 64
  dropout: 0.2
loss:
  use_pos_weight: true
```

experiment で次の差分を試す.

1. `loss.use_pos_weight: false` で AP がどう変わるか
2. `data.pos_ratio: 0.01`

## 確かめる ML-Found の機能

- [ ] `Trainer` を継承し, `step_fn` を渡さずに `step` を上書きできる
- [ ] `on_epoch_end` の戻り値が `train/auc`, `valid/auc` として記録され, 学習曲線も描かれる
- [ ] `on_epoch_end` で返した指標 (`valid/ap`) を `monitor` にできる
- [ ] `self.epoch` を継承したクラスから参照できる

## 期待する結果

- 検証用データの ROC-AUC が 0.90 以上, AP が 0.6 前後

## 予想される問題

- `pos_weight` は学習用データの正例の割合から決まるので, config の値だけでは書けず, train.py で計算して損失関数に渡す形になる (ML-Found の問題ではない)
- 学習時の予測は Dropout が効いた状態なので `train/auc` は参考程度にしかならない. 学習用データを推論モードで評価し直す仕組みは ML-Found に無い

## 実行方法

```bash
python tasks/03_imbalanced_cls/train.py
```

## 結果のメモ

(実行後に記入)
