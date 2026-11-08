# 01 表形式の多クラス分類

## 目的

最も単純なタスクで, ML-Found の基本の流れが README の通りに動くことを確かめる.
以降のタスクの基準になるので, ML-Found の既定値をなるべくそのまま使う.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | Iris (`sklearn.datasets.load_iris`) |
| 取得方法 | scikit-learn に同梱, ダウンロード不要 |
| 規模 | 150 サンプル, 特徴量 4 つ (がく片と花弁の長さと幅), 3 クラス (各 50) |
| 分割 | 学習用 80 % と検証用 20 %, クラスの比率を保って分割 (`stratify`) |
| 前処理 | 学習用データで `StandardScaler` を当てはめ, 両方に適用 |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `nn.Linear(4, 16) → nn.ReLU() → nn.Linear(16, 3)` の 2 層の MLP
- 隠れ層の次元数は config の `model.hidden_dim` で変えられるようにする

## step の設計

- `batch` は `(x, y)` のタプル (`TensorDataset` + `DataLoader`)
- `nn.CrossEntropyLoss()` で損失を求め, `{"loss": loss, "acc": acc}` を返す

## config

```yaml
# default
experiment:
  name: iris
optimizer:
  name: Adam
  params:
    lr: 0.01
trainer:
  epochs: 200
  monitor: valid/acc
  mode: max
  early_stopping:
    use: true
    patience: 20
data:
  valid_size: 0.2
  batch_size: 16
model:
  hidden_dim: 16
```

experiment で次の差分を順に試す.

1. 差分なし (空のファイル)
2. `optimizer.params.lr: 0.001` と `model.hidden_dim: 32`
3. `optimizer.name: SGD` に変え, `params` に `lr` と `momentum` を書く (default の `params` が捨てられることの確認)
4. default に無い項目 (例: `model.dropout`) を書いてエラーになることの確認

## 確かめる ML-Found の機能

- [ ] `load_config` と `start_experiment` で `outputs/iris/<日付>/<時刻>/` が作られる
- [ ] `config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線, `best.pt`, `last.pt`, `final_model.pt` がそろう
- [ ] `diff.yaml` と `train.log` に experiment の差分が正しく出る
- [ ] `monitor: valid/acc` と `mode: max` でベストモデルが選ばれる
- [ ] Early Stopping で止まり, `trainer.model` がベストモデルの重みに戻っている
- [ ] クラス名を変えると default の `params` が捨てられる
- [ ] default に無い項目を experiment に書くとエラーになる
- [ ] `git_diff.patch` が未コミットの変更がある場合だけ作られる

## 期待する結果

- 検証用データの正解率が 0.93 以上
- 数秒で学習が終わる

## 予想される問題

特に無い. ML-Found の `examples/minibatch` (wine) とほぼ同じ形のため.

## 実行方法

```bash
python tasks/01_tabular_cls/train.py
```

## 結果のメモ

(実行後に記入)
