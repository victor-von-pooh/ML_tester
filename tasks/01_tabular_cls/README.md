# 01 表形式の多クラス分類

## 目的

最も単純なタスクで, ML-Found の基本の流れが README の通りに動くことを確かめる.
以降のタスクの基準になるので, ML-Found の既定値をなるべくそのまま使う.

## ディレクトリ構成

```
01_tabular_cls/
├── configs/
│   ├── default/train.yaml          # 基準の config
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 4 の差分
│           ├── lr_hidden.yaml
│           ├── sgd.yaml
│           └── invalid_key.yaml
├── model.py                        # IrisClassifier
├── train.py                        # データ, step, 学習の実行
└── outputs/                        # ML-Found が作る (git の管理外)
```

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

experiment で次の差分を順に試す. 2 から 4 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `optimizer.params.lr: 0.001` と `model.hidden_dim: 32` (`lr_hidden.yaml`)
3. `optimizer.name: SGD` に変え, `params` に `lr` と `momentum` を書く. default の `params` が捨てられることの確認 (`sgd.yaml`)
4. default に無い項目 `model.dropout` を書いてエラーになることの確認 (`invalid_key.yaml`)

## 確かめる ML-Found の機能

- [x] `load_config` と `start_experiment` で `outputs/iris/<日付>/<時刻>/` が作られる
- [x] `config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線, `best.pt`, `last.pt`, `final_model.pt` がそろう
- [x] `diff.yaml` と `train.log` に experiment の差分が正しく出る
- [x] `monitor: valid/acc` と `mode: max` でベストモデルが選ばれる
- [x] Early Stopping で止まり, `trainer.model` がベストモデルの重みに戻っている
- [x] クラス名を変えると default の `params` が捨てられる
- [x] default に無い項目を experiment に書くとエラーになる
- [x] `git_diff.patch` が未コミットの変更がある場合だけ作られる

## 期待する結果

- 検証用データの正解率が 0.93 以上
- 数秒で学習が終わる

## 予想される問題

特に無い. ML-Found の `examples/minibatch` (wine) とほぼ同じ形のため.

## 実行方法

リポジトリのルートで実行する. 出力先は `tasks/01_tabular_cls/outputs/` になる.

```bash
# 実験 1 (差分なし)
python tasks/01_tabular_cls/train.py

# 実験 2 から 4
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/lr_hidden.yaml
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/sgd.yaml
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/invalid_key.yaml
```

`--experiment` の相対パスはタスクのディレクトリから辿る.

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, macOS (Apple Silicon), Python 3.13 で実行.

| 実験 | デバイス | 止まったエポック | ベストのエポック | 検証用データの正解率 | 学習時間 |
| --- | --- | --- | --- | --- | --- |
| 1 差分なし | mps | 39 | 19 | 0.9667 | 0.79 秒 |
| 1 差分なし | cpu | 39 | 19 | 0.9667 | 0.12 秒 |
| 2 `lr_hidden` | mps | 71 | 51 | 0.9000 | - |
| 3 `sgd` | mps | 48 | 28 | 1.0000 | - |
| 4 `invalid_key` | - | - | - | エラー (想定通り) | - |

- 1 から 4 まで, README に書かれた通りに動いた. 出力物もすべてそろい, `final_model.pt` は別のスクリプトから読み込めた
- 実験 3 の `diff.yaml` には `optimizer.params.momentum: <未定義> -> 0.9` と出て, default の `params` が捨てられて置き換わったことが分かる
- 実験 4 は `ValueError: experiment に default に無い項目があります: model.dropout` で止まり, 出力先のディレクトリは作られなかった
- 同じデバイスで 2 回学習すると `history.csv` が完全に一致した (mps, cpu とも). mps と cpu の間では一致しない
- 実験 2 は学習率が小さく学習が遅いため, 正解率が 0.9 のまま 20 エポック続いて Early Stopping で止まった. その間も `valid/loss` は 0.253 から 0.184 まで下がり続けており, まだ学習の途中だった (下の 1 つ目の気付きと同じ原因)

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- 検証用データが 30 件と少なく, 正解率が同じ値を取り続ける. `mode: max` では同じ値は改善とみなされないので, ベストモデルは最初にその値に達したエポック 19 のままになった. その後も `valid/loss` は 0.099 から 0.073 まで下がり続けていたが, 判定には使われない
- Early Stopping で止まると, 進捗バーの最後の表示が 1 エポック少ない (エポック 39 で止まったのに `38/200`)
- `device: auto` では mps が選ばれるが, この規模のモデルでは cpu の方が 6 倍以上速かった
- experiment の config のパスは利用者が `train.py` に書くので, 複数の実験を切り替えるには毎回 `configs/experiment/train.yaml` を書き換えるか, 自分で引数を用意する必要がある. このタスクでは `--experiment` を自分で用意した
