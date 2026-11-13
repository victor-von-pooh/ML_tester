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
│       └── examples/               # 下の「config」の実験 2 から 5 の差分
│           ├── lr_hidden.yaml
│           ├── sgd.yaml
│           ├── invalid_key.yaml
│           └── cpu.yaml
├── model.py                        # IrisClassifier
├── train.py                        # データ, step, 学習の実行
└── outputs/iris/<日付>/<時刻>/     # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | Iris (`sklearn.datasets.load_iris`) |
| 取得方法 | scikit-learn に同梱, ダウンロード不要 (scikit-learn はルートの README の導入で入る) |
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
  seed: 42
optimizer:
  name: Adam
  params:
    lr: 0.01
    weight_decay: 0.0   # Adam の既定値と同じ. クラスを変えると捨てられることの確認に使う
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

experiment で次の差分を試す. 2 から 5 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `optimizer.params.lr: 0.001` と `model.hidden_dim: 32` (`lr_hidden.yaml`)
3. `optimizer.name: SGD` に変え, `params` に `lr` と `momentum` だけを書く. default の `weight_decay` が捨てられることの確認 (`sgd.yaml`)
4. default に無い項目 `model.dropout` を書いてエラーになることの確認 (`invalid_key.yaml`)
5. `experiment.device: cpu` で mps との速度と結果を比べる (`cpu.yaml`)

## 確かめる ML-Found の機能

- [x] `load_config` と `start_experiment` で `outputs/iris/<日付>/<時刻>/` が作られる
- [x] `config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線 (`acc_curve.png`, `loss_curve.png`, `lr_curve.png`), `best.pt`, `last.pt`, `final_model.pt` がそろう
- [x] `diff.yaml` と `train.log` に experiment の差分が正しく出る
- [x] `monitor: valid/acc` と `mode: max` でベストモデルが選ばれる
- [x] Early Stopping で止まり, `trainer.model` がベストモデルの重みに戻っている (`final_model.pt` が `best.pt` のモデルの重みと一致し, `last.pt` とは異なる)
- [x] クラス名を変えると default の `params` が捨てられる
- [x] default に無い項目を experiment に書くとエラーになる
- [x] `git_diff.patch` が未コミットの変更がある場合だけ作られる (変更がある状態では作られ, クリーンな状態では作られなかった)

## 期待する結果

- 検証用データの正解率が 0.93 以上
- 数秒で学習が終わる

## 予想される問題

特に無い. ML-Found の `examples/minibatch` (wine) とほぼ同じ形のため.

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/01_tabular_cls/outputs/iris/<日付>/<時刻>/` になる.

```bash
# リポジトリのルートから
python tasks/01_tabular_cls/train.py                                                  # 実験 1
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/lr_hidden.yaml    # 実験 2
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/sgd.yaml          # 実験 3
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/invalid_key.yaml  # 実験 4
python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/cpu.yaml          # 実験 5
```

`--experiment` のパスは次の順に探す.

1. 絶対パスならそのまま使う (`~` も使える)
2. タスクのディレクトリ (`tasks/01_tabular_cls/`) から辿る
3. 2 で見つからなければ, 実行した場所から辿る (`tasks/01_tabular_cls/configs/...` と書いても動く)

学習後のモデルは次のように読み込める.

```python
import torch
from model import IrisClassifier   # tasks/01_tabular_cls/ を import できる場所から

model = IrisClassifier(input_dim=4, hidden_dim=16, num_classes=3)
model.load_state_dict(torch.load("outputs/iris/<日付>/<時刻>/final_model.pt"))
```

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, macOS (Apple Silicon), Python 3.13 で実行.

| 実験 | デバイス | 止まったエポック | ベストのエポック | 検証用データの正解率 | 学習時間 |
| --- | --- | --- | --- | --- | --- |
| 1 差分なし | mps | 39 | 19 | 0.9667 | 0.79 から 1.07 秒 (初回は mps の準備で 2.28 秒) |
| 2 `lr_hidden` | mps | 71 | 51 | 0.9000 | 1.34 秒 |
| 3 `sgd` | mps | 48 | 28 | 1.0000 | 0.84 秒 |
| 4 `invalid_key` | - | - | - | エラー (想定通り) | - |
| 5 `cpu` | cpu | 39 | 19 | 0.9667 | 0.11 から 0.12 秒 |

- 1 から 5 まで, README に書かれた通りに動いた. 出力物もすべてそろい, `final_model.pt` は `IrisClassifier` に読み込めた
- 実験 3 の `diff.yaml` には `optimizer.params.weight_decay: 0.0 -> <未定義>` と出て, クラスを変えたことで default の `params` が捨てられたことが分かる
- 実験 4 は `ValueError: experiment に default に無い項目があります: model.dropout` で止まり, 出力先のディレクトリは作られなかった
- 同じデバイスで複数回学習すると `history.csv` が完全に一致した (mps, cpu とも). mps と cpu の間では一致しない (小数第 4 位までは一致)
- 実験 1 は epoch 19 で初めて正解率 0.9667 に達し, その後 0.9333 と 0.9667 を行き来したまま 20 エポック後の 39 で止まった. この間 `valid/loss` は上下しながらも 0.099 から 0.073 まで下がる傾向が続いていた
- 実験 2 は学習率が小さく学習が遅いため, 正解率が 0.9 のまま 20 エポック続いて止まった. その間も `valid/loss` は 0.253 から 0.184 まで単調に下がり続けており, まだ学習の途中だった
- 学習時間は, エポックあたり mps で約 17 ミリ秒, cpu で約 2.6 ミリ秒だった

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- 指標が最良値と同じ値になっても改善とみなされない (`training/callbacks.py` の `is_improved` が `>` と `<` で比べる). そのため, 検証用データが少なく正解率のように同じ値を取りやすい指標を `monitor` にすると, ベストモデルは最初にその値に達したエポックのままになり, Early Stopping も最良値に並んだだけのエポックを「改善なし」と数えて止まる. 実験 1 と 2 は, `valid/loss` が下がり続けている途中で止まった (`3aca743` の時点. その後 ML-Found の PR #1 で `trainer.monitor_tiebreak` が加わり, 同じ値の場合に別の指標 (例: `valid/loss`) で比べられるようになった)
- Early Stopping で止まると, 進捗バーの最後の表示が 1 エポック少ない (エポック 39 で止まったのに `38/200`). `for epoch in pbar` の tqdm は次の要素を取り出すときに数えるので, `break` した最後のエポックが数えられない (`3aca743` の時点. その後 ML-Found の PR #1 で直った. 下の「最新の ML-Found での再実行」を参照)
- `device: auto` では mps が選ばれるが, このタスクでは cpu の方が 6 倍以上速かった
- experiment の config のパスは利用者が `train.py` に書くので, 複数の実験を切り替えるには毎回 `configs/experiment/train.yaml` を書き換えるか, 自分で引数を用意する必要がある. このタスクでは `--experiment` を自分で用意した (その後 ML-Found の PR #19 の `load_project_config` に移し, 「項目=値」の上書きも受け取れるようになった)
- `train.log` の「未コミットの変更: N ファイル」は `git status --porcelain` の行数なので, 未追跡のディレクトリが 1 行にまとめられ, `git_diff.patch` に含まれるファイル数より少なく出る (このタスクの最初の実行では 3 ファイルと出たが, patch には 7 ファイル含まれていた) (`3aca743` の時点. その後 ML-Found の PR #1 で, 未追跡のディレクトリの中のファイルも 1 つずつ数えるようになった)

### 最新の ML-Found での再実行 (2026-11-13)

上の結果は ML-Found `3aca743` で得たもの. 2026-11-13 に, 全ての実験をその後の ML-Found (`24f18c2`) で流し直し, 学習時間は `e2df46e` でも測り直した. 止まったエポック, ベストのエポック, 正解率は全て同じで, `history.csv` もバイト単位で一致した. 実験 4 は同じエラーで, 出力先は作られない.

- 学習時間 (`e2df46e`): 実験 1 は 1.00 秒, 実験 5 (cpu) は 0.13 秒で, 以前とほぼ同じ (`24f18c2` では状態のファイルの書き込みが重く, cpu で 0.19 秒だった. ML-Found の PR #18 で直した)
- Early Stopping で止まると進捗バーが 1 少ない件は直り, `39/200` と出る
