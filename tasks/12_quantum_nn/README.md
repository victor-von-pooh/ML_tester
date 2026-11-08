# 12 量子ニューラルネット

## 目的

qiskit の量子回路を `TorchConnector` で `nn.Module` にした, 普通とは性質の違うモデルで ML-Found が使えるかを確かめる.
量子回路のシミュレーションは CPU で動き, 重みの数が少なく, 1 バッチの計算が遅い.
また, qiskit は PyTorch や NumPy とは別に乱数を持つ.
このような外部ライブラリのモデルを載せたときに, デバイス, 乱数, チェックポイントがどう扱われるかを記録するのが主な目的になる.

## ディレクトリ構成

```
12_quantum_nn/
├── configs/
│   ├── default/train.yaml          # 基準の config (moons, Estimator, 量子層だけ)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 7 の差分
│           ├── auto_device.yaml
│           ├── sampler.yaml
│           ├── iris.yaml
│           ├── hybrid.yaml
│           ├── mlp.yaml
│           └── no_qiskit_seed.yaml
├── model.py                        # QEstimator, QSampler, QuantumNeuralNetwork (OddNN から), HybridQNN, ClassicalMLP
├── train.py                        # データ, モデルの切り替え, step, 学習の実行
└── outputs/qnn/<日付>/<時刻>/       # ML-Found が実験ごとに作る (git の管理外)
```

## データ

量子ビットの数を特徴量の数に合わせるので, 特徴量が少なく, サンプル数も少ないデータにする.

| 項目 | 内容 |
| --- | --- |
| データセット | 2 つの半月 (`sklearn.datasets.make_moons`, 既定) と, Iris の 2 クラス (versicolor と virginica) |
| 取得方法 | scikit-learn で生成, または同梱, ダウンロード不要 |
| 規模 | make_moons は 300 サンプル (雑音 0.1), 特徴量 2 (= 量子ビット 2). Iris は 100 サンプル, 特徴量 4 (= 量子ビット 4) |
| 分割 | 学習用 80 % と検証用 20 %, `stratify`. 検証用は moons で 60 件, Iris で 20 件 |
| 前処理 | 回転角として回路に入れるので, 学習用データの基準で `MinMaxScaler` で `[0, π]` に変換 |

用意してもらう必要があるデータは無い.

## モデル

[OddNN の `models/qnn_base.py`](https://github.com/victor-von-pooh/OddNN/blob/main/models/qnn_base.py) の `QuantumNeuralNetwork` を `model.py` に写して使う.

- 特徴マップ (`zz_feature_map`) と ansatz (`real_amplitudes`, linear, reps 2) を 2 回交互に重ねる Re-Upload の回路. 同じ ansatz を 2 回使うので重みは共有され, 2 量子ビットで重みは 6 個
- 出力の種類は 2 つあり, 両方を試す
    - `Estimator` (既定): 観測量 `Z...Z` の期待値 `[-1, 1]` (= 1 の数が偶数の確率 - 奇数の確率) を出力し, 2 クラス目のロジットとして `nn.BCEWithLogitsLoss()` に渡す. `EstimatorQNN` は既定で精度 0.015625 の雑音を期待値に加える
    - `Sampler`: 測定結果 (1024 回) の 1 の数の偶奇で 2 クラスの確率にまとめ, その対数を `nn.NLLLoss()` に渡す
- `model.kind` で次の 3 つを切り替える
    - `qnn` (既定): 量子層だけ
    - `hybrid`: `Linear` で特徴量を `[-π, π]` の回転角に変換してから量子層に入れ, 出力を `Linear` で 1 つのロジットに変換する. 前の `Linear` を学習するために, 量子層で入力に対する勾配も計算する (`input_gradients=True`)
    - `mlp`: 比較用の古典の MLP (隠れ層 8, パラメータ 33 個)

OddNN の `qnn_base.py` から直した点.

- `net_type="Estimator"` で `observables` を省略すると `self.observables` が未定義のまま参照されて `AttributeError` になるので, 全ビットの `Z` を既定にした
- `net_type="Sampler"` で `output_dim` を省略すると `self.output_dim` が未定義になるので, 既定を 2 にした
- `SamplerQNN` に `interpret` を渡していないので, `output_shape` は警告とともに無視され, 出力が全ての測定結果 (2 の量子ビット数乗) の確率になる. 1 の数の偶奇でまとめる `interpret` を渡した
- `net_type` が Estimator でも Sampler でもない場合に何もしない (`pass`) ので, エラーにした

依存ライブラリに `qiskit` と `qiskit-machine-learning` (どちらも Apache 2.0) が増える.

## step の設計

- `batch` は `(x, y)` のタプル
- モデルの出力の種類に合わせて, `make_step` で `step` を作る. `Sampler` (量子層だけ) は確率の対数で `nn.NLLLoss()`, それ以外は 1 つのロジットで `nn.BCEWithLogitsLoss()`
- どちらも `{"loss": loss, "acc": acc}` を返す
- qiskit の乱数 (初期の重みに使う `algorithm_globals`, Estimator の雑音と Sampler の測定に使う `seed`) は ML-Found のシード固定の対象外なので, `train.seed_qiskit: true` の場合は `experiment.seed` で自分で固定する. Estimator と Sampler は呼ばれるたびに同じ `seed` から乱数を作り直すので, 固定すると毎回同じ雑音の列になる

## config

```yaml
# default
experiment:
  name: qnn
  seed: 42
  device: cpu        # 量子回路のシミュレーションは cpu で動く
optimizer:
  name: Adam
  params:
    lr: 0.05
trainer:
  epochs: 50
  monitor: valid/acc
  mode: max
data:
  dataset: moons     # moons または iris
  n_samples: 300
  noise: 0.1
  valid_size: 0.2
  batch_size: 16
model:
  kind: qnn          # qnn, hybrid, mlp
  net_type: Estimator
  num_reuploads: 2
  feature_map_reps: 1
  ansatz_reps: 2
  hidden_dim: 8      # mlp の隠れ層
train:
  seed_qiskit: true  # qiskit の乱数を experiment.seed で固定するかどうか
```

experiment で次の差分を試す. 2 から 7 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`). 同じ条件で 2 回学習して結果が一致するかも見る
2. `experiment.device: auto`. mps や cuda が選ばれたときにどうなるか (`auto_device.yaml`)
3. `model.net_type: Sampler` (`sampler.yaml`)
4. `data.dataset: iris`, 4 量子ビット (`iris.yaml`)
5. `model.kind: hybrid` (`hybrid.yaml`)
6. `model.kind: mlp`, 比較用の古典の MLP (`mlp.yaml`)
7. `train.seed_qiskit: false` で 2 回学習し, 結果が一致するか (`no_qiskit_seed.yaml`)

## 確かめる ML-Found の機能

- [x] `experiment.device: auto` で GPU が選ばれる環境では失敗するか → 失敗しなかった. mps が選ばれ, `TorchConnector` が入力を cpu の numpy に移して計算し, 結果を入力と同じデバイスに戻すので動いた. 結果は cpu と小数第 7 位程度まで一致し, 時間は 141 秒と cpu (127 秒) より長かった (転送のためと考えられるが, 内訳は測っていない)
- [x] `fix_seed` は qiskit の乱数を固定しないので, 再現性がどうなるか → `train.seed_qiskit: false` の 2 回は結果が異なった (エポック 1 の学習用の正解率から差があり, 検証用の正解率は 50 エポック中 16 エポックで最大 0.033 違った). 初期の重みに加えて, Estimator の雑音も呼ぶたびに変わるため. `true` の 2 回は `history.csv` が完全に一致した
- [ ] 再開時の乱数の状態にも qiskit の乱数が含まれない → 再開は試していない. ML-Found の `get_rng_state` は Python, NumPy, PyTorch の乱数だけを保存する. ただし `seed_qiskit: true` では Estimator と Sampler が呼ぶたびに同じ `seed` から乱数を作り直し, `algorithm_globals` は初期の重みにしか使わない (再開ではチェックポイントの重みで上書きされる) ので, 再開しても qiskit の側は通しの学習と同じ振る舞いになると考えられる. 問題になるのは `seed` を渡さない場合
- [x] `TorchConnector` の重みが `state_dict` に入り, `best.pt`, `last.pt`, `final_model.pt` から戻せる → 記述の検証のレビューで, 6 つの実験の 3 つのファイルを `weights_only=True` で組み直したモデルに読み込めた. 検証用の正解率も表と一致したが, 実験 1 と 2 は学習時と同じ 16 件ずつのバッチで評価した場合だけで, 60 件を 1 つのバッチにすると 0.683 になった (Estimator の雑音がバッチの分け方で変わり, ロジットが 0 に近い 1 件の判定が変わるため)
- [x] `log_settings` のモデルの構造とパラメータ数の表示が, 量子層で意味のある内容になるか → パラメータ数は量子回路の重みの数 (6 や 12) が正しく出た. モデルの構造は, 量子層の部分が `(qnn_est): TorchConnector()` とだけ出て, 回路の中身は分からない
- [x] 1 エポックが遅い場合の進捗バーの見え方 → Estimator で 1 エポック 2.5 秒, Sampler で 9.5 秒程度なので困らなかった

## 期待する結果

- make_moons で検証用データの正解率 0.85 以上, Iris の 2 クラスで 0.9 以上
- 古典の MLP と同程度か少し劣る

## 予想される問題

- `experiment.device: auto` では mps や cuda が選ばれてテンソルが GPU に移り, qiskit が受け取れずに失敗する → 外れた. `TorchConnector` がデバイスの移動を受け持つので動いた (遅くはなる)
- qiskit の乱数は ML-Found のシード固定と再開の対象外なので, 利用者が `experiment.seed` を回路にも渡す必要がある → シード固定はその通りだった
- OddNN の `QuantumNeuralNetwork` は `observables` や `output_dim` を省略すると `AttributeError` になる → 上の「直した点」の通り直した

## 実行方法

事前にルートの README の「導入」 (qiskit と qiskit-machine-learning を含む) を済ませ, `.venv` を有効にしておく.

`train.py` はどこから実行してもよく, 出力先は常に `tasks/12_quantum_nn/outputs/qnn/<日付>/<時刻>/` になる.

```bash
# リポジトリのルートから (時間は Apple Silicon の cpu での目安)
python tasks/12_quantum_nn/train.py                                                              # 実験 1 (約 2 分)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/auto_device.yaml      # 実験 2 (約 2.5 分)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/sampler.yaml          # 実験 3 (約 8 分)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/iris.yaml             # 実験 4 (約 4 分)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/hybrid.yaml           # 実験 5 (約 3 分)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/mlp.yaml              # 実験 6 (学習は 1 秒未満, 起動を含め数秒)
python tasks/12_quantum_nn/train.py --experiment configs/experiment/examples/no_qiskit_seed.yaml   # 実験 7 (約 2 分, 2 回)
```

`--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す.

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, qiskit 2.6.2, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, macOS (Apple Silicon), Python 3.13 で実行.

| 実験 | データ | モデル | パラメータ数 | ベストのエポック | 検証用の正解率 (ベスト) | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 差分なし (2 回) | moons | 量子, Estimator | 6 | 32 | 0.700 | 127 秒 |
| 2 `auto_device` (mps) | moons | 量子, Estimator | 6 | 32 | 0.700 | 141 秒 |
| 3 `sampler` | moons | 量子, Sampler | 6 | 17 | **0.833** | 482 秒 |
| 4 `iris` | Iris | 量子, Estimator | 12 | 10 | 0.800 | 230 秒 |
| 5 `hybrid` | moons | 古典 + 量子 + 古典 | 14 | 46 | 0.850 | 176 秒 |
| 6 `mlp` | moons | 古典の MLP | 33 | 34 | **0.867** | 0.2 秒 |
| 7 `no_qiskit_seed` (2 回) | moons | 量子, Estimator | 6 | 2 | 0.717 | 119 秒, 120 秒 |

- 量子層だけ (実験 1) は期待 (0.85) を大きく下回る 0.70 だった. 学習用の正解率も 0.73 前後にとどまった
- Sampler (実験 3) は同じ回路でも 0.83 まで上がった. Estimator の期待値 `Z⊗Z` と Sampler の偶奇の確率は同じ量を表すので, 違いは損失への渡し方にある. Estimator は `[-1, 1]` の値をそのままロジットにするので確率が 0.27 から 0.73 の範囲にしかならず, 勾配が弱い. Sampler は確率の対数を損失に渡す. 1 エポックの時間は Estimator の約 4 倍
- Iris (実験 4) は期待 (0.9) を下回る 0.80 (検証用 20 件中 16 件)
- ハイブリッド (実験 5) は量子層だけより良い 0.85 で, パラメータ 33 個の古典の MLP (実験 6, 0.87) に近づいたが, 時間は 800 倍以上かかった
- 実験 5 は最初, 量子層で入力に対する勾配を計算していなかったため, 前の `Linear` が学習されないまま 0.817 だった (コードのレビューで発見して直し, 学習し直した)
- 実験 1 の 2 回は乱数を固定しているので一致して当然だが, 実験 7 の 2 回でベストの値が同じ (0.717) なのは偶然

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- 外部ライブラリの乱数 (qiskit の `algorithm_globals`, Estimator と Sampler の `seed`) は `fix_seed` と再開時の乱数の保存の対象外で, 利用者が自分で固定する必要がある. 固定しないと同じ config でも結果がずれる. 外部の乱数を登録して一緒に固定, 保存できる仕組みがあると良い
- `experiment.device: auto` は, 計算の中身が cpu で行われるモデルでも mps を選ぶ. 今回は `TorchConnector` が移動を受け持ったので動いたが, 転送のぶん遅くなった. 01 と同じく, 小さなモデルや cpu で計算するモデルでは `cpu` を明示する方が良い
- `train.log` のモデルの構造は `print(model)` なので, 量子層は `TorchConnector()` とだけ出て量子回路の中身は記録されない. モデルが自分で構造の説明 (`extra_repr` など) を持たない限り, 再現に必要な情報が `train.log` に残らない
