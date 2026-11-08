# 12 量子ニューラルネット

## 目的

qiskit の量子回路を `TorchConnector` で `nn.Module` にした, 普通とは性質の違うモデルで ML-Found が使えるかを確かめる.
量子回路のシミュレーションは CPU でしか動かず, 重みの数が少なく, 1 バッチの計算が遅い.
また, qiskit は PyTorch や NumPy とは別に乱数を持つ.
このような外部ライブラリのモデルを載せたときに, デバイス, 乱数, チェックポイントがどう扱われるかを記録するのが主な目的になる.

## データ

量子ビットの数を特徴量の数に合わせるので, 特徴量が少なく, サンプル数も少ないデータにする.

| 項目 | 内容 |
| --- | --- |
| データセット | 2 つの半月 (`sklearn.datasets.make_moons`) と, Iris の 2 クラス (setosa 以外) |
| 取得方法 | scikit-learn で生成, または同梱, ダウンロード不要 |
| 規模 | make_moons は 300 サンプル, 特徴量 2. Iris は 100 サンプル, 特徴量 4 (= 量子ビット 4) |
| 分割 | 学習用 80 % と検証用 20 %, `stratify` |
| 前処理 | 特徴量を `[0, π]` に収まるように `MinMaxScaler` で変換 (回転角として回路に入れるため) |

用意してもらう必要があるデータは無い.

## モデル

[OddNN の `models/qnn_base.py`](https://github.com/victor-von-pooh/OddNN/blob/main/models/qnn_base.py) の `QuantumNeuralNetwork` を使う.

- 特徴マップ (`zz_feature_map`) と ansatz (`real_amplitudes`) を交互に重ねる Re-Upload の回路
- 出力の種類は 2 つあり, 両方を試す
    - `Estimator`: 観測量 (例: `SparsePauliOp("Z" * n)`) の期待値 `[-1, 1]` を出力. 2 値分類ではロジットとして扱う
    - `Sampler`: 測定結果の確率を出力. パリティなどでクラスに対応付ける
- 比較用に, 前後に `nn.Linear` を付けたハイブリッドなモデル (古典 → 量子 → 古典) と, 同じ程度のパラメータ数の古典の MLP も用意する
- OddNN のファイルはコピーして `model.py` に置き, 必要な修正はそちらに入れる (下の「予想される問題」を参照)

依存ライブラリに `qiskit` と `qiskit-machine-learning` が増える.

## step の設計

- `batch` は `(x, y)` のタプル
- `Estimator`: 出力を `(B, 1)` のロジットとみなし, `nn.BCEWithLogitsLoss()` で損失を求める
- `Sampler`: 出力の確率の対数をとり, `nn.NLLLoss()` で損失を求める
- どちらも `{"loss": loss, "acc": acc}` を返す

## config

```yaml
# default
experiment:
  name: qnn_moons
  device: cpu        # 量子回路のシミュレーションは CPU でしか動かない
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
  net_type: Estimator
  num_reuploads: 2
  feature_map_reps: 1
  ansatz_reps: 2
  hybrid: false
```

experiment で次の差分を試す.

1. `experiment.device: auto` に戻す (MPS や CUDA が選ばれたときに, どこでどんなエラーになるか)
2. `model.net_type: Sampler`
3. `data.dataset: iris` (4 量子ビット)
4. `model.hybrid: true`
5. 同じ config で 2 回学習し, 結果が一致するか (qiskit の乱数が `experiment.seed` で固定されるか)

## 確かめる ML-Found の機能

- [ ] `experiment.device: auto` で GPU が選ばれる環境では失敗する. 分かりやすいエラーになるか, モデルごとにデバイスを固定する方法が要るか
- [ ] `fix_seed` は qiskit の乱数 (`algorithm_globals.random_seed`, Estimator と Sampler の `seed`) を固定しないので, 再現性がどうなるか
- [ ] 再開時の乱数の状態にも qiskit の乱数が含まれない
- [ ] `TorchConnector` の重みが `state_dict` に入り, `best.pt`, `last.pt`, `final_model.pt` から戻せる (`weights_only=True` で読めるか)
- [ ] `log_settings` のモデルの構造とパラメータ数の表示が, 量子層で意味のある内容になるか
- [ ] 1 エポックが遅い場合の進捗バーの見え方 (エポック単位なので, 最初のエポックが終わるまで何も表示されない)

## 期待する結果

- make_moons で検証用データの正解率 0.85 以上, Iris の 2 クラスで 0.9 以上
- 古典の MLP と同程度か少し劣る

## 予想される問題

- `experiment.device: auto` では mps や cuda が選ばれてテンソルが GPU に移り, qiskit が受け取れずに失敗する. 利用者が `cpu` を明示する必要がある
- qiskit の乱数は ML-Found のシード固定と再開の対象外なので, 利用者が `experiment.seed` を回路にも渡す必要がある
- OddNN の `QuantumNeuralNetwork` は, `net_type="Estimator"` で `observables` を省略すると `self.observables` が未定義のまま参照されて `AttributeError` になる. `Sampler` で `output_dim` を省略した場合も同じ. コピーした `model.py` では既定値を用意して直す

## 実行方法

```bash
pip install qiskit qiskit-machine-learning
python tasks/12_quantum_nn/train.py
```

## 結果のメモ

(実行後に記入)
