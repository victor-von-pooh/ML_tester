# 06 関数近似

## 目的

全データを 1 度に流すフルバッチ学習で, 2 階の情報を使う LBFGS が使えるかを確かめる.
現在の ML-Found は closure を渡す更新手順に対応していないので, **動かないことを確かめ, どこでどう失敗するかを記録する** のがこのタスクの主な目的になる.
あわせて, `Trainer` を継承して closure を渡す回避策がどこまで書けるかも確かめる.
ML-Found の TODO の項目 2 (closure を渡す更新手順への対応) の動作確認にも使う.

## ディレクトリ構成

```
06_func_approx/
├── configs/
│   ├── default/
│   │   ├── train.yaml              # train.py の基準の config (Adam)
│   │   └── pinn.yaml               # train_pinn.py の基準の config
│   └── experiment/
│       ├── train.yaml              # train.py の普段使う差分 (空)
│       ├── pinn.yaml               # train_pinn.py の普段使う差分 (空)
│       └── examples/               # 下の実験 2 から 4 と 6 の差分
│           ├── lbfgs.yaml
│           ├── lbfgs_closure.yaml
│           ├── adam_closure.yaml
│           └── pinn_force_grad.yaml
├── model.py                        # FunctionMLP
├── train.py                        # データ, step, ClosureTrainer (Trainer を継承), 学習の実行
├── train_pinn.py                   # 微分方程式を PINN で解く (実験 5 と 6)
└── outputs/                        # func_approx/ と pinn/ に実験ごとに作られる (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成データ, 1 次元の関数 `y = sin(2πx) + 0.5 sin(6πx)` |
| 取得方法 | その場で生成, ダウンロード不要 |
| 学習用 | `x ∈ [0, 1]` の一様乱数 1,000 点に, 標準偏差 0.01 の雑音を加えた値 |
| 検証用 | `[0, 1]` を等間隔に区切った 200 点の, 雑音の無い値 |
| 形 | `{"x": (N, 1), "y": (N, 1)}` の辞書. `fit([train_data], [valid_data])` のように要素が 1 つのリストで渡す |

学習用の値には分散 0.0001 の雑音が入っているので, 学習用の MSE は 0.0001 前後が下限になる.
検証用は雑音の無い値なので, 検証用の MSE は関数そのものをどれだけ近似できたかを表す.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `FunctionMLP`: `Linear → Tanh` を 3 回と 1 次元の出力層の MLP (隠れ層 64, パラメータ数 8,513)
- 滑らかな関数なので活性化関数は `nn.Tanh()` にする

## step と Trainer の設計

- `step` は MSE の損失 (`nn.MSELoss()`) と, 絶対誤差の最大値 `max_err` を返す
- LBFGS のために, `Trainer` を継承した `ClosureTrainer` を用意し, config の `train.use_closure: true` で切り替える
    - `step`: `update_parameters` は損失しか受け取らないので, 学習時のバッチを `self.current_batch` に保存してから元の `step` を呼ぶ
    - `update_parameters`: 保存したバッチから損失を計算し直して逆伝播する closure を作り, `optimizer.step(closure)` を呼ぶ
    - 元の `update_parameters` にある勾配クリッピング (closure の中で逆伝播の後に適用) と, `interval: step` の Scheduler の更新も写す

## config

```yaml
# default
experiment:
  name: func_approx
  seed: 42
  device: cpu          # 小さな MLP なので cpu の方が速い (01 で確認)
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 5000         # フルバッチなので 1 エポックが 1 回の更新
  progress_bar: false
data:
  n_train: 1000
  n_valid: 200
  noise: 0.01
model:
  hidden_dim: 64
  n_layers: 3
train:
  use_closure: false   # ClosureTrainer を使うかどうか
```

experiment で次の差分を試す. 2 から 4 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, Adam を ML-Found の `Trainer` で)
2. `optimizer.name: LBFGS` (`lr: 1.0`, `max_iter: 20`, `history_size: 50`, `line_search_fn: strong_wolfe`), `trainer.epochs: 200` を ML-Found の `Trainer` で (`lbfgs.yaml`)
3. 2 と同じ LBFGS を `ClosureTrainer` で (`lbfgs_closure.yaml`)
4. Adam を `ClosureTrainer` で. 実験 1 と同じ結果になるかの確認 (`adam_closure.yaml`)

## 確かめる ML-Found の機能

- [x] フルバッチ (`[data]` を渡す形) で数千エポック回したときの速度とログの量. 5000 エポックで `train.log` は 5,108 行 (845 KB). 学習時間の大部分はチェックポイントの保存だった (下の「気付いたこと」)
- [x] LBFGS を指定したときに, どこで, どんなエラーになるか → `Trainer.update_parameters` の `optimizer.step()` で `TypeError: LBFGS.step() missing 1 required positional argument: 'closure'`
- [x] エラーが分かりやすいか. 学習を始める前に気付けるか → torch のエラーがそのまま出るので原因は分かるが, ML-Found が closure に対応していないことは分からない. 最初のバッチの更新で初めて失敗し, 学習結果の無い出力先が残る. エラーは端末にだけ出て, `train.log` には残らない
- [x] (PINN を試す場合) 検証時に `torch.set_grad_enabled(False)` の中で `torch.autograd.grad` を呼べずに失敗するか → 失敗した. エポック 1 の検証で `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`. `step` の中で `torch.enable_grad()` を使えば回避できた (実験 5 と 6)

## 期待する結果

- Adam: 検証用データの MSE が 1e-4 程度
- LBFGS: 現状は失敗する. 対応後は Adam より少ないエポックで MSE が 1e-5 程度まで下がる

## 予想される問題

- LBFGS は `optimizer.step(closure)` が必要だが, `Trainer.update_parameters` は `optimizer.step()` を固定で呼ぶ → その通りだった
- エポック数が多いと, 毎エポックの `last.pt` の保存とログの出力が学習そのものより遅くなる可能性がある. 保存やログの間隔を設定できると良さそう → 保存はその通りで, 学習時間の 8 割以上を占めた. ログの出力は 2000 エポックで 0.1 秒程度で問題にならなかった
- PINN では検証時にも入力に対する勾配が必要だが, `run_epoch` は検証時に勾配の計算を止める → その通りだった (実験 5)

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく (有効にしない場合は `python` を `.venv/bin/python` に読み替える).
`train.py` はどこから実行してもよく, 出力先は常に `tasks/06_func_approx/outputs/func_approx/<日付>/<時刻>/` になる.

```bash
# リポジトリのルートから
python tasks/06_func_approx/train.py                                                            # 実験 1 (約 45 秒)
python tasks/06_func_approx/train.py --experiment configs/experiment/examples/lbfgs.yaml          # 実験 2 (エラー)
python tasks/06_func_approx/train.py --experiment configs/experiment/examples/lbfgs_closure.yaml  # 実験 3
python tasks/06_func_approx/train.py --experiment configs/experiment/examples/adam_closure.yaml   # 実験 4 (約 45 秒)
```

- `.venv` を有効にしない場合は, リポジトリの `.venv/bin/python` を絶対パスか, 実行した場所からの相対パスで指定する
- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 実験 2 はエラーで止まり, `config.yaml`, `diff.yaml`, `train.log` (未コミットの変更があれば `git_diff.patch`) だけを含む出力先が残る. エラーの内容は端末にだけ出る. この出力先は消してよい
- 進捗バーは表示しない設定にしている. 学習時間は `train.log` の最後の「学習時間」に出る. 時間の目安は Apple Silicon の Mac での値

## PINN (実験 5 と 6)

検証時に勾配の計算が止まる ML-Found の仕様と, 損失の中で入力に対する微分を使う学習が衝突するかを確かめる.
`train_pinn.py` で, 微分方程式 `u''(x) = -π² sin(πx)`, `u(0) = u(1) = 0` (厳密解 `u = sin(πx)`) を, `FunctionMLP` (隠れ層 32, 3 層, パラメータ数 2,209) で解く.

- 損失は, 内部の 200 点 (一様乱数) での残差 `u'' - (-π² sin(πx))` の二乗平均と, 両端での `u` の二乗平均の和. `u''` は `torch.autograd.grad` を 2 回使って求める
- 検証は等間隔の 101 点で同じ損失を求め, 厳密解との最大誤差 `max_err` も記録する
- `train.py` とは別の config (`configs/default/pinn.yaml`, `configs/experiment/pinn.yaml`) を使う. default はコードで固定し, `--experiment` の既定は `configs/experiment/pinn.yaml` (空). `train.py` 用の `examples/` (`lbfgs*.yaml`, `adam_closure.yaml`) は使えない. 1 つのプロジェクトのディレクトリに複数の config の組を置いても ML-Found は問題なく動いた
- 保存が学習時間の大部分を占めること (実験 1) を踏まえ, `trainer.checkpoint.save_best: false` と `save_last: false` にした

`train.py` と同じく, `.venv` を有効にした状態で, どこから実行してもよい. 出力先は `tasks/06_func_approx/outputs/pinn/<日付>/<時刻>/` になる.

```bash
python tasks/06_func_approx/train_pinn.py                                                               # 実験 5 (エラー)
python tasks/06_func_approx/train_pinn.py --experiment configs/experiment/examples/pinn_force_grad.yaml  # 実験 6 (学習時間 約 3 秒)
```

| 実験 | 検証時の勾配 | 結果 | ベストのエポック | 検証用の損失 | ベストのエポックでの厳密解との最大誤差 | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 回避策なし | ML-Found のまま (無効) | エポック 1 の検証で `RuntimeError` | - | - | - | - |
| 6 `pinn_force_grad` | `step` の中で `torch.enable_grad()` | 3000 エポック学習 | 2988 | 1.11e-5 | **2.47e-4** | 3.2 秒 |

- 実験 5 は学習のエポック 1 は通り, 検証 (`fit` から呼ばれる 2 回目の `run_epoch`) の `step` で失敗した. `x` は `requires_grad_(True)` にしているが, 勾配の計算が止まった中で求めた `u` が計算グラフを持たないので, `torch.autograd.grad(u.sum(), x)` が失敗する. 学習結果の無い出力先が残り, エラーは端末にだけ出る (`train.log` には残らない). この出力先は消してよい
- 実験 6 は, `step` の中で順伝播と微分を `torch.enable_grad()` で囲めば, 検証でも 2 階微分を計算できた. ML-Found の `run_epoch` が `step` を `torch.set_grad_enabled(False)` で囲んでいても, 内側で有効にし直せる. 検証では `x` に対する微分だけを求めて `backward` はしないので, パラメータの勾配には影響しない
- 実験 6 の最大誤差は `valid/loss` で選んだベストのエポック (2988) の値で, 最後のエポック (3000) では 1.55e-4 だった

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, macOS (Apple Silicon), Python 3.13, cpu で実行.
同じ環境なら同じ結果になる (再現のレビューで全ての `history.csv` が一致した). PyTorch の版や OS が違う場合は一致しないことがある.

| 実験 | Trainer | エポック数 | ベストのエポック | 検証用 MSE | 検証用の最大誤差 | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Adam | `Trainer` | 5000 | 4952 | **1.85e-6** | 3.64e-3 | 44.9 秒 |
| 2 LBFGS | `Trainer` | - | - | エラー (想定通り) | - | - |
| 3 LBFGS | `ClosureTrainer` | 200 | 19 | **5.88e-6** | 1.34e-2 | 1.7 秒 |
| 4 Adam | `ClosureTrainer` | 5000 | 4952 | 1.85e-6 | 3.64e-3 | 42.3 秒 |

- 実験 1 は予想 (1e-4) より 2 桁小さい検証用 MSE まで下がった. 学習用 MSE はエポック 5000 で 1.05e-4 と, 雑音の分散 (1e-4) の下限に達していた
- 実験 2 は `TypeError: LBFGS.step() missing 1 required positional argument: 'closure'` で止まった
- 実験 3 は, エポック 19 (LBFGS の 1 回の `step` は最大 20 回の内部反復, ここまでで損失と勾配の計算は約 450 回) で学習用 MSE が 1.09e-4 と雑音の下限の近くまで下がった. その後は LBFGS の `tolerance_change` による終了判定で, パラメータが全く動かなくなった (エポック 20 以降の記録は全て同じ値). Adam が 4952 エポックかけた所に 19 エポックで近づいた (検証用 MSE は Adam の 3 倍程度, 学習用 MSE は Adam の 1.05e-4 より少し高い 1.09e-4)
- 実験 4 は `history.csv` が実験 1 と完全に一致した. `ClosureTrainer` は勾配法の Optimizer でも同じ結果になる (ただし closure の中で順伝播をもう 1 回行うので, 順伝播の回数は 2 倍)
- 実験 1 の学習時間は 44.9 秒だったが, `trainer.checkpoint.save_last: false` と `save_best: false` にすると 6.1 秒になった (記録は同じ). 素の PyTorch で同じ 5000 回の学習と検証を回すと 3.5 秒だった. 学習時間の 8 割以上がチェックポイントの保存 (`torch.save` の中の pickle) で, cProfile でも保存の pickle が大部分を占めた

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- closure を渡す更新手順に対応していないので, LBFGS は最初の更新で `TypeError` になる (TODO の項目 2 の通り). エラーは torch のものがそのまま出るので, ML-Found が対応していないことは利用者には分からない
- `Trainer` を継承して回避できたが, `update_parameters` は損失しか受け取らないので, closure で損失を計算し直すためにバッチを `step` で `self` に保存する必要があった. また元の `update_parameters` の勾配クリッピングと Scheduler の処理を自分で写す必要がある (このタスクでは省いた)
- 記録される `train/loss` は, `update_parameters` の前に `step` で計算した損失, つまり前のエポックの更新の後のパラメータでの損失. これはどの Optimizer でも同じだが, LBFGS は 1 回の `step` の中でパラメータが大きく動くので, 記録と現在のパラメータの差が大きい
- チェックポイントは, `last.pt` を毎エポック, `best.pt` (中身は同じ) を改善したエポックごとに保存する (実験 1 では 5000 エポック中 2351 回改善). どちらにもモデルと Optimizer の状態, `history` の全ての記録, 乱数の状態, ベストモデルの重みが入る. そのため 1 エポックが短いフルバッチ学習では保存が学習時間の大部分を占め (5000 エポックで 44.9 秒が保存をやめると 6.1 秒), 記録が増えるほど 1 回の保存も重くなる (`last.pt` は 2000 エポックで 459 KB, 5000 エポックで 906 KB). 保存の間隔を設定できると良い
