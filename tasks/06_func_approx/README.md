# 06 関数近似

## 目的

全データを 1 度に流すフルバッチ学習で, 2 階の情報を使う LBFGS が使えるかを確かめる.
現在の ML-Found は closure を渡す更新手順に対応していないので, **動かないことを確かめ, どこでどう失敗するかを記録する** のがこのタスクの主な目的になる.
ML-Found の TODO の項目 2 (closure を渡す更新手順への対応) の動作確認にも使う.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成データ, 1 次元の関数 `y = sin(2πx) + 0.5 sin(6πx)` に小さなノイズを加えたもの |
| 取得方法 | その場で生成, ダウンロード不要 |
| 規模 | 学習用 1,000 点 (`x ∈ [0, 1]` の一様乱数), 検証用 200 点 (等間隔) |
| 形 | `batch` は `{"x": (N, 1), "y": (N, 1)}` の辞書を 1 つだけ含むリスト |

余力があれば, 2 次元の関数 (Rosenbrock 関数など) や, 物理を組み込んだニューラルネット (PINN) で微分方程式を解く問題に広げる.
PINN は損失の中で入力に対する微分 (`torch.autograd.grad`) を使うので, 検証時に勾配を切る ML-Found の仕様と衝突するかを確かめられる.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `Linear → Tanh` を 3 回と出力層の MLP (隠れ層 64)
- 滑らかな関数なので活性化関数は `nn.Tanh()` にする

## step の設計

- `{"loss": mse}` だけを返す. 指標として最大誤差 (`max_err`) も返す

## config

```yaml
# default
experiment:
  name: func_approx
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 5000
  progress_bar: true
data:
  n_train: 1000
  n_valid: 200
  noise: 0.01
model:
  hidden_dim: 64
  n_layers: 3
```

experiment で次の差分を試す.

1. `optimizer.name: LBFGS`, `params: {lr: 1.0, max_iter: 20, history_size: 50, line_search_fn: strong_wolfe}`, `trainer.epochs: 200`

## 確かめる ML-Found の機能

- [ ] フルバッチ (`[data]` を渡す形) で数千エポック回したときの速度とログの量 (`train.log` に毎エポック 1 行)
- [ ] LBFGS を指定したときに, どこで, どんなエラーになるか (想定: `optimizer.step()` に closure が渡らず `TypeError`)
- [ ] エラーが分かりやすいか. 学習を始める前に気付けるか
- [ ] (PINN を試す場合) 検証時に `torch.set_grad_enabled(False)` の中で `torch.autograd.grad` を呼べずに失敗するか

## 期待する結果

- Adam: 検証用データの MSE が 1e-4 程度
- LBFGS: 現状は失敗する. 対応後は Adam より少ないエポックで MSE が 1e-5 程度まで下がる

## 予想される問題

- LBFGS は `optimizer.step(closure)` が必要だが, `Trainer.update_parameters` は `optimizer.step()` を固定で呼ぶ
- エポック数が多いと, 毎エポックの `last.pt` の保存とログの出力が学習そのものより遅くなる可能性がある. 保存やログの間隔を設定できると良さそう
- PINN では検証時にも入力に対する勾配が必要だが, `run_epoch` は検証時に勾配の計算を止める

## 実行方法

```bash
python tasks/06_func_approx/train.py
```

## 結果のメモ

(実行後に記入)
