# 10 GAN

## 目的

生成器 (Generator) と識別器 (Discriminator) の 2 つのモデルを, 2 つの Optimizer で交互に更新する学習を ML-Found でどう書けるかを比べる.
書き方は 4 つ試す. ML-Found の仕組みのまま損失を 1 つにまとめる A (失敗する例), `Trainer` を継承して更新の手順を書き換える B と C, config の `optimizers` に名前付きで 2 つの Optimizer を書く D (`train_named.py`).
GAN の損失は画像の良し悪しを表さないので, `trainer.select_best: false` でベストモデルを選ばず, 最後のエポックの重みを残す.

## ディレクトリ構成

```
10_gan/
├── configs/
│   ├── default/train.yaml          # 基準の config (書き方 A, B, C)
│   ├── default/named.yaml          # 書き方 D の基準の config (optimizers)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       ├── named.yaml              # D の普段使う差分 (空)
│       └── examples/
│           ├── variant_a.yaml      # 書き方 A
│           └── variant_c.yaml      # 書き方 C
├── data.py                         # MNIST の読み込みと DataLoader
├── model.py                        # Generator, Discriminator, GAN
├── train.py                        # GANTrainer (Trainer を継承, 3 つの書き方), 学習の実行
├── train_named.py                  # NamedGANTrainer (書き方 D), 学習の実行
└── outputs/mnist_gan/<日付>/<時刻>/ # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が初回に自動でダウンロード (リポジトリの `data/torchvision/`, 08 と共有) |
| 分割 | 学習用 60,000 枚を全て使う. GAN には検証用の損失の意味が薄いので, 検証用データは使わない (`fit(train_loader)` だけ) |
| 前処理 | `ToTensor()` と `Normalize((0.5,), (0.5,))` で画素値を -1 から 1 にする (生成器の出力の Tanh に合わせる) |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない. DCGAN の小さな版.

- `Generator`: ノイズ (100 次元) → `Linear` → `BatchNorm1d` → `ReLU` → 7×7 の特徴マップ → `ConvTranspose2d` を 2 回 (7 → 14 → 28, 間に `BatchNorm2d` と `ReLU`) → `Tanh`
- `Discriminator`: `Conv2d` を 2 回 (28 → 14 → 7, `LeakyReLU(0.2)`, 2 回目に `BatchNorm2d`) → `Linear` で 1 つのロジット
- `GAN`: 2 つを `model.generator` と `model.discriminator` として持ち, `forward` はノイズから画像を生成する. 2 つ合わせてパラメータ数は 92 万

## 学習の設計

### 試す書き方

A から C は config の `train.variant` で切り替え, D は別のスクリプト `train_named.py` で動かす. 損失は `nn.BCEWithLogitsLoss()` で, 識別器は本物を 1, 生成画像を 0 と判定させ, 生成器は生成画像を 1 と判定させる.

#### A. 損失を 1 つにまとめる (ML-Found の仕組みのまま)

- `step` で `d_loss + g_loss` を返し, ML-Found が config から作る 1 つの Optimizer で両方のパラメータを更新する
- 生成器と識別器が同じ損失の和を最小化してしまい, 敵対的にならないので **正しく学習できないはず**. それを確かめる
- 生成画像についての 2 つの項 (0 と判定させる項と 1 と判定させる項) の和は, 識別器が生成画像を 0.5 と判定したときに最小になる. 識別器は自分だけでその状態にできるので, 生成器が何を生成しても勾配がほぼ 0 になる見込み

#### B. `Trainer` を継承して `update_parameters` を書き換える (既定)

- 生成器の Optimizer は config の `optimizer` から作って `Trainer` に渡し, 識別器の Optimizer は config の `discriminator_optimizer` から `build_optimizer` で作る
- `step` は生成器の損失を `loss` として返し, 識別器の損失は `self.d_loss` に保存する
- `update_parameters` で, まず生成器を ML-Found の元の手順 (`super().update_parameters`) で更新し, 次に識別器を 2 つ目の Optimizer で更新する. 識別器を先に更新すると, 生成器の損失の計算グラフの中の識別器の重みが書き換わり, 逆伝播でエラーになる (PyTorch の in-place の検出) ため, 一般的な順 (識別器 → 生成器) と逆になる
- 2 つ目の Optimizer の状態はチェックポイントに入らないので, `state_dict` と `resume` を上書きして加える. ML-Found の `resume` は読み込んだ状態を外に出さないので, 同じ規則でパスを解決してもう一度読み込む (ML-Found の `extra_state_dict` と `load_extra_state_dict` を使えば読み込み直さずに済むが, このタスクは上書きのまま)
- `trainer.grad_clip_norm` は使わない. ML-Found は `Trainer` の Optimizer (生成器) が更新するパラメータだけをクリップするので, 識別器の更新はクリップされない. config の Scheduler も生成器の Optimizer にしか付かない

#### C. `step` の中で手動で更新する

- `step` の中で識別器と生成器の Optimizer を自分で `zero_grad`, `backward`, `step` する (識別器 → 生成器の一般的な順)
- ML-Found には `g_loss.detach().requires_grad_(True)` (勾配がパラメータに流れない損失) を返す. ML-Found の `update_parameters` はこれで逆伝播して `optimizer.step()` を呼ぶが, その前の `zero_grad` で勾配が `None` になるので何も更新しない. `update_parameters` を上書きして逆伝播と更新を飛ばす書き方でもよい
- PyTorch Lightning の `automatic_optimization = False` に当たる書き方. ML-Found の `trainer.manual_optimization: true` にすれば, 勾配の流れない損失を返す工夫なしに同じことができる (このタスクの C は使っていない)

#### D. config の `optimizers` に名前付きで書く (`train_named.py`)

- `configs/default/named.yaml` の `optimizers` に, 識別器 (`modules: [discriminator]`) と生成器 (`modules: [generator]`) の Optimizer を名前付きで書く. ML-Found が 2 つを作り, 更新, 保存, 再開する
- `step` は `{"loss": {"discriminator": d_loss, "generator": g_loss}}` を返すだけで, `update_parameters`, `state_dict`, `resume` は上書きしない. ML-Found は損失ごとに受け持つ部分のパラメータだけの勾配を求めるので, 生成器の損失の勾配は識別器に入らない
- 2 つの損失の勾配を更新の前の同じパラメータで求めてから両方を更新する (同時の更新) ので, B と同じ計算になる. B と同じ乱数で学習すると, 記録, 生成画像, 最後の重みが一致するはず
- 記録は `train/loss_discriminator`, `train/loss_generator` と合計の `train/loss`, 学習率は `lr_discriminator` と `lr_generator`. 合計は良し悪しを表さないので, 監視する指標は `train/loss_generator` にし, B と同じく `select_best: false`
- 識別器を更新した後の値で生成器の損失を求める順 (C) にしたい場合は, `manual_optimization` で `self.optimizer["generator"]` などを使う

### step と Trainer の設計

- `step` は `{"loss", "d_loss", "g_loss", "d_real", "d_fake"}` を返す. `d_real` と `d_fake` は識別器が本物と生成画像を本物と判定した確率の平均
- `on_epoch_end` で, エポック 1, `visualize.sample_every` (5) の倍数, 最後のエポックに, 固定したノイズから 64 枚を生成して `samples/epoch_XXX.png` に保存する
- `train/loss` は B と C では生成器の損失, A では 2 つの和で, 画像の質を表さないので, `trainer.select_best: false` でベストモデルを選ばない. `best.pt` は作られず, `final_model.pt` と学習後のモデルは最後のエポックの重みになる
- 学習後に, 最後のエポックの重みで生成し, `last_model.png` に保存する

### config

```yaml
# default
experiment:
  name: mnist_gan
  seed: 42
optimizer:                  # A では両方, B と C では生成器だけ
  name: Adam
  params:
    lr: 0.0002
    betas: [0.5, 0.999]
trainer:
  epochs: 20
  monitor: train/loss       # 検証用データが無いので学習用の指標だけ
  mode: min
  select_best: false        # GAN の損失は良し悪しの指標にならないので, ベストモデルを選ばない
train:
  variant: B                # A, B, C
discriminator_optimizer:    # B と C で識別器に使う 2 つ目の Optimizer (ML-Found の外の設定)
  name: Adam
  params:
    lr: 0.0002
    betas: [0.5, 0.999]
data:
  root: ../../data/torchvision
  batch_size: 128
  num_workers: 2
model:
  noise_dim: 100
  g_channels: 64
  d_channels: 64
visualize:
  n_images: 64
  sample_every: 5
```

## 確かめること

- [x] 2 つ目の Optimizer を config で表せるか → B と C では `discriminator_optimizer` を自分で足し, ML-Found の `build_optimizer` で作る (ML-Found の `optimizer` の項目と違い, クラスを変えたときに引数を捨てる扱いなどは効かない). D では ML-Found の `optimizers` に名前付きで書ける
- [x] `update_parameters` を上書きして交互の更新が書けるか → 書けた (B). ただし生成器を先に更新する順にする必要がある
- [x] 2 つ目の Optimizer の状態をチェックポイントに含められるか → `state_dict` と `resume` を上書きして含められる (B, C). D では ML-Found が保存して戻す
- [x] ベストモデルの判定と学習後の巻き戻しを止められるか → `trainer.select_best: false` で止められる. `best.pt` は作られず, `run.json` は `select_best: false`, `best_epoch: null`, `final_weights: last` になり, `final_model.pt` は最後のエポックの重みになる
- [x] 固定したノイズから生成した画像をエポックごとに保存できる (`on_epoch_end` で)
- [x] ML-Found の `optimizers` で, 2 つ目の Optimizer の作成, 更新, 保存, 再開を書かずに GAN を学習できるか → できた (D). B と結果が一致した (下の結果を参照)

## 期待する結果

- A: 生成画像が数字にならない (想定通り失敗する)
- B か C: 20 エポックで数字と分かる画像が生成される

### 予想される問題

- Optimizer, Scheduler, チェックポイントがすべて 1 つの Optimizer を前提にしている → `optimizer` の項目はその通りで, B と C では 2 つ目を自分で扱う. `optimizers` を使う D では ML-Found が扱う
- `check_outputs` が `loss` を必須にし, 学習時はその `loss` で必ず `backward` するので, C の書き方では勾配の流れない損失を返す工夫が要る → `check_outputs` は `loss` がテンソルであることしか確かめないので通るが, 元の `update_parameters` の `backward` で失敗する. 勾配の流れない損失を返すか, `update_parameters` を上書きするか, `trainer.manual_optimization: true` にする必要がある
- 生成器の損失 (`train/loss`) でベストモデルを判定すると, 最後の重みにならない → その通り. 判定すると識別器が最も弱いエポック 1 がベストになるので, `select_best: false` にする

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/10_gan/outputs/mnist_gan/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps での目安, 他の学習と並行すると 2 倍程度かかる)
python tasks/10_gan/train.py                                                         # B (約 6 分)
python tasks/10_gan/train.py --experiment configs/experiment/examples/variant_a.yaml  # A (約 5 分)
python tasks/10_gan/train.py --experiment configs/experiment/examples/variant_c.yaml  # C (約 5 分)
python tasks/10_gan/train_named.py                                                   # D (約 5 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 出力先には ML-Found の出力に加えて, `samples/` (学習中の生成画像) と, A から C では `last_model.png` (最後のエポックの重みでの生成画像) ができる. `best.pt` は作られない

## 結果のメモ

PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps での値.
同じ環境なら, `history.csv` と生成画像は実行ごとに完全に一致する.

| 書き方 | エポック 20 の `d_loss` | `g_loss` | `d_real` | `d_fake` | 最後の重みの生成画像 |
| --- | --- | --- | --- | --- | --- |
| B `update_parameters` を上書き | 0.323 | 2.93 | 0.879 | 0.122 | 数字が読める |
| D `optimizers` | 0.323 | 2.93 | 0.879 | 0.122 | 数字が読める (B と同じ) |

- B は 20 エポックで数字と分かる画像を生成した (`last_model.png`, `samples/epoch_020.png`)
- B の生成器の損失は上下しながらも全体として大きくなり (エポック 1 の 2.23 からエポック 20 の 2.93), 識別器が最も弱いエポック 1 で最小だった. `train/loss` (= `g_loss`) でベストモデルを判定すると崩れた画像のモデルが選ばれるので, `select_best: false` にしている
- D は B と, 20 エポックの識別器と生成器の損失, 判定の確率 (`d_real`, `d_fake`) が全て一致した (差は 0). `samples/` の生成画像 (エポック 1, 5, 10, 15, 20) と `final_model.pt` の重みも一致した. D は B と同じ同時の更新なので, 期待どおり. D の記録の列は `train/loss_discriminator`, `train/loss_generator` と合計の `train/loss`
- `train_named.py` は, B で上書きしている `update_parameters`, `state_dict`, `resume` と, 2 つ目の Optimizer の作成が要らない (`GANTrainer` は 3 つの書き方を含めて約 260 行, `NamedGANTrainer` は `__init__`, `step`, `on_epoch_end` だけで約 100 行). 2 つ目の Optimizer も config の確認 (範囲, typo) とダッシュボードの学習の起動のフォームの対象になる
- A と C の数値は載せていない. 実行して, `history.csv` のエポック 20 の行 (`d_loss`, `g_loss`, `d_real`, `d_fake`) と `last_model.png` を見る. A は生成画像が数字にならず, 生成画像の判定 `d_fake` が 0.5 付近, `g_loss` が log 2 (約 0.69) 付近から動かないはず. C は B と同じく数字が読める画像になるはず (更新の順が違うので値は B と一致しない)

## 気付いたこと

ML-Found について気付いたこと. リポジトリをまたいで残っている項目は [TODO.md](../../TODO.md) にまとめた.

- `optimizer` の項目と `update_parameters` は 1 つの Optimizer と 1 つの損失を前提にしているので, 2 つ目を自分で扱う B と C では, 2 つの損失を `step` から `self` を経由して渡す必要があり, 計算グラフの都合で更新の順も制約された. `optimizers` を使う D ではこれらを書かずに済む
- 識別器を更新した後の値で生成器の損失を求める順 (C) は, `optimizers` の同時の更新では書けないので, `trainer.manual_optimization: true` で `step` の中で更新する
- このタスクの `train.py` は `state_dict` と `resume` を上書きして識別器の Optimizer の状態を保存して戻している (`extra_state_dict` は使っていない). そのため `STOP` に `now` と書いてエポックの途中で止めた場合, モデルと生成器の Optimizer などはメモリの上で前のエポックの状態に戻るが, 識別器の Optimizer の状態は戻らない. `last.pt` はその場合に書かれないので再開には影響せず, `STOP` を使わないこのタスクの実験にも影響しない
