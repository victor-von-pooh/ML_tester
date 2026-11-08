# 10 GAN

## 目的

生成器 (Generator) と識別器 (Discriminator) の 2 つのモデルを, 2 つの Optimizer で交互に更新する学習が ML-Found で書けるかを確かめる.
今の ML-Found は Optimizer が 1 つで更新手順も固定なので, **書ける方法を探り, どこで無理が出るかを記録する** のが主な目的になる.

## ディレクトリ構成

```
10_gan/
├── configs/
│   ├── default/train.yaml          # 基準の config (書き方 B)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── variant_a.yaml      # 書き方 A
│           └── variant_c.yaml      # 書き方 C
├── data.py                         # MNIST の読み込みと DataLoader
├── model.py                        # Generator, Discriminator, GAN
├── train.py                        # GANTrainer (Trainer を継承, 3 つの書き方), 学習の実行
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

## 試す書き方

config の `train.variant` で次の 3 通りを切り替える. 損失は `nn.BCEWithLogitsLoss()` で, 識別器は本物を 1, 生成画像を 0 と判定させ, 生成器は生成画像を 1 と判定させる.

### A. 損失を 1 つにまとめる (ML-Found の仕組みのまま)

- `step` で `d_loss + g_loss` を返し, ML-Found が config から作る 1 つの Optimizer で両方のパラメータを更新する
- 生成器と識別器が同じ損失の和を最小化してしまい, 敵対的にならないので **正しく学習できないはず**. それを確かめる

### B. `Trainer` を継承して `update_parameters` を書き換える (既定)

- 生成器の Optimizer は config の `optimizer` から作って `Trainer` に渡し, 識別器の Optimizer は config の `discriminator_optimizer` から `build_optimizer` で作る
- `step` は生成器の損失を `loss` として返し, 識別器の損失は `self.d_loss` に保存する
- `update_parameters` で, まず生成器を ML-Found の元の手順 (`super().update_parameters`) で更新し, 次に識別器を 2 つ目の Optimizer で更新する. 識別器を先に更新すると, 生成器の損失の計算グラフの中の識別器の重みが書き換わり, 逆伝播でエラーになる (PyTorch の in-place の検出) ため, 一般的な順 (識別器 → 生成器) と逆になる
- 2 つ目の Optimizer の状態はチェックポイントに入らないので, `state_dict` と `resume` も上書きして加える
- `trainer.grad_clip_norm` は使わない. ML-Found は `model.parameters()` (識別器も含む) でクリップするので, 生成器のクリップに捨てる予定の識別器の勾配が混ざり, 識別器の本当の更新はクリップされない. config の Scheduler も生成器の Optimizer にしか付かない

### C. `step` の中で手動で更新する

- `step` の中で識別器と生成器の Optimizer を自分で `zero_grad`, `backward`, `step` する (識別器 → 生成器の一般的な順)
- ML-Found には `g_loss.detach().requires_grad_(True)` (勾配がパラメータに流れない損失) を返す. ML-Found の `update_parameters` はこれで逆伝播して `optimizer.step()` を呼ぶが, その前の `zero_grad` で勾配が `None` になるので何も更新しない. `update_parameters` を上書きして逆伝播と更新を飛ばす書き方でもよい
- PyTorch Lightning の `automatic_optimization = False` に当たる書き方

## step と Trainer の設計

- `step` は `{"loss", "d_loss", "g_loss", "d_real", "d_fake"}` を返す. `d_real` と `d_fake` は識別器が本物と生成画像を本物と判定した確率の平均
- `on_epoch_end` で, エポック 1, `visualize.sample_every` (5) の倍数, 最後のエポックに, 固定したノイズから 64 枚を生成して `samples/epoch_XXX.png` に保存する
- 学習後に, ML-Found が戻したベストモデルと, `last.pt` に入っている最後のエポックの重みの両方で生成し, `best_model.png` と `last_model.png` に保存する. `train/loss` は B と C では生成器の損失, A では 2 つの和なので, ベストモデルは画像の質を表さない

## config

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

## 確かめる ML-Found の機能

- [x] 2 つ目の Optimizer を config で表す方法が無く, プロジェクト固有の項目として書く必要がある → その通りだった. `discriminator_optimizer` を自分で足し, ML-Found の `build_optimizer` で作った. ML-Found の `optimizer` の項目と違い, クラスを変えたときに引数を捨てる扱いなどは効かない
- [x] `update_parameters` を上書きして交互の更新が書けるか → 書けた (B). ただし生成器を先に更新する順にする必要があった
- [x] 2 つ目の Optimizer の状態をチェックポイントに含められるか → `state_dict` と `resume` を上書きして含められた. 1 エポック学習した後に再開して 2 エポック目まで学習し, 識別器の Optimizer のステップ数が 469 から 938 に続いていることを確かめた. ただし ML-Found の `resume` は読み込んだ状態を外に出さないので, 同じパスの解決の規則を写してもう一度読み込んだ
- [x] ベストモデルの判定 (GAN の損失は良し悪しの指標にならない) を実質的に無効にできるか → 判定を止める設定は無く, `monitor` を指定しないと `train/loss` が使われる. `on_epoch_end` で毎エポック増える値 (例: エポック数) を返して `mode: max` で監視すれば毎エポック更新させられるが, 見せかけの指標が要る
- [x] 学習後にベストモデルの重みに戻される動作が GAN では不要で, 最後の重みのまま残せるか → 設定では残せなかった (上の見せかけの指標を使えば残せる). B と C ではエポック 1 (生成器の損失が最も小さい) がベストと判定され, `final_model.pt` と学習後の `trainer.model` はエポック 1 の重みになった (下の結果を参照)
- [x] 固定したノイズから生成した画像をエポックごとに保存できる (`on_epoch_end` で)

## 期待する結果

- A: 生成画像が数字にならない (想定通り失敗する)
- B か C: 20 エポックで数字と分かる画像が生成される

## 予想される問題

- Optimizer, Scheduler, チェックポイントがすべて 1 つの Optimizer を前提にしている → その通りだった
- `check_outputs` が `loss` を必須にし, 学習時はその `loss` で必ず `backward` するので, C の書き方では勾配の流れない損失を返す工夫が要る → `check_outputs` は `loss` がテンソルであることしか確かめないので通るが, 元の `update_parameters` の `backward` で失敗する. 勾配の流れない損失を返すか, `update_parameters` を上書きする必要があった
- ベストモデルへの巻き戻しを止める設定が無い (`monitor` を指定しないと `train/loss` で判定され, 最後の重みにならない) → その通りで, 最も困った点だった

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/10_gan/outputs/mnist_gan/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps での目安, 他の学習と並行すると 2 倍程度かかる)
python tasks/10_gan/train.py                                                         # B (約 6 分)
python tasks/10_gan/train.py --experiment configs/experiment/examples/variant_a.yaml  # A (約 5 分)
python tasks/10_gan/train.py --experiment configs/experiment/examples/variant_c.yaml  # C (約 5 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 出力先には ML-Found の出力に加えて, `samples/` (学習中の生成画像), `best_model.png` (ML-Found が戻したベストモデル), `last_model.png` (最後のエポックの重み) ができる

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps で実行.
学習時間は他の学習と GPU を分け合った状態での値.

| 書き方 | ML-Found のベストのエポック | エポック 20 の `d_loss` | `g_loss` | `d_real` | `d_fake` | 最後の重みの生成画像 | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 損失を 1 つに | 8 | 0.694 | 0.694 | 1.000 | 0.500 | 格子状の模様 (失敗) | 319 秒 |
| B `update_parameters` を上書き | **1** | 0.323 | 2.93 | 0.879 | 0.122 | 数字が読める | 343 秒 |
| C `step` で手動で更新 | **1** | 0.295 | 3.13 | 0.886 | 0.114 | 数字が読める | 303 秒 |

- A は想定通り失敗した. 識別器は本物を確率 1.000 で本物と判定する一方, 生成画像は 0.500 のままで, 生成器の損失 (0.694 ≈ log 2) も識別器の損失もエポック 1 からほとんど動かなかった. 生成画像についての 2 つの項 (0 と判定させる項と 1 と判定させる項) の和は, 識別器が生成画像を 0.5 と判定したときに最小になる. 識別器は自分だけでその状態にできるので, 生成器が何を生成しても勾配がほぼ 0 になる
- B と C は 20 エポックで数字と分かる画像を生成した (`last_model.png`, `samples/epoch_020.png`). 更新の順は B が生成器 → 識別器, C が識別器 → 生成器と違うが, 結果の傾向は同じだった
- B と C では, 生成器の損失は上下しながらも全体として大きくなる傾向で, 識別器が最も弱いエポック 1 で最小だった. そのため `train/loss` (= `g_loss`) で判定したベストモデルはエポック 1 になった. 学習後に ML-Found がエポック 1 の重みに戻すので, `best_model.png` と `final_model.pt` は崩れた画像を出すモデルになった. 最後の重みは `last.pt` の `model` から読み込み直す必要があった

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- Optimizer, Scheduler, チェックポイントがすべて 1 つの Optimizer を前提にしているので, 2 つ目の Optimizer は config, 作成, 更新, 保存, 再開の全てを利用者が書く必要があった
- `update_parameters` は損失を 1 つしか受け取らないので, 2 つの損失を交互に更新するには `step` から `self` を経由して渡す必要があり, 計算グラフの都合で更新の順も制約された
- C のように更新を全て自分で行う場合も, 元の `update_parameters` が `loss` で必ず `backward` するので, `detach().requires_grad_(True)` を返すか `update_parameters` を上書きする必要がある. 自動の更新を止める設定 (Lightning の `automatic_optimization = False` に当たるもの) があると良い
- 勾配クリッピングは `model.parameters()` 全体に掛かるので, Optimizer を分けた場合は Optimizer ごとのパラメータでクリップできない
- ベストモデルの判定と学習後の巻き戻しを止める設定が無いので, 損失が良し悪しの指標にならない GAN では, 何もしないと最初のエポックの崩れたモデルが `final_model.pt` に残る. 09 の β-VAE と同じ問題で, GAN では最も影響が大きかった
- `resume` が読み込んだチェックポイントを外に出さないので, 追加の状態を戻すにはパスの解決の規則を写してもう一度読み込む必要があった
