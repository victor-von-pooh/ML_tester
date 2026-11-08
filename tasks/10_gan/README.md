# 10 GAN

## 目的

生成器 (Generator) と識別器 (Discriminator) の 2 つのモデルを, 2 つの Optimizer で交互に更新する学習が ML-Found で書けるかを確かめる.
今の ML-Found は Optimizer が 1 つで更新手順も固定なので, **書ける方法を探り, どこで無理が出るかを記録する** のが主な目的になる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が自動でダウンロード (`data/torchvision/`, 08 と共有) |
| 分割 | 学習用 60,000 枚を全て使う. GAN には検証用の損失の意味が薄いので, 検証用データは使わない |
| 前処理 | `ToTensor()` と `Normalize((0.5,), (0.5,))` で画素値を -1 から 1 にする |

余力があれば, Fashion-MNIST や CelebA (64×64 に縮小) に広げる.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- DCGAN の小さな版
    - Generator: ノイズ (100 次元) → `Linear` → `ConvTranspose2d` を 2 回 (`BatchNorm2d` と `ReLU`) → `Tanh` で 28×28
    - Discriminator: `Conv2d` を 2 回 (`LeakyReLU`) → `Linear` で 1 つのロジット
- 2 つを 1 つの `nn.Module` (`GAN`) にまとめ, `model.generator` と `model.discriminator` として持つ

## 試す書き方

次の 3 通りを順に試し, それぞれ README の「結果のメモ」に書きやすさと結果を記録する.

### A. 損失を 1 つにまとめる (ML-Found の仕組みのまま)

- `step` で `d_loss + g_loss` を返し, 1 つの Optimizer で両方のパラメータを更新する
- 生成器と識別器が同じ損失の和を最小化してしまい, 敵対的にならないので **正しく学習できないはず**. それを確かめる

### B. `Trainer` を継承して更新手順を書き換える

- `__init__` で Discriminator 用の Optimizer を別に作り, `update_parameters` を上書きして D と G を順に更新する
- `step` は `{"loss": g_loss, "d_loss": d_loss}` を返すが, D と G で計算グラフを分ける必要があり, `step` と `update_parameters` の分担に無理が出る見込み
- 2 つ目の Optimizer の状態はチェックポイントに入らないので, 再開すると失われる. `state_dict` と `resume` も上書きが必要になるかを確かめる

### C. 手動で更新する (自動最適化を切る)

- `step` の中で D と G の Optimizer を自分で `zero_grad`, `backward`, `step` し, Trainer の更新には何もしない損失を返す
- PyTorch Lightning の `automatic_optimization = False` に当たる書き方が ML-Found でできるかを確かめる

## config

```yaml
# default
experiment:
  name: mnist_gan
optimizer:
  name: Adam
  params:
    lr: 0.0002
    betas: [0.5, 0.999]
trainer:
  epochs: 30
  monitor: train/loss
data:
  root: ../../data/torchvision
  batch_size: 128
model:
  noise_dim: 100
  g_channels: 64
  d_channels: 64
discriminator_optimizer:    # B と C で使う 2 つ目の Optimizer (ML-Found の外の設定)
  name: Adam
  params:
    lr: 0.0002
    betas: [0.5, 0.999]
visualize:
  n_images: 64
```

## 確かめる ML-Found の機能

- [ ] 2 つ目の Optimizer を config で表す方法が無く, プロジェクト固有の項目として書く必要がある
- [ ] `update_parameters` を上書きして交互の更新が書けるか
- [ ] 2 つ目の Optimizer の状態をチェックポイントに含められるか (`state_dict` と `resume` の上書き)
- [ ] ベストモデルの判定 (GAN の損失は良し悪しの指標にならない) を実質的に無効にできるか
- [ ] 学習後にベストモデルの重みに戻される動作が GAN では不要で, 最後の重みのまま残せるか
- [ ] 固定したノイズから生成した画像をエポックごとに保存できるか

## 期待する結果

- A: 生成画像が数字にならない (想定通り失敗する)
- B か C: 30 エポックで数字と分かる画像が生成される

## 予想される問題

- Optimizer, Scheduler, チェックポイントがすべて 1 つの Optimizer を前提にしている
- `check_outputs` が `loss` を必須にし, 学習時はその `loss` で必ず `backward` するので, C の書き方では空の損失を返す工夫が要る
- ベストモデルへの巻き戻しを止める設定が無い (`monitor` を指定しないと `train/loss` で判定され, 最後の重みにならない)

## 実行方法

```bash
python tasks/10_gan/train.py --variant B
```

## 結果のメモ

(実行後に記入)
