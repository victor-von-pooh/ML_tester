# 08 オートエンコーダ

## 目的

正解ラベルを使わない教師なし学習で, ML-Found の書き方 (`step` と指標) がそのまま使えるかを確かめる.
学習の途中で再構成画像を保存するなど, 指標以外の出力を残す方法も確かめる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が自動でダウンロード (`data/torchvision/` に保存, 約 60 MB). 09 から 11 と共有 |
| 規模 | 学習用 60,000 枚, テスト用 10,000 枚, 28×28 のグレースケール |
| 分割 | 学習用の 60,000 枚を学習用 54,000 枚と検証用 6,000 枚に分割 |
| 前処理 | `ToTensor()` だけ (画素値は 0 から 1). ラベルは学習に使わず, 潜在空間の可視化の色分けにだけ使う |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `Encoder`, `Decoder` と, それらを持つ `AutoEncoder` の 3 つのクラスに分ける (Pooh の VGAE と OddNN の VAE と同じ分け方)
    - `Encoder`: `Conv2d` (stride 2) を 2 回と `Linear` で, 潜在変数 (16 次元) に圧縮
    - `Decoder`: `Linear` と `ConvTranspose2d` を 2 回で 28×28 に戻し, 最後に `nn.Sigmoid()`
    - `AutoEncoder`: `forward` は `outputs, z` を返し, 可視化用に `encode` と `decode` のメソッドも持つ
- 潜在変数の次元数は config の `model.latent_dim` で変えられるようにする
- 09 の VAE と `Decoder` の形をそろえ, 再構成の質を比べられるようにする

## step と Trainer の設計

- `batch` は `(x, label)` のタプルで, `label` は使わない
- `nn.BCELoss()` で再構成の損失を求め, `{"loss": loss, "mse": mse}` を返す
- `Trainer` を継承し, `on_epoch_end("valid")` で検証用データの先頭 16 枚と再構成画像を並べて `run_dir/recon/epoch_XXX.png` に保存する
- 学習後に, 潜在変数を 2 次元に落として (`latent_dim: 2` か PCA) ラベルで色分けした図を保存する

## config

```yaml
# default
experiment:
  name: mnist_ae
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 30
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
model:
  latent_dim: 16
visualize:
  n_images: 16
  every_n_epochs: 5
```

experiment で次の差分を試す.

1. `model.latent_dim: 2` で潜在空間を直接図にする

## 確かめる ML-Found の機能

- [ ] 正解ラベルを使わない `step` でも問題なく学習できる
- [ ] `on_epoch_end` の中で `self.experiment.run_dir` を使って画像を保存できる
- [ ] `on_epoch_end` で何も記録しない場合に `{}` を返せばよい
- [ ] `on_epoch_end` の中で再構成のために推論するとき, 検証時と同じく勾配が止まっているか (止まっていなければ `torch.no_grad()` が必要)

## 期待する結果

- 検証用データの BCE が 0.08 前後まで下がり, 再構成画像で数字が読める

## 予想される問題

- `on_epoch_end` は指標を返すための仕組みなので, 画像の保存に使うと役割が混ざる. 学習後の可視化用のフックがあると良さそう
- `on_epoch_end` が呼ばれる時点で `torch.set_grad_enabled` の範囲から抜けているので, 推論するときに自分で `torch.no_grad()` を書く必要がある

## 実行方法

```bash
python tasks/08_autoencoder/train.py
```

## 結果のメモ

(実行後に記入)
