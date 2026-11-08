# 11 拡散モデル

## 目的

拡散モデル (DDPM) のように, 損失そのものに乱数 (ノイズと時刻) を含む学習で, 検証用の損失やベストモデルの判定が意味を持つかを確かめる.
拡散モデルで一般的な EMA (重みの指数移動平均) を ML-Found に組み込めるかも確かめる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が自動でダウンロード (`data/torchvision/`, 08 と共有) |
| 分割 | 学習用 54,000 枚, 検証用 6,000 枚 |
| 前処理 | `ToTensor()` と `Normalize((0.5,), (0.5,))` で画素値を -1 から 1 にする. 必要なら 32×32 にパディング (U-Net の解像度を 2 の冪にするため) |

余力があれば, Fashion-MNIST や CIFAR-10 (32×32 のカラー) に広げる.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. まずは事前学習済みモデルは使わない.

- 小さな U-Net (解像度 32 → 16 → 8, チャネル 32 → 64 → 128)
    - 時刻 `t` は正弦波の埋め込みと MLP で各ブロックに加える
    - 残差ブロックは `Conv2d → GroupNorm → SiLU` を 2 回
- ノイズのスケジュール: 線形の β (1e-4 から 0.02), `T = 1000`
- 生成は DDPM の祖先サンプリング (1,000 ステップ). 速くしたい場合は DDIM (50 ステップ) も試す

比較用の選択肢として, Hugging Face の `diffusers` の `UNet2DModel` と `DDPMScheduler` を使う版も考えられる.
ただし事前学習済みの重みは MNIST では使わず, モデルの定義だけを借りる形になる.

## step と Trainer の設計

- `step` の中で, バッチごとに時刻 `t` とノイズ `eps` を乱数で選び, ノイズを加えた画像から `eps` を予測する
- 損失は予測したノイズと `eps` の MSE (`nn.MSELoss()`)
- EMA: `Trainer` を継承して `update_parameters` を上書きし, `super().update_parameters(loss)` の後に EMA の重みを更新する (`torch.optim.swa_utils.AveragedModel` を使う)
- 生成: `on_epoch_end("valid")` で数エポックごとに EMA のモデルから 64 枚を生成して保存する

## config

```yaml
# default
experiment:
  name: mnist_ddpm
optimizer:
  name: AdamW
  params:
    lr: 0.0002
scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 50
trainer:
  epochs: 50
  grad_clip_norm: 1.0
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
  image_size: 32
model:
  base_channels: 32
  channel_mults: [1, 2, 4]
diffusion:
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
ema:
  use: true
  decay: 0.999
sampling:
  n_images: 64
  every_n_epochs: 10
```

## 確かめる ML-Found の機能

- [ ] 検証用の損失が `t` と `eps` の乱数によってばらつき, ベストモデルの判定が不安定にならないか. 検証時だけ乱数を固定する方法があるか
- [ ] `update_parameters` を上書きして EMA を更新できる
- [ ] EMA のモデルがチェックポイントに入らず, 再開すると失われる. ベストモデルとして保存されるのも EMA でない方の重みになる
- [ ] 検証を EMA のモデルで行いたいが, `run_epoch` は `self.model` を使うので差し替えが必要
- [ ] 生成に時間がかかる処理を `on_epoch_end` に入れたときに, 進捗バーやログが分かりにくくならないか
- [ ] 学習時間が長い (MPS で数十分から 1 時間程度) ので, 途中で止めて再開する流れ

## 期待する結果

- 50 エポックで, 生成画像の大半が数字と分かる
- EMA ありの方が生成画像がきれいになる

## 予想される問題

- EMA の重み, 生成用の状態など, モデル以外に保存したい状態をチェックポイントに加える仕組みが無い
- 検証用の損失が乱数に依存するので, そのままでは `valid/loss` でのベストモデルの判定や Early Stopping の信頼性が低い
- 検証に使うモデルを差し替える (EMA で検証する) フックが無い

## 実行方法

```bash
python tasks/11_diffusion/train.py
```

## 結果のメモ

(実行後に記入)
