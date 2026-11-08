# 02 画像分類

## 目的

画像をミニバッチで学習する標準的な形で, GPU/MPS の利用, step ごとの Scheduler, 勾配クリッピングを確かめる.
学習後に `final_model.pt` を読み込んでテスト用データで評価し, 学習後のモデルの使いやすさも確かめる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | Fashion-MNIST (`torchvision.datasets.FashionMNIST`) |
| 取得方法 | torchvision が自動でダウンロード (`data/torchvision/` に保存, 約 30 MB) |
| 規模 | 学習用 60,000 枚, テスト用 10,000 枚, 28×28 のグレースケール, 10 クラス |
| 分割 | 学習用の 60,000 枚を学習用 54,000 枚と検証用 6,000 枚に分割, テスト用は学習後の評価だけに使う |
| 前処理 | `ToTensor()` と `Normalize` (学習用データの平均と標準偏差), データ拡張は使わない |

MNIST ではなく Fashion-MNIST にするのは, MNIST では正解率がすぐ 0.99 近くまで上がり, 設定による差が見えにくいため.
MNIST は 08 から 11 の生成系のタスクで使う.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- 畳み込み 2 ブロック (`Conv2d → BatchNorm2d → ReLU → MaxPool2d`) と全結合 2 層の小さな CNN
- チャネル数は config の `model.channels` (例: `[32, 64]`) で変えられるようにする
- `BatchNorm2d` を入れて, `model.train()` と `model.eval()` の切り替えが効いていることも確かめる

余力があれば, torchvision の事前学習済み ResNet-18 を CIFAR-10 で転移学習する版も追加する.
その場合は, 一部の層だけを学習するためのパラメータグループの渡し方も確かめる対象になる.

## step の設計

- `batch` は `(x, y)` のタプル
- `nn.CrossEntropyLoss()` で損失を求め, `{"loss": loss, "acc": acc}` を返す

## config

```yaml
# default
experiment:
  name: fashion_mnist
optimizer:
  name: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001
scheduler:
  name: OneCycleLR
  params:
    max_lr: 0.01
    total_steps: ???   # epochs × バッチ数, train.py で決めるか config で計算
  interval: step
trainer:
  epochs: 15
  monitor: valid/acc
  mode: max
  grad_clip_norm: 1.0
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
  num_workers: 2
model:
  channels: [32, 64]
```

experiment で次の差分を試す.

1. `scheduler.name: StepLR` と `scheduler.interval: epoch` に変える
2. `experiment.device: cpu` で速度を比べる
3. `trainer.grad_clip_norm: null`

## 確かめる ML-Found の機能

- [ ] `experiment.device: auto` で cuda または mps が選ばれ, バッチが自動でデバイスに移る
- [ ] `scheduler.interval: step` で学習率がバッチごとに更新され, `lr_curve.png` に反映される
- [ ] `OneCycleLR` の `total_steps` を config で無理なく指定できるか
- [ ] 勾配クリッピングが効いている
- [ ] `final_model.pt` を別のスクリプトから読み込んでテスト用データで評価できる
- [ ] `DataLoader` の `num_workers > 0` で問題が起きない (macOS の spawn を含む)

## 期待する結果

- テスト用データの正解率が 0.91 以上
- MPS や CUDA で数分以内に終わる

## 予想される問題

- `OneCycleLR` の `total_steps` はバッチ数に依存するので, config だけでは決めにくい. train.py で計算して `Trainer` に Scheduler を渡す形になりそう
- 進捗バーがエポック単位なので, 1 エポックが長い場合にバッチの進み具合が見えない

## 実行方法

```bash
python tasks/02_image_cls/train.py
python tasks/02_image_cls/evaluate.py outputs/fashion_mnist/<日付>/<時刻>
```

## 結果のメモ

(実行後に記入)
