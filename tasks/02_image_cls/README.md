# 02 画像分類

## 目的

画像をミニバッチで学習する標準的な形で, GPU/MPS の利用, step ごとの Scheduler, 勾配クリッピングを確かめる.
学習後に `final_model.pt` を読み込んでテスト用データで評価し, 学習後のモデルの使いやすさも確かめる.

## ディレクトリ構成

```
02_image_cls/
├── configs/
│   ├── default/train.yaml          # 基準の config
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 5 の差分
│           ├── steplr.yaml
│           ├── no_clip.yaml
│           ├── workers0.yaml
│           └── cpu.yaml
├── data.py                         # データの分割, 標準化, DataLoader (train.py と evaluate.py で共有)
├── model.py                        # ConvBlock, FashionCNN
├── train.py                        # step, 学習の実行
├── evaluate.py                     # 学習したモデルをテスト用データで評価
└── outputs/fashion_mnist/<日付>/<時刻>/   # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | Fashion-MNIST (`torchvision.datasets.FashionMNIST`) |
| 取得方法 | torchvision が初回に自動でダウンロード (リポジトリの `data/torchvision/` に保存, ダウンロードは約 30 MB, 展開後を含め約 85 MB) |
| 規模 | 学習用 60,000 枚, テスト用 10,000 枚, 28×28 のグレースケール, 10 クラス |
| 分割 | 学習用の 60,000 枚を, `experiment.seed` で並べ替えて学習用 54,000 枚と検証用 6,000 枚に分割. テスト用は学習後の評価だけに使う |
| 前処理 | `ToTensor()` と `Normalize`. 平均と標準偏差は学習用の 54,000 枚だけから求める (平均 0.2858, 標準偏差 0.3529). データ拡張は使わない |

MNIST ではなく Fashion-MNIST にするのは, MNIST では正解率がすぐ 0.99 近くまで上がり, 設定による差が見えにくいため.
MNIST は 08 から 11 の生成系のタスクで使う.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `ConvBlock`: `Conv2d (3×3) → BatchNorm2d → ReLU → MaxPool2d` で縦横を半分にする部品
- `FashionCNN`: `ConvBlock` を `model.channels` の数だけ重ね (既定は `[32, 64]` で 28 → 14 → 7), 全結合 2 層 (`model.hidden_dim` は 128) でクラスごとのスコアを出す. パラメータ数は約 42 万
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
  seed: 42
optimizer:
  name: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001
scheduler:
  name: OneCycleLR
  params:
    max_lr: 0.01
    epochs: ${trainer.epochs}   # trainer.epochs に合わせる
    # steps_per_epoch はバッチ数で決まるので, 学習を始めるときに ML-Found が補う
  interval: step
trainer:
  epochs: 15
  monitor: valid/acc
  mode: max
  grad_clip_norm: 1.0
data:
  root: ../../data/torchvision   # 相対パスはタスクのディレクトリから辿る
  valid_size: 6000
  batch_size: 128
  num_workers: 2
model:
  channels: [32, 64]
  hidden_dim: 128
```

`OneCycleLR` は全体のステップ数 (`epochs × steps_per_epoch`) が必要だが, `steps_per_epoch` はバッチ数で決まる.
config に書かない場合は, ML-Found が `fit` で学習用データのバッチ数から補う (このタスクでは 422, `interval: epoch` なら 1). 補った値は `config.yaml` と `train.log` に残る.

experiment で次の差分を試す. 2 から 5 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `scheduler.name: StepLR` (`step_size: 5`, `gamma: 0.5`) と `scheduler.interval: epoch` に変える (`steplr.yaml`)
3. `trainer.grad_clip_norm: null` (`no_clip.yaml`)
4. `data.num_workers: 0` でワーカーを使わない場合と比べる (`workers0.yaml`)
5. `experiment.device: cpu` で速度を比べる. 時間がかかるので `trainer.epochs: 3` にする (`cpu.yaml`)

## 確かめる ML-Found の機能

- [x] `experiment.device: auto` で cuda または mps が選ばれ, バッチが自動でデバイスに移る (この環境では mps)
- [x] `scheduler.interval: step` で学習率がバッチごとに更新される. `lr_curve.png` にはエポックの開始時点の値が出て, 更新ごとの学習率は `lr_steps.csv` と `lr_steps.png` に残る
- [x] `OneCycleLR` の `steps_per_epoch` を config で無理なく指定できるか → 書かなければ ML-Found が学習用データのバッチ数から補う
- [ ] 勾配クリッピングが効いている → `trainer.log_grad_norm: true` を加えると, クリップ前の勾配のノルムのエポックの平均が `train/grad_norm` に記録される. このタスクではまだ確かめていない. 実験 3 (クリッピングなし) と比べる
- [x] `final_model.pt` を別のスクリプト (`evaluate.py`) から読み込んでテスト用データで評価できる
- [x] `DataLoader` の `num_workers > 0` で問題が起きない (macOS の spawn を含む). `train.py` は `if __name__ == "__main__":` で守っているので問題ない
- [x] 検証時に `model.eval()` になっている. 実験 1 のベストモデルを推論モードで検証用データに使った正解率 0.9300 が `history.csv` の `valid/acc` と一致し, 学習モード (BatchNorm がバッチの統計を使う) では 0.9282 と異なる

## 期待する結果

- テスト用データの正解率が 0.91 以上
- MPS や CUDA で数分以内に終わる

## 予想される問題

- `OneCycleLR` の `steps_per_epoch` はバッチ数に依存するので, config だけでは決めにくい → 書かなければ ML-Found が補うので, config に書く必要は無い
- 進捗バーがエポック単位なので, 1 エポックが長い場合にバッチの進み具合が見えない → このタスクでは 1 エポックが数秒なので困らない. 長い場合は `trainer.progress_bar_batches` でバッチ単位の進捗バーを出せる

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/02_image_cls/outputs/fashion_mnist/<日付>/<時刻>/` になる.
初回はデータのダウンロードが入る.

```bash
# リポジトリのルートから
python tasks/02_image_cls/train.py                                                         # 実験 1
python tasks/02_image_cls/train.py --experiment configs/experiment/examples/steplr.yaml    # 実験 2
python tasks/02_image_cls/train.py --experiment configs/experiment/examples/no_clip.yaml   # 実験 3
python tasks/02_image_cls/train.py --experiment configs/experiment/examples/workers0.yaml  # 実験 4
python tasks/02_image_cls/train.py --experiment configs/experiment/examples/cpu.yaml       # 実験 5

# テスト用データでの評価 (<日付>/<時刻> は train.py が最後に表示する「保存先」の YYYY-MM-DD/HH-MM-SS)
python tasks/02_image_cls/evaluate.py outputs/fashion_mnist/<日付>/<時刻>

# タスクのディレクトリからでも同じ
cd tasks/02_image_cls
python train.py --experiment configs/experiment/examples/cpu.yaml
python evaluate.py outputs/fashion_mnist/<日付>/<時刻>
```

- `--experiment` と `evaluate.py` の出力先のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す. そのため `outputs/fashion_mnist/...` とも `tasks/02_image_cls/outputs/fashion_mnist/...` とも書ける
- `evaluate.py` は出力先の `config.yaml` からモデルの形とデータの分割を復元し, `final_model.pt` を cpu に読み込んでから評価に使うデバイスに移す

## 結果のメモ

PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13 での値.

| 実験 | デバイス | エポック数 | ベストのエポック | 検証用データの正解率 | テスト用データの正解率 |
| --- | --- | --- | --- | --- | --- |
| 1 差分なし (OneCycleLR) | mps | 15 | 15 | 0.9300 | **0.9258** |
| 2 `steplr` | mps | 15 | 11 | 0.9305 | 0.9222 |
| 3 `no_clip` | mps | 15 | - | - | - |
| 4 `workers0` | mps | 15 | - | - | - |
| 5 `cpu` | cpu | 3 | 3 | 0.9250 | 0.9200 |

- 実験 3 と 4 は数値を載せていない. 流した場合は, `history.csv` の `valid/acc` と `evaluate.py` のテスト用データの正解率を実験 1 と比べる. 実験 4 は学習時間 (`train.log` の最後の「学習時間」) を実験 1 と比べる
- 載せた実験はテスト用データの正解率が目標の 0.91 を超える. クラスごとでは Shirt (実験 1 で 0.775) が最も低い
- 実験 1 は学習用データの正解率が 0.999 に達し, `valid/loss` はエポック 7 で最小 (0.238) になった後, エポック 9 以降は上がり続ける (15 で 0.400, 過学習). 一方で `valid/acc` は上下しながらも最後のエポックで最大になるので, `valid/acc` を監視したベストモデルは最後のエポックになる
- 実験 1 の `train.log` には「OneCycleLR の引数を学習用データから補いました: steps_per_epoch=422」と出る
- 実験 1 の `lr_curve.png` は OneCycleLR の山形 (0.0004 → 約 0.0099 → 0.0002) になる. エポックの開始時点の値なので, 本当の最大値 0.01 は出ない. 更新ごとの `lr_steps.csv` と `lr_steps.png` には, 最大値 0.01 がエポック 5 の途中 (6330 ステップ中のステップ 1899) に出る. `interval: step` の実験 (1, 3, 4, 5) は `lr_steps.csv` と `lr_steps.png` が出る
- 実験 2 の学習率は 5 エポックごとに 0.001 → 0.0005 → 0.00025 と半分になる. Scheduler のクラスを変えたので, default の `max_lr` と `epochs` が捨てられたことが `diff.yaml` に出る
- 実験 5 では `trainer.epochs: 3` に合わせて `scheduler.params.epochs` も 3 になり, `diff.yaml` にも出る. `${trainer.epochs}` の参照は ML-Found の config でそのまま使える
- 学習時間は `train.log` の最後の「学習時間」に出る. 他の学習と GPU を分け合うと mps の学習時間は大きく延びる

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した (ML-Found で解消したものは FEEDBACK.md から消し, 残っているものだけを載せる).

- `final_model.pt` は cpu のテンソルで保存されるので, mps の無い環境でもそのまま読める. `evaluate.py` は `map_location="cpu"` も付けている
