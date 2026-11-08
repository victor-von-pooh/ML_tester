# 08 オートエンコーダ

## 目的

正解ラベルを使わない教師なし学習で, ML-Found の書き方 (`step` と指標) がそのまま使えるかを確かめる.
学習の途中で再構成画像を保存するなど, 指標以外の出力を残す方法も確かめる.

## ディレクトリ構成

```
08_autoencoder/
├── configs/
│   ├── default/train.yaml          # 基準の config
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── latent2.yaml        # 下の「config」の実験 2 の差分
├── data.py                         # MNIST の分割と DataLoader
├── model.py                        # Encoder, Decoder, AutoEncoder
├── train.py                        # AETrainer (Trainer を継承), 潜在空間の図, 学習の実行
└── outputs/mnist_ae/<日付>/<時刻>/  # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が初回に自動でダウンロード (リポジトリの `data/torchvision/` に保存, ダウンロードは約 12 MB, 展開後を含め約 65 MB). 09 から 11 と共有し, 既にあればダウンロードしない |
| 規模 | 学習用 60,000 枚, 28×28 のグレースケール |
| 分割 | 学習用の 60,000 枚を, `experiment.seed` で並べ替えて学習用 54,000 枚と検証用 6,000 枚に分割 |
| 前処理 | `ToTensor()` だけ (画素値は 0 から 1). ラベルは学習に使わず, 潜在空間の図の色分けにだけ使う |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.
ルートの README の「モデルの書き方」の通り, `Encoder`, `Decoder` と, それらを持つ `AutoEncoder` の 3 つのクラスに分ける.

- `Encoder`: `Conv2d` (4×4, stride 2) を 2 回 (28 → 14 → 7) と `Linear` で, 潜在変数 (`model.latent_dim`, 既定 16 次元) に圧縮
- `Decoder`: `Linear` と `ConvTranspose2d` を 2 回 (7 → 14 → 28) で画像に戻し, 最後に `nn.Sigmoid()`
- `AutoEncoder`: `forward` は `outputs, z` を返し, 潜在空間の図のために `encode` と `decode` のメソッドも持つ. パラメータ数は 17 万 (潜在変数 16 次元), 8.2 万 (2 次元)
- 09 の VAE (`model.type: cnn`) と畳み込み・転置畳み込みの部分をそろえ, 再構成の質を比べられるようにした (09 は潜在変数と畳み込みの間に隠れ層を 1 つ挟み, Sigmoid を掛けずにロジットを出す点が違う)

## step と Trainer の設計

`Trainer` を継承した `AETrainer` で, 次のメソッドを上書きする.

- `step`: `batch` の `(x, label)` のうち `label` は使わずに, `nn.BCELoss()` で再構成の損失を求め, 比較用の平均二乗誤差 `mse` と一緒に返す
- `on_epoch_end`: 検証の後, エポック 1, `visualize.every_n_epochs` (5) の倍数, `trainer.epochs` のエポックで, 検証用の先頭 16 枚と再構成画像を上下に並べて `run_dir/recon/epoch_XXX.png` に保存する. 指標は返さないので `{}` を返す. あわせて, エポック 1 で `on_epoch_end` の中の勾配の計算が有効かどうかを `train.log` に記録する
- 学習後に, ベストモデルの再構成画像を `recon/best.png` に保存する (Early Stopping で `trainer.epochs` より前に止まった場合も, 最後の状態を残すため)
- 学習後に, ベストモデルで検証用データを潜在変数に変換し, 2 次元 (`latent_dim: 2` ならそのまま, それ以外は特異値分解による主成分分析) に落としてラベルで色分けした図を `latent.png` に保存する. `torch.pca_lowrank` は乱数を使う近似で, 図が実行ごとに変わりうるので使わない

## config

```yaml
# default
experiment:
  name: mnist_ae
  seed: 42
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 30
data:
  root: ../../data/torchvision   # 相対パスはタスクのディレクトリから辿る
  valid_size: 6000
  batch_size: 128
  num_workers: 2
model:
  latent_dim: 16
visualize:
  n_images: 16
  every_n_epochs: 5
```

experiment で次の差分を試す.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `model.latent_dim: 2` で潜在空間を直接図にする (`configs/experiment/examples/latent2.yaml`)

## 確かめる ML-Found の機能

- [x] 正解ラベルを使わない `step` でも問題なく学習できる
- [x] `on_epoch_end` の中で `self.experiment.run_dir` を使って画像を保存できる
- [x] `on_epoch_end` で何も記録しない場合に `{}` を返せばよい
- [x] `on_epoch_end` の中で再構成のために推論するとき, 検証時と同じく勾配が止まっているか → 止まっていない (`train.log` に「on_epoch_end での勾配の計算: True」と記録された). 自分で `torch.no_grad()` を書く必要がある

## 期待する結果

- 検証用データの BCE が 0.08 前後まで下がり, 再構成画像で数字が読める

## 予想される問題

- `on_epoch_end` は指標を返すための仕組みなので, 画像の保存に使うと役割が混ざる. 学習後の可視化用のフックがあると良さそう → その通りだった. 学習後の図は `fit` の後に `train.py` で作った
- `on_epoch_end` が呼ばれる時点で `torch.set_grad_enabled` の範囲から抜けているので, 推論するときに自分で `torch.no_grad()` を書く必要がある → その通りだった

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/08_autoencoder/outputs/mnist_ae/<日付>/<時刻>/` になる.
初回はデータのダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps での目安)
python tasks/08_autoencoder/train.py                                                         # 実験 1 (約 3 分)
python tasks/08_autoencoder/train.py --experiment configs/experiment/examples/latent2.yaml    # 実験 2 (約 3 分)

# タスクのディレクトリからでも同じ
cd tasks/08_autoencoder
python train.py --experiment configs/experiment/examples/latent2.yaml
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 出力先には ML-Found の出力 (`config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線, `best.pt`, `last.pt`, `final_model.pt`, 未コミットの変更があれば `git_diff.patch`) に加えて, `recon/` (学習中とベストモデルの再構成画像) と `latent.png` (潜在空間の図) ができる

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps で実行.

| 実験 | 潜在変数の次元数 | ベストのエポック | 検証用 BCE | 検証用 MSE | 学習時間 |
| --- | --- | --- | --- | --- | --- |
| 1 差分なし | 16 | 28 / 30 | **0.0836** | 0.0073 | 118 秒 |
| 2 `latent2` | 2 | 28 / 30 | 0.1861 | 0.0419 | 163 秒 |

- 実験 1 は予想通り検証用 BCE が 0.08 前後まで下がり, `recon/epoch_030.png` の再構成画像では 16 枚とも数字が読めた. 16 次元の潜在変数を主成分分析で 2 次元に落とした `latent.png` では, 1 と 0 が両端に分かれ, 4, 7, 9 が下側にまとまり, 残りは中央で重なった
- 再構成画像と主成分分析の修正 (セルフレビューの指摘) の後に 2 つとも学習し直し, `history.csv` は修正前と完全に一致した. 学習時間は他の学習と GPU を分け合った状態での値
- 実験 2 は 2 次元に圧縮するので再構成がぼやけ, BCE は実験 1 の 2 倍以上だった. 4 が 9 に再構成されるなど, 別の数字になったものもあった. `latent.png` では 0 と 1 がはっきり分かれ, 6 と 7 もおおむね分かれた一方, 2, 3, 5, 8 と 4, 9 は重なった
- どちらもエポック 28 まで小さな上下を挟みつつゆっくり下がり, 29 と 30 はわずかに悪化した. 下げ止まりかけで, 明確な過学習ではない

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- 正解ラベルを使わない教師なし学習でも, `step` で再構成の損失を返すだけで問題なく書けた
- `on_epoch_end` は勾配の計算を止める範囲 (`torch.set_grad_enabled(train)`) の外で呼ばれるので, 検証の後でも勾配の計算が有効になっている. 中で推論すると, `torch.no_grad()` を書かない限り計算グラフが作られる. README にこのことが書かれていない
- エポックの途中や学習後に画像などを残す仕組みは無いので, 指標を返すための `on_epoch_end` を流用した. 学習後の処理は `fit` の後に書けば済むが, 出力先や間隔の扱いは自分で書く必要がある
