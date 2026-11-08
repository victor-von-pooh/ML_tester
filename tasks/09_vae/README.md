# 09 VAE

## 目的

複数の項からなる損失 (再構成誤差と KL ダイバージェンス) を, 内訳ごとに指標として記録できるかを確かめる.
エポックに応じて KL の重みを上げる KL annealing のように, `step` の中でエポックを参照する書き方も確かめる.

## ディレクトリ構成

```
09_vae/
├── configs/
│   ├── default/train.yaml          # 基準の config (全結合, 潜在変数 16 次元)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 5 の差分
│           ├── cnn.yaml
│           ├── latent2.yaml
│           ├── no_warmup.yaml
│           └── beta4.yaml
├── data.py                         # MNIST の分割と DataLoader (08 と同じ)
├── model.py                        # Encoder, Decoder, VAE
├── train.py                        # VAETrainer (Trainer を継承), 画像の生成, 学習の実行
└── outputs/mnist_vae/<日付>/<時刻>/ # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が初回に自動でダウンロード (リポジトリの `data/torchvision/`, 08 と共有) |
| 分割 | 学習用の 60,000 枚を, `experiment.seed` で並べ替えて学習用 54,000 枚と検証用 6,000 枚に分割 |
| 前処理 | `ToTensor()` だけ (画素値は 0 から 1) |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.
OddNN の `models/vae_base.py` と Pooh の VGAE の形に合わせ, `Encoder`, `Decoder`, `VAE` の 3 つのクラスにする.

- `Encoder`: 隠れ層 (400) を通して, 平均 `mu` と対数分散 `log_var` (それぞれ `latent_dim` 次元) を出力し, 再パラメータ化 (`z = mu + exp(log_var / 2) * eps`) でサンプリングして `mu, log_var, z` を返す
- `Decoder`: 潜在変数から隠れ層を通して画像に戻す. 損失を安定して計算できるように, Sigmoid を掛けずにロジットを出力する (`nn.BCEWithLogitsLoss(reduction="sum")` を使う)
- `VAE`: `forward` は `outputs, mu, log_var, z` を返し, 生成用に `encode` と `decode` (Sigmoid を掛けた画像) のメソッドも持つ
- `model.type` で構造を切り替える
    - `mlp` (既定): OddNN と同じ全結合 (784 → 400 → 潜在変数). パラメータ数 65 万
    - `cnn`: 08 と同じ畳み込み (28 → 14 → 7) の後に全結合の隠れ層を挟む. パラメータ数 260 万

OddNN の `vae_base.py` から直した点.

- `Decoder.forward` が `self.relu` を使っているのに `__init__` で定義していなかったので, `self.relu = nn.ReLU()` を定義した
- `Decoder` の出力の Sigmoid を外してロジットにし, 損失関数の側で Sigmoid と合わせて計算するようにした
- 再パラメータ化の `torch.randn_like(torch.exp(log_var))` は `torch.randn_like(mu)` と同じなので, 簡単な方にした

## step と Trainer の設計

`Trainer` を継承した `VAETrainer` で `step` を上書きする.

- 損失は画像 1 枚あたりの負の ELBO: `recon + beta * kl`. `kl` は `mu` と `log_var` から解析的に求める
- `{"loss": loss, "recon": recon, "kl": kl, "nelbo": recon + kl, "beta": beta}` を返し, 内訳を学習曲線に出す
- KL annealing: `beta = loss.beta * min(1, epoch / loss.kl_warmup_epochs)` (エポック 1 で `beta / 10`). エポックを参照するため `Trainer` を継承して `step` を上書きし, `self.epoch` を使う (`step_fn` の関数には引数としてエポックが渡らない)
- `beta` が変わる間は `loss` が比べられないので, ベストモデルの判定には重み 1 の負の ELBO (`valid/nelbo`) を使う
- 学習後に, ベストモデルで標準正規分布から 64 枚を生成して `samples.png` に保存する. `latent_dim: 2` の場合は潜在空間の -3 から 3 を 20×20 の格子でたどった画像も `traversal.png` に保存する

## config

```yaml
# default
experiment:
  name: mnist_vae
  seed: 42
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 30
  monitor: valid/nelbo   # KL の重みが変わる間も比べられる重み 1 の負の ELBO
  mode: min
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
  num_workers: 2
model:
  type: mlp              # mlp または cnn
  hidden_dim: 400
  latent_dim: 16
loss:
  beta: 1.0              # KL の重みの最終値
  kl_warmup_epochs: 10   # 最終値に達するまでのエポック数, 0 の場合は最初から最終値
visualize:
  n_images: 64
  n_steps: 20
```

experiment で次の差分を試す. 2 から 5 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`)
2. `model.type: cnn` (`cnn.yaml`)
3. `model.latent_dim: 2` (`latent2.yaml`)
4. `loss.kl_warmup_epochs: 0`, annealing なし (`no_warmup.yaml`)
5. `loss.beta: 4.0`, β-VAE (`beta4.yaml`)

## 確かめる ML-Found の機能

- [x] 損失の内訳 (`recon`, `kl`) が `train/` と `valid/` を付けて記録され, それぞれ学習曲線 (`recon_curve.png`, `kl_curve.png`) が描かれる
- [x] `step_fn` (関数を渡す形) からはエポックを参照できないので, 継承する必要があるか → 引数では渡らないので, 継承するか, closure の中から作った `trainer` の `epoch` を参照する必要がある. このタスクでは継承した `step` で `self.epoch` を使った
- [x] `beta` が変わる間は `valid/loss` が比べられないので, ベストモデルの判定を何にするか → 重み 1 の負の ELBO (`valid/nelbo`) を別に返して `monitor` にした. ただし実験 5 では, この判定がかえって学習の途中のモデルを選んだ (下の結果を参照)
- [ ] 検証時の再パラメータ化のサンプリングが乱数を消費し, 再開や再現性に影響しないか → 同じ条件の 2 回の学習は完全に一致した (再現のレビューで実験 3 と 4 を別のディレクトリから学習し直し, `history.csv` と画像が一致). 再開は試していない. なお検証の負の ELBO は再パラメータ化の 1 回のサンプルから求めるので, 少しばらつきを含む

## 期待する結果

- `latent_dim: 16` で, 検証用データの負の ELBO が 1 枚あたり 100 前後
- 生成画像で数字らしい形が出る

## 予想される問題

- `step_fn(model, batch, phase)` にはエポックが渡らないので, KL annealing のようにエポックに依存する損失は関数の形では書けない → 引数では渡らなかった. 関数の形でも closure で `trainer` を参照すれば書けるが, 継承する方が素直
- `valid/loss` の定義がエポックによって変わる (annealing 中) ので, ベストモデルの判定に向かない. 利用者が `monitor` を変える必要がある → その通りだった
- `beta` のような指標でない値 (スケジュールの値) も指標として学習曲線が作られる → その通りだった (`beta_curve.png` ができ, `train/beta` と `valid/beta` が記録された)

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/09_vae/outputs/mnist_vae/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps での目安)
python tasks/09_vae/train.py                                                          # 実験 1 (約 1.5 分)
python tasks/09_vae/train.py --experiment configs/experiment/examples/cnn.yaml          # 実験 2 (約 3 分)
python tasks/09_vae/train.py --experiment configs/experiment/examples/latent2.yaml      # 実験 3 (約 1.5 分)
python tasks/09_vae/train.py --experiment configs/experiment/examples/no_warmup.yaml    # 実験 4 (約 1.5 分)
python tasks/09_vae/train.py --experiment configs/experiment/examples/beta4.yaml        # 実験 5 (約 1.5 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 出力先には ML-Found の出力に加えて, `samples.png` (生成画像) と, 潜在変数が 2 次元の場合は `traversal.png` ができる

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps で実行.
学習時間は他の学習と GPU を分け合った状態での値.
同じ環境なら同じ結果になる (再現のレビューで実験 3 と 4 の `history.csv` と画像が一致した).

| 実験 | ベストのエポック | 検証用の負の ELBO | 検証用の再構成誤差 | 検証用の KL | 学習時間 |
| --- | --- | --- | --- | --- | --- |
| 1 全結合, 16 次元 | 30 / 30 | **105.1** | 81.1 | 24.0 | 79 秒 |
| 2 `cnn` | 29 / 30 | **98.7** | 75.5 | 23.2 | 187 秒 |
| 3 `latent2` | 29 / 30 | 152.3 | 146.0 | 6.3 | 71 秒 |
| 4 `no_warmup` | 30 / 30 | 105.0 | 81.4 | 23.6 | 70 秒 |
| 5 `beta4` | **4** / 30 | 114.5 | 96.0 | 18.6 | 75 秒 |

表の値は全てベストのエポックでの値.

- 実験 1 と 2 は予想 (100 前後) の通りで, 畳み込みの実験 2 の方が 6 ほど良かった. `samples.png` では, 実験 1 は半分ほどが数字とはっきり読め, 実験 2 の方が明瞭だった
- 実験 3 (2 次元) は負の ELBO が 150 程度と悪いが, `traversal.png` では 1, 7, 9, 3, 5, 6, 0 などが潜在空間の中で滑らかにつながって並んだ
- 実験 4 (annealing なし) は実験 1 とほぼ同じ結果で, このモデルの大きさでは annealing の効果は見えなかった. エポック 1 の負の ELBO は annealing なしの方が良い (131.6 と 154.0) が, エポック 10 では同じ程度 (108.4) になった
- 実験 5 (β-VAE) は, KL の重みがエポック 10 で 4 まで上がるにつれて重み 1 の負の ELBO が悪くなるので, `valid/nelbo` で判定したベストモデルは, 重みがまだ 1.6 のエポック 4 のものになった. ML-Found はベストモデルの重みに戻すので, 最後に残る `final_model.pt` と `samples.png` は β が 4 に上がりきる前 (β = 1.6) のモデルになる. β-VAE として判定するなら, 最終値の β で重み付けた `recon + 4 * kl` を別に返して `monitor` にする必要がある. ここでは ML-Found の振る舞いを記録するため, このままにした
- `beta` は指標として記録され, 小数の誤差で `1.2000000000000008` のような値になった (バッチごとの値をサンプル数で重み付けして平均するため)

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- `step_fn(model, batch, phase)` には引数としてエポックが渡らないので, KL annealing のようにエポックに依存する損失は, `Trainer` を継承して `self.epoch` を使うか, closure の中から `trainer.epoch` を参照する必要がある
- 学習の途中で損失の定義が変わる場合 (annealing, β のスケジュール), `monitor` の指標を何にするかで選ばれるベストモデルが大きく変わり, 最後にその重みに戻される. 実験 5 では学習の途中のモデルが残った. ベストモデルに戻さない設定 (最後の重みを残す) があると良い
- 指標でない値 (`beta` のようなスケジュールの値) も `step` で返すと指標として扱われ, `train/` と `valid/` の両方に記録されて学習曲線も描かれる. 記録だけしたい値を返す方法があると良い
