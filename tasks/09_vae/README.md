# 09 VAE

## 目的

複数の項からなる損失 (再構成誤差と KL ダイバージェンス) を, 内訳ごとに指標として記録できるかを確かめる.
エポックに応じて KL の重みを上げる KL annealing のように, `step` の中でエポックを参照する書き方も確かめる.

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が自動でダウンロード (`data/torchvision/`, 08 と共有) |
| 分割 | 学習用 54,000 枚, 検証用 6,000 枚 |
| 前処理 | `ToTensor()` だけ |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

OddNN の `models/vae_base.py` と Pooh の VGAE の形に合わせ, `Encoder`, `Decoder`, `VAE` の 3 つのクラスにする.

- `Encoder`: 08 と同じ形の CNN で, 平均 `mu` と対数分散 `log_var` (それぞれ `latent_dim` 次元) を出力し, 再パラメータ化 (`z = mu + exp(log_var / 2) * eps`) でサンプリングして `mu, log_var, z` を返す
- `Decoder`: 08 と同じ形で, 出力はロジット (`nn.BCEWithLogitsLoss(reduction="sum")` を使う)
- `VAE`: `forward` は `outputs, mu, log_var, z` を返し, 生成用に `encode` と `decode` のメソッドも持つ
- `latent_dim` は 2 (潜在空間の図) と 16 (生成の質) の 2 通りを試す
- まず OddNN と同じ全結合の版 (入力 784 次元) で動かし, その後に CNN の版に広げる

## step と Trainer の設計

- 損失は画像 1 枚あたりの負の ELBO: `recon + beta * kl`. `kl` は `mu` と `log_var` から解析的に求める
- `{"loss": loss, "recon": recon, "kl": kl, "beta": beta}` を返し, 内訳を学習曲線に出す
- KL annealing: `beta = min(1, epoch / warmup_epochs)`. エポックを参照するため `Trainer` を継承して `step` を上書きし `self.epoch` を使う
- 学習後に, 標準正規分布から `z` をサンプルして 64 枚の画像を生成し, 保存する. `latent_dim: 2` の場合は潜在空間を格子状にたどった画像も保存する

## config

```yaml
# default
experiment:
  name: mnist_vae
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 50
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
model:
  latent_dim: 16
loss:
  beta: 1.0
  kl_warmup_epochs: 10
```

experiment で次の差分を試す.

1. `model.latent_dim: 2`
2. `loss.kl_warmup_epochs: 0` (annealing なし)
3. `loss.beta: 4.0` (β-VAE)

## 確かめる ML-Found の機能

- [ ] 損失の内訳 (`recon`, `kl`) が `train/` と `valid/` を付けて記録され, それぞれ学習曲線が描かれる
- [ ] `step_fn` (関数を渡す形) からはエポックを参照できないので, 継承する必要があるか
- [ ] `beta` が変わる間は `valid/loss` が比べられないので, ベストモデルの判定を何にするか (`valid/recon` を `monitor` にする)
- [ ] 検証時の再パラメータ化のサンプリングが乱数を消費し, 再開や再現性に影響しないか

## 期待する結果

- `latent_dim: 16` で, 検証用データの負の ELBO が 1 枚あたり 100 前後
- 生成画像で数字らしい形が出る

## 予想される問題

- `step_fn(model, batch, phase)` にはエポックが渡らないので, KL annealing のようにエポックに依存する損失は関数の形では書けない
- `valid/loss` の定義がエポックによって変わる (annealing 中) ので, ベストモデルの判定に向かない. 利用者が `monitor` を変える必要がある
- `beta` のような指標でない値 (スケジュールの値) も指標として学習曲線が作られる

## 実行方法

```bash
python tasks/09_vae/train.py
```

## 結果のメモ

(実行後に記入)
