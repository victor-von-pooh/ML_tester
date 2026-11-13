# 11 拡散モデル

## 目的

拡散モデル (DDPM) のように, 損失そのものに乱数 (ノイズと時刻) を含む学習で, 検証用の損失やベストモデルの判定が意味を持つかを確かめる.
拡散モデルで一般的な EMA (重みの指数移動平均) を ML-Found に組み込めるかも確かめる.

## ディレクトリ構成

```
11_diffusion/
├── configs/
│   ├── default/train.yaml          # 基準の config (EMA あり, 30 エポック)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── no_ema.yaml         # 実験 2: EMA なし
│           └── smoke.yaml          # 動作確認用 (2 エポック, 時刻 50)
├── data.py                         # MNIST の分割, 32×32 への余白, DataLoader
├── model.py                        # TimeEmbedding, ResBlock, UNet, Diffusion
├── train.py                        # DiffusionTrainer (Trainer を継承), 学習の実行
└── outputs/mnist_ddpm/<日付>/<時刻>/ # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | MNIST (`torchvision.datasets.MNIST`) |
| 取得方法 | torchvision が初回に自動でダウンロード (リポジトリの `data/torchvision/`, 08 と共有) |
| 分割 | 学習用の 60,000 枚を, `experiment.seed` で並べ替えて学習用 54,000 枚と検証用 6,000 枚に分割 |
| 前処理 | 上下左右に 2 画素の余白を足して 32×32 にし (U-Net の解像度を 2 の冪にするため), `Normalize((0.5,), (0.5,))` で画素値を -1 から 1 にする |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `UNet`: 解像度 32 → 16 → 8 (チャネル 32 → 64 → 128) の小さな U-Net. パラメータ数 178 万
    - `TimeEmbedding`: 時刻 `t` を正弦波で表して MLP で変換し, 各ブロックに加える
    - `ResBlock`: `GroupNorm → SiLU → Conv2d` を 2 回と残差接続. 間に時刻の埋め込みを各チャネルに足す
    - 縮小は stride 2 の畳み込み, 拡大は転置畳み込み. 同じ解像度の特徴をつないで戻す
- `Diffusion`: 線形の β (1e-4 から 0.02), `T = 1000`. 時刻 `t` までのノイズを一度に加える `add_noise` と, 1,000 ステップの祖先サンプリングの `sample` を持つ

## step と Trainer の設計

`Trainer` を継承した `DiffusionTrainer` で, 次のメソッドを上書きする.

- `step`: バッチごとに時刻 `t` とノイズ `eps` を乱数で選び, ノイズを加えた画像から `eps` を予測した MSE (`nn.MSELoss()`) を求める
    - 学習時はこの損失を `loss` とし, `loss_random` にも同じ値を入れる (`train/loss_random` は `train/loss` と常に同じ)
    - 検証時は, 毎エポック同じ乱数 (専用の乱数生成器を `on_epoch_start` で `experiment.seed` に戻す) で `t` と `eps` を選び直した損失を `loss` とし, 毎回違う乱数で求めた損失を `loss_random` として比較用に記録する
- `update_parameters`: `super().update_parameters(loss)` の後に EMA の重みを更新する (`torch.optim.swa_utils.AveragedModel` と `get_ema_multi_avg_fn(0.999)`)
- `state_dict` と `resume`: 10 と同じく, EMA の重みと更新回数をチェックポイントに加え, 再開時に戻す
- `on_epoch_end`: 検証の後, 10 エポックごとと最後のエポックに, EMA の重み (EMA を使わない場合は学習中の重み) で 64 枚を生成して `samples/epoch_XXX.png` に保存する
- 学習後に, ML-Found が戻したベストモデルの重みで生成して `best_model.png` に, EMA を使う場合は最後の EMA の重みを `ema_model.pt` に保存し, それで生成して `ema_model.png` に保存する. EMA が一度も更新されていない場合 (学習が終わった状態から再開した場合など) は保存しない

## config

```yaml
# default
experiment:
  name: mnist_ddpm
  seed: 42
optimizer:
  name: AdamW
  params:
    lr: 0.0002
scheduler:
  name: CosineAnnealingLR
  params:
    T_max: ${trainer.epochs}
trainer:
  epochs: 30
  grad_clip_norm: 1.0
data:
  root: ../../data/torchvision
  valid_size: 6000
  batch_size: 128
  num_workers: 2
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

experiment で次の差分を試す.

1. 差分なし (`configs/experiment/train.yaml`, EMA あり)
2. `ema.use: false` (`no_ema.yaml`)

## 確かめる ML-Found の機能

- [x] 検証用の損失が `t` と `eps` の乱数によってばらつき, ベストモデルの判定が不安定にならないか. 検証時だけ乱数を固定する方法があるか → 毎回違う乱数の `loss_random` は 29 回のエポック間の変化のうち 10 回で上がり, それで判定するとベストはエポック 24 になった. 毎エポック同じ乱数の `loss` は上がったのが 3 回で, ベストは最後のエポック 30 だった. ML-Found `3aca743` には検証の乱数を固定する仕組みが無いので, 専用の乱数生成器を `on_epoch_start` で戻して自分で固定した (その後 ML-Found の PR #1 で `trainer.valid_seed` が加わった)
- [x] `update_parameters` を上書きして EMA を更新できる
- [x] EMA のモデルがチェックポイントに入らず, 再開すると失われる. ベストモデルとして保存されるのも EMA でない方の重みになる → その通り. `state_dict` と `resume` を上書きして EMA をチェックポイントに加えた (コードのレビューで指摘を受けて追加). 学習用 256 枚, 時刻 50 の小さな設定で, 1 エポック学習した後に 2 エポック目まで再開し, EMA の更新回数が 4 から 8 に続くことを確かめた. ベストモデルと `final_model.pt` は EMA でない方の重みのまま
- [x] 検証を EMA のモデルで行いたいが, `run_epoch` は `self.model` を使うので差し替えが必要 → 差し替えなかった. 検証の損失は EMA でない方の重みのもので, EMA の重みは生成画像にだけ使った
- [x] 生成に時間がかかる処理を `on_epoch_end` に入れたときに, 進捗バーやログが分かりにくくならないか → 1 回の生成 (1,000 ステップ, 64 枚) は約 20 から 22 秒で, 実験 2 では生成するエポックが約 85 秒, しないエポックが約 65 秒だった. ML-Found の進捗バーはエポック単位なので, 生成の間に限らずエポックの途中は止まって見える. 生成は 3 回だけなので問題にならなかった
- [x] 学習時間が長い (MPS で数十分から 1 時間程度) ので, 途中で止めて再開する流れ → 短い設定で確かめた (下の「学習の再開」)

## 期待する結果

- 50 エポックで, 生成画像の大半が数字と分かる (このタスクでは 30 エポックにした)
- EMA ありの方が生成画像がきれいになる

## 予想される問題

- EMA の重み, 生成用の状態など, モデル以外に保存したい状態をチェックポイントに加える仕組みが無い → その通り. 10 と同じく `state_dict` と `resume` を上書きして加えた (その後 ML-Found の PR #1 で `extra_state_dict` と `load_extra_state_dict` が加わった)
- 検証用の損失が乱数に依存するので, そのままでは `valid/loss` でのベストモデルの判定や Early Stopping の信頼性が低い → その通り (上のチェックボックスを参照)
- 検証に使うモデルを差し替える (EMA で検証する) フックが無い → その通り

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/11_diffusion/outputs/mnist_ddpm/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps で, 他の学習と並行した場合の目安)
python tasks/11_diffusion/train.py                                                       # 実験 1 (約 35 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/no_ema.yaml   # 実験 2 (約 35 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml    # 動作確認 (2 から 6 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 既定の進捗バーはエポック単位なので, 1 エポック (約 1 分, smoke では他の学習と並行して約 3 分) の間は 0 のまま止まって見える. `trainer.progress_bar_batches: true` にするとエポックの中の進み具合も出る
- 動作確認 (smoke) は, ベストのエポック 2, `valid/loss` 約 0.0759 になれば正しく動いている (同じ環境なら完全に一致する). 出力先は約 70 MB あるので, 確認の後は消してよい
- 出力先には ML-Found の出力に加えて, `samples/` (学習中の生成画像), `best_model.png`, EMA を使う場合は `ema_model.pt` と `ema_model.png` ができる

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps で実行.
学習時間は他の学習と GPU を分け合った状態での値.
表の 2 つの学習は EMA をチェックポイントに加える前のコードで行った. 加えた後も学習と検証の計算は変わらない (チェックポイントの中身が増えるだけ).

| 実験 | ベストのエポック | 検証用の損失 (固定した乱数) | 検証用の損失 (毎回違う乱数) | 学習時間 |
| --- | --- | --- | --- | --- |
| 1 EMA あり | 30 / 30 | 0.01756 | 0.01774 | 2224 秒 |
| 2 EMA なし | 30 / 30 | 0.01756 | 0.01774 | 2020 秒 |

学習時間の差 (約 200 秒) は, 実験 1 のエポック 8 から 13 が他の学習と GPU を取り合って遅くなったためで, EMA の更新の時間ではない (他のエポックは同じ時間).

- 2 つの実験は, 生成画像 (`samples/`, `ema_model.png`) と `ema_model.pt` 以外は同じだった. `history.csv`, `final_model.pt`, `best_model.png` はバイト単位で一致した. EMA は学習中の重みを変えず, 生成はどちらもエポック 10, 20, 30 に同じ枚数と時刻の数で行うので, 学習の乱数の使い方も同じになる. 逆に言うと, 生成は学習と同じ乱数を使うので, `sampling.every_n_epochs` を変えると学習そのものが変わる
- 実験 1 の `ema_model.png` (EMA の重み) と `best_model.png` (学習中の重みのベストモデル, エポック 30) は, どちらも大半が数字と読めた. 学習率を CosineAnnealing で最後に 5.5e-7 まで下げたので, 最後の重みと EMA の重みの差が小さく, 見た目の差はほとんど無かった
- エポック 10 の生成画像 (`samples/epoch_010.png`) でも, 多くが数字の形になっていた. EMA ありとなしのエポック 10 の画像は同じノイズから始まり, 質もほとんど同じだった. 期待した「EMA ありの方がきれい」は, この設定 (学習率を最後に 0 近くまで下げる) では確かめられなかった
- 毎回違う乱数で求めた検証用の損失は, 固定した乱数のものと最大 0.00095 (損失の 5 % 程度) ずれ, 最後の 10 エポックでも上下した

### 学習の再開

時刻 50, 生成を毎エポック, `data.num_workers: 0` の短い設定で, 1 エポックで止めて 2 エポックまで再開した学習と, 通しで 2 エポック学習した結果を比べた (mps, EMA あり).

| Scheduler の `T_max` | 再開した学習と通しの学習 |
| --- | --- |
| `${trainer.epochs}` (既定) | **エポック 2 から異なった**. 再開した学習の学習率は 0.0, 通しの学習は 0.0001 |
| 2 に固定 | `history.csv`, `ema_model.pt`, `ema_model.png` が完全に一致した |

- `T_max: ${trainer.epochs}` は, 1 エポックで止めた学習では `T_max = 1` で Scheduler が作られ, その状態がチェックポイントに保存される. 再開すると Scheduler の状態が引数ごと戻るので, config で `epochs: 2` にしても `T_max = 1` のまま動き, エポック 2 の学習率が 0 になった. 一方で再開した学習の `config.yaml` には `T_max: 2` と記録され, 実際の設定と食い違う
- `T_max` を固定すると完全に一致したので, EMA の状態 (`state_dict` と `resume` の上書き) は再開で正しく戻り, 検証の固定した乱数 (毎エポック `experiment.seed` に戻す) は再開の影響を受けない
- 09 で見つけたとおり, `num_workers > 0` (`persistent_workers=True`) では再開した学習は通しの学習と一致しないので, ここでは 0 にした

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- 再開すると Scheduler の状態が引数 (`T_max` など) ごとチェックポイントから戻るので, config で引数を変えても効かず, `config.yaml` の記録と実際の設定が食い違う (上の「学習の再開」)
- 損失に乱数を含む学習では, 検証用の損失がエポックごとに乱数で揺れ, ベストモデルの判定と Early Stopping が乱数に左右される. 検証の乱数を固定する仕組みが無いので, 専用の乱数生成器を自分で用意して `on_epoch_start` で戻した
- EMA の重みはチェックポイントにも `final_model.pt` にも入らない (`state_dict` と `resume` を上書きすれば加えられる). 検証や判定に EMA の重みを使うフックも無い. EMA は拡散モデルや画像の生成で一般的なので, `trainer.ema_decay` のような設定で, 更新, 保存, 再開, 検証に使うモデルの切り替えまで面倒を見てもらえると良い

### 最新の ML-Found での再実行 (2026-11-13)

上の結果は ML-Found `3aca743` で得たもの. 2026-11-13 に, 全ての実験をその後の ML-Found (`24f18c2`) で流し直した. 2 つの実験の `history.csv`, EMA の重み, 生成した画像は全て一致し, 実験 1 と 2 の `history.csv`, `final_model.pt`, `best_model.png` がバイト単位で一致することも確かめた.

- 学習時間 (`24f18c2`, 他の学習を並べずに流した): 2016 秒 (EMA あり), 2085 秒 (EMA なし). EMA なしの方は一部のエポックだけ遅く, 時間の差は EMA ではないという結論を裏付ける

「学習の再開」の確認も `e2df46e` で流し直した. 結果は以前と同じで, `T_max: ${trainer.epochs}` ではエポック 2 から異なり (再開した学習の学習率は 0.0), `T_max` を 2 に固定すると `history.csv`, `ema_model.pt`, `ema_model.png` が完全に一致した. 今は前者の再開で, `train.log` に「再開元の Scheduler の引数が config と違うので, 再開元の値を使います: T_max (再開元 1, config 2)」の警告が出る.
