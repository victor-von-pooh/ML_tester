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
│           ├── ema_eval.yaml       # 実験 3: EMA の重みで検証し, ベストモデルを選ぶ
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
- `extra_state_dict` と `load_extra_state_dict`: EMA の重みと更新回数をチェックポイントに加え, 再開時に戻す
- `eval_model`: `ema.eval: true` の場合は EMA のモデルを返し, 検証, ベストモデルの判定, `final_model.pt` に EMA の重みを使う. `step` の検証でもこのモデルで予測する
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
  eval: false
sampling:
  n_images: 64
  every_n_epochs: 10
```

experiment で次の差分を試す.

1. 差分なし (`configs/experiment/train.yaml`, EMA あり)
2. `ema.use: false` (`no_ema.yaml`)
3. `ema.eval: true` (`ema_eval.yaml`, EMA の重みで検証し, ベストモデルを選ぶ)

## 確かめる ML-Found の機能

- [x] 検証用の損失が `t` と `eps` の乱数によってばらつき, ベストモデルの判定が不安定にならないか. 検証時だけ乱数を固定する方法があるか → 毎回違う乱数で求めた `loss_random` はエポックごとに乱数で揺れるので, それで判定するとベストモデルが乱数に左右される. このタスクは専用の乱数生成器を `on_epoch_start` で `experiment.seed` に戻し, 毎エポック同じ乱数で求めた `loss` で判定する. ML-Found の `trainer.valid_seed` でも検証の乱数を固定できる (このタスクは使っていない)
- [x] `update_parameters` を上書きして EMA を更新できる
- [x] EMA のモデルはそのままではチェックポイントに入らず, 再開すると失われる → `extra_state_dict` と `load_extra_state_dict` で EMA の重みと更新回数をチェックポイントに加え, 再開時に戻せる (下の「学習の再開」). `ema.eval: false` ではベストモデルと `final_model.pt` は EMA でない方の重み, `ema.eval: true` では EMA の重みになる
- [x] 検証を EMA のモデルで行えるか → `eval_model` を上書きして EMA のモデルを返せば, 検証, ベストモデルの判定, `final_model.pt` に EMA の重みが使われる (`ema.eval: true`, 実験 3)
- [x] 生成に時間がかかる処理を `on_epoch_end` に入れたときに, 進捗バーやログが分かりにくくならないか → 生成 (1,000 ステップ, 64 枚) をするエポックはそのぶん長くなる. ML-Found の既定の進捗バーはエポック単位なので, 生成の間に限らずエポックの途中は止まって見える (`trainer.progress_bar_batches: true` で中の進み具合も出る). 既定の設定では生成は 3 回だけなので問題にならない
- [x] 学習時間が長い (MPS で 30 分以上) ので, 途中で止めて再開する流れ → 短い設定で確かめた (下の「学習の再開」)

## 期待する結果

- 50 エポックで, 生成画像の大半が数字と分かる (このタスクでは 30 エポックにした)
- EMA ありの方が生成画像がきれいになる

## 予想される問題

- EMA の重み, 生成用の状態など, モデル以外に保存したい状態をチェックポイントに加える仕組みが要る → ML-Found の `extra_state_dict` と `load_extra_state_dict` で加えられる
- 検証用の損失が乱数に依存するので, そのままでは `valid/loss` でのベストモデルの判定や Early Stopping の信頼性が低い → その通り (上のチェックボックスを参照)
- 検証に使うモデルを差し替える (EMA で検証する) フックが要る → ML-Found の `eval_model` で差し替えられる

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/11_diffusion/outputs/mnist_ddpm/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps で, 他の学習と並行した場合の目安)
python tasks/11_diffusion/train.py                                                       # 実験 1 (約 35 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/no_ema.yaml   # 実験 2 (約 35 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/ema_eval.yaml # 実験 3 (約 35 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml    # 動作確認 (2 から 6 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 既定の進捗バーはエポック単位なので, 1 エポック (約 1 分, smoke では他の学習と並行して約 3 分) の間は 0 のまま止まって見える. `trainer.progress_bar_batches: true` にするとエポックの中の進み具合も出る
- 動作確認 (smoke) は, ベストのエポック 2, `valid/loss` 約 0.0759 になれば正しく動いている (同じ環境なら完全に一致する. `data.num_workers=0` にすると 0.0757 になる). 出力先は約 70 MB あるので, 確認の後は消してよい
- 出力先には ML-Found の出力に加えて, `samples/` (学習中の生成画像), `best_model.png`, EMA を使う場合は `ema_model.pt` と `ema_model.png` ができる

## 結果のメモ

PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps での値.

### 30 エポックの実験 (実験 1 から 3)

1 回に 30 分以上かかるので, 数値は載せていない. 実行して次を見る.

- `history.csv` の `valid/loss` (毎エポック同じ乱数) と `valid/loss_random` (毎回違う乱数): `loss_random` の方がエポックごとの上下が多く, 判定に使うとベストのエポックが乱数で動く. `train.log` の「ベストモデルのエポック」
- `samples/epoch_010.png`, `epoch_020.png`, `epoch_030.png`, `best_model.png`, `ema_model.png`: 期待する結果は, 大半が数字と分かる画像. 学習率を CosineAnnealing で最後に 0 近くまで下げるので, 最後の重みと EMA の重みの差は小さく, 見た目の差も小さくなりうる
- 実験 1 と 2 の比べ方: EMA は学習中の重みを変えず, 生成はどちらもエポック 10, 20, 30 に同じ枚数と時刻の数で行うので, 学習の乱数の使い方も同じになる. そのため `history.csv`, `final_model.pt`, `best_model.png` は一致し, 違うのは EMA の重みでの生成画像 (`samples/`, `ema_model.png`) と `ema_model.pt` だけになるはず. 逆に言うと, 生成は学習と同じ全体の乱数を使うので, `sampling.every_n_epochs` を変えると学習そのものが変わる
- 実験 3 (`ema.eval: true`): 学習 (`train/loss`), EMA の重み, 生成画像は実験 1 と同じで, 違うのは検証の損失とベストモデルの重み (`final_model.pt`, `best_model.png`) だけになるはず. 検証の損失は EMA の重みのもので, 減衰率 0.999 では 1 エポック (422 回の更新) の時点では初期の重みが多く残るので, エポック 1 の `valid/loss` は実験 1 より大きい. ベストモデルがエポック 30 なら, `final_model.pt` は最後の EMA の重み (`ema_model.pt`) と一致する

### 学習の再開

動作確認の設定 (`smoke.yaml`: 時刻 50, 生成を毎エポック) に `data.num_workers=0` を足し, 1 エポックで止めて 2 エポックまで再開した学習と, 通しで 2 エポック学習した結果を比べた (mps, EMA あり).

```bash
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml trainer.epochs=1 data.num_workers=0
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml data.num_workers=0 \
    trainer.resume_from=outputs/mnist_ddpm/<日付>/<時刻>/last.pt trainer.resume_scheduler=config
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml data.num_workers=0
```

- `T_max: ${trainer.epochs}` なので, 1 エポックで止めた学習の Scheduler は `T_max = 1` で作られ, その状態がチェックポイントに保存される. `trainer.resume_scheduler: config` で再開すると, `train.log` に「再開元と config で Scheduler の引数が違うので, config の引数で作り直して 1 ステップ進めました: T_max (再開元 1, config 2)」が出て, エポック 2 の学習率は通しの学習と同じ 0.0001 になる
- この再開した学習と通しの学習で, `history.csv`, `ema_model.pt`, `ema_model.png`, `best_model.png`, `final_model.pt` が完全に一致した. EMA の状態 (`extra_state_dict`) は再開で正しく戻り, 検証の固定した乱数 (毎エポック `experiment.seed` に戻す) は再開の影響を受けない
- 既定 (`trainer.resume_scheduler: checkpoint`) で再開すると, Scheduler の状態が引数ごと再開元から戻るので `T_max = 1` のまま動き, 「再開元の Scheduler の引数が config と違うので, 再開元の値を使います」の警告が出て, エポック 2 の学習率が 0 になる. 通しの学習とは一致しない
- 09 で見つけたとおり, `num_workers > 0` (`persistent_workers=True`) では再開した学習は通しの学習と一致しないので, ここでは 0 にした

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した (ML-Found で解消したものは FEEDBACK.md から消し, 残っているものだけを載せる).

- 再開すると, 既定では Scheduler の状態が引数 (`T_max` など) ごとチェックポイントから戻るので, config で引数を変えても効かない (警告は出る). `${trainer.epochs}` のように引数がエポック数に従う config で, エポック数を延ばして再開する場合は `trainer.resume_scheduler: config` にする
- 損失に乱数を含む学習では, 検証用の損失がエポックごとに乱数で揺れ, ベストモデルの判定と Early Stopping が乱数に左右される. `trainer.valid_seed` で検証の乱数を固定できる
- EMA は ML-Found の設定には無く, 更新 (`update_parameters`), 保存と再開 (`extra_state_dict`, `load_extra_state_dict`), 検証に使うモデルの切り替え (`eval_model`) を継承して書く. EMA は拡散モデルや画像の生成で一般的なので, `trainer.ema_decay` のような設定で面倒を見てもらえると良い
