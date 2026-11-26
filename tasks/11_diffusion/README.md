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

## 学習の設計

`Trainer` を継承した `DiffusionTrainer` で, 次のメソッドを上書きする.

- `step`: バッチごとに時刻 `t` とノイズ `eps` を乱数で選び, ノイズを加えた画像から `eps` を予測した MSE (`nn.MSELoss()`) を求める
    - 学習時はこの損失を `loss` とし, `loss_random` にも同じ値を入れる (`train/loss_random` は `train/loss` と常に同じ)
    - 検証時は, 毎エポック同じ乱数 (専用の乱数生成器を `on_epoch_start` で `experiment.seed` に戻す) で `t` と `eps` を選び直した損失を `loss` とし, 毎回違う乱数で求めた損失を `loss_random` として比較用に記録する
- `eval_model`: `ema.eval: false` (既定) の場合は学習しているモデルを返し, 検証, ベストモデルの判定, `final_model.pt` に学習しているモデルの重みを使う (EMA は生成だけに使う). `ema.eval: true` の場合は ML-Found の既定 (`super().eval_model()`. EMA を使う場合は EMA のモデル) を返す. `step` の検証でもこのモデルで予測する
- `on_epoch_end`: 検証の後, 10 エポックごとと最後のエポックに, EMA の重み (`self.ema_model.module`. EMA を使わない場合は `self.ema_model` が `None` なので学習中の重み) で 64 枚を生成して `samples/epoch_XXX.png` に保存する
- 学習後に, ML-Found が戻したベストモデルの重みで生成して `best_model.png` に, EMA を使う場合は最後の EMA の重み (`trainer.ema_model.module`) を `ema_model.pt` に保存し, それで生成して `ema_model.png` に保存する. EMA が一度も更新されていない場合 (EMA を使わなかった再開元から, 学習を進めずに終えた場合など) は保存しない

EMA (重みの指数移動平均) は ML-Found の設定 `trainer.ema.decay` で持つ. ML-Found がパラメータの更新ごとに EMA を更新し (`torch.optim.swa_utils.AveragedModel` と `get_ema_multi_avg_fn`), EMA の重みと更新回数をチェックポイントの `ema` に入れて再開時に戻すので, このタスクでは更新と保存を書かない. ML-Found の設定の EMA は既定で検証, ベストモデル, `final_model.pt` を EMA の重みにするので, EMA を生成だけに使う既定の形は `eval_model` の上書きで作る

### config

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
  ema:
    decay: 0.999
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
  eval: false
sampling:
  n_images: 64
  every_n_epochs: 10
```

experiment で次の差分を試す.

1. 差分なし (`configs/experiment/train.yaml`, EMA あり)
2. `trainer.ema.decay: null` (`no_ema.yaml`, EMA なし)
3. `ema.eval: true` (`ema_eval.yaml`, ML-Found の既定のとおり EMA の重みで検証し, ベストモデルを選ぶ)

## 確かめること

- [x] 検証用の損失が `t` と `eps` の乱数によってばらつき, ベストモデルの判定が不安定にならないか. 検証時だけ乱数を固定する方法があるか → 毎回違う乱数で求めた `loss_random` はエポックごとに乱数で揺れるので, それで判定するとベストモデルが乱数に左右される. このタスクは専用の乱数生成器を `on_epoch_start` で `experiment.seed` に戻し, 毎エポック同じ乱数で求めた `loss` で判定する. ML-Found の `trainer.valid_seed` でも検証の乱数を固定できる (このタスクは使っていない)
- [x] EMA を ML-Found の設定で持てるか → `trainer.ema.decay` を書くと ML-Found が更新ごとに EMA を更新する
- [x] EMA が再開で戻るか → ML-Found が EMA の重みと更新回数をチェックポイントの `ema` に入れて戻すので, 再開した学習は通しの学習と一致する (下の「学習の再開」)
- [x] EMA を持ったまま, 検証を学習しているモデルで行えるか → `eval_model` を上書きして学習しているモデルを返せば, 検証, ベストモデルの判定, `final_model.pt` に学習しているモデルの重みが使われ, EMA の重みは生成と `ema_model.pt` にだけ使う (既定の `ema.eval: false`). 上書きで ML-Found の既定を返すと, EMA の重みで検証し, ベストモデルと `final_model.pt` も EMA の重みになる (`ema.eval: true`, 実験 3)
- [x] 生成に時間がかかる処理を `on_epoch_end` に入れたときに, 進捗バーやログが分かりにくくならないか → 生成 (1,000 ステップ, 64 枚) をするエポックはそのぶん長くなる. ML-Found の既定の進捗バーはエポック単位なので, 生成の間に限らずエポックの途中は止まって見える (`trainer.progress_bar_batches: true` で中の進み具合も出る). 既定の設定では生成は 3 回だけなので問題にならない
- [x] 学習時間が長い (MPS で 30 分以上) ので, 途中で止めて再開する流れ → 短い設定で確かめた (下の「学習の再開」)

## 期待する結果

- 50 エポックで, 生成画像の大半が数字と分かる (このタスクでは 30 エポックにした)
- EMA ありの方が生成画像がきれいになる

### 予想される問題

- EMA の重み, 生成用の状態など, モデル以外に保存したい状態をチェックポイントに加える仕組みが要る → EMA は ML-Found の `trainer.ema.decay` がチェックポイントに入れる. 他の状態は ML-Found の `extra_state_dict` と `load_extra_state_dict` で加えられる
- 検証用の損失が乱数に依存するので, そのままでは `valid/loss` でのベストモデルの判定や Early Stopping の信頼性が低い → その通り (上のチェックボックスを参照)
- 検証に使うモデルを差し替える (EMA で検証する) フックが要る → ML-Found の `eval_model` で差し替えられる

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/11_diffusion/outputs/mnist_ddpm/<日付>/<時刻>/` になる.
MNIST が `data/torchvision/` に無い場合は, 初回にダウンロードが入る (ネットワークが必要).

```bash
# リポジトリのルートから (時間は Apple Silicon の mps で, 実験 1 から 3 を並行した場合の目安)
python tasks/11_diffusion/train.py                                                       # 実験 1 (約 90 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/no_ema.yaml   # 実験 2 (約 90 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/ema_eval.yaml # 実験 3 (約 90 分)
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml    # 動作確認 (2 から 6 分)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 既定の進捗バーはエポック単位なので, 1 エポック (約 1 分, smoke では他の学習と並行して約 3 分) の間は 0 のまま止まって見える. `trainer.progress_bar_batches: true` にするとエポックの中の進み具合も出る
- 動作確認 (smoke) は, ベストのエポック 2, `valid/loss` 約 0.0759 になれば正しく動いている (同じ環境なら完全に一致する. `data.num_workers=0` にすると 0.0757 になる). 出力先は約 70 MB あるので, 確認の後は消してよい
- 出力先には ML-Found の出力に加えて, `samples/` (学習中の生成画像), `best_model.png`, EMA を使う場合 (`trainer.ema.decay` が `null` でない場合) は `ema_model.pt` と `ema_model.png` ができる

## 結果のメモ

PyTorch 2.15.0, torchvision 0.29.2, macOS (Apple Silicon), Python 3.13, mps での値.

### 30 エポックの実験 (実験 1 から 3)

3 つを並行して流し, 学習時間は 1 つ 5,352 秒から 5,745 秒.

| 実験 | ベストのエポック | ベストの `valid/loss` | その `valid/loss_random` | エポック 1 の `valid/loss` | `train/loss` (エポック 30) |
| --- | --- | --- | --- | --- | --- |
| 1. 既定 (EMA は生成だけ) | 30 | 0.01756 | 0.01774 | 0.03255 | 0.01741 |
| 2. `no_ema.yaml` | 30 | 0.01756 | 0.01774 | 0.03255 | 0.01741 |
| 3. `ema_eval.yaml` (EMA で検証) | 30 | 0.01757 | 0.01775 | 0.34421 | 0.01741 |

- 検証の乱数: 実験 1 の `valid/loss` (毎エポック同じ乱数) が前のエポックより上がったのは 29 回のうち 3 回 (エポック 12, 21, 24) で, `valid/loss_random` (毎回違う乱数) は 10 回. `valid/loss_random` の最小はエポック 24 (0.01748) なので, それで判定するとベストのエポックが 24 になり, 乱数で動く. `valid/loss` の最小はエポック 30
- 実験 1 と 2: `history.csv`, `final_model.pt`, `best_model.png` が完全に一致した. EMA は学習中の重みを変えず, 生成はどちらもエポック 10, 20, 30 に同じ枚数と時刻の数で行うので, 学習の乱数の使い方も同じになるため. 違うのは EMA の重みでの生成画像 (`samples/`) と, 実験 1 にだけある `ema_model.pt` と `ema_model.png`. 逆に言うと, 生成は学習と同じ全体の乱数を使うので, `sampling.every_n_epochs` を変えると学習そのものが変わる
- 実験 3: `train/loss`, `ema_model.pt`, `ema_model.png`, `samples/` は実験 1 と完全に一致し, 違うのは検証の損失とベストモデルの重み (`final_model.pt`, `best_model.png`) だけ. 検証の損失は EMA の重みのもので, 減衰率 0.999 では 1 エポック (422 回の更新) の時点では初期の重みが多く残るので, エポック 1 の `valid/loss` は 0.34421 と実験 1 の約 10 倍. その後は 30 エポックの間に一度も上がらずに下がる. ベストモデルはエポック 30 なので, `final_model.pt` は最後の EMA の重み (`ema_model.pt`) と一致した. `train.log` に「評価に使うモデル: EMA (decay=0.999, use_buffers=False)」が出る (実験 1 は `eval_model` で学習しているモデルを返すので出ない)
- 生成画像: `samples/epoch_030.png`, `best_model.png`, `ema_model.png` は, どれも大半が数字と分かる. 学習率を CosineAnnealing で最後に 0 近くまで下げるので, 最後の重みと EMA の重みの見た目の差は小さく, EMA ありの方がはっきりきれいとは言えない. エポック 10 の EMA の重み (実験 1) と学習中の重み (実験 2) の生成画像も, 見た目の差は小さい

### 学習の再開

動作確認の設定 (`smoke.yaml`: 時刻 50, 生成を毎エポック) に `data.num_workers=0` を足し, 1 エポックで止めて 2 エポックまで再開した学習と, 通しで 2 エポック学習した結果を比べた (mps, EMA あり).

```bash
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml trainer.epochs=1 data.num_workers=0
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml data.num_workers=0 \
    trainer.resume_from=outputs/mnist_ddpm/<日付>/<時刻>/last.pt trainer.resume_scheduler=config
python tasks/11_diffusion/train.py --experiment configs/experiment/examples/smoke.yaml data.num_workers=0
```

- `T_max: ${trainer.epochs}` なので, 1 エポックで止めた学習の Scheduler は `T_max = 1` で作られ, その状態がチェックポイントに保存される. `trainer.resume_scheduler: config` で再開すると, `train.log` に「再開元と config で Scheduler の引数が違うので, config の引数で作り直して 1 ステップ進めました: T_max (再開元 1, config 2)」が出て, エポック 2 の学習率は通しの学習と同じ 0.0001 になる
- この再開した学習と通しの学習で, `history.csv`, `ema_model.pt`, `ema_model.png`, `best_model.png`, `final_model.pt` が完全に一致した (ベストのエポック 2, `valid/loss` 0.07568). EMA の状態 (ML-Found がチェックポイントの `ema` に入れる) は再開で正しく戻り, 検証の固定した乱数 (毎エポック `experiment.seed` に戻す) は再開の影響を受けない
- 既定 (`trainer.resume_scheduler: checkpoint`) で再開すると, Scheduler の状態が引数ごと再開元から戻るので `T_max = 1` のまま動き, 「再開元の Scheduler の引数が config と違うので, 再開元の値を使います」の警告が出て, エポック 2 の学習率が 0 になる. 通しの学習とは一致しない
- 09 で見つけたとおり, `num_workers > 0` (`persistent_workers=True`) では再開した学習は通しの学習と一致しないので, ここでは 0 にした

## 気付いたこと

ML-Found について気付いたこと. リポジトリをまたいで残っている項目は [TODO.md](../../TODO.md) にまとめた.

- 再開すると, 既定では Scheduler の状態が引数 (`T_max` など) ごとチェックポイントから戻るので, config で引数を変えても効かない (警告は出る). `${trainer.epochs}` のように引数がエポック数に従う config で, エポック数を延ばして再開する場合は `trainer.resume_scheduler: config` にする
- 損失に乱数を含む学習では, 検証用の損失がエポックごとに乱数で揺れ, ベストモデルの判定と Early Stopping が乱数に左右される. `trainer.valid_seed` で検証の乱数を固定できる
- ML-Found の設定の EMA (`trainer.ema.decay`) は, 検証, ベストモデル, `final_model.pt` を必ず EMA の重みにする. EMA を生成だけに使う場合は `eval_model` を上書きして学習しているモデルを返す. このとき EMA の重みは `final_model.pt` にならないので, 生成や保存には `trainer.ema_model.module` を自分で使う
