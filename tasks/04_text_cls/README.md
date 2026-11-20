# 04 テキスト分類

## 目的

長さの違う系列を辞書のバッチで扱うタスクで, ML-Found のバッチの転送とサンプル数の推定が正しく働くかを確かめる.
自前の小さなモデルと, Hugging Face の事前学習済みモデルの 2 段階で試し, 外部のモデルを ML-Found に載せたときの書きやすさも確かめる.

## ディレクトリ構成

```
04_text_cls/
├── configs/
│   ├── default/
│   │   ├── train.yaml              # 段階 1 (train.py) の基準の config
│   │   └── hf.yaml                 # 段階 2 (train_hf.py) の基準の config
│   └── experiment/
│       ├── train.yaml              # 段階 1 の普段使う差分 (空)
│       ├── hf.yaml                 # 段階 2 の普段使う差分 (空)
│       └── examples/
│           ├── hf_smoke.yaml       # 段階 2 の動作確認用 (件数を絞って 1 エポック)
│           └── hf_config_warmup.yaml   # 段階 2 の Scheduler を ML-Found の config の warmup で作る
├── data.py                         # CSV の読み込み, ラベルの分解, 単語の分割と語彙, パディング
├── model.py                        # FastTextClassifier
├── train.py                        # 段階 1: MultiLabelTrainer (Trainer を継承), 学習の実行
├── train_hf.py                     # 段階 2: DistilBERT の微調整
└── outputs/                        # arxiv_cls/ と arxiv_hf/ に実験ごとに作られる (git の管理外)
```

## データ

arXiv の論文のタイトルと要旨から, その論文がどの分野に属するかを当てる.

| 項目 | 内容 |
| --- | --- |
| データセット | arXiv の論文 212,561 件 (2022 年 1 月から 2025 年 12 月の投稿) |
| 取得方法 | **用意してもらったもの**. `data/arxiv/arXiv.csv` に置く (git の管理外, 約 280 MB) |
| 列 | `Title`, `Abstract`, `URL` (`http://arxiv.org/abs/2201.00260v2` の形), `Label` (1 から 7 の整数) |
| 入力 | タイトルと要旨をつないだ文章 |
| 正解 | 3 つの分野 (量子物理, 機械学習, 最適化) のそれぞれに属するかどうか (複数ラベル分類) |
| 分割 | URL の arXiv ID の先頭 4 桁 (投稿年月) で分け, 2025 年 6 月までを学習用 (182,416 件), 2025 年 7 月以降を検証用 (30,145 件) |

### ラベルの読み方

`Label` の定義はデータに付いていないので, 3 つの分野を表すビットの和と **推測して** 使う. 各ラベルの論文のタイトル (量子, 学習, 最適化に関する語の出現) を見ると, 次の対応と考えられる.

| `Label` | 分野 | 件数 |
| --- | --- | --- |
| 1 | 量子物理 (quant-ph) | 49,340 |
| 2 | 機械学習 | 138,033 |
| 3 | 量子物理 + 機械学習 | 1,702 |
| 4 | 最適化 (math.OC) | 17,950 |
| 5 | 量子物理 + 最適化 | 353 |
| 6 | 機械学習 + 最適化 | 5,123 |
| 7 | 3 つ全て | 60 |

分野が重なる論文を含む複数ラベル分類として使う.
学習用と検証用での分野の割合はほぼ同じ (学習用: 量子物理 24.1 %, 機械学習 68.2 %, 最適化 11.2 %, 検証用: 24.9 %, 68.2 %, 9.9 %).

## モデル

### 段階 1: 自前のモデル (`train.py`)

- 単語の分割は, 小文字にして英数字の並び (単語の中のハイフンを含む) を切り出すだけの簡単なもの. 学習用データだけで語彙を作る (出現回数の多い 30,000 語, パディングと未知語を含む)
- 各論文の先頭 256 語を使う
- `FastTextClassifier`: `nn.Embedding` (128 次元) → パディングを除いた平均 → `Dropout` → `nn.Linear` で 3 つのロジット. パラメータ数 384 万 (ほとんどが埋め込み)

### 段階 2: Hugging Face の事前学習済みモデル (`train_hf.py`)

- `transformers` の `AutoModelForSequenceClassification` で `distilbert-base-uncased` (Apache 2.0) を微調整する. `problem_type="multi_label_classification"` で 3 つの出力にする. パラメータ数 6,696 万
- トークナイザも同じモデルのもので, 最大 256 トークン
- 重いので, 学習用から 20,000 件, 検証用から 5,000 件を乱数で取り出して使う
- 依存に `transformers` (Apache 2.0) が増える

## 学習の設計

- 段階 1: `collate` でパディングし, `batch` は `{"input_ids", "attention_mask", "labels"}` の辞書にする
- 段階 2: トークナイザの出力 (`BatchEncoding`) に `labels` を加えたものをそのままバッチにする
- 損失は分野ごとに独立した二値交差エントロピー (`nn.BCEWithLogitsLoss()`). 段階 2 は `labels` を渡すと Hugging Face のモデルが同じ損失を自分で計算するので, それを使う
- `Trainer` を継承した `MultiLabelTrainer` (段階 2 はそれを継承した `HFTrainer`) で
    - `step`: 損失と, 3 つの分野が全て当たった割合 (`exact`) を返し, 予測を貯める
    - `on_epoch_end`: 貯めた予測から分野ごとの F1 (`f1_qph`, `f1_ml`, `f1_oc`), その単純平均 (`macro_f1`), 全体で数えた F1 (`micro_f1`) を求める
- ベストモデルの判定は `valid/macro_f1` (`mode: max`)
- 段階 2 の Scheduler は, 既定では `transformers` の `get_linear_schedule_with_warmup` (最初の 10 % で warmup, その後は線形に減衰) で作って `Trainer` に渡す. 全体のステップ数がバッチ数で決まるため. experiment で Scheduler か warmup を書くと, 代わりに ML-Found が作る (下の「config の warmup」を参照)

### config

```yaml
# default (段階 1)
experiment:
  name: arxiv_cls
  seed: 42
  device: cpu              # 小さなモデルなので cpu
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 10
  monitor: valid/macro_f1
  mode: max
  early_stopping:
    use: true
    patience: 3
data:
  path: ../../data/arxiv/arXiv.csv
  valid_from: "2507"       # この年月 (yymm) 以降の投稿を検証用にする
  vocab_size: 30000
  max_len: 256
  batch_size: 256
model:
  embed_dim: 128
  dropout: 0.1
```

段階 2 (`configs/default/hf.yaml`) は, `optimizer.name: AdamW`, `lr: 2.0e-5`, `weight_decay: 0.01`, `scheduler.interval: step` (Scheduler 自体は `train_hf.py` で作る), `trainer.epochs: 2`, `warmup_ratio: 0.1`, `data.n_train: 20000`, `data.n_valid: 5000`, `data.batch_size: 32`, `model.name: distilbert-base-uncased`. `experiment.device` は既定の `auto` (この環境では mps).

## 確かめること

- [x] 辞書のバッチが `move_to_device` で丸ごとデバイスに移る → 段階 1 は cpu なので移動は起きない. 段階 2 の `BatchEncoding` は `.to` を持つので, `move_to_device` の `.to` の分岐で mps に丸ごと移る. `dict` の分岐を GPU で通す確認はしていない
- [x] 辞書のバッチから `get_batch_size` がサンプル数を正しく推定する (最初に見つかったテンソルの 0 次元目) → 段階 1 の `dict` も, 段階 2 の `BatchEncoding` (`dict` の子クラスではない `UserDict`) も Mapping として推定できる. 段階 2 の `step` は `batch_size` も明示している
- [x] Hugging Face のモデルを `Trainer` にそのまま渡せる
- [x] `transformers.get_linear_schedule_with_warmup` で作った Scheduler (中身は `LambdaLR`) を `Trainer` に渡せる. `train.log` には `Scheduler: LambdaLR` と出る
- [x] Hugging Face の warmup に必要な総ステップ数を config から決めにくくないか → ML-Found の `auto` と `scheduler.warmup_ratio` で config だけで書ける (`hf_config_warmup.yaml`). 既定の `train_hf.py` のように Scheduler を自分で作って渡すと, `config.yaml` の `scheduler` には残らず, `train.log` の `Scheduler: LambdaLR` だけになる
- [x] モデルが大きい場合のチェックポイントの保存時間と容量 → 容量は下の「実行方法」を参照. 保存の時間はログに分けて出ないので測っていない

## 期待する結果

- 段階 1: 検証用データで, 分野ごとの F1 が 0.8 以上
- 段階 2: 段階 1 より高い

### 予想される問題

- `last.pt` にベストモデルの重みと Optimizer の状態が入るので, 事前学習済みモデルではファイルが大きく (DistilBERT で 1 GB 近く), 毎エポックの保存に時間がかかる → 容量はその通り (下の「実行方法」). `trainer.checkpoint` の設定で減らせる. 時間は測っていない
- warmup の総ステップ数は 02 の `OneCycleLR` と同じく, config だけでは決めにくい → ML-Found の `auto` と `warmup_ratio` で config だけで書ける (下の「config の warmup」)
- Hugging Face のモデルは `labels` を渡すと自分で損失を計算するので, ML-Found の `step` と役割が重なる → その通りで, モデルが計算した損失をそのまま使う

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく. 段階 2 は `transformers` も必要.

```bash
pip install transformers==5.22.2
```

`train.py` と `train_hf.py` はどこから実行してもよく, 出力先は `tasks/04_text_cls/outputs/arxiv_cls/` (段階 1) と `tasks/04_text_cls/outputs/arxiv_hf/` (段階 2) になる.
データは `data/arxiv/arXiv.csv` に置く (別の場所に置く場合は config の `data.path` を変える).
段階 2 は初回に DistilBERT (約 270 MB) を Hugging Face からダウンロードする (ネットワークが必要, トークンは不要). 読み込むときに, 分類の出力層の重みが無いという表 (LOAD REPORT) が出るが, 新しく出力層を付けるので想定通り.
段階 2 は 1 回の実行で出力先に約 2.1 GB (`best.pt` が約 0.8 GB, `last.pt` が約 1.07 GB, `final_model.pt` が 270 MB) を書くので, 空き容量に注意する. `trainer.checkpoint.best_weights_only: true` と `last_with_best: false` にすると約 1.3 GB (`best.pt` 268 MB, `last.pt` 804 MB). 容量はデータの件数によらないので, 動作確認の設定でも同じになる.

```bash
# リポジトリのルートから
python tasks/04_text_cls/train.py       # 段階 1 (cpu で約 2 分)
python tasks/04_text_cls/train_hf.py    # 段階 2 (Apple Silicon の mps で約 30 分, cpu ではずっと長い)

# 段階 2 の動作確認 (件数を絞って 1 エポック, 数十秒)
python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_smoke.yaml
# 段階 2 の Scheduler を ML-Found の config の warmup で作る (約 30 分)
python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_config_warmup.yaml
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- `train_hf.py` は `configs/default/hf.yaml` と `configs/experiment/hf.yaml` を使う

## 結果のメモ

PyTorch 2.15.0, transformers 5.22.2, macOS (Apple Silicon), Python 3.13 での値.

### 段階 1

| モデル | デバイス | 学習用 / 検証用 | エポック数 | ベストのエポック | macro F1 | micro F1 | 完全一致 | F1 量子物理 | F1 機械学習 | F1 最適化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastText 風 | cpu | 182,416 / 30,145 | 10 | 7 | 0.929 | 0.964 | 0.939 | 0.984 | 0.975 | 0.828 |

- 1 エポック目から macro F1 が 0.84 で, エポック 7 で 0.929 になる. その後は学習用の指標だけが上がり, ベストのエポック 7 から 3 エポック改善しないので, `epochs: 10` の上限と同じエポック 10 で Early Stopping が掛かる
- 分野ごとでは, 最適化の F1 (0.83) が他の 2 つ (0.98, 0.98) より低い. 件数が少なく, 機械学習の論文との境界が曖昧なためと考えられる
- cpu で同じ結果になる (流し直すと `history.csv` が一致する)

### 段階 2

本番の設定 (`python tasks/04_text_cls/train_hf.py`) は mps で約 30 分かかるので, 数値はここに載せない. 流した場合は次を見る.

- `history.csv` の `valid/macro_f1`, `valid/micro_f1`, `valid/exact`, 分野ごとの `valid/f1_*` を段階 1 と比べる. 段階 2 は検証用が 5,000 件の部分集合なので, 公平に比べるには段階 1 のモデルも同じ 5,000 件で評価し直す
- エポック 2 でもまだ上がっているかどうか (上がっていれば, 学習用データを増やすかエポックを延ばす余地がある)
- mps の計算は決定的でないので, 同じ設定でも値が少しずれる
- `history.csv` の `lr` はエポックの開始時点の学習率なので, warmup が学習率 0 から始まる本番の設定ではエポック 1 が 0.0 になる. 更新ごとの学習率は `lr_steps.csv` に残る. 動作確認の設定 (`hf_smoke.yaml`) では warmup のステップ数が 0 になるので, エポック 1 から 2e-5 になる
- 1 エポックが長いので, `trainer.progress_bar_batches: true` でバッチ単位の進捗バーを出すと進み具合が分かる. 途中で止めたい場合は, 出力先の `STOP` に `now` と書くとエポックの途中でも止まる

### config の warmup

ML-Found の config には, Scheduler の前に学習率を線形に上げる warmup (`scheduler.warmup_steps` と `warmup_ratio`) を書ける. `get_linear_schedule_with_warmup` (warmup の後に 0 まで線形に下げる) は, 次のように書ける (`hf_config_warmup.yaml`). `train_hf.py` は, config に Scheduler か warmup を書いた場合は ML-Found に作らせ, 書かない場合は transformers の関数で作る.

```yaml
scheduler:
  name: LinearLR
  params:
    start_factor: 1.0
    end_factor: 0.0
    total_iters: auto      # warmup の後の残りのステップ数
  warmup_ratio: 0.1
  warmup_start_factor: 1.0e-8
```

学習用データを 640 件 (20 ステップ, warmup は 2 ステップ) にした動作確認で, 2 つの作り方を比べられる. この場合, トップレベルの `warmup_ratio` (transformers の関数に使う値) は使わない.

```bash
# transformers の Scheduler
python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_smoke.yaml data.n_train=640
# ML-Found の config の warmup
python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_config_warmup.yaml trainer.epochs=1 data.n_train=640 data.n_valid=32
```

- `train.log` に「LinearLR の引数を学習用データから補いました: warmup_steps=2, total_iters=18」と出て, `config.yaml` に補った値が残る. Scheduler は `SequentialLR` (warmup の `LinearLR` と減衰の `LinearLR`) になり, 中の Scheduler ごとの状態も `train.log` に残る
- 更新ごとの学習率 (`lr_steps.csv`) は, warmup の 2 ステップだけ `warmup_start_factor` (1e-8) の分ずれる (1 ステップ目は 0 と 2e-13, 2 ステップ目は 1e-05 と 1.00000001e-05). `warmup_start_factor` は 0 より大きい必要がある. 3 ステップ目以降は浮動小数の誤差の範囲で一致する
- エポック 1 の `train/loss` は 0.58847997 と 0.58847998 で, 7 桁目まで同じ (mps の計算の非決定性の範囲)
- warmup の回数の求め方は, transformers が切り捨て (`int`), ML-Found が Python の `round` (ちょうど半分は偶数へ丸める) なので, 総ステップ数 × 割合の小数部が 0.5 を超える場合と, 整数部が奇数で小数部がちょうど 0.5 の場合は 1 回違う. 段階 2 の本番 (20,000 件, バッチ 32, 2 エポックで 1,250 ステップ) では 125 回で同じになり, ML-Found は `warmup_steps=125, total_iters=1125` を補う

## 気付いたこと

ML-Found について気付いたこと. リポジトリをまたいで残っている項目は [TODO.md](../../TODO.md) にまとめた.

- Scheduler を自分で作って `Trainer` に渡すと, `config.yaml` の `scheduler` は `name: null` のままで, ML-Found が残す記録は `train.log` の `Scheduler: LambdaLR` だけになる. このタスクでは `warmup_ratio` を自分の config の項目にしたので, 引数は `config.yaml` に残る. config の warmup で書けば Scheduler も `config.yaml` に残る
