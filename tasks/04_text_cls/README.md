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
│           └── hf_smoke.yaml       # 段階 2 の動作確認用 (件数を絞って 1 エポック)
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

最初にこの README で想定していた「直近 1 か月の ML 系の論文と primary category」とは違う形だが, 分野が重なる論文を含む複数ラベル分類として使う.
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

## step と Trainer の設計

- 段階 1: `collate` でパディングし, `batch` は `{"input_ids", "attention_mask", "labels"}` の辞書にする
- 段階 2: トークナイザの出力 (`BatchEncoding`) に `labels` を加えたものをそのままバッチにする
- 損失は分野ごとに独立した二値交差エントロピー (`nn.BCEWithLogitsLoss()`). 段階 2 は `labels` を渡すと Hugging Face のモデルが同じ損失を自分で計算するので, それを使う
- `Trainer` を継承した `MultiLabelTrainer` (段階 2 はそれを継承した `HFTrainer`) で
    - `step`: 損失と, 3 つの分野が全て当たった割合 (`exact`) を返し, 予測を貯める
    - `on_epoch_end`: 貯めた予測から分野ごとの F1 (`f1_qph`, `f1_ml`, `f1_oc`), その単純平均 (`macro_f1`), 全体で数えた F1 (`micro_f1`) を求める
- ベストモデルの判定は `valid/macro_f1` (`mode: max`)
- 段階 2 の Scheduler は, 既定では `transformers` の `get_linear_schedule_with_warmup` (最初の 10 % で warmup, その後は線形に減衰) で作って `Trainer` に渡す. 全体のステップ数がバッチ数で決まるため (ML-Found の PR #35 の後は, experiment で Scheduler か warmup を書くと ML-Found が作る. 下の「config の warmup」を参照)

## config

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

## 確かめる ML-Found の機能

- [x] 辞書のバッチが `move_to_device` で丸ごとデバイスに移る → 段階 1 は cpu なので移動は起きていない. 段階 2 の `BatchEncoding` は `.to` を持つので, `move_to_device` の `.to` の分岐で mps に丸ごと移った. `dict` の分岐を GPU で通す確認はしていない
- [x] 辞書のバッチから `get_batch_size` がサンプル数を正しく推定する (最初に見つかったテンソルの 0 次元目) → 段階 1 の `dict` では推定できる. 段階 2 の `BatchEncoding` は `dict` の子クラスではない (`UserDict`) ので推定できず, 1 として扱われる. そのため段階 2 では `step` で `batch_size` を明示した (その後 ML-Found の PR #1 で `dict` 以外の Mapping も推定するようになった)
- [x] Hugging Face のモデルを `Trainer` にそのまま渡せる
- [x] `transformers.get_linear_schedule_with_warmup` で作った Scheduler (中身は `LambdaLR`) を `Trainer` に渡せる. `train.log` には `Scheduler: LambdaLR` と出る
- [x] Hugging Face の warmup に必要な総ステップ数を config から決めにくくないか → 決められなかった. 02 の `OneCycleLR` と同じく, バッチ数から `train_hf.py` で計算した. 02 と違い Scheduler を自分で作って渡したので, `config.yaml` には Scheduler の名前も引数も残らない (`3aca743` の時点. その後 ML-Found の PR #29 の `auto` と PR #35 の `warmup_ratio` で, config だけで書けるようになった)
- [x] モデルが大きい場合のチェックポイントの保存時間と容量 → `best.pt` と `last.pt` はそれぞれ 1.07 GB (モデル 268 MB に, Optimizer の状態とベストモデルの重みが加わる). 保存の時間はログに分けて出ないので測っていない

## 期待する結果

- 段階 1: 検証用データで, 分野ごとの F1 が 0.8 以上
- 段階 2: 段階 1 より高い → ほぼ同じだった (下の結果を参照)

(最初の README の「正解率 0.6 から 0.7」は, 主カテゴリを当てる多クラス分類を想定した値だったので, データに合わせて変えた)

## 予想される問題

- `last.pt` にベストモデルの重みと Optimizer の状態が入るので, 事前学習済みモデルではファイルが大きく (DistilBERT で 1 GB 近く), 毎エポックの保存に時間がかかる → 容量はその通りだった (1.07 GB). 時間は測っていない
- warmup の総ステップ数は 02 の `OneCycleLR` と同じく, config だけでは決めにくい → その通りだった
- Hugging Face のモデルは `labels` を渡すと自分で損失を計算するので, ML-Found の `step` と役割が重なる → その通りで, モデルが計算した損失をそのまま使った

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく. 段階 2 は `transformers` も必要.

```bash
pip install transformers==5.22.2
```

`train.py` と `train_hf.py` はどこから実行してもよく, 出力先は `tasks/04_text_cls/outputs/arxiv_cls/` (段階 1) と `tasks/04_text_cls/outputs/arxiv_hf/` (段階 2) になる.
データは `data/arxiv/arXiv.csv` に置く (別の場所に置く場合は config の `data.path` を変える).
段階 2 は初回に DistilBERT (約 270 MB) を Hugging Face からダウンロードする (ネットワークが必要, トークンは不要). 読み込むときに, 分類の出力層の重みが無いという表 (LOAD REPORT) が出るが, 新しく出力層を付けるので想定通り.
段階 2 は 1 回の実行で出力先に約 2.1 GB (`best.pt` が約 0.8 GB, `last.pt` が約 1.07 GB, `final_model.pt` が 270 MB) を書くので, 空き容量に注意する. `trainer.checkpoint.best_weights_only: true` と `last_with_best: false` にすると約 1.3 GB (`best.pt` 268 MB, `last.pt` 804 MB). 下の結果のときは `best.pt` と `last.pt` が 1.07 GB ずつだった.

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

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, transformers 5.22.2, macOS (Apple Silicon), Python 3.13 で実行.

| 段階 | モデル | デバイス | 学習用 / 検証用 | エポック数 | ベストのエポック | macro F1 | micro F1 | 完全一致 | F1 量子物理 | F1 機械学習 | F1 最適化 | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fastText 風 | cpu | 182,416 / 30,145 | 10 | 7 | 0.929 | 0.964 | 0.939 | 0.984 | 0.975 | 0.828 | 94 秒 |
| 2 | DistilBERT | mps | 20,000 / 5,000 | 2 | 2 | 0.922 | 0.959 | 0.936 | 0.976 | 0.973 | 0.816 | 1784 秒 (他の学習と GPU を分け合った状態) |
| (参考) 1 を段階 2 の検証用 5,000 件で評価 | fastText 風 | cpu | 182,416 / 5,000 | - | 7 | 0.921 | 0.960 | 0.933 | 0.981 | 0.973 | 0.810 | - |

- 段階 1 は 1 エポック目から macro F1 が 0.84 で, エポック 7 で 0.929 になった. その後は学習用の指標だけが上がり, ベストのエポック 7 から 3 エポック改善しなかったので, `epochs: 10` の上限と同じエポック 10 で Early Stopping が掛かった
- 分野ごとでは, 最適化の F1 (0.83) が他の 2 つ (0.98, 0.98) より低い. 件数が少なく, 機械学習の論文との境界が曖昧なためと考えられる
- 段階 2 (DistilBERT) は, 学習用 20,000 件で 2 エポック学習して macro F1 0.922 だった. 段階 1 のモデルを同じ検証用 5,000 件で評価し直すと 0.921 で, ほぼ同じだった. 段階 2 は学習用データが 9 分の 1 でも並んだが, 期待した「段階 1 より高い」にはならなかった. 段階 2 はエポック 2 でもまだ上がっており, 学習用データを増やすかエポックを延ばせば上回る可能性がある
- 段階 2 は, 他の学習と GPU を分け合った状態でエポック 1 が約 17.5 分, 2 エポックで約 30 分 (1,784 秒) かかった
- 段階 2 の `history.csv` の `lr` は, エポック 1 が 0.0 だった. warmup は学習率 0 から始まり, ML-Found はエポックの開始時点の学習率だけを記録するため (02 の FB の具体例). 動作確認の設定のように warmup のステップ数が 0 になる場合は, エポック 1 から 2e-5 になる

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- Hugging Face のトークナイザの出力 (`BatchEncoding`) は `dict` の子クラスではないので, ML-Found の `get_batch_size` がサンプル数を見つけられず 1 として扱う. `step` で返す指標 (`loss`, `exact`) がバッチ数の平均になり, 小さなバッチがサンプル数の割に重く効く (段階 2 の検証用では 8 件のバッチが 32 件のバッチと同じ重み). `on_epoch_end` で求める F1 には影響しない. `move_to_device` は `BatchEncoding` が `.to` を持つので正しく動く (`3aca743` の時点. その後 ML-Found の PR #1 で `dict` 以外の Mapping のバッチもサンプル数を推定するようになり, PR #21 で推定できずに 1 とした場合は `train.log` に警告が 1 回出るようになった)
- warmup 付きの Scheduler の総ステップ数がバッチ数で決まるので, Scheduler を自分で作って `Trainer` に渡した. この場合 `config.yaml` の `scheduler` は `name: null` のままで, ML-Found が残す記録は `train.log` の `Scheduler: LambdaLR` だけになる. このタスクでは `warmup_ratio` を自分の config の項目にしたので, 引数は `config.yaml` に残った (`3aca743` の時点. その後 ML-Found の PR #35 で warmup を config に書けるようになった. 下の「config の warmup」を参照)
- 事前学習済みモデルでは `best.pt` と `last.pt` がそれぞれ 1 GB を超える. ベストモデルの重みを `last.pt` にも入れるので, 2 つで 2 GB を超える (`3aca743` の時点. その後 ML-Found の PR #10 で `best.pt` がベストモデルの重みを重ねて持たなくなり, `trainer.checkpoint.last_with_best: false` で `last.pt` からベストモデルの重みを外し, `best_weights_only: true` で `best.pt` を重みだけにできるようになった)
- 進捗バーがエポック単位なので, 1 エポックが 17.5 分かかった段階 2 では, エポックの中の進み具合が全く分からない (02 で予想した問題が実際に起きた) (`3aca743` の時点. その後 ML-Found の PR #1 で `trainer.progress_bar_batches` でバッチ単位の進捗バーを出せるようになり, PR #11 で `STOP` に `now` と書くとエポックの途中でも止められるようになった)

### 最新の ML-Found での再実行 (2026-11-13)

上の結果は ML-Found `3aca743` で得たもの. 2026-11-13 に, 全ての実験をその後の ML-Found (`24f18c2`) で流し直した. 段階 1 (cpu) は `history.csv` が一致した (85 秒). 段階 2 (DistilBERT, mps) は mps の計算の非決定性で小さな差があった.

| 指標 | 以前 | 今回 |
| --- | --- | --- |
| macro F1 | 0.922 | 0.922 (0.92170) |
| micro F1 | 0.959 | 0.959 (0.95896) |
| 完全一致 | 0.936 | 0.935 |
| F1 量子物理, 機械学習, 最適化 | 0.976, 0.973, 0.816 | 0.976, 0.973, 0.816 |
| 学習時間 | 1784 秒 | 1506 秒 (他の学習と GPU を分け合っていない) |

- エポック 1 は約 13.0 分 (以前は 17.5 分), エポック 2 は約 12.1 分で以前と同じ. 以前の遅さは GPU を分け合ったエポック 1 だけだった
- `best.pt` は 1.07 GB から 0.80 GB になった (ベストモデルの重みを重ねて持たなくなったため). `last.pt` は 1.07 GB のまま
- 段階 1 とほぼ同じという結論は変わらない

### config の warmup (2026-11-13)

ML-Found の PR #35 で, Scheduler の前に学習率を線形に上げる warmup (`scheduler.warmup_steps` と `warmup_ratio`) を config に書けるようになった. `get_linear_schedule_with_warmup` (warmup の後に 0 まで線形に下げる) は, 次のように書ける (`hf_config_warmup.yaml`). `train_hf.py` は, config に Scheduler か warmup を書いた場合は ML-Found に作らせ, 書かない場合はこれまでどおり transformers の関数で作る.

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

学習用データを 640 件 (20 ステップ, warmup は 2 ステップ) にした動作確認で, 2 つの作り方を ML-Found `1c5c47d` で比べた. コマンドは `python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_smoke.yaml data.n_train=640` (transformers) と `python tasks/04_text_cls/train_hf.py --experiment configs/experiment/examples/hf_config_warmup.yaml trainer.epochs=1 data.n_train=640 data.n_valid=32` (ML-Found). この場合, トップレベルの `warmup_ratio` (transformers の関数に使う値) は使わない.

- `train.log` に「LinearLR の引数を学習用データから補いました: warmup_steps=2, total_iters=18」と出て, `config.yaml` に補った値が残った. Scheduler は `SequentialLR` (warmup の `LinearLR` と減衰の `LinearLR`) になり, 中の Scheduler ごとの状態も `train.log` に残る
- 更新ごとの学習率 (`lr_steps.csv`) は, warmup の 2 ステップだけ `warmup_start_factor` (1e-8) の分ずれた (1 ステップ目は 0 と 2e-13, 2 ステップ目は 1e-05 と 1.00000001e-05). `warmup_start_factor` は 0 より大きい必要がある. 3 ステップ目以降は浮動小数の誤差の範囲で一致した
- エポック 1 の `train/loss` は 0.5884799764 と 0.5884799689 で, 7 桁目まで同じだった
- warmup の回数の求め方は, transformers が切り捨て (`int`), ML-Found が Python の `round` (ちょうど半分は偶数へ丸める) なので, 総ステップ数 × 割合の小数部が 0.5 を超える場合と, 整数部が奇数で小数部がちょうど 0.5 の場合は 1 回違う. 段階 2 の本番 (20,000 件, バッチ 32, 2 エポックで 1,250 ステップ) では 125 回で同じ

本番の設定 (20,000 件, 2 エポック, 1,250 ステップ) でも `hf_config_warmup.yaml` で流し, transformers の Scheduler で流した以前の結果 (2026-11-13, ML-Found `24f18c2`) と比べた.

- 補った値は `warmup_steps=125, total_iters=1125` で, 更新ごとの学習率は 1,250 ステップ全てで差が 2e-13 以下だった (最初のステップの 0 と 2e-13)
- 結果はほぼ同じだった. ベストのエポックは同じ 2

| | `train/loss` (エポック 2) | `valid/loss` | `valid/macro_f1` | `valid/micro_f1` | `valid/exact` | `valid/f1_oc` | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformers の Scheduler | 0.0700 | 0.0802 | 0.9217 | 0.9590 | 0.9354 | 0.8157 | 1506 秒 |
| ML-Found の config の warmup | 0.0699 | 0.0802 | 0.9222 | 0.9590 | 0.9356 | 0.8173 | 1517 秒 |

- 違いは, 学習率の 1e-8 倍のずれ (warmup の 2 ステップ) と mps の計算の非決定性の範囲. 以前の流し直しでも, 段階 2 は mps で同じ結果にならなかった

