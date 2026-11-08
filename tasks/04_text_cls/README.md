# 04 テキスト分類

## 目的

長さの違う系列を辞書のバッチで扱うタスクで, ML-Found のバッチの転送とサンプル数の推定が正しく働くかを確かめる.
自前の小さなモデルと, Hugging Face の事前学習済みモデルの 2 段階で試し, 外部のモデルを ML-Found に載せたときの書きやすさも確かめる.

## データ

arXiv の論文のタイトルと要旨から, その論文の primary category を当てる.

| 項目 | 内容 |
| --- | --- |
| データセット | 直近 1 か月に投稿された ML 系の arXiv 論文のリスト |
| 取得方法 | **用意してもらう** (`data/arxiv/papers.csv` に置く) |
| 入力 | タイトルと要旨をつないだ文字列 |
| 正解 | primary category, 件数の多い上位 5 から 8 カテゴリに絞り, それ以外は除く |
| 分割 | 投稿日で分け, 古い 80 % を学習用, 新しい 20 % を検証用 (時間の漏れを防ぐ) |

### 用意してほしいデータ

| 列名 | 内容 | 例 | 必須 |
| --- | --- | --- | --- |
| `id` | arXiv ID | `2610.01234` | ◯ |
| `title` | タイトル | `Scaling Laws for ...` | ◯ |
| `abstract` | 要旨 | | ◯ |
| `primary_category` | 主カテゴリ | `cs.LG` | ◯ |
| `categories` | 全カテゴリ (空白区切り) | `cs.LG stat.ML cs.AI` | 推奨 |
| `published` | 初版の投稿日 | `2026-10-15` | 推奨 |

- 対象カテゴリ: `cs.LG`, `cs.CV`, `cs.CL`, `cs.AI`, `stat.ML`, `cs.RO`, `cs.IR`, `cs.NE` あたり (いずれかを `categories` に含む論文)
- 件数: 5,000 件以上あると良い. 1 か月で足りなければ期間を延ばしてかまわない
- 形式: UTF-8 の CSV (要旨に改行やカンマを含むので, クォートで囲む). JSON Lines でも可
- 改訂版 (v2 以降) は除き, 1 論文 1 行にする

`categories` があれば, 後から複数ラベル分類 (`BCEWithLogitsLoss`) にも広げられる.

## モデル

2 段階で用意する.

### 段階 1: 自前のモデル (ベースライン)

- 単語分割は空白と記号で区切るだけの簡単なものを自前で書き, 学習用データで語彙を作る (上位 30,000 語)
- `nn.Embedding → 平均プーリング (パディングをマスク) → nn.Linear` の fastText 風のモデル
- 依存ライブラリを増やさずに動く形

### 段階 2: Hugging Face の事前学習済みモデル

- `transformers` の `AutoModelForSequenceClassification` で微調整する
- モデルの候補: `distilbert-base-uncased` (軽い) または `allenai/scibert_scivocab_uncased` (論文向け)
- トークナイザも同じモデルのものを使い, 最大長は 256 トークン
- 依存に `transformers` が増えるので, 段階 2 は別のスクリプト (`train_hf.py`) にする

## step の設計

- `collate_fn` でパディングし, `batch` は `{"input_ids", "attention_mask", "labels"}` の辞書にする
- 段階 1: モデルに `input_ids` と `attention_mask` を渡し, `nn.CrossEntropyLoss()` で損失を求める
- 段階 2: Hugging Face のモデルの出力 (`outputs.loss`, `outputs.logits`) から損失と正解率を取り出す
- どちらも `{"loss": loss, "acc": acc}` を返す. 余力があれば `on_epoch_end` で macro F1 も記録する

## config

```yaml
# default (段階 1)
experiment:
  name: arxiv_cls
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 20
  monitor: valid/acc
  mode: max
  early_stopping:
    use: true
    patience: 5
data:
  path: ../../data/arxiv/papers.csv
  top_k_categories: 6
  valid_ratio: 0.2
  vocab_size: 30000
  max_len: 400
  batch_size: 64
model:
  embed_dim: 128
```

段階 2 では `optimizer.name: AdamW`, `lr: 2.0e-5`, `weight_decay: 0.01` とし, warmup 付きの線形減衰の Scheduler を使う.

## 確かめる ML-Found の機能

- [ ] 辞書のバッチが `move_to_device` で丸ごとデバイスに移る
- [ ] 辞書のバッチから `get_batch_size` がサンプル数を正しく推定する (最初に見つかったテンソルの 0 次元目)
- [ ] Hugging Face のモデルを `Trainer` にそのまま渡せる (`nn.Module` なので渡せるはず)
- [ ] `transformers.get_linear_schedule_with_warmup` で作った Scheduler (中身は `LambdaLR`) を `Trainer` に渡せる
- [ ] Hugging Face の warmup に必要な総ステップ数を config から決めにくくないか
- [ ] モデルが大きい場合のチェックポイントの保存時間と容量 (毎エポック `last.pt` に Optimizer の状態とベストモデルの重みも入る)

## 期待する結果

- 段階 1: 検証用データの正解率 0.6 から 0.7 程度 (カテゴリ同士の境界が曖昧なため高くはならない)
- 段階 2: 段階 1 より数ポイント高い

## 予想される問題

- `last.pt` にベストモデルの重みと Optimizer の状態が入るので, 事前学習済みモデルではファイルが大きく (DistilBERT で 1 GB 近く), 毎エポックの保存に時間がかかる
- warmup の総ステップ数は 02 の `OneCycleLR` と同じく, config だけでは決めにくい
- Hugging Face のモデルは `labels` を渡すと自分で損失を計算するので, ML-Found の `step` と役割が重なる

## 実行方法

```bash
python tasks/04_text_cls/train.py       # 段階 1
python tasks/04_text_cls/train_hf.py    # 段階 2 (pip install transformers が必要)
```

## 結果のメモ

(実行後に記入)
