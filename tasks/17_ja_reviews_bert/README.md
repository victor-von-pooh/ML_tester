# 17 日本語のレビューを, 画面で組んだ BERT で分類する

## 目的

「機械学習を全て GUI 上で行う」流れを, 難しめの問題で通しで確かめる. データの分析 (ML-Fathom), モデルの作成 (ML-CreateArchitecture), 学習と評価 (ML-Found) の 3 つの段階をつなぐ.

- **問題**: 日本語の商品レビューから, **本体**の評価が良いか悪いかを当てる (2 クラス)
- **難しさ**: レビューは本体のほかに 1 から 3 つの点 (配送, 梱包, 説明書など) も, 本体と関係なく良い・悪いと書く. 句の順は無作為で, 述語には否定 (「良くない」) と悪いことの否定 (「悪くない」, 「不満はない」) を混ぜる. どの評価が本体のものかは文字の並びを読まないと分からないので, 文字を数えるだけのモデルでは解けない
- **データの汚れ**: Shift_JIS の csv, 投稿日時の列 (時系列と判定される), 答えから作った列 (星の数), ID, 同じ文の 2 回の投稿, 空の文, 極端に長い文, 半角と全角の混在 (「ﾘｭｯｸ」と「リュック」), クラスの偏り (7 : 3)
- **前処理**: ML-Fathom で「読み方」を「文 (review)」にし, 気付きから手順を作り (同じ文を落とす, 空の文を除く), 文字ごとの語彙で番号の並びにしたレシピ (`recipe.yaml`) を使う
- **モデル**: ML-CreateArchitecture の画面で, 部品から小さな BERT (`models/ja_bert.yaml`) と, 比べるための文字の平均のモデル (`models/char_bag.yaml`) を組んだ
- **学習**: ラベルの無いレビュー 10 万件で, 隠した文字を当てる事前学習 (MLM) をしてから, ラベル付きのレビューで微調整する

## ディレクトリ構成

```
17_ja_reviews_bert/
├── configs/
│   ├── default/train.yaml          # 基準の config (微調整. model.init: outputs/pretrained.pt)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── pretrain.yaml       # 事前学習 (pretrain.py で使う)
│           ├── scratch.yaml        # 事前学習をしない同じ BERT
│           ├── char_bag.yaml       # 文字の平均のモデル
│           ├── pretrain_subword.yaml   # 部分語のデータセットでの事前学習
│           ├── subword.yaml        # 部分語のデータセットでの微調整
│           ├── subword_scratch.yaml    # 部分語で, 事前学習をしない BERT
│           ├── subword_bag.yaml    # 部分語の平均のモデル
│           ├── pretrain_natural_char.yaml, natural_char.yaml, natural_char_scratch.yaml, natural_char_bag.yaml
│           │                       # 言い方を増やした版の文字ごとの事前学習, 微調整, 事前学習なし, 平均のモデル
│           ├── pretrain_natural_subword.yaml, natural_subword.yaml, natural_subword_scratch.yaml, natural_subword_bag.yaml
│           │                       # 同じく部分語
│           └── quick.yaml          # 1 エポックで終える (動作の確認用)
├── models/
│   ├── ja_bert.yaml                # ML-CreateArchitecture で組んだ BERT (部分モデル encoder_block を 2 回)
│   ├── char_bag.yaml               # ML-CreateArchitecture で組んだ文字の平均のモデル
│   ├── ja_bert_subword.yaml        # ja_bert の語の数と入力の長さを部分語に合わせたもの
│   ├── subword_bag.yaml            # char_bag の語の数を部分語に合わせたもの
│   ├── ja_bert_natural_char.yaml, ja_bert_natural_subword.yaml   # 言い方を増やした版に合わせた ja_bert
│   ├── ja_bert_natural_char_deep.yaml   # ja_bert_natural_char の encoder_block を 4 回にしたもの
│   ├── ja_bert_natural_char_wide.yaml   # ja_bert_natural_char の次元を 128 にしたもの
│   └── bag_natural_char.yaml, bag_natural_subword.yaml           # 言い方を増やした版に合わせた平均のモデル
├── make_data.py                    # data/reviews_ja.csv (ラベル付き) と data/reviews_ja_unlabeled.csv (ラベル無し) を作る. --natural で言い方を増やした版 (reviews_ja_natural*.csv) を, --rows で件数を変えた版を別に作る
├── recipe.yaml                     # ML-Fathom で作って保存した前処理のレシピ (文字ごと)
├── recipe_subword.yaml             # ML-Fathom が文の区切り方を比べて勧めた部分語のレシピ
├── recipe_natural_char.yaml, recipe_natural_subword.yaml   # 言い方を増やした版の, 文字ごとと部分語 (勧め) のレシピ
├── prepare.py                      # 分析と書き出し (outputs/dataset) と, ラベル無しの番号の並び (outputs/corpus.npy). --recipe, --out, --corpus, --source, --unlabeled で変えられる
├── tokens.py                       # 特別な番号 ([MASK], [CLS]), [CLS] を足す, 定義からモデルを作る
├── pretrain.py                     # 事前学習 (MLM). 重みを outputs/pretrained.pt に書く
├── train.py                        # 学習 (微調整) とテスト用の評価, 見分けの確かめ
└── outputs/                        # 書き出したデータ, 事前学習の重み, 実験 (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した日本語のレビュー (`make_data.py`. 乱数を固定) |
| 規模 | ラベル付き 12,200 行 (重複 200 行を含む), ラベル無し 100,000 件 |
| 列 | `review_id`, `posted` (投稿日時), `product`, `stars` (星の数), `review`, `label` (良い / 悪い) |
| 分割 | 層化して学習用 70 %, 検証用 20 %, テスト用 10 % (`recipe.yaml`. 同じ文と空の文を落とした後) |

用意してもらう必要があるデータは無い. ML-EDA と ML-ModelControl はルートの README の「導入」で入れる.

### データの分析 (ML-Fathom)

1. csv を登録すると, 投稿日時の列があるので**時系列**と判定され, 時刻の順に並べ替えられる. 星の数のリーク (`target_leak`), ID (`id_like`), 表記のゆれ (`mixed_forms`), 時刻の重なり (`irregular_interval`. 同じ時刻 200 か所は重複投稿) の気付きが出るが, 文の列は「文」の型として分析されるだけで, 学習に使えない
2. 右上の「読み方」を「文 (文の列: review)」にすると, 1 行 1 文の文のデータになる. 気付きは同じ文 (`duplicate_items`, 294 件) と空の文 (`empty_items`, 42 件) に変わる
3. 「気付きから手順を足す」で `drop_duplicates` (review) と `filter_rows` (length > 0) が, 「学習用の配列にする手順を作る」で文字ごとの語彙 (`vocabulary`, unit: char, 長さ 78) ができる. 分割を層化, テスト用を 0.1 にして保存した (`recipe.yaml`)

`prepare.py` は, このレシピで ML-EDA から同じデータセットを書き出し, ラベル無しのレビューに `transform_item` で同じ語彙をかける. 語彙は学習用の文で作った 70 文字 (詰め物と語彙に無い文字を含む).

## モデル

モデルの作成の段階 (ML-CreateArchitecture) の画面で組み, `models/` に置いた定義 (ML-ModelControl).

定義は画面の操作で作った (操作は Playwright で動かした).

- `encoder_block`: 入力 `x` (64 次元) と `pad_mask` → `nn.MultiheadAttention` (4 ヘッド. key_padding_mask に pad_mask) → 残差 (`op.Add`) と `nn.LayerNorm` → 全結合 (64 → 128 → 64, GELU) → 残差と `nn.LayerNorm`. 33.5K パラメータ
- `ja_bert`: `ids` → 語の埋め込み (72 = 語彙 70 + [MASK] + [CLS]. 0 は詰め物) と, `op.Positions` → 位置の埋め込み → 足して正規化 → `sub.encoder_block` を 2 回 (`op.Equal` で詰め物の印を作ってつなぐ) → `op.Index` で先頭の [CLS] → `nn.Linear` で 2 クラス (`logits`). 事前学習用に全ての位置の文字を当てる `nn.Linear` (`mlm`) も出す. 84.7K パラメータ
- `char_bag`: `nn.EmbeddingBag` (詰め物を除いた平均) → `nn.Linear`. 4.7K パラメータ

## 学習の設計

- `pretrain.py`: ラベル無しの番号の並びの先頭に [CLS] を足し, 詰め物と [CLS] 以外の 15 % を [MASK] にして元の文字を当てる (BERT の 80/10/10 の置き換えはせず, 全て [MASK]). 5 % を検証用にし, 検証の隠す位置は毎エポック同じにする (`trainer.valid_seed`). 損失と正解率は隠した文字の数で重み付けして平均する. ベストのエポックの重みを `pretrain.output` (既定 `outputs/pretrained.pt`) に書く
- `train.py`: 書き出したデータセットに [CLS] を足して分類を学習する. `model.init` の重み (事前学習の後) を読んでから始める. 学習後にテスト用の正解率と, 文字が全く同じで本体の評価だけが違う 3 組 (6 文) の予測 (「見分けの確かめ」) を `train.log` に残す. 最後に, 推論の段階 (ML-Inference) の一式を実験のディレクトリの `inference/` に作る ([CLS] は前処理の外で足すので, 一式の `prepend` に書く. 検証用の正解率が記録と合うかを `train.log` に残す. 19 で使う)
- どちらも ML-ModelControl の `build_model` で定義から作り, ML-Found の `Trainer` で学習する (AdamW, warmup 1 割と余弦の学習率)

## 確かめること

- [x] Shift_JIS の汚れた csv を ML-Fathom で文のデータとして読み, 気付きから前処理の手順を作って書き出せる
- [x] ML-CreateArchitecture の画面だけで, 自己注意のブロックを部分モデルにした BERT を組める (誤り 0 件, 定義を ML-ModelControl で組み立てられる)
- [x] ML-Found で, 画面で組んだ定義のモデルを事前学習し, 重みを読んで微調整できる
- [x] 文字を数えるだけのモデル (char_bag) では解けず, BERT は事前学習をすると解ける (テスト用の正解率と見分けの確かめ)
- [x] 新しい文に `transform_item` で同じ語彙をかけて予測できる
- [x] ML-Fathom が文の区切り方 (文字, 語, 部分語) を比べて勧めた部分語で書き出し, 語の数を合わせた同じ BERT を事前学習から微調整まで通せる
- [x] 言い方を増やした版 (`--natural`) では, ラベル付きを 4 倍にすると, 勧めた部分語の BERT だけが平均のモデルを大きく上回る (文字ごとの BERT は学べない. 下の「言い方を増やした版で比べた結果」)

## 期待する結果

- 2 で, 表として読んだ場合の気付き (`id_like`, `mixed_forms`, `target_leak`, `high_cardinality`) と, 文として読んだ場合の気付き (`duplicate_items`, `empty_items`) が出て, 形 `train [8305, 78]`, `valid [2373, 78]`, `test [1186, 78]` と事前学習用 `(100000, 78)` に書き出す
- 3 で, 検証用の隠した文字の正解率が 0.9 を超える
- 4 で, テスト用の正解率が 0.95 を超え, 見分けの確かめが 6/6 になる. 5 と 6 は 0.7 から 0.8 ほどで, 見分けの確かめは 3/6 (組の 2 つに同じ答え) になる
- 7 で, 語彙 500 (`<pad>`, `<unk>` を含む) の部分語で, 形 `train [8305, 16]` と事前学習用 `(100000, 16)` に書き出す
- 9 から 11 は, 部分語の平均のモデル (11) も含めてテスト用の正解率が 0.97 を超え, 見分けの確かめが 6/6 になる (下の「部分語で比べた結果」)
- 13 で, 語彙 2000 の部分語で形 `train [8372, 24]` に書き出す. 15 から 18 と文字ごとは, どれも 0.7 から 0.82 ほど
- 21 のテスト用の正解率が 0.94 ほどで見分けの確かめが 6/6, 22 は 0.87 ほど. 文字ごとは BERT も平均も 0.78 ほど

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/17_ja_reviews_bert/make_data.py     # 1. 元のデータ (data/reviews_ja.csv, data/reviews_ja_unlabeled.csv)
python tasks/17_ja_reviews_bert/prepare.py       # 2. 分析と書き出し (outputs/dataset, outputs/corpus.npy)
python tasks/17_ja_reviews_bert/pretrain.py --experiment configs/experiment/examples/pretrain.yaml   # 3. 事前学習 (outputs/pretrained.pt)
python tasks/17_ja_reviews_bert/train.py         # 4. 微調整
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/scratch.yaml    # 5. 事前学習なし
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/char_bag.yaml   # 6. 文字の平均のモデル
# 部分語 (ML-Fathom が勧めた区切り方) で比べる (1 の後なら 2 から 6 は要らない)
python tasks/17_ja_reviews_bert/prepare.py --recipe recipe_subword.yaml --out outputs/dataset_subword --corpus outputs/corpus_subword.npy   # 7. 書き出し
python tasks/17_ja_reviews_bert/pretrain.py --experiment configs/experiment/examples/pretrain_subword.yaml   # 8. 事前学習 (outputs/pretrained_subword.pt)
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/subword.yaml          # 9. 微調整
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/subword_scratch.yaml  # 10. 事前学習なし
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/subword_bag.yaml      # 11. 部分語の平均のモデル
# 言い方を増やした版 (元の版のデータと結果はそのまま)
python tasks/17_ja_reviews_bert/make_data.py --natural   # 12. data/reviews_ja_natural.csv, data/reviews_ja_natural_unlabeled.csv
python tasks/17_ja_reviews_bert/prepare.py --source data/reviews_ja_natural.csv --unlabeled data/reviews_ja_natural_unlabeled.csv --recipe recipe_natural_subword.yaml --out outputs/dataset_natural_subword --corpus outputs/corpus_natural_subword.npy   # 13. 部分語で書き出し
python tasks/17_ja_reviews_bert/pretrain.py --experiment configs/experiment/examples/pretrain_natural_subword.yaml   # 14. 事前学習
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword.yaml          # 15. 微調整 (12 エポック)
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword.yaml trainer.epochs=30   # 16. 微調整 (30 エポック)
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword_scratch.yaml  # 17. 事前学習なし
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword_bag.yaml      # 18. 部分語の平均のモデル
```

文字ごとで比べる場合は, 13 から 18 の `subword` を `char` に変える (13 のレシピは `recipe_natural_char.yaml`). 文字ごとの学習率を変える場合は 16 の後ろに `optimizer.params.lr=0.0003` を足す.

ラベル付きを 4 倍にした版 (事前学習はしない):

```bash
python tasks/17_ja_reviews_bert/make_data.py --natural --rows 48000   # 19. data/reviews_ja_natural_48000.csv (と data/reviews_ja_natural_unlabeled_48000.csv)
python tasks/17_ja_reviews_bert/prepare.py --source data/reviews_ja_natural_48000.csv --unlabeled data/reviews_ja_natural_unlabeled_48000.csv --recipe recipe_natural_subword.yaml --out outputs/dataset_natural48k_subword --corpus outputs/corpus_natural48k_subword.npy   # 20. 書き出し
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword_scratch.yaml data.dataset=outputs/dataset_natural48k_subword data.corpus=outputs/corpus_natural48k_subword.npy experiment.name=natural_subword_scratch_48k   # 21. 事前学習なしの BERT
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/natural_subword_bag.yaml data.dataset=outputs/dataset_natural48k_subword data.corpus=outputs/corpus_natural48k_subword.npy experiment.name=natural_subword_bag_48k   # 22. 平均のモデル
```

23. 文字ごとは, 20 から 22 の `subword` を `char` に変える. 4 層の BERT は, 文字ごとの 21 の後ろに `model.spec=models/ja_bert_natural_char_deep.yaml` を足し, `experiment.name` を `natural_char_deep_48k` にする (128 次元は `ja_bert_natural_char_wide.yaml` と `natural_char_wide_48k`).

CPU (Apple Silicon の Mac) で, 2 は 2 秒ほど, 3 は 2 分半ほど, 4 と 5 は 45 秒ほど, 6 は 5 秒ほど. 7 は 4 秒ほど, 8 は 45 秒ほど, 9 と 10 は 15 秒ほど, 11 は 5 秒ほど. 12 と 13 は数秒, 14 は 1 分半ほど (文字は 4 分ほど), 15 と 17 は 20 秒ほど (文字は 70 秒ほど), 16 は 50 秒ほど, 18 は 3 秒ほど (Python の起動を除く). 21 は 75 秒ほど (文字は 4 分半ほど, 4 層は 8 分ほど, 128 次元は 5 分ほど), 22 は 10 秒ほど. テスト用の正解率と見分けの確かめは, 端末ではなく各実験の `train.log` に残る.

7 から 11 は 1 の元のデータだけを使う (2 から 6 の文字ごとの出力は要らない). 9 は 8 の重み (`outputs/pretrained_subword.pt`) を読むので, 先に 8 を通す. 8 は事前学習の重みを `pretrain.output` に書くので, 動作の確認で 1 エポックだけ動かすと本番の重みを上書きする (3 も同じ). 残したい場合は `pretrain.output=outputs/pretrained_try.pt` のように書き出す先を変える. 8 から 11 の動作の確認には, コマンドの後ろに `trainer.epochs=1` を足す (1 エポックでも見分けの確かめが 6/6 になることがあるので, 確かめには使わない).

動作の確認には, 4 は `--experiment configs/experiment/examples/quick.yaml` (1 エポック), 3, 5, 6 は差分の config に `trainer.epochs=1` を足す (`--experiment` は 1 つだけ). 4 は事前学習の重み (`outputs/pretrained.pt`) が要るので, 先に 3 を通すか `model.init=null` を足す.

モデルの定義を画面で直す場合は, ML-CreateArchitecture の「開く」で `models/` の定義を開いて直し, 保存した定義を `models/` に写す (実験ごとの定義の残し方の決まりは, TASK_TYPES.md の 6 の節の 17).

## 結果のメモ

PyTorch 2.15.0, CPU (Apple Silicon の Mac) での値. 乱数を固定しているので, 実行するたびに同じ値になる. 事前学習を除く全ての実験で, 学習の最後の推論の一式の確かめ (検証用の分割を一式で推論した値が学習の記録と合うか) が合う.

| 実験 | モデル | 事前学習 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 見分けの確かめ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ja_bert | あり (MLM 4 エポック) | 12 | 10 | 0.9452 | 0.9612 | 6/6 |
| 5 (`scratch.yaml`) | ja_bert | なし | 12 | 10 | 0.7952 | 0.8010 | 3/6 |
| 6 (`char_bag.yaml`) | char_bag | なし | 30 | 20 | 0.7383 | 0.7378 | 3/6 |

- 事前学習 (3) の検証用の隠した文字の正解率は 0.9599 (4 エポック, ベストは 4)
- 多いクラス (良い) を答え続けた場合の正解率は約 0.7. 文字の平均のモデルはそれとほぼ同じで, 本体以外の評価に引きずられる. 見分けの確かめでは, 組の 2 つ (文字が全く同じ) に同じ答えを返すので 3/6 にしかならない
- 事前学習の無い BERT は, 最後のエポックで学習用の正解率 0.888 に対して検証用 0.788 と差が開き, 検証用は 0.8 で止まる (学習用の 8,305 件では, 「本体は」と述語の結び付きより個々の文を覚えてしまう). ラベル無しのレビューで文字の並びを先に学ぶと, 同じモデルでテスト用 0.96 になる (最後のエポックで学習用 0.984, 検証用 0.944)

### 部分語で比べた結果

ML-Fathom に csv を登録すると, 前処理のタブの「決めてほしいこと」に, 目的の列の候補 (label が先頭), 「列 review は文です」 (文として読む), 時刻の順でない分割が出る. 目的の列に label を選び, 文として読み, 「中身を配列にする手順がありません」の「分析から作る」を押すと, 分析が比べた区切り方の表 (下) から部分語を勧め, 中身の手順が `vocabulary` (unit: subword, length 16, max_tokens 500) になる. 分割と表の手順は `recipe.yaml` と同じにして保存したのが `recipe_subword.yaml`.

| 区切り方 | 語彙 | トークンの数 (平均 / 95 %) | 語彙に無い語 | 勧める長さ |
| --- | --- | --- | --- | --- |
| 文字 | 70 | 42.0 / 66 | 0.0 % | 72 |
| 語 | 70 | 42.0 / 66 | 0.0 % | 72 |
| 部分語 (勧める) | 500 | 7.6 / 11 | 0.0 % | 16 |
| 部分語 | 647 | 7.4 / 11 | 0.0 % | 16 |

(分析の見本 5,000 件を 8 対 2 に分けて測った値. 部分語の 2 段目は上限 2000 と 8000 で同じ語彙になったもの)

モデルの定義は, 画面で組んだ `ja_bert.yaml` と `char_bag.yaml` の語の数 (72 → 502 = 部分語 500 + [MASK] + [CLS]) と入力の形の長さ (80 → 17) だけを変えた (ML-CreateArchitecture の部品の引数の欄で変えるのと同じ. 140.2K と 32.3K パラメータ).

| 実験 | モデル | 事前学習 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 見分けの確かめ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 (`subword.yaml`) | ja_bert_subword | あり (MLM 4 エポック) | 12 | 6 | 1.0000 | 0.9983 | 6/6 |
| 10 (`subword_scratch.yaml`) | ja_bert_subword | なし | 12 | 8 | 0.9907 | 0.9874 | 6/6 |
| 11 (`subword_bag.yaml`) | subword_bag | なし | 30 | 19 | 0.9768 | 0.9764 | 6/6 |

- 事前学習 (8) の検証用の隠した部分語の正解率 (端末の「隠したトークンの正解率」) は 0.4350 (4 エポック, ベストは 4). 部分語は句の大きさなので, 前後から当てにくい
- 部分語にすると, 事前学習の無い BERT (10) も, 並びを見ない平均のモデル (11) も解ける. 部分語の区切りは「本体は良い」「配送は不満はない」のような句をまるごと 1 つのトークンにすることが多いので, どの評価が本体のものかが多くの場合トークン 1 つで分かる (「▁届いてすぐ見たが本体は」と「期待通りだ」のように分かれる文もある). 文字ごとでは難しかった「本体は」と述語の結び付きを, 区切り方が先に解いている
- これはこのデータが決まった句の組み合わせで作られているため (句の種類が少ない) で, 実際の文では部分語は句より細かくなる. 文字ごとの結果 (上) は「並びを読むモデルが要る」ことを確かめ, 部分語の結果は「区切り方の勧めに従うと, 同じ問題が短い並び (66 → 11) と軽い学習 (微調整 45 秒 → 15 秒) で解ける」ことを確かめたもの

### 言い方を増やした版で比べた結果

元の版は句の型が少なく, 部分語が「本体は良い」のような句をまるごと 1 つのトークンにするので, 並びを見ない平均のモデルでも解けた (上の 11). そこで, 言い方を増やした版を `make_data.py --natural` で別に作る (元の版はそのまま). 作り方 (本体の評価がラベル, 他の点は無作為, 否定と悪いことの否定, 汚れ) は同じで, 次の言い方を増やしてある.

- 主語: 本体の言い方 9 種 (「商品自体」「{商品}本体」「この{商品}自体」など. 商品の名前は表の `product` の列と同じ), 他の点の言い方を 3 種ずつ
- 助詞 (は, も, については, に関しては), 主語と述語の間の言葉 (「思ったより」「値段の割に」など. 良し悪しに関係しない)
- 述語: どの点にも使う 12 組 (「良かった」「気に入った」「文句なしだ」と, その逆) と, 点ごとに足す 2 から 3 組 (本体は「丈夫だ」「使いやすい」, 配送は「早かった」など. 本体に足す述語は他の点にも付くので, 述語だけでは本体か分からない)
- 句のつなぎ (「けど、」「。ただ、」「。一方で」など)

例: 「正直に言うと本体の作りはしっかりしているけど、包装については思ったより期待外れだ!」 (ラベルは 良い)

ML-Fathom で同じ手順 (目的の列 → 文として読む → 中身の手順を分析から作る, 分割は層化) をすると, 区切り方の比較は次のようになり, 部分語 (語彙 2000, 長さ 24) を勧める (`recipe_natural_subword.yaml`). 比べるために, 表の「文字」の「使う」で文字ごと (長さ 112) にしたものも保存してある (`recipe_natural_char.yaml`).

| 区切り方 | 語彙 | トークンの数 (平均 / 95 %) | 語彙に無い語 | 勧める長さ |
| --- | --- | --- | --- | --- |
| 文字 | 162 | 71.5 / 107 | 0.0 % | 112 |
| 語 | 162 | 71.5 / 107 | 0.0 % | 112 |
| 部分語 | 500 | 16.0 / 23 | 0.0 % | 24 |
| 部分語 (勧める) | 2,000 | 12.2 / 18 | 0.0 % | 24 |
| 部分語 | 6,277 | 11.3 / 17 | 0.0 % | 24 |

部分語の区切りは, 主語と述語を別のトークンにすることが多い (例: 「▁正直に言うと / 本体の作りは / しっかりしているけど / 、 / 包装 / については思ったより期待外れだ / !」). モデルの定義は `ja_bert.yaml` と `char_bag.yaml` の語の数 (文字 164, 部分語 2002) と入力の形の長さ (113, 25) だけを変えた.

| 区切り方 | モデル | 事前学習 | 学習率 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 見分けの確かめ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 部分語 | ja_bert (15) | あり | 1e-3 | 12 | 7 | 0.7981 | 0.8018 | 4/6 |
| 部分語 | ja_bert (16) | あり | 1e-3 | 30 | 9 | 0.8002 | 0.8152 | 2/6 |
| 部分語 | ja_bert (17) | なし | 1e-3 | 12 | 4 | 0.7809 | 0.7726 | 3/6 |
| 部分語 | 平均 (18) | なし | 1e-3 | 30 | 14 | 0.8089 | 0.8060 | 2/6 |
| 文字 | ja_bert | なし | 1e-3 | 12 | 8 | 0.7458 | 0.7375 | 3/6 |
| 文字 | 平均 | なし | 1e-3 | 30 | 21 | 0.7517 | 0.7450 | 3/6 |

- 部分語の事前学習 (14) の検証用の隠したトークンの正解率は 0.3064 (4 エポック, ベストは 4). 文字ごとの事前学習と, その重みからの微調整 (12 エポック, 30 エポック, 学習率 3e-4 と 1e-4) の値は載せていない. 文字ごとの事前学習は CPU で 4 分ほどかかる. 見るのは端末の「隠したトークンの正解率」と, 各実験の `train.log` のテスト用の正解率と見分けの確かめ
- ラベル付きの 8,372 件 (学習用) では, どのモデルも 0.70 から 0.82 で, BERT は平均のモデルとほぼ同じか下回る. 部分語の BERT は学習用の正解率が 1.0 になっても検証用は 0.79 ほどで (16 の最後のエポック), 個々の文を覚えてしまう

そこで, データが足りないのかを確かめるために, 同じ作り方でラベル付きを 4 倍にした版 (`make_data.py --natural --rows 48000`. 学習用 33,572 件) を作り, 同じレシピとモデルの定義 (語彙の大きさも文字 162, 部分語 2000 で同じ) で, 事前学習なしの BERT と平均のモデルを学習する (実行方法の 19 から 23).

| 区切り方 | モデル | ラベル付き | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 見分けの確かめ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 部分語 | ja_bert (事前学習なし) | 12,000 | 12 | 4 | 0.7809 | 0.7726 | 3/6 |
| 部分語 | ja_bert (事前学習なし) | 48,000 | 12 | 12 | 0.9416 | **0.9445** | **6/6** |
| 部分語 | 平均 | 12,000 | 30 | 14 | 0.8089 | 0.8060 | 2/6 |
| 部分語 | 平均 | 48,000 | 30 | 14 | 0.8718 | 0.8693 | 2/6 |
| 文字 | ja_bert (事前学習なし) | 12,000 | 12 | 8 | 0.7458 | 0.7375 | 3/6 |
| 文字 | ja_bert (事前学習なし) | 48,000 | 12 | 10 | 0.7878 | 0.7857 | 3/6 |
| 文字 | 平均 | 12,000 | 30 | 21 | 0.7517 | 0.7450 | 3/6 |
| 文字 | 平均 | 48,000 | 30 | 21 | 0.7836 | 0.7834 | 3/6 |
| 文字 | ja_bert 4 層 (事前学習なし) | 48,000 | 12 | 7 | 0.7853 | 0.7882 | 4/6 |
| 文字 | ja_bert 128 次元 (事前学習なし) | 48,000 | 12 | 6 | 0.7885 | 0.7865 | 3/6 |

- 部分語の BERT は, 4 倍のデータで 0.945 になり, 平均のモデル (0.869) を 7.5 ポイント上回り, 見分けの確かめも 6/6 になる. 12,000 件では, この大きさのモデルが主語と述語の結び付きを学ぶにはラベル付きのデータが足りない (ラベルの無い 10 万件での事前学習 (部分語の隠したトークンの正解率 0.31) では補えない)
- 文字ごとの BERT は, 4 倍のデータでも 0.786 で, 文字の平均のモデル (0.783) と同じ. データの件数を増やしても, 文字ごとの並び (113) からは主語と述語の結び付きを学べない. 層の数 (`ja_bert_natural_char_deep.yaml`. encoder_block を 4 回, 16.3 万パラメータ. 2 層は 9.7 万) と次元 (`ja_bert_natural_char_wide.yaml`. 64 から 128 に, 全結合の中間も 256 に. 32.4 万パラメータ) を変えても (実行方法の 23), テスト用の正解率は 0.7882 と 0.7865 で上がらない. 最後のエポックでは学習用の正解率が 0.908 と 0.935 まで上がるのに検証用は 0.768 で, 大きくしたモデルは個々の文を覚えるだけになる
- まとめ: 言い方を増やした版は, 並びを読むモデルが要り, ラベル付きのデータも多く要る問題. 区切り方の勧め (部分語) に従うと, 同じ BERT が学べる問題になり, 文字ごとでは学べない. 元の版の結果 (上) と合わせると, 区切り方の勧めは, 元の版では「短い並びで同じ問題が解ける」, 言い方を増やした版では「文字ごとでは解けない問題が解ける」ことにつながる

## 気付いたこと

無し
