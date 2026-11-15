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
│           └── quick.yaml          # 1 エポックで終える (動作の確認用)
├── models/
│   ├── ja_bert.yaml                # ML-CreateArchitecture で組んだ BERT (部分モデル encoder_block を 2 回)
│   └── char_bag.yaml               # ML-CreateArchitecture で組んだ文字の平均のモデル
├── make_data.py                    # data/reviews_ja.csv (ラベル付き) と data/reviews_ja_unlabeled.csv (ラベル無し) を作る
├── recipe.yaml                     # ML-Fathom で作って保存した前処理のレシピ
├── prepare.py                      # 分析と書き出し (outputs/dataset) と, ラベル無しの番号の並び (outputs/corpus.npy)
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

## 3 つの段階でしたこと

### データの分析 (ML-Fathom)

1. csv を登録すると, 投稿日時の列があるので**時系列**と判定され, 時刻の順に並べ替えられる. 星の数のリーク (`target_leak`), ID (`id_like`), 表記のゆれ (`mixed_forms`) の気付きが出るが, 文の列は「文」の型として分析されるだけで, 学習に使えない
2. 右上の「読み方」を「文 (文の列: review)」にすると, 1 行 1 文の文のデータになる. 気付きは同じ文 (`duplicate_items`, 294 件) と空の文 (`empty_items`, 42 件) に変わる
3. 「気付きから手順を足す」で `drop_duplicates` (review) と `filter_rows` (length > 0) が, 「学習用の配列にする手順を作る」で文字ごとの語彙 (`vocabulary`, unit: char, 長さ 78) ができる. 分割を層化, テスト用を 0.1 にして保存した (`recipe.yaml`)

`prepare.py` は, このレシピで ML-EDA から同じデータセットを書き出し, ラベル無しのレビューに `transform_item` で同じ語彙をかける. 語彙は学習用の文で作った 70 文字 (詰め物と語彙に無い文字を含む).

### モデルの作成 (ML-CreateArchitecture)

画面の操作で 2 つの定義を作った (操作は Playwright で動かし, 動画にした).

- `encoder_block`: 入力 `x` (64 次元) と `pad_mask` → `nn.MultiheadAttention` (4 ヘッド. key_padding_mask に pad_mask) → 残差 (`op.Add`) と `nn.LayerNorm` → 全結合 (64 → 128 → 64, GELU) → 残差と `nn.LayerNorm`. 33.5K パラメータ
- `ja_bert`: `ids` → 語の埋め込み (72 = 語彙 70 + [MASK] + [CLS]. 0 は詰め物) と, `op.Positions` → 位置の埋め込み → 足して正規化 → `sub.encoder_block` を 2 回 (`op.Equal` で詰め物の印を作ってつなぐ) → `op.Index` で先頭の [CLS] → `nn.Linear` で 2 クラス (`logits`). 事前学習用に全ての位置の文字を当てる `nn.Linear` (`mlm`) も出す. 84.7K パラメータ
- `char_bag`: `nn.EmbeddingBag` (詰め物を除いた平均) → `nn.Linear`. 4.7K パラメータ

### 学習と評価 (ML-Found)

- `pretrain.py`: ラベル無しの番号の並びの先頭に [CLS] を足し, 詰め物と [CLS] 以外の 15 % を [MASK] にして元の文字を当てる. 5 % を検証用にする
- `train.py`: 書き出したデータセットに [CLS] を足して分類を学習する. `model.init` の重み (事前学習の後) を読んでから始める. 学習後にテスト用の正解率と, 文字が全く同じで本体の評価だけが違う 3 組 (6 文) の予測 (「見分けの確かめ」) を `train.log` に残す
- どちらも ML-ModelControl の `build_model` で定義から作り, ML-Found の `Trainer` で学習する (AdamW, warmup 1 割と余弦の学習率)

## 確かめること

- [x] Shift_JIS の汚れた csv を ML-Fathom で文のデータとして読み, 気付きから前処理の手順を作って書き出せる
- [x] ML-CreateArchitecture の画面だけで, 自己注意のブロックを部分モデルにした BERT を組める (誤り 0 件, 定義を ML-ModelControl で組み立てられる)
- [x] ML-Found で, 画面で組んだ定義のモデルを事前学習し, 重みを読んで微調整できる
- [x] 文字を数えるだけのモデル (char_bag) では解けず, BERT は事前学習をすると解ける (テスト用の正解率と見分けの確かめ)
- [x] 新しい文に `transform_item` で同じ語彙をかけて予測できる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/17_ja_reviews_bert/make_data.py     # 1. 元のデータ (data/reviews_ja.csv, data/reviews_ja_unlabeled.csv)
python tasks/17_ja_reviews_bert/prepare.py       # 2. 分析と書き出し (outputs/dataset, outputs/corpus.npy)
python tasks/17_ja_reviews_bert/pretrain.py --experiment configs/experiment/examples/pretrain.yaml   # 3. 事前学習 (outputs/pretrained.pt)
python tasks/17_ja_reviews_bert/train.py         # 4. 微調整
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/scratch.yaml    # 5. 事前学習なし
python tasks/17_ja_reviews_bert/train.py --experiment configs/experiment/examples/char_bag.yaml   # 6. 文字の平均のモデル
```

CPU (Apple Silicon の Mac) で, 2 は 2 秒ほど, 3 は 2 分半ほど, 4 と 5 は 45 秒ほど, 6 は 5 秒ほど. 動作の確認には, 4 から 6 に `--experiment configs/experiment/examples/quick.yaml` (1 エポック) を使う. 4 は事前学習の重み (`outputs/pretrained.pt`) が要るので, 先に 3 を通すか `model.init=null` を足す.

モデルの定義を画面で直す場合は, ML-CreateArchitecture の「開く」で `models/` の定義を開いて直し, 保存した定義を `models/` に写す (定義を ML-CreateArchitecture の置き場と結ぶ方法は, VISION.md で保留中).

## 期待する結果

- 2 で, 表として読んだ場合の気付き (`id_like`, `mixed_forms`, `target_leak`, `high_cardinality`) と, 文として読んだ場合の気付き (`duplicate_items`, `empty_items`) が出て, 形 `train [8305, 78]`, `valid [2373, 78]`, `test [1186, 78]` と事前学習用 `(100000, 78)` に書き出す
- 3 で, 検証用の隠した文字の正解率が 0.9 を超える
- 4 で, テスト用の正解率が 0.95 を超え, 見分けの確かめが 6/6 になる. 5 と 6 は 0.7 から 0.8 で, 見分けの確かめは 3/6 (組の 2 つに同じ答え) になる

## 結果のメモ

ML-Found `c1fde7e`, ML-EDA `df94bd8`, ML-ModelControl `88730ae`, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した. モデルの定義は ML-CreateArchitecture `88bc747` で組んだ.

| 実験 | モデル | 事前学習 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 見分けの確かめ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ja_bert | あり (MLM 4 エポック) | 12 | 12 | 0.9612 | 0.9663 | 6/6 |
| 5 (`scratch.yaml`) | ja_bert | なし | 12 | 10 | 0.7952 | 0.8010 | 3/6 |
| 6 (`char_bag.yaml`) | char_bag | なし | 30 | 20 | 0.7383 | 0.7378 | 3/6 |

- 事前学習 (3) の検証用の隠した文字の正解率は 0.9608 (4 エポック, 学習 146 秒)
- 多いクラス (良い) を答え続けた場合の正解率は約 0.7. 文字の平均のモデルはそれとほぼ同じで, 本体以外の評価に引きずられる. 見分けの確かめでは, 組の 2 つ (文字が全く同じ) に同じ答えを返すので 3/6 にしかならない
- 事前学習の無い BERT は, 最後のエポックで学習用の正解率 0.888 に対して検証用 0.788 と差が開き, 検証用は 0.8 で止まる (学習用の 8,305 件では, 「本体は」と述語の結び付きより個々の文を覚えてしまう). ラベル無しのレビューで文字の並びを先に学ぶと, 同じモデルで 0.96 になった (最後のエポックで学習用 0.983, 検証用 0.961)
- 作業の途中で見つけて各リポジトリで直したこと:
    - ML-EDA: 表の文の列を文のデータとして読めなかった (#11. `text_column`), 同じ文, 空の文, 表記のゆれの気付きが無かった (#11)
    - ML-Fathom: 表の読み方 (表, 時系列, 文) を画面で選べなかった (#8)
    - ML-ModelControl: 位置の番号, 詰め物の印, 先頭のトークンの取り出しの部品が無く, 学習のスクリプトで作って入力に渡す必要があった (#46)
    - ML-CreateArchitecture: 自己注意 (出力の口が 2 つ) の後に `op.Add` を置くと線が引かれず, 注意の出力がつながらないまま誤り 0 件になった (#71)
