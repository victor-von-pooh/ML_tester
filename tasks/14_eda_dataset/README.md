# 14 ML-EDA で前処理したデータセットの学習

## 目的

データの分析の段階 (ML-EDA, ML-Fathom) で前処理して書き出したデータセットを, ML-Found でそのまま学習できることを確かめる.
データの分析と学習の 2 つの段階の交点を, ML-EDA の README の「学習で使う」の手順どおりに使う.

- 元のデータは, 前処理が要る顧客の表 (欠損, 文字のカテゴリ, カンマ付きの数, 偏った分布, ID, 答えから作った列)
- 分析 (`profile`) が見つけた気付きをもとに, 前処理のレシピ (`recipe.yaml`) を書く. ML-Fathom の「前処理と書き出し」で作るものと同じ形
- 学習のスクリプトは, 書き出したデータセットを `load_dataset` で読むだけで, 前処理を書かない
- 前処理の前の形の新しいデータに, 書き出したときと同じ前処理をかけて予測できるかも確かめる (推論の段階の準備)

## ディレクトリ構成

```
14_eda_dataset/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/quick.yaml     # 3 エポックで終える (動作の確認用)
├── make_data.py                    # 元のデータ (data/churn.csv) を作る
├── recipe.yaml                     # ML-EDA の前処理のレシピ
├── prepare.py                      # 分析して気付きを表示し, レシピで書き出す
├── model.py                        # ChurnClassifier (2 層の MLP)
├── train.py                        # 書き出したデータセットで学習
└── outputs/                        # dataset/ (書き出したデータセット) と churn/<日付>/<時刻>/ (実験. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した顧客の表 (`make_data.py`. 乱数を固定) |
| 規模 | 3000 行, 10 列. 目的の列 `churn` (yes / no) |
| 前処理が要る所 | `customer_id` (ID), `monthly_fee` の 15 % の欠損, `plan` の 5 % の欠損, 文字のカテゴリ (`plan`, `contract`), カンマ付きの数 (`total_paid`), 偏った分布 (`data_gb`), 答えから作った列 (`cancel_reason_code`. 解約しなければ 0) |
| 分割 | 層化して学習用 70 %, 検証用 20 %, テスト用 10 % (`recipe.yaml`) |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる.

## 前処理のレシピ

`prepare.py` が表示する気付きに合わせて書いた.

| 手順 | 理由 (気付き) |
| --- | --- |
| `drop_columns` で `customer_id`, `cancel_reason_code` を落とす | `id_like` (ID らしい列), `target_leak` (目的の列との関連の強さ 1.00) |
| `impute` で `monthly_fee` を中央値, `plan` を最も多い値で埋める | 欠損 (`monthly_fee` には欠損の印の列も足す) |
| `log` で `data_gb` の対数を取る | `skewed` (歪度 4.1) |
| `one_hot` で `plan`, `contract` を 0 と 1 の列にする | 文字のカテゴリ |
| `standardize` で数値の列を標準化する | 値の大きさをそろえる |

値を学ぶ手順 (補完, 標準化, ワンホット) は, ML-EDA が分割した後の学習用のデータだけで当てはめる.

## 確かめること

- [x] ML-EDA が気付き (ID, リーク, 偏った分布, 外れ値) を見つけ, カンマ付きの数を数として読む
- [x] レシピで前処理して書き出したデータセットを, 学習のスクリプトが `load_dataset` と `tensors` だけで使える (入力の次元とクラスの数もデータセットの説明から)
- [x] 書き出したデータセットに, 学習の前に直すことの注意が無い (全て数値, 欠損なし)
- [x] テスト用の分割で評価できる
- [x] 前処理の前の形の新しいデータ (ID と欠損を含む) に, `transform` で同じ前処理をかけて予測できる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/14_eda_dataset/make_data.py   # 1. 元のデータ (data/churn.csv)
python tasks/14_eda_dataset/prepare.py     # 2. 分析と書き出し (outputs/dataset)
python tasks/14_eda_dataset/train.py       # 3. 学習
python tasks/14_eda_dataset/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (3 エポック)
```

CPU で, 2 は 1 秒ほど, 3 は 5 秒ほど (Python の起動を含む).

## 期待する結果

- 2 で `id_like`, `target_leak`, `skewed`, `outliers` の気付きが出て, 分割 `{'train': 2100, 'valid': 600, 'test': 300}`, 学習に使う列 12, 注意なし
- 3 で検証用の正解率が, 多いクラス (no) を答え続けた場合 (約 0.69) より高くなる. `train.log` にテスト用の正解率と, 新しい顧客の解約の確率が残る

## 結果のメモ

ML-Found `c1fde7e`, ML-EDA `9ac69a9`, pandas 3.1.1, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した.

| 項目 | 値 |
| --- | --- |
| 学習したエポック | 15 (Early Stopping) |
| ベストのエポック | 5 |
| 検証用の正解率 (ベスト) | 0.7450 |
| テスト用の正解率 | 0.7067 |
| 新しい顧客 (月ごとの契約, 問い合わせ 6 回, 利用 3 か月) の解約の確率 | 0.866 |

- 答えから作った列 (`cancel_reason_code`) を落とさないと, 検証用の正解率は 1 に近くなる (リーク). ML-EDA の `target_leak` の気付きで見つけて落とした
- 新しい顧客の確率は, 解約しやすい条件 (月ごとの契約, 多い問い合わせ, 短い利用期間) に合わせて高く出た. `monthly_fee` の欠損は学習用の中央値で埋まり, 欠損の印の列が 1 になる
- この ML-EDA の版では, `tensors` が書き込めない配列から作ったテンソルを返し, PyTorch の警告が 1 回出る. 書き込める配列を返す修正は ML-EDA の次の版に入る
