# 14 ML-EDA で前処理したデータセットの学習

## 目的

データの分析の段階 (ML-EDA, ML-Fathom) で前処理して書き出したデータセットを, ML-Found でそのまま学習できることを確かめる.
データの分析と学習の 2 つの段階の交点を, ML-EDA の README の「学習で使う」の手順どおりに使う.

- 元のデータは, 前処理が要る顧客の表 (欠損, 文字のカテゴリ, カンマ付きの数, 偏った分布, ID, 答えから作った列)
- 分析 (`profile`) が見つけた気付きをもとに, 前処理のレシピ (`recipe.yaml`) を書く. ML-Fathom の「前処理と書き出し」で作るものと同じ形
- 学習のスクリプトは, 書き出したデータセットを `load_dataset` で読むだけで, 前処理を書かない
- 元のデータと同じ形の新しいデータを `load_data` で読み, 書き出したときと同じ前処理 (`transform`) をかけて予測できるかも確かめる (推論の段階の準備)
- 答えから作った列 (リーク) を残すとどうなるかも, 別のレシピで確かめる

## ディレクトリ構成

```
14_eda_dataset/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── quick.yaml          # 3 エポックで終える (動作の確認用)
│           └── leak.yaml           # リークを残したデータセットで学習
├── make_data.py                    # 元のデータ (data/churn.csv) を作る
├── recipe.yaml                     # ML-EDA の前処理のレシピ
├── recipe_leak.yaml                # 答えから作った列を残したレシピ (リークの確かめ用)
├── prepare.py                      # 分析して気付きを表示し, レシピで書き出す
├── model.py                        # ChurnClassifier (2 層の MLP)
├── train.py                        # 書き出したデータセットで学習
└── outputs/                        # dataset/, dataset_leak/ (書き出したデータセット) と churn*/<日付>/<時刻>/ (実験. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した顧客の表 (`make_data.py`. 乱数を固定) |
| 規模 | 3000 行, 10 列. 目的の列 `churn` (yes / no) |
| 前処理が要る所 | `customer_id` (ID), `monthly_fee` の 15 % の欠損, `plan` の 5 % の欠損, 文字のカテゴリ (`plan`, `contract`), カンマ付きの数 (`total_paid`), 偏った分布 (`data_gb`), 答えから作った列 (`cancel_reason_code`. 解約しなければ 0) |
| 分割 | 層化して学習用 70 %, 検証用 20 %, テスト用 10 % (`recipe.yaml`) |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる.

### 前処理のレシピ

`prepare.py` が表示する気付きと, 列の型を見て書いた.

| 手順 | 理由 |
| --- | --- |
| `drop_columns` で `customer_id`, `cancel_reason_code` を落とす | 気付き `id_like` (ID らしい列) と `target_leak` (目的の列との関連の強さ 1.00) |
| `impute` で `monthly_fee` を中央値, `plan` を最も多い値で埋める | 欠損 (気付きは 30 % 以上で出るので出ないが, 列の分析の欠損の数で分かる. `monthly_fee` には欠損の印の列も足す) |
| `log` で `data_gb` の対数を取る | 気付き `skewed` (歪度 4.1). `outliers` の気付きもあるが, 対数を取ると外れ値も縮むので切り詰めない |
| `one_hot` で `plan`, `contract` を 0 と 1 の列にする | 文字のカテゴリ (学習には数値が要る) |
| `standardize` で数値の列を標準化する | 値の大きさをそろえる |

値を学ぶ手順 (補完, 標準化, ワンホット) は, ML-EDA が分割した後の学習用のデータだけで当てはめる. `recipe_leak.yaml` は 1 つ目の手順で `customer_id` だけを落とし, 他は同じ.

## モデル

`model.py` の `ChurnClassifier`. 全結合の 2 層 (隠れ層 32, ReLU) の MLP. 入力の次元とクラスの数は書き出したデータセットの説明から決める.

## 学習の設計

`step` は交差エントロピーの損失と正解率を返す. Adam (学習率 0.003), `valid/acc` を監視して Early Stopping (待つ回数 10). 学習の後に, テスト用の分割の正解率と, 新しい顧客に `transform` で同じ前処理をかけた解約の確率を `train.log` に残す.

## 確かめること

- [x] ML-EDA が気付き (ID, リーク, 偏った分布, 外れ値) を見つけ, カンマ付きの数を数として読む
- [x] レシピで前処理して書き出したデータセットを, 学習のスクリプトが `load_dataset` と `tensors` だけで使える (入力の次元とクラスの数もデータセットの説明から)
- [x] 書き出したデータセットに, 学習の前に直すことの注意が無い (全て数値, 欠損なし)
- [x] テスト用の分割で評価できる
- [x] 元のデータと同じ形の新しいデータ (ID, 欠損, カンマ付きの数を含む csv) を `load_data` で読み, `transform` で同じ前処理をかけて予測できる
- [x] 答えから作った列を残すと検証用とテスト用の正解率が 1 になり, 予測のときにはその列が無いので前処理が誤りになる (リークに気付ける)

## 期待する結果

- 2 で `id_like`, `target_leak`, `skewed`, `outliers` の気付きが出て, 分割 `{'train': 2100, 'valid': 600, 'test': 300}`, 学習に使う列 12, 注意なし
- 3 で検証用の正解率が, 多いクラス (no) を答え続けた場合 (検証用 419/600 = 0.698) より高くなる. `train.log` にテスト用の正解率と, 新しい顧客の解約の確率が残る
- 5 で検証用の正解率が 1 になり, `train.log` に「新しい顧客を予測できません: 前処理の後に, 学習に使う列が足りません: cancel_reason_code」の警告が残る

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/14_eda_dataset/make_data.py   # 1. 元のデータ (data/churn.csv)
python tasks/14_eda_dataset/prepare.py     # 2. 分析と書き出し (outputs/dataset)
python tasks/14_eda_dataset/train.py       # 3. 学習
python tasks/14_eda_dataset/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (3 エポック)
python tasks/14_eda_dataset/prepare.py --recipe recipe_leak.yaml --out outputs/dataset_leak  # 4. リークを残して書き出す
python tasks/14_eda_dataset/train.py --experiment configs/experiment/examples/leak.yaml      # 5. リークを残して学習
```

CPU で, 2 は 1 秒ほど, 3 は 5 秒ほど (Python の起動を含む. 新しい環境の最初の 1 回は 10 秒ほど).

## 結果のメモ

ルートの README の「導入」で固定した版と pandas 3.1.1, PyTorch 2.15.0, CPU (Apple Silicon の Mac) での値.

| 実験 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 新しい顧客の解約の確率 |
| --- | --- | --- | --- | --- | --- |
| 3 (`recipe.yaml`) | 15 (Early Stopping) | 5 | 0.7450 | 0.7067 | 0.860 |
| quick (3 エポック) | 3 | 3 | 0.7433 | 0.7267 | 0.893 |
| 5 (`recipe_leak.yaml`) | 16 (Early Stopping) | 6 | 1.0000 | 1.0000 | 予測できない (前処理の誤り) |

- 多いクラスを答え続けた場合の正解率は, 検証用が 0.698 (419/600), テスト用が 0.697 (209/300). 3 のテスト用の正解率 0.7067 はそれより 3 件分だけ高い. この表は解約の理由の多くを乱数で作ったので, 予測できる部分が小さい. quick の方がテスト用の正解率が高いのは, 300 件のテスト用の揺れの範囲
- 新しい顧客 (月ごとの契約, 問い合わせ 6 回, 利用 3 か月) の確率は, 解約しやすい条件に合わせて高く出る. `monthly_fee` の欠損は学習用の中央値で埋まり, 欠損の印の列が 1 になる. `total_paid` の "1,210" は `load_data` が数にする
- 5 は答えから作った列 (`cancel_reason_code`. 解約しなければ 0) を残したので, 検証用もテスト用も正解率が 1 になる. 予測のときにはこの列が無いので, ML-EDA の `transform` が足りない列を誤りにする. ML-EDA の `target_leak` の気付きで, 学習の前に見つけて落とせる

## 気付いたこと

無し
