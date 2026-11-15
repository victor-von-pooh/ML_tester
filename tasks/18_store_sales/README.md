# 18 店ごとの日々の売上を予測する (汚れた表と複数の系列)

## 目的

表と時系列の汚れたデータで, データの分析 (ML-EDA, ML-Fathom), モデルの作成 (ML-CreateArchitecture), 学習 (ML-Found) を通しで確かめる. 17 (文) と対になる, 表と時系列の難しめのタスク.

- **問題**: 12 店の 2 年分の日々の売上から, その日の売上を予測する (回帰). 使えるのは, その日の気温と販促の有無と, 前の日までの売上
- **データの汚れ**:
    - 売上は「738,984円」の文字
    - 日付は `2024/01/03`, `2024-01-03`, `2024年1月3日` の 3 つの書き方が混在
    - 店の名前の表記のゆれ (「渋谷店」と「渋谷店 」)
    - 休みの日 (行が無い)
    - 同じ行の重複 (150 行)
- **複数の系列**: 12 店の売上が同じ日付に並ぶ. 過去の値 (前の日, 前の週) は店ごとにずらす必要があり, 時刻の順に並べた表をそのままずらすと, 前の行の**別の店**の値を使ってしまう
- **前処理**: ML-Fathom の「学習できる形にする手順を作る」で作ったレシピ (`recipe.yaml`) と, 比べるために店ごとのずらしを外したレシピ (`recipe_naive.yaml`)
- **モデル**: ML-CreateArchitecture の画面で組んだ全結合のモデル (`models/sales_mlp.yaml`. 25 → 64 → 64 → 1, ReLU)

## ディレクトリ構成

```
18_store_sales/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── naive.yaml          # 店をまたいでずらしたデータセットで学習
│           └── quick.yaml          # 3 エポックで終える (動作の確認用)
├── models/sales_mlp.yaml           # ML-CreateArchitecture で組んだモデル
├── make_data.py                    # 元のデータ (data/store_sales.csv) を作る
├── recipe.yaml                     # ML-Fathom で作って保存したレシピ (店ごとにずらす)
├── recipe_naive.yaml               # 店ごとのずらしを外したレシピ (比べる用)
├── prepare.py                      # 分析して気付きを表示し, レシピで書き出す
├── train.py                        # 学習とテスト用の期間の評価
└── outputs/                        # dataset/, dataset_naive/ と実験 (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した店ごとの売上 (`make_data.py`. 乱数を固定) |
| 規模 | 8,652 行 (12 店 × 730 日から休みの日の約 3 % を除き, 重複 150 行を足した) |
| 売上の決まり方 | 店ごとの水準と週末の倍率, 季節, 気温の効き (店ごとに違う), 販促 (+30 %), 前の日からの揺れの続き (自己相関 0.7) |
| 分割 | 時刻の順に, 学習用 70 %, 検証用 15 %, テスト用 15 % (検証用とテスト用は学習用より後の期間) |

用意してもらう必要があるデータは無い. ML-EDA と ML-ModelControl はルートの README の「導入」で入れる.

## 3 つの段階でしたこと

### データの分析 (ML-Fathom)

ML-EDA の `ee16d4f` より前は, このデータは表として読まれ, 売上は文字, 日付は 2,171 種類の文字のままで, 学習に使えなかった (ML-EDA #12 で直した). 今は次のように読む.

1. csv を登録すると, 日付と店の組が重ならないので**店ごとの時系列** (役割 `time = date`, `series = store`) になり, 売上は数になる
2. 気付き: 表記のゆれ (`mixed_forms`. store), 重複 (`duplicate_rows`, 150 行), 複数の系列 (`panel_series`. 店の 13 本の系列. 表記のゆれで渋谷店が 2 本に分かれている), 間隔のゆれ (`irregular_interval`). 「時系列」のタブの周期の候補は 7 (最も長い系列の売上の自己相関)
3. タスクを「予測 (時系列)」, 分割を「時刻の順」, 検証用とテスト用を 0.15 にして, 「学習できる形にする手順を作る」を押すと, 次のレシピになる

| 手順 | 中身 |
| --- | --- |
| `drop_duplicates` | 重複した行を落とす |
| `normalize_text` (store) | 表記のゆれをそろえる (12 本の系列にする) |
| `lag` (by: store) | 数値の列の 1 つ前と 7 つ前の値と 7 つの移動平均を, **店ごとに**足す |
| `drop_missing` | 過去が足りない始めの行を落とす |
| `datetime_parts` (date) | 曜日と月の列を足す (日付の列は分割に使うので残す) |
| `one_hot` (store) | 店を 0 と 1 の列にする |
| `standardize` | 学習用の平均と標準偏差で標準化する |

学習に使う列は 25 (気温, 販促, 売上と気温と販促の lag1, lag7, mean7, 曜日, 月, 12 店).

### モデルの作成 (ML-CreateArchitecture)

画面で `x` (形 [64, 25]) → `nn.Linear` (25 → 64) → `nn.ReLU` → `nn.Linear` (64 → 64) → `nn.ReLU` → `nn.Linear` (64 → 1) → 出力 `y` を組んだ (5.9K パラメータ). 操作は Playwright で動かした.

### 学習と評価 (ML-Found)

`train.py` は売上を学習用の平均と標準偏差で標準化して二乗誤差で学び, 円での平均絶対誤差 (`mae`) を記録して, 検証用の `mae` が最も小さいエポックを残す. テスト用の期間では, 平均絶対誤差と平均の割合の誤差を, 「前の週の同じ曜日の売上をそのまま答える」場合 (`sales_lag7` を標準化の前に戻したもの) と比べて `train.log` に残す.

## 確かめること

- [x] 円の付いた売上, 3 つの書き方の日付を読み, 店ごとの時系列として分析できる
- [x] ML-Fathom の手順の作成で, 重複と表記のゆれをそろえ, 店ごとにずらし, 曜日を足したレシピができる
- [x] ML-CreateArchitecture で組んだモデルを ML-Found で学習し, テスト用の期間で評価できる
- [x] 店ごとにずらすと, 店をまたいでずらした場合より誤差が小さい. どちらも前の週の同じ曜日を答えるより良い

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/18_store_sales/make_data.py      # 1. 元のデータ (data/store_sales.csv)
python tasks/18_store_sales/prepare.py        # 2. 分析と書き出し (outputs/dataset)
python tasks/18_store_sales/train.py          # 3. 学習
python tasks/18_store_sales/prepare.py --recipe recipe_naive.yaml --out outputs/dataset_naive   # 4. 店をまたいでずらす
python tasks/18_store_sales/train.py --experiment configs/experiment/examples/naive.yaml        # 5. その学習
```

CPU で, 2 と 4 は 2 秒ほど, 3 と 5 は 5 秒ほど (Python の起動を含む). テスト用の評価は端末ではなく `train.log` に残る. 動作の確認には 3 に `--experiment configs/experiment/examples/quick.yaml` を使う.

## 期待する結果

- 2 で, 種類 `timeseries`, 系列を分ける列 `store` (13 本), 気付き `mixed_forms`, `duplicate_rows`, `panel_series`, `irregular_interval` が出て, 分割 `{'train': 5892, 'valid': 1263, 'test': 1263}`, 学習に使う列 25, 注意なし
- 3 のテスト用の平均の割合の誤差が 10 % を下回り, 前の週の同じ曜日を答えた場合 (17 % ほど) の半分ほどになる
- 5 は 3 より誤差が大きい (11 % ほど). 5 の「前の週の同じ曜日」は別の店の値なので 44 % ほどになる

## 結果のメモ

ML-Found `c1fde7e`, ML-EDA `ee16d4f`, ML-ModelControl `88730ae`, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した. モデルの定義は ML-CreateArchitecture `88bc747` で組み, レシピは ML-Fathom の PR #9 のブランチ (`ee16d4f` の ML-EDA) で作った.

| 実験 | レシピ | 学習したエポック | ベストのエポック | 検証用の平均絶対誤差 (ベスト) | テスト用の平均絶対誤差 | テスト用の平均の割合の誤差 | 前の週の同じ曜日を答えた場合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | `recipe.yaml` (店ごと) | 51 (Early Stopping) | 41 | 58,060 円 | 59,019 円 | 8.601 % | 120,420 円 (17.453 %) |
| 5 | `recipe_naive.yaml` (店をまたぐ) | 65 (Early Stopping) | 55 | 76,315 円 | 79,853 円 | 11.390 % | 272,453 円 (44.305 %) |

- 店ごとにずらすと, 誤差は前の週の同じ曜日を答えた場合の半分ほどになった. 売上の揺れは前の日から続く (自己相関 0.7) ので, 前の日の売上 (`sales_lag1`) が効く
- 店をまたいでずらすと, 1 つ前の値は前の行 (多くは同じ日の別の店) の売上になる. モデルは店の列と気温, 販促, 曜日から店の水準は学べるが, 前の日からの揺れの続きを使えないので誤差が大きい. 5 の「前の週の同じ曜日」は 7 行前の別の店の値なので, 誤差が 44 % になる
- ML-Fathom の「学習できる形にする手順を作る」は, 分析で見つけた系列を分ける列 (`series_by`) から `by` を付ける. 付けないと気付かないうちに 5 のようになる (誤りにはならない)
