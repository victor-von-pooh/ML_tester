# 15 ML-EDA で配列にした画像のデータセットの学習

## 目的

データの分析の段階 (ML-EDA, ML-Fathom) で, クラスごとのフォルダに分けた画像を学習用の配列に書き出し, ML-Found でそのまま学習できることを確かめる.
14 は表のデータだったので, ここでは中身の手順 (`item_steps`) で 1 件ずつ配列にする画像を扱う.

- 元のデータは, 前処理が要る図形の画像 (大きさが 3 種類, 灰色とカラーの混在, 途中で切れて読めないファイル)
- 分析 (`profile`) の気付きをもとにレシピ (`recipe.yaml`) を書く. ML-Fathom の「前処理と書き出し」で作るものと同じ形
- 書き出す前に, 中身を読まずにレシピを確かめる (`prepare(..., read_items=False)`. ML-Fathom の前と後の様子と同じ). 気付きを見ずに書いたレシピ (`recipe_naive.yaml`) で, 書き出しの前に注意が出ることも確かめる
- 学習のスクリプトは, 書き出したデータセットを `load_dataset` で読むだけで, 画像の前処理を書かない
- 大きさも色の形式も違う新しい画像に, 書き出したときと同じ中身の手順 (`transform_item`) をかけて予測できるかも確かめる (推論の段階の準備)

## ディレクトリ構成

```
15_eda_images/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── quick.yaml          # 3 エポックで終える (動作の確認用)
├── make_data.py                    # 元のデータ (data/shapes/<クラス>/*.png) を作る
├── recipe.yaml                     # ML-EDA の前処理のレシピ
├── recipe_naive.yaml               # 気付きを見ずに書いたレシピ (注意と誤りの確かめ用)
├── prepare.py                      # 分析して気付きを表示し, 中身を読まずに確かめてから書き出す
├── model.py                        # ShapeCNN (2 段の畳み込み)
├── train.py                        # 書き出したデータセットで学習
└── outputs/                        # dataset/ (書き出したデータセット) と shapes/<日付>/<時刻>/ (実験. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した図形の画像 (`make_data.py`. 乱数を固定). 雑音のある暗い背景に, 位置, 大きさ, 色をばらつかせた円, 四角, 三角 |
| 規模 | クラスごとに 300 枚 (900 枚) と, 読めないファイル 3 枚. 目的の列 `label` (フォルダの名前) |
| 前処理が要る所 | 大きさ (48x48, 64x64, 64x48), 灰色 (約 30 %) とカラーの混在, 途中で切れたファイル (`broken.png`) |
| 分割 | 層化して学習用 70 %, 検証用 20 %, テスト用 10 % (`recipe.yaml`) |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる (Pillow は `ml-eda[all]` に含まれる).

## 前処理のレシピ

`prepare.py` が表示する気付きを見て書いた.

| 手順 | 理由 |
| --- | --- |
| 表の手順 `filter_rows` で `width` が `notna` の件だけを残す | 気付き `unreadable_items` (読めないファイルは読み込みの表で大きさが欠損になる) |
| 中身の手順 `rgb` | 気付き `mixed_modes` (RGB と L). 色が分類の手がかりにならない図形なら `grayscale` でもよい |
| 中身の手順 `resize` で 32x32 | 気付き `mixed_sizes` (3 種類) |
| 中身の手順 `normalize` | 学習用の画像の平均と標準偏差 (チャンネルごと) で正規化し, (チャンネル, 高さ, 幅) にする |

`normalize` の平均と標準偏差は, ML-EDA が分割した後の学習用の画像だけで当てはめる.

## 確かめること

- [x] ML-EDA がクラスのフォルダを `label` 付きの画像として読み, 大きさのばらつき, 色の形式の混在, 読めないファイルの気付きを出す
- [x] 気付きを見ずに書いたレシピは, 中身を読む前に「読めない件」「灰色とカラーの混在」の注意が出て, 書き出しは読めないファイルの場所と除き方を添えた誤りで止まる
- [x] `recipe.yaml` では注意が無く, (件数, 3, 32, 32) の配列に書き出せる
- [x] 学習のスクリプトが `load_dataset` と `tensors` だけで使え, 入力のチャンネル数とクラスの数もデータセットの説明から決まる
- [x] テスト用の分割で評価できる
- [x] 学習に無い大きさ (100x80) の灰色の画像に `transform_item` で同じ中身の手順をかけ, 予測できる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/15_eda_images/make_data.py   # 1. 元のデータ (data/shapes)
python tasks/15_eda_images/prepare.py --recipe recipe_naive.yaml --out outputs/naive   # 2. 気付きを見ずに書いたレシピ (注意と誤り)
python tasks/15_eda_images/prepare.py     # 3. 分析と書き出し (outputs/dataset)
python tasks/15_eda_images/train.py       # 4. 学習
python tasks/15_eda_images/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (3 エポック)
```

CPU で, 3 は 1 秒ほど, 4 は 7 秒ほど (Python の起動を含む).

## 期待する結果

- 2 と 3 で `mixed_sizes`, `mixed_modes`, `unreadable_items` の気付きが出る. 2 では 「読めない件が 3 件あります」「灰色とカラーの画像が混ざっています」の注意が出て, 書き出しは「circle/broken.png を読めません」で止まる
- 3 で分割 `{'train': 630, 'valid': 180, 'test': 90}`, 注意なし, 配列の形 `[630, 3, 32, 32]` など
- 4 で検証用とテスト用の正解率が 1 に近くなり, `train.log` に新しい画像の予測 (三角) が残る

## 結果のメモ

ML-Found `c1fde7e`, ML-EDA `829bcea`, Pillow 13.0.0, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した.

| 実験 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 新しい画像の予測 |
| --- | --- | --- | --- | --- | --- |
| 4 (`recipe.yaml`) | 22 (Early Stopping) | 16 | 1.0000 | 1.0000 | triangle (0.987) |
| quick (3 エポック) | 3 | 2 | 0.5333 | 0.5889 | circle (0.438) |

- 合成の図形ははっきり分かれるので, 正解率が 1 になるのは想定どおり. このタスクで確かめたいのは, 前処理が要る画像のフォルダから学習と推論までが, 学習のスクリプトに前処理を書かずにつながること
- 新しい画像は大きさ (100x80) も色の形式 (灰色) も学習用と違うが, `transform_item` が `rgb`, `resize`, 学習用の平均と標準偏差での正規化をかけて (3, 32, 32) にした
- 最初の実行 (ML-EDA `ab94f12`) では, 分析の気付きに `path` の `id_like` (ID らしい列) も出た. 画像などでは読み込みの表の列を学習に使わないので不要な気付きで, ML-EDA の PR #6 で出さないようにした. 上の結果はその後の再実行 (結果は同じ)
