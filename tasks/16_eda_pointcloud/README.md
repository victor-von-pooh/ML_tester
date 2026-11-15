# 16 ML-EDA で配列にした点群のデータセットの学習

## 目的

データの分析の段階 (ML-EDA, ML-Fathom) で, クラスごとのフォルダに分けた点群を学習用の配列に書き出し, ML-Found でそのまま学習できることを確かめる.
15 (画像) と同じく中身の手順 (`item_steps`) を使うが, 点群に特有の問題 (ファイルの形式の混在, 点の数の違い, 座標が数でない点) を含める.

- 元のデータは, 4 つの形 (球, 立方体, 円柱, 円錐) の表面の点群. PLY (バイナリ), PCD (ASCII), xyz を混ぜ, 点の数を件ごとに変え, PCD には距離のセンサーの欠けとして NaN の点を入れる. 途中で切れて読めないファイルも 1 つ置く
- 分析 (`profile`) の気付きをもとにレシピ (`recipe.yaml`) を書く. ML-Fathom の「前処理と書き出し」で作るものと同じ形
- 書き出す前に, 中身を読まずにレシピを確かめる (`prepare(..., read_items=False)`). 気付きを見ずに書いたレシピ (`recipe_naive.yaml`) で, 注意と誤りが出ることも確かめる
- 学習のスクリプトは, 書き出したデータセットを `load_dataset` で読むだけで, 点群の前処理を書かない. モデルは点の並びに依らない小さな PointNet 風のもの
- 点の数も欠けも違う新しい点群に, 書き出したときと同じ中身の手順 (`transform_item`) をかけて予測できるかも確かめる (推論の段階の準備)

## ディレクトリ構成

```
16_eda_pointcloud/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── quick.yaml          # 3 エポックで終える (動作の確認用)
├── make_data.py                    # 元のデータ (data/objects/<クラス>/*.ply, *.pcd, *.xyz) を作る
├── recipe.yaml                     # ML-EDA の前処理のレシピ
├── recipe_naive.yaml               # 気付きを見ずに書いたレシピ (注意と誤りの確かめ用)
├── prepare.py                      # 分析して気付きを表示し, 中身を読まずに確かめてから書き出す
├── model.py                        # PointClassifier (点ごとの 1x1 の畳み込み 3 段と最大値でまとめる)
├── train.py                        # 書き出したデータセットで学習
└── outputs/                        # dataset/ (書き出したデータセット) と objects/<日付>/<時刻>/ (実験. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成した形の表面の点群 (`make_data.py`. 乱数を固定). z 軸の周りの回転, 大きさ (0.5 から 3 倍), 位置, 雑音をばらつかせる |
| 規模 | クラスごとに 60 件 (240 件) と, 読めないファイル 1 件. 目的の列 `label` (フォルダの名前) |
| 形式 | 3 件に 1 件ずつ PLY (バイナリ, 81 件. 読めない 1 件を含む), PCD (ASCII, 80 件), xyz (80 件) |
| 前処理が要る所 | 点の数 (302 から 1996), PCD の点の約 5 % が NaN, 途中で切れたファイル (`cone/broken.ply`) |
| 分割 | 層化して学習用 70 %, 検証用 20 %, テスト用 10 % (`recipe.yaml`) |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる.

## 前処理のレシピ

`prepare.py` が表示する気付きを見て書いた.

| 手順 | 理由 |
| --- | --- |
| 表の手順 `filter_rows` で `error` が `isna` の件だけを残す | 気付き `unreadable_items` (読めない点群は読み込みの表の `error` に理由が入る) |
| 中身の手順 `sample_points` で 512 点 | 気付き `mixed_points`. 多い点群は間引き, 少ない点群は重ねる. 座標が数でない点は使わない (気付き `invalid_points`) |
| 中身の手順 `normalize_cloud` | 中心を原点にし, 最も遠い点までを 1 にする (位置と大きさのばらつきを消す) |

## 確かめること

- [x] ML-EDA が PLY, PCD, xyz の混ざったクラスのフォルダを `label` 付きの点群として読み, 点の数の違い, 座標が数でない点, 読めないファイルの気付きを出す
- [x] 気付きを見ずに書いたレシピは, 中身を読む前に「読めない件」「点の数がそろっていません」の注意が出て, 書き出しは形がそろわない理由を添えた誤りで止まる
- [x] `recipe.yaml` では注意が無く, (件数, 512, 3) の配列に書き出せる
- [x] 学習のスクリプトが `load_dataset` と `tensors` だけで使え, クラスの数もデータセットの説明から決まる
- [x] テスト用の分割で評価できる
- [x] 学習に無い点の数 (3000) で 1 割が NaN の新しい点群に `transform_item` で同じ中身の手順をかけ, (512, 3) にして予測できる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/16_eda_pointcloud/make_data.py   # 1. 元のデータ (data/objects)
python tasks/16_eda_pointcloud/prepare.py --recipe recipe_naive.yaml --out outputs/naive   # 2. 気付きを見ずに書いたレシピ (注意と誤り)
python tasks/16_eda_pointcloud/prepare.py     # 3. 分析と書き出し (outputs/dataset)
python tasks/16_eda_pointcloud/train.py       # 4. 学習
python tasks/16_eda_pointcloud/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (3 エポック)
```

CPU で, 3 は 1 秒ほど, 4 は 12 から 14 秒ほど (Python の起動を含む. 学習は 9 から 10 秒). `--recipe` と `--out` の相対パスは, タスクのディレクトリから辿る (2 の `outputs/naive` は `tasks/16_eda_pointcloud/outputs/naive`).

## 期待する結果

- 2 と 3 で `mixed_points`, `invalid_points`, `unreadable_items` の気付きが出る. 2 は意図どおり誤りで止まる (終了コード 1. `outputs/naive` は作られない). 中身を読む前に「読めない件が 1 件あります」「点の数がそろっていません」の注意が出て, 書き出しは「中身の形がそろいません」で止まる (読めないファイルより先に, 形のそろわない学習用の点群に当たる)
- 3 で分割 `{'train': 168, 'valid': 48, 'test': 24}`, 注意なし, 配列の形 `[168, 512, 3]` など
- 4 で検証用の正解率が 1 に近くなる (端末に出る). テスト用の正解率と新しい点群の予測 (円錐) は `train.log` に残る

## 結果のメモ

ML-Found `c1fde7e`, ML-EDA `28db489`, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した.

| 実験 | 学習したエポック | ベストのエポック | 検証用の正解率 (ベスト) | テスト用の正解率 | 新しい点群の予測 |
| --- | --- | --- | --- | --- | --- |
| 4 (`recipe.yaml`) | 20 (Early Stopping) | 12 | 1.0000 | 1.0000 | cone (0.970) |
| quick (3 エポック) | 3 | 2 | 0.5000 | 0.5000 | cone (0.280) |

- 合成の形ははっきり分かれるので, 正解率が 1 になるのは想定どおり. このタスクで確かめたいのは, 形式, 点の数, 欠けの混ざる点群のフォルダから学習と推論までが, 学習のスクリプトに前処理を書かずにつながること
- 最初の実行では, 読み込みは NaN の点を `invalid_points` の列に数えていたが, 気付きが出なかった. ML-EDA の PR #10 で `invalid_points` の気付きを加え, 上の結果はその後の実行
- 新しい点群は点の数 (3000) も欠け (1 割) も学習用と違うが, `transform_item` が有限の点から 512 点を選び, 中心と大きさをそろえた
