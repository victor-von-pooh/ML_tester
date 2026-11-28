# 22 画像と領域の対を画面の操作で学習する

## 目的

画像と領域の対 (入力も正解もファイルの組. ルートの TASK_TYPES.md の相談 16 の「画像と領域の対」) を, 利用者が画面だけで, 問題の設定から始めて書き出し, 骨組みで学習し, 新しい画像を推論するまでの手順を示す. ライブラリの関数を呼ぶコードは書かない. 手で用意するのは元のデータ (`make_data.py`) だけ.

- 元のデータは ML-EDA の合成の見本 (`write_image_masks`). 雑音の上に円か四角を 1 つ置いた 32x32 の画像と, その領域 (0 が背景, 1 が円, 2 が四角)
- 画像の大きさがそろっているので, ML-Fathom の書き出しで中身の手順 (`resize`) はかけない (大きさの混ざる画像は 27 で確かめる)
- 正解の出どころ (`target.origin`) は `synthetic`. ML-Fathom の書き出しの画面で入れ, 書き出したデータセットと実験の `task.yaml` に残る
- モデルは ML-CreateArchitecture の同梱の例 `unet_segmentation`. 学習は ML-Pathfinder が書き出した骨組みを書き換えずに ML-Dashboard で起動する

**測っているもの**: 指標は正解の出どころとの一致. ここでは出どころが `synthetic` (作った正確な正解) なので, 正解そのものとの一致になる. 出どころが `process` (今ある処理の出力) なら, 測れるのはその処理との一致で, モデルはその処理を超えられない.

`journey.py` は, 下の「実行方法」の画面の操作を, 画面と同じ API で順に流して確かめる (段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りか). 部品は 27 の `journey.py` と `pairs.py`, アプリの起動と記録は 25 の `apps.py`.

## ディレクトリ構成

```
22_image_masks/
├── make_data.py   # 元のデータ (data/image_masks の images, masks) を ML-EDA の合成の見本で作る
├── journey.py     # 実行方法の画面の操作を API で流して確かめる
└── README.md
```

プロジェクト (骨組み, 書き出したデータセット, モデルの定義, 実験) は, ML-Pathfinder で選んだ書き出し先にできる. このディレクトリには置かない.

## データ

| 項目 | 内容 |
| --- | --- |
| 元のデータ | `write_image_masks(count=60, size=(32, 32), seed=0)`. 同じ stem の PNG の組 (`images/sample_000.png` と `masks/sample_000.png`). `make_data.py` が `data/image_masks` に作る |
| 書き出し | ML-Fathom のファイルの組の書き出し. 版 2, `role: predict`, `target.kind: mask`, `target.origin.type: synthetic`, 領域の値の名前 (0 `background`, 1 `circle`, 2 `square`) |
| 分割 | ML-Fathom の下書きの分割 (1 件ごとに無作為, 検証用とテスト用 15 %, 種 0). 学習用 42, 検証用 9, テスト用 9 |
| 推論 | 別の種 (1) で同じく作った 4 枚. 正解の領域は推論の結果と比べるだけに使う (`journey.py` が作る) |

用意してもらう必要があるデータは無い. ML-EDA はルートの README の「導入」で入れる.

## モデル

ML-CreateArchitecture の同梱の例 `unet_segmentation` (2 段の U-Net. 入力 `[4, 3, 32, 32]`). 「データセットから入力と出力を作る」が背景を含む 3 クラスを返し, 最後の畳み込み (`head`) の `out_channels` を 3 にする. 入力の形とチャンネル数は書き出したデータセットと同じなので直さない.

## 学習の設計

ML-Pathfinder が書き出した骨組みの config のまま (エポックの数 30, Adam の学習率 0.001, `valid/iou` を監視して Early Stopping. 待つ回数 10). 損失, 指標 (背景を除いた IoU の平均), テスト用の IoU の表, 領域の見本は骨組みの `train.py` が書く.

## 確かめること

- [x] 空の問題の設定から, 予測するで正解の種類が画像の領域の受け持ちを答え, 元の画像と領域のディレクトリを入れて骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から始め, 入力と正解のディレクトリと分割を下書きで埋め, 正解の出どころと領域の値の名前を足してプロジェクトの `outputs/dataset` に書き出せる (場所を写さない)
- [x] 正解の出どころ (`type: synthetic`, `name`, `settings`) が書き出した `dataset.yaml` と実験の `task.yaml` に残る
- [x] ML-CreateArchitecture で問題の設定を一覧から選び, データセットから入力と出力 (クラスの数) が決まる
- [x] 同梱の例 `unet_segmentation` から組んだモデルで, 骨組みの `train.py` を書き換えずに学習できる (起動の欄に何も打たない)
- [x] 学習の後に実験の中に推論の一式ができ, 一式の確かめ (`valid/iou`) が学習の記録と合う
- [x] ML-Dashboard でテスト用の IoU の表と, 領域の見本を凡例付きで見る
- [x] ML-Provide で実験の一式を一覧から登録でき, 新しい画像の推論の出力が領域の値 (0, 1, 2) だけを持ち, 元の画像の大きさになる

## 期待する結果

誤りの段階 0, 手の段階 0.

## 実行方法

準備は 25 と同じ (5 つのアプリを clone して `make setup`). 画面の操作の手順.

1. 元のデータを作る. ルートのディレクトリで `.venv/bin/python tasks/22_image_masks/make_data.py` (`data/image_masks` に `images/` と `masks/`)
2. ML-Pathfinder: 空の問題の設定を作り, 問いに答える (予測する, 1 件は写真 1 枚, 単位は組 (`pair`), 出力の形は領域 (`region`), 正解の種類は画像の領域 (`mask`), 正解はファイル, 指標は IoU). 元のデータに 1 のディレクトリを入れ, 骨組みを書き出す (書き出し先が問題の設定に記録される)
3. ML-Fathom: 問題の設定の一覧から 2 の設定を選んで始める. 書き出しの画面 (画像と領域) に入力と正解のディレクトリ (`images`, `masks`) と分割が下書きで入る. 正解の出どころ (種類 `synthetic`, 名前 `ml_eda.synthetic.write_image_masks`, 設定) と領域の値の名前 (0 `background`, 1 `circle`, 2 `square`) を足し, プロジェクトへ書き出す
4. ML-CreateArchitecture: 問題の設定の一覧から 2 の設定を選ぶ (保存先が決まる). 同梱の例 `unet_segmentation` を開き, 「データセットから入力と出力を作る」を押し, プロジェクトに保存する (`models/model.yaml`)
5. ML-Dashboard: 書き出したプロジェクトを一覧から登録し, 学習の準備を見て「環境を作る」, 「学習を起動」(config の欄は骨組みのまま). 結果でテスト用の IoU の表と領域の見本を見る
6. ML-Provide: 実験の一式を一覧から登録し, 新しい画像を推論して出力の配列をダウンロードする

この手順を API で流して確かめるには, ルートのディレクトリで実行する.

```bash
.venv/bin/python tasks/22_image_masks/journey.py               # 1 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/22_image_masks/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1. データは作業用の場所に作る (1 と同じ設定).

## 結果のメモ

5 つのアプリの main (ML-Pathfinder 387dd26, ML-Fathom b226609, ML-CreateArchitecture ee46a89, ML-Dashboard 1a0cac8, ML-Provide 599afa7) を手元 (CPU, Apple Silicon の Mac) で `journey.py` で動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (受け持ち, 正解の種類, 元のデータ) | OK | 元の `images/` と `masks/` を持つディレクトリを入れる |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | データセット `outputs/dataset` とモデルの定義 `models/model.yaml` の場所が記録される |
| ML-Fathom で問題の設定を一覧から始める | OK | 入力と正解のディレクトリ (`images`, `masks`) と分割 (random, 検証用とテスト用 15 %, 種 0) が下書きで入る |
| ML-Fathom でプロジェクトへ書き出す | OK | 版 2, 学習用 42, 検証用 9, テスト用 9. 中身の手順 無し, 注意 無し. `target.origin` は `synthetic`, `ml_eda.synthetic.write_image_masks`, 設定 (count 60, size 32x32, seed 0) |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 保存先が問題の設定の記録から決まる |
| ML-CreateArchitecture (同梱の例, 入力と出力) | OK | 入力 `[4, 3, 32, 32]` と 3 クラス (`background`, `circle`, `square`) を返し, `head` の `out_channels` を 3 にする |
| ML-CreateArchitecture でプロジェクトに保存 | OK | `models/model.yaml` |
| ML-Dashboard で書き出したプロジェクトを一覧から登録 | OK | |
| 学習の準備 | OK | データセットとモデルの定義がある. 足りない機能 eda, modelcontrol, inference |
| 環境を作る (画面から) | OK | 17 秒 |
| 学習の起動の config | OK | 骨組みのまま (30 エポック) |
| 学習 | OK | 23 エポックで Early Stopping. ベストはエポック 13 の valid/iou 0.3000. 21 秒 |
| テスト用の評価 | OK | IoU は背景 0.987, 円 0.100, 四角 0.575. 平均 (`mean` の行) 0.338 |
| 学習の後の推論の一式 | OK | 一式で推論した valid/iou 0.3000 が記録と合う |
| 見本を見る | OK | 領域の見本 6 件. 凡例は `background` (黒), `circle` (`#800000`), `square` (`#008000`) |
| ML-Provide で実験の一式を一覧から登録 | OK | 一式の確かめ (valid の iou 0.3) が記録と合う |
| 問題の設定をつなぐ (候補) | OK | ファイルの組の受け持ちは意思決定が無いので候補に出さない (ML-Provide の決まり) |
| 新しい件の推論と出力 | OK | 4 件. 値は 0, 1, 2 だけ. 正解との画素の一致 0.905. 出力は元の画像の大きさ (32x32) |

手の段階 0, 誤りの段階 0.

- 実験の `task.yaml` に `target.origin` (type, name, settings) が書き出しのとおり残る
- IoU が低いのは, 骨組みの既定 (学習率 0.001, 待つ回数 10) で 60 組を学習し, 円をまだ塗り分けられないため. 確かめたいのは段階のつながりで, モデルの良し悪しではない

## 気付いたこと

- ライブラリの関数で書き出して手で学習のスクリプトを書く道では, 領域の値から番号の表, クラスの数, チャンネル数, 正解の出どころの `task.yaml` への写し, IoU の数え方, 見本の書き方を学習のスクリプトに書いた. 画面の道ではその全てを骨組みと ML-Fathom の書き出しが受け持ち, 利用者が打つのは正解の出どころと領域の値の名前だけ
