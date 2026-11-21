# 27 画像と領域を画面の API だけで通す (大きさの混ざる画像)

## 目的

節目「点群と領域を画面で通す」(ROADMAP.md の M2) の確かめのうち, 画像の領域 (J6). 25 と同じく, 問題の設定から推論までを 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. ライブラリの関数は呼ばない (合成のデータを作るところと, 推論の結果を正解と比べるところだけ ML-EDA, NumPy, Pillow を使う). 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか, コードを書くか (手), 誤りかを表にする.

- **課題**: 写真の中の物の領域 (背景, 円, 四角) を画素ごとに塗る (架空). 問題の設定の見本が無いので, 空の問題の設定から問いに答える
- **データの分析**: 大きさの混ざる画像を, ML-Fathom のファイルの組の書き出しの中身の手順 (`resize`) で 32x32 にそろえる (同梱の U-Net の例は 4 の倍数の大きさ)
- **モデル**: ML-CreateArchitecture の同梱の例 `unet_segmentation`
- 撮影と点群は 28 で確かめる. 通しの部品 (`pairs.py`) は 28 も使い, アプリの起動と記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
27_m2_image_masks/
├── pairs.py     # ファイルの組の受け持ちを API で通す部品 (28 も使う)
├── journey.py   # 合成のデータを作り, 画像と領域で通す
└── README.md
```

作業用の場所の中身は 25 と同じ.

## データ

`journey.py` が作業用の場所に作る. ML-EDA の `write_image_masks` で, 32x32 の 30 組と 48x40 (高さ x 幅) の 30 組を作り, 1 つの `images/` と `masks/` に並べる (`data/image_masks`. 乱数を固定). 領域の値は 0 (背景), 1 (円), 2 (四角). 推論には, 別の乱数の種で同じく作った 4 枚 (2 つの大きさを 2 枚ずつ. `data/image_masks_new`) を使い, 正解の領域は結果と比べるだけに使う. 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の同梱の例 `unet_segmentation` (2 段の U-Net. 入力 `[4, 3, 32, 32]`, 背景を含む 2 クラス). クラスの数を 3 にするため, 最後の畳み込み (`head`) の `out_channels` を 3 にする. ML-CreateArchitecture の「データセットから入力と出力を作る」(`POST /api/dataset/inputs`) が作れれば, 画面と同じく返った値を例に当て (`journey.py` の `apply_unet_info`), 残る違いだけを手で直す. 作れなければ, 画面でノードの欄を書き換えるのと同じく手で直す (`adapt_unet` がデータセットの説明から直す所を決める). 入力の形とチャンネル数は書き出したデータセットと同じなので直さない.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`project/train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. エポックの数は骨組みの既定の 30 (60 組の 32x32 なので 10 秒ほど). 学習の環境は ML-Dashboard の「環境を作る」で入れる.

段階と API (画面のボタンに当たるもの).

| 段階 | アプリ | API |
| --- | --- | --- |
| 空の問題の設定を作り, 問いに答え, 元のディレクトリを入れる | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から始め, 書き出しの下書きに正解の出どころ, 領域の値の名前, 中身の手順を足してプロジェクトへ書き出す. 開き直して中身の手順が残るか | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `GET /api/problems/pairs`, `POST /api/problems/pairs/export` |
| 問題の設定を一覧から選ぶ (選べなければプロジェクトのディレクトリを打つ), 同梱の例を開き, データセットから入力と出力を作る (作れなければ手で直す), プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `GET /api/examples/{name}`, `POST /api/dataset/inputs`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 学習の準備, 環境を作る, 学習 (config の欄), 結果, ログ, 見本 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results`, `GET <実験>/log` |
| 実験の一式を一覧から登録 (一式の確かめ), 問題の設定の候補, 新しい画像の推論, 出力の配列のダウンロード | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `GET /api/bundles/{id}`, `GET /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/artifacts/{行}/file` |

起動と停止, 時間の上限は 25 と同じ.

## 確かめること

- [x] 空の問題の設定から, 予測するで正解の種類が画像の領域の受け持ちを答え, 元の画像と領域のディレクトリを入れて骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から始め, 入力と正解のディレクトリと分割を下書きで埋め, プロジェクトの `outputs/dataset` に書き出せる (場所を写さない)
- [x] 中身の手順 (`resize`) をかけた書き出しが版 3 になり, 問題の設定に書き戻した中身の手順が開き直した画面に残る
- [ ] ML-CreateArchitecture で問題の設定を一覧から選べる (選べない. 下の「結果のメモ」)
- [ ] ML-CreateArchitecture でデータセットから入力と出力 (クラスの数) が決まる (決まらない)
- [x] 同梱の例 `unet_segmentation` から組んだモデルで, 骨組みの `train.py` を書き換えずに学習できる
- [ ] 骨組みの config がモデルの定義の場所を持ち, 起動の欄に場所を打たずに済む (打つ)
- [x] 学習の後に実験の中に推論の一式ができ, 一式の確かめ (`valid/iou`) が学習の記録と合う (19 と同じ確かめ)
- [ ] ML-Dashboard で領域の見本を凡例付きで見る (骨組みが見本を書かない)
- [x] ML-Provide で実験の一式を一覧から登録でき, 新しい画像の推論の出力が領域の値 (0, 1, 2) だけを持つ
- [ ] 推論の出力が元の画像の大きさになる (中身の手順の後の大きさ)

## 期待する結果

誤りの段階 0. 手の段階は 0 を目指す (今は 5. 下の「結果のメモ」).

## 実行方法

準備は 25 と同じ (5 つのアプリを clone して `make setup`). ルートのディレクトリで実行する.

```bash
.venv/bin/python tasks/27_m2_image_masks/journey.py               # 30 秒ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/27_m2_image_masks/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1.

## 結果のメモ

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (受け持ち, 正解の種類, 元のデータ) | OK | 元の `images/` と `masks/` を持つディレクトリを入れる |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | データセットの場所 `outputs/dataset` は記録される. モデルの定義の場所は記録されない (骨組みの config の `model.spec` が null) |
| ML-Fathom で問題の設定を一覧から始める | OK | 書き出しの画面 (画像と領域) に移り, 入力と正解のディレクトリ (`images`, `masks`) と分割 (random, 検証用とテスト用 15 %) が下書きで入る |
| ML-Fathom でプロジェクトへ書き出す | OK | 版 3, 学習用 42, 検証用 9, テスト用 9. 中身の手順 `resize` (32x32) と, 領域に自動でかかる `resize_mask` |
| 開き直した書き出しの画面に中身の手順が残る | OK | 問題の設定の書き出しの設定 (`options.item_steps`) に残る |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | 手 | 一覧で「プロジェクトを書き出していません」となり選べない. プロジェクトのディレクトリを打つと, 保存先が既定の `models/model.yaml` になる |
| ML-CreateArchitecture (同梱の例, 入力と出力) | 手 | 「データセットから入力と出力を作る」が版 3 のデータセットを読めない (422). 例の `head` の `out_channels` を 2 から 3 に手で直す |
| ML-CreateArchitecture でプロジェクトに保存 | OK | `models/model.yaml` |
| ML-Dashboard で書き出したプロジェクトを一覧から登録 | OK | |
| 学習の準備 | OK | 足りない機能 eda, inference. モデルの定義は config に無いので確かめの外 |
| 環境を作る (画面から) | OK | 10 秒ほど (uv のキャッシュあり) |
| 学習の起動の config | 手 | 起動の欄の config に `model.spec: models/model.yaml` と `requires` の `modelcontrol` を打つ (打たないと仮のモデルで学習し, 一式ができない) |
| 学習 | OK | 30 エポック. ベストの valid/iou 0.3356 |
| テスト用の評価 | OK | IoU は背景 0.988, 円 0.186, 四角 0.673. 背景を除く平均 0.430 |
| 学習の後の推論の一式 | OK | 一式で推論した valid/iou 0.3356 が記録と合う |
| 見本を見る | 手 | 見本が 0 件. 骨組みの `train.py` が `save_sample` を呼ばない |
| ML-Provide で実験の一式を一覧から登録 | OK | 一式の確かめ (valid の iou) が記録と合う |
| 問題の設定をつなぐ (候補) | OK | ファイルの組の受け持ちは意思決定が無いので候補に出さない (ML-Provide の決まり) |
| 新しい件の推論と出力 | 手 | 4 件. 値は 0, 1, 2 だけ. 正解を出力の大きさにして比べた画素の一致 0.873. 48x40 の 2 枚の出力は 32x32 (中身の手順の後の大きさ) で, 元の画像に重ねるには手で戻す |

手の段階 5, 誤りの段階 0.

- IoU が低いのは学習が 30 エポック (10 秒ほど) で短いため. 確かめたいのは段階のつながりで, モデルの良し悪しではない

## 気付いたこと

手の段階は, 持ち主の TODO.md の次の項目で無くなる (順は ML_tester の TODO.md).

- **ML-Quest**: ファイルの組の骨組みの config の `model.spec` が null なので, 問題の設定にモデルの定義の置き場が記録されない. このため ML-CreateArchitecture の一覧で選べず, ML-Dashboard の起動の欄に定義の場所と `modelcontrol` を打つ. 列の予測の骨組みと同じく定義の置き場を書いて記録すると, 2 つの手が無くなる
- **ML-Quest**: ファイルの組の骨組みの `train.py` が見本を書かないので, ML-Dashboard の領域の見本 (凡例) を見られない. テスト用の数件の入力, 正解, 予測を `save_sample` (`kind="mask"` と `classes`) で書くと見られる
- **ML-CreateArchitecture**: 一覧の理由「プロジェクトを書き出していません」は, プロジェクトを書き出していても, モデルの定義の置き場の記録が無ければ出る. 理由の文が実際と違う
- **ML-EDA, ML-CreateArchitecture**: `model_inputs` が版 2 と版 3 のデータセット (ファイルの組) を読まないので, 「データセットから入力と出力を作る」が使えない. 同梱の例のクラスの数 (と, 大きさやチャンネル数が違えば入力の形) を手で直す
- **ML-Inference**: 版 3 の一式は中身の手順をかけた後の大きさで領域を返す (ML-Inference の README に書いてある決まり). 元の画像に重ねるには, 画像ごとに元の大きさへ戻す (最も近い画素で) 必要がある
