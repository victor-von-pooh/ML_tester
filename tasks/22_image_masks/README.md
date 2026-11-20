# 22 画像と領域の対 (書き出しの版 2) の学習

## 目的

ML-EDA の書き出しの約束 `dataset.yaml` の版 2 の, 最初の利用者の 1 つ (ルートの TASK_TYPES.md の相談 16 の「画像と領域の対」). 入力も正解もファイル (画像と領域の画像) の組を, 版 2 で書き出し, `load_items` で読んで ML-Found で学習するまでの道具のつながりを確かめる. モデルの良し悪しは目的ではないので, データは小さく (32x32 の 60 組), 学習は数秒で終える.

- 元のデータは `ml_eda.synthetic.write_image_masks` で作る. 雑音の上に円か四角を 1 つ置いた画像と, その領域 (0 が背景, 1 が円, 2 が四角)
- `export_image_masks` で版 2 に書き出す. 正解は見本を作る関数の出力そのものなので, 正解の出どころ (`target.origin`) を `synthetic` にし, 作った関数の名前と設定を書く
- 学習のスクリプトは `load_items` の分割を torch の `Dataset` に包むだけで, ファイルの組を自分で探さない. クラスの数と名前, 画像のチャンネル数は `dataset.yaml` から決める
- 指標は背景を除いた IoU の平均 (`valid/miou`). エポック全体の画素数から求めるので `on_epoch_end` で返す

**測っているもの**: 指標は正解の出どころとの一致. ここでは出どころが `synthetic` (作った正確な正解) なので, 正解そのものとの一致になる. 出どころが `process` (今ある処理の出力) なら, 測れるのはその処理との一致で, モデルはその処理を超えられない. 学習のスクリプトは出どころを ML-Found の `task.yaml` (`task_from_dataset` で `dataset.yaml` から埋める) と `train.log` に残す.

学習の後に, テスト用の IoU の表を `evaluations/test_iou.json` (`save_evaluation`) に, テスト用の最初の 3 件の入力の画像と, 正解と予測の領域 (`kind="mask"`. クラスの名前を `classes` に) を `samples/test_<id>_input/`, `samples/test_<id>_truth/`, `samples/test_<id>_pred/` の `epoch_<ベストのエポック>.png` (`save_sample`) に書く. 守る指標 (`guard`) の表は書かない (問題の設定で守る指標を決めていない課題なので).

## ディレクトリ構成

```
22_image_masks/
├── configs/
│   ├── default/train.yaml          # 基準の config (data.dataset: outputs/dataset)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── quick.yaml          # 3 エポックで終える (動作の確認用)
├── make_data.py                    # 元のデータ (data/image_masks/images, masks) を ML-EDA の合成の見本で作る
├── prepare.py                      # 版 2 に書き出し (正解の出どころは synthetic), 読み直して確かめる
├── model.py                        # TinyUNet (1 段だけ縮めて戻す U-Net)
├── train.py                        # load_items を torch の Dataset に包んで学習, IoU を記録
└── outputs/                        # dataset/ (書き出したデータセット) と masks/<日付>/<時刻>/ (実験. task.yaml, evaluations/, samples/ を含む. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | `write_image_masks(count=60, size=(32, 32), seed=0)`. 同じ stem の PNG の組 (`images/sample_000.png` と `masks/sample_000.png`) |
| 版 2 の説明 | `format_version: 2`, `role: predict`, `target.kind: mask`, `target.origin.type: synthetic`. 項目 `image` (入力, 形 `[32, 32, 3]`, uint8) と `mask` (正解, 形 `[32, 32]`, 値 0, 1, 2 と名前) |
| 分割 | 1 件ごとに無作為に, 学習用 36, 検証用 12, テスト用 12 (`split: {method: random, valid: 0.2, test: 0.2, seed: 0}`) |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる.

## モデル

`model.py` の `TinyUNet`. 畳み込みの塊 (3x3 の畳み込み 2 回, BatchNorm, ReLU) で 16 チャンネルにし, 最大値のプーリングで縮めて 32 チャンネルの塊を通し, 転置畳み込みで戻して縮める前の特徴とつなぎ, 1x1 の畳み込みで画素ごとのクラスのスコアを出す. 入力のチャンネル数とクラスの数は `dataset.yaml` から決める.

## 学習の設計

`load_items` の分割を torch の `Dataset` に包んで学習する. `step` は画素ごとの交差エントロピーの損失を返し, 背景を除いた IoU の平均 (`valid/miou`) はエポック全体の画素数から `on_epoch_end` で求める. Adam (学習率 0.003), `valid/miou` を監視して Early Stopping (待つ回数 8). 学習の後に `task.yaml`, テスト用の IoU の表 (`evaluations/`), 見本 (`samples/`) を書く (上の「目的」).

## 確かめること

- [x] `export_image_masks` が組を stem で作り, 版 2 の `dataset.yaml` (受け持ち, 正解の種類, 正解の出どころ, 項目の形と型, 領域の値の名前) を書く
- [x] 正解の出どころに `type: synthetic`, `name`, `settings` を渡すと, そのまま `target.origin` に残る
- [x] `load_items` の分割 (`split("train")`) を torch の `Dataset` に包み, `DataLoader` で学習できる
- [x] クラスの数と名前, 入力のチャンネル数を `dataset.yaml` から決められる (学習のスクリプトに書かない)
- [x] 学習のスクリプトが正解の出どころを `train.log` に残す
- [x] IoU を `on_epoch_end` で求め, `valid/miou` でベストモデルを選ぶ. テスト用の分割でも同じ数え方で評価する
- [x] `task_from_dataset` で `dataset.yaml` (版 2) から受け持ち, 正解の種類と出どころを埋め, 出力の形 (`region`), 意味, 作り方 (`scratch`) を足して `write_task` で `task.yaml` に書ける
- [x] テスト用の IoU の表を `save_evaluation` で, 入力の画像と, 正解と予測の領域 (`kind="mask"`) を `save_sample` で実験のディレクトリに書ける

## 期待する結果

- 2 で `版: 2, 受け持ち: predict, 正解の種類: mask`, 正解の出どころ `{'type': 'synthetic', 'name': 'ml_eda.synthetic.write_image_masks', ...}`, 分割 `{'train': {'items': 36}, 'valid': {'items': 12}, 'test': {'items': 12}}`, 注意なし
- 3 で検証用の mIoU が 1 に近くなる. テスト用の IoU と正解の出どころは `train.log` にも残る
- 実験のディレクトリに `task.yaml`, `evaluations/test_iou.json`, `samples/index.json` と, 3 件の入力, 正解, 予測の 9 つの見本 (`samples/test_<id>_input/epoch_<ベストのエポック>.png` と `_truth`, `_pred`) ができる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/22_image_masks/make_data.py   # 1. 元のデータ (data/image_masks)
python tasks/22_image_masks/prepare.py     # 2. 版 2 に書き出し (outputs/dataset)
python tasks/22_image_masks/train.py       # 3. 学習
python tasks/22_image_masks/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (3 エポック)
```

CPU で, 1 と 2 は 1 秒ほど, 3 は 5 秒ほど (Python の起動を含む).

## 結果のメモ

PyTorch 2.15.0, CPU (Apple Silicon の Mac) での値.

| 実験 | 学習したエポック | ベストのエポック | 検証用の mIoU (ベスト) | テスト用の IoU (円 / 四角 / 背景) | テスト用の mIoU |
| --- | --- | --- | --- | --- | --- |
| 3 (`train.yaml`) | 30 | 29 | 0.9796 | 0.9656 / 0.9947 / 1.0000 | 0.9802 |
| quick (3 エポック) | 3 | 2 | 0.2557 | 0.1152 / 0.0000 / 0.9733 | 0.0576 |

- 測っているのは, 正解の出どころ `synthetic` (`ml_eda.synthetic.write_image_masks`, count 60, size 32x32, seed 0) との一致. 合成の正確な正解なので, IoU が 1 に近いのは「正解そのものに近い」ことを表す. 図形がはっきりしているので, 高い値は想定どおり
- 別のディレクトリから実行しても `history.csv` が完全に一致した
- 版 2 の書き出しから学習まで, 学習のスクリプトに組の探し方, クラスの数, チャンネル数を書かずにつながった
- `task.yaml` には `role: predict`, `target.kind: mask`, `target.origin` (type, name, settings), `dataset` (`path: outputs/dataset`, `name: image_masks`, `format_version: 2`) が埋まり, 足した `output: region`, `what`, `method: scratch` と一緒に書かれた. `describe_origin` は「正解は作った正確な値 (ml_eda.synthetic.write_image_masks). 実際のデータでの性能は別に確かめる」を返した
- `evaluations/test_iou.json` は `epoch: 29`, `guard: false` で, クラスごとの行と `miou` の行を持つ. 見本は `samples/test_sample_005_input/epoch_0029.png` など 9 つ (32x32). 正解と予測は `kind: mask` で, 決まった色の表で色分けした PNG と, 一覧の `classes` に番号, 名前, 色 (`background` が黒, `circle` が `#800000`, `square` が `#008000`) が残る

## 気付いたこと

### ML-EDA

- `load_items` が返す画像の配列は書き込めない (`np.asarray` で Pillow の画像から作るため). そのまま `torch.from_numpy` に渡すと PyTorch の警告 (`The given NumPy array is not writable`) が出る. ここでは `np.array` で写してから渡した. 領域の配列は書き込めるので出ない (同じ `load_items` でも項目によって違う)
- 版 2 の説明の `classes` は `value` と `name` の並び (値の順) で, 領域の値をそのままクラスの番号にしてよいかは書かれない. 値から番号の表は学習のスクリプトで作った. 値が 0 から続かない領域 (255 を境界にするなど) でも同じ書き方で済む

### ML-Found

- `output`, `what`, `method` は `dataset.yaml` に無いので, 学習のスクリプトが書く. `output` の値 (ここでは `region`) は ML-Quest の出力の形から選ぶが, 学習の側からは問題の設定の受け渡しを読む決まった口が無く, 手で写した
- 領域の見本は `save_sample(..., kind="mask", classes=names)` にクラスの番号の配列を渡すだけで, 色分けと番号の表は ML-Found が作る. 入力と並べた 1 枚の画像にはならないので, 入力は別の見本 (`kind="image"`) にした
- 評価の表の 1 行目からクラスごとの行で, 平均 (`miou`) も同じ列の行として足した. 表の中の「まとめの行」を区別する決まりは無い
