# 23 多視点の撮影から点群を作る (書き出しの版 2 と検証の間隔)

## 目的

ML-EDA の書き出しの約束 `dataset.yaml` の版 2 の, 最初の利用者の 1 つ (ルートの TASK_TYPES.md の相談 16 の「単純な形から作った多視点の撮影と正確な点群」). 撮影 (画像, 姿勢, 深度のまとまり) と正解の点群の組を版 2 で書き出し, `load_items` で読んで ML-Found で学習するまでの道具のつながりを確かめる. あわせて, 重い評価 (Chamfer 距離) を数エポックおきにだけ行う ML-Found の `trainer.valid_every_n_epochs` を使い, 検証したエポックだけでベストモデルを選ぶことを確かめる (相談 6). モデルの良し悪しは目的ではないので, データは小さく (40 件, 1 件 8 枚の 24x32), 学習は数秒で終える.

- 元のデータは `ml_eda.synthetic.write_shape_captures` で作る. 球と箱の面の上の正確な点群 (2000 点) と, 物の周りの決まった姿勢から光線を当てて求めた画像, 深度, 信頼度の撮影. 位置, 大きさ, 色を変えた 40 個の形を渡す (既定の形は 4 つなので)
- `export_capture_clouds` で版 2 に書き出す. 1 件ごとに, 姿勢だけから求めた物の中心を原点にする平行移動を, 撮影の姿勢と正解の点群の両方にかける (回転と拡大縮小はかけず, 実寸のメートルを残す). 正解の出どころ (`target.origin`) は `synthetic`
- 学習のスクリプトは `load_items` で読み, 各枚の深度を内部パラメータと姿勢で世界の座標に戻した点から 512 点を選んで入力にする. モデルは点の並びに依らない特徴 (PointNet 風) から全結合の層で 256 点を出す
- 損失は 2 乗の距離の Chamfer (plain PyTorch の `torch.cdist`). 指標は距離の Chamfer (メートル) と 1 cm 以内の F 値で, `on_epoch_end` の検証のときだけ求める. `trainer.valid_every_n_epochs: 3` なので, 検証はエポック 3, 6, 9, ... と最後のエポックだけ. `valid/chamfer` (小さいほど良い) でベストモデルを選ぶ
- 書き出しに入れない新しい撮影に `normalize_capture` で同じ正規化をかけて予測し, `shift` を引いて元の座標に戻して元の点群と比べる (推論の段階の準備)

**測っているもの**: 指標は正解の出どころとの一致. ここでは出どころが `synthetic` (形の式から置いた面の上の点) なので, 本当の形との一致になる. 実際の撮影で, 正解が今ある処理 (別の復元の道具など) の出力 (`process`) なら, 測れるのはその処理との一致で, モデルはその処理を超えられない. 学習のスクリプトは出どころを ML-Found の `task.yaml` (`task_from_dataset` で `dataset.yaml` から埋める) と `train.log` に残す.

学習の後に, テスト用の Chamfer 距離と F 値の表を, 入力の点そのものを比べる相手 (`baseline: input_points`) にして `evaluations/test_chamfer.json` (`save_evaluation`) に, テスト用の最初の 2 件の予測と正解の点群を `samples/test_<id>_pred/` と `samples/test_<id>_truth/` の `epoch_<ベストのエポック>.ply` (`save_sample`) に, 元の座標に戻す値の決まり (`convention`. ML-EDA の `shift` を符号を変えずに写す) を添えて書く. 書いた正解の見本を `load_samples` と `original_coordinates` で元の座標に戻し, 元の点群と重なることを確かめる. 守る指標 (`guard`) の表は書かない (問題の設定で守る指標を決めていない課題なので).

## ディレクトリ構成

```
23_capture_clouds/
├── configs/
│   ├── default/train.yaml          # 基準の config (valid_every_n_epochs: 3, monitor: valid/chamfer)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── quick.yaml          # 4 エポックで終える (検証はエポック 3 と最後の 4)
├── make_data.py                    # 元のデータ (data/capture_clouds と, 新しい撮影の data/capture_clouds_new) を ML-EDA の合成の見本で作る
├── prepare.py                      # 版 2 に書き出し (正解の出どころは synthetic), 読み直して正規化を確かめる
├── model.py                        # CloudPredictor (点の特徴と, 決まった数の点を出す全結合の層)
├── train.py                        # 深度から点に戻して学習, 検証のエポックだけ Chamfer 距離, 新しい撮影の予測
└── outputs/                        # dataset/ (書き出したデータセット) と clouds/<日付>/<時刻>/ (実験. task.yaml, evaluations/, samples/ を含む. git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | `write_shape_captures(views=8, size=(24, 32), points=2000, seed=0)` に, 球と箱を交互に 40 個 (中心は x と z が -2 から 2 m, y が 0 から 0.3 m. 球の半径 0.08 から 0.2 m, 箱の辺 0.1 から 0.35 m) |
| 版 2 の説明 | `format_version: 2`, `role: predict`, `target.kind: point_cloud`, `target.origin.type: synthetic`. 項目 `capture` (入力, `placement: directory`, 形 `[8, 24, 32]`, OpenGL のカメラの軸, 世界の上は +Y) と `cloud` (正解, 形 `[2000, 3]`, メートル). `normalization` は `view_center` (平行移動だけ, `normalized = original + shift`) |
| 分割 | 1 件ごとに無作為に, 学習用 24, 検証用 8, テスト用 8 (`split: {method: random, valid: 0.2, test: 0.2, seed: 0}`) |
| 新しい撮影 | 違う乱数の箱 1 つ (`new_001`). 書き出しには入れない |

用意してもらう必要があるデータは無い (`make_data.py` が作る). ML-EDA はルートの README の「導入」で入れる. ML-Found は submodule のものを使う (`trainer.valid_every_n_epochs` と, `task.yaml`, 評価, 値の決まり (`convention`) 付きの見本を書く関数, 見本の一覧の版 2 (`shift` が ML-EDA と同じ向き) と `original_coordinates` が要る).

### 深度から点に戻す

`load_items` の撮影は, `intrinsics` (`fl_x`, `fl_y`, `cx`, `cy`), `poses` (カメラから世界への 4x4. 無い姿勢は NaN), `depth` ((枚数, 高さ, 幅) のメートル, 欠けは 0) を返す. 画素 (行 r, 列 c) の深度 d を, カメラの座標 ((c + 0.5 - cx) / fl_x * d, -(r + 0.5 - cy) / fl_y * d, -d) にし, 姿勢で世界に戻す. カメラの軸は OpenGL (+X 右, +Y 上, -Z 前) で, 深度はカメラの前の向き (-Z) の距離とした (合成の見本の作り方に合わせた. 下の「気付いたこと」).

## モデル

`model.py` の `CloudPredictor`. 点ごとの 1x1 の畳み込み 3 段 (チャンネル 32, 64, 128) の後に最大値でまとめる特徴 (`PointEncoder`. 点の並びに依らない) から, 全結合 3 層 (隠れ層 256) で 256 点の座標を出す (`PointDecoder`).

## 学習の設計

損失は 2 乗の距離の Chamfer. 距離の Chamfer と 1 cm 以内の F 値は `on_epoch_end` の検証のときだけ求める. Adam (学習率 0.001), `trainer.valid_every_n_epochs: 3`, `valid/chamfer` を監視して Early Stopping (待つ回数 5. 検証の回数で数える). 学習の後に `task.yaml`, テスト用の表 (`evaluations/`), 予測と正解の点群の見本 (`samples/`) を書く (上の「目的」).

## 確かめること

- [x] `export_capture_clouds` が撮影と点群を名前で組にし, 版 2 の `dataset.yaml` (受け持ち, 正解の種類, 正解の出どころ, 正規化, 項目の座標の決まり) を書く
- [x] 正解の出どころに `type: synthetic`, `name`, `settings` を渡すと, そのまま `target.origin` に残る
- [x] 正規化した後の正解の点群の中心が原点の近くに来る (姿勢だけから決めるので, ちょうど 0 ではない)
- [x] `load_items` の撮影 (内部パラメータ, 姿勢, 深度) から深度を世界の点に戻せ, 正解の点群と同じ座標に重なる (入力の点そのものの Chamfer 距離が約 1 cm)
- [x] 学習のスクリプトが正解の出どころを `train.log` に残す
- [x] `trainer.valid_every_n_epochs: 3` で, 検証 (と `on_epoch_end` の Chamfer 距離) はエポック 3 の倍数と最後のエポックだけ. 検証しないエポックの `valid/chamfer` は `History.get` で `None`, `history.csv` で空になる
- [x] ベストモデルは検証したエポックの中から選ばれ, Early Stopping の待つ回数は検証の回数で数える
- [x] 学習曲線 (`chamfer_curve.png`) は検証したエポックだけを点で描く
- [x] 新しい撮影に `normalize_capture` で同じ正規化をかけられ, その `shift` は `capture_shift` と一致する. 予測から `shift` を引くと元の座標で元の点群と比べられる
- [x] `task_from_dataset` で `dataset.yaml` (版 2) から受け持ち, 正解の種類と出どころを埋め, 出力の形 (`shape`), 意味, 作り方 (`scratch`) を足して `write_task` で `task.yaml` に書ける
- [x] テスト用の表を比べる相手つきで `save_evaluation` で, 予測と正解の点群を `save_sample` (`kind="point_cloud"`. PLY. 元の座標に戻す `convention` 付き) で実験のディレクトリに書ける
- [x] 見本の `convention` に ML-EDA の `shift` (正規化した座標 = 元の座標 + shift) を符号を変えずに写せ, `load_samples` の値の決まりを `original_coordinates` に渡すと, 正解の見本が元の点群の座標に戻る

## 期待する結果

- 2 で `版: 2, 受け持ち: predict, 正解の種類: point_cloud`, 正解の出どころ `{'type': 'synthetic', 'name': 'ml_eda.synthetic.write_shape_captures', ...}`, 分割 `{'train': {'items': 24}, 'valid': {'items': 8}, 'test': {'items': 8}}`, 注意なし. 正解の点群の中心は原点から数 mm
- 3 で端末に出る「検証したエポック」が 3 の倍数 (と最後のエポック) だけになる. quick では `[3, 4]`. `train.log` には出さないので, 後から見る場合は `history.csv` の `valid/chamfer` の空でない行を見る
- 3 でテスト用の Chamfer 距離が `train.log` に残る. 深度から戻した点をそのまま答えにした場合 (入力の点) も並べて記録する
- 実験のディレクトリに `task.yaml`, `evaluations/test_chamfer.json`, `samples/index.json` (`format_version: 2`) と 4 つの点群 (`samples/test_<id>_pred/epoch_<ベストのエポック>.ply` と `_truth`) ができる
- 3 で「正解の見本 <id> を元の座標に戻した点群と元の点群の点ごとのずれの最大」が 2 件とも 1e-6 m 未満 (PLY に書いた桁の分だけずれる)

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/23_capture_clouds/make_data.py   # 1. 元のデータ (data/capture_clouds と data/capture_clouds_new)
python tasks/23_capture_clouds/prepare.py     # 2. 版 2 に書き出し (outputs/dataset)
python tasks/23_capture_clouds/train.py       # 3. 学習
python tasks/23_capture_clouds/train.py --experiment configs/experiment/examples/quick.yaml   # 動作の確認 (4 エポック)
```

CPU で, 1 と 2 は 1 秒ほど, 3 は 6 秒ほど (Python の起動を含む).

## 結果のメモ

PyTorch 2.15.0, CPU (Apple Silicon の Mac) での値.

| 実験 | 学習したエポック | 検証したエポック | ベストのエポック | 検証用の Chamfer 距離 (ベスト) | テスト用の Chamfer 距離 | テスト用の F 値 (1 cm) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 (`train.yaml`) | 96 (Early Stopping) | 3, 6, ..., 96 (32 回) | 81 | 0.01949 m | 0.02766 m | 0.0968 |
| quick (4 エポック) | 4 | 3, 4 | 3 | 0.03006 m | 0.03671 m | 0.0335 |
| (参考) 入力の点そのもの | - | - | - | - | 0.01191 m | 0.5220 |

- 測っているのは, 正解の出どころ `synthetic` (`ml_eda.synthetic.write_shape_captures`, 40 個, 8 枚, 24x32, 2000 点, seed 0) との一致. 合成の正確な点群なので, 距離は本当の形からの距離
- **検証の間隔**: 3 では, 96 エポックのうち `valid/chamfer` があるのは 32 エポックだけで, 残りの 64 エポックは `None`. ベストのエポック 81 の後, 84, 87, 90, 93, 96 の 5 回の検証で改善せず (`patience: 5`), エポック 96 で止まった (15 エポック). quick では予定の最後のエポック 4 (3 で割り切れない) も検証した. どちらも ML-Found の docs/DESIGN.md の「検証の間隔」の説明どおり
- **モデルは入力の点そのものより悪い**: 入力 (深度から戻した 512 点) は見えている面の上の点なので, それだけで Chamfer 距離が約 1.2 cm になる. モデルは点群全体を 1 つの特徴にまとめてから 256 点を出すので, 24 件の学習では細部を戻せず 2.8 cm にとどまった. ここで確かめたいのは道具のつながりで, モデルの良し悪しではない. 本気で点群を作るなら, 入力の点を足し込む形 (残差) や点の数を増やす
- **新しい撮影**: `new_001` (箱) の `shift` は `[-1.3108, -0.1228, -0.1984]` で, `normalize_capture` の値と `capture_shift` の値が一致した. 予測から `shift` を引いた点群と元の点群の Chamfer 距離は 0.02275 m (F 値 0.0951). 正規化した座標で学習したモデルを, 元の座標の撮影に使えることを確かめた
- 別のディレクトリから実行しても `history.csv` が完全に一致した. `data/` と `outputs/` はどこから実行してもリポジトリの中に作られ, `--experiment` の相対パスはタスクのディレクトリから辿る
- **F 値 (1 cm) の読み方**: 正解の 2000 点の間隔は 4 から 8 mm ほどで, 予測は 256 点しかないので, 面を正しく当てても 1 cm 以内に予測のある正解の点の割合 (再現率) に上限がある (入力の 512 点そのものでも 0.52). F 値はモデルどうしの比べに使い, 値の大きさそのものは読まない. quick の新しい撮影は Chamfer 距離 0.02611 m, F 値 0.0876

## 気付いたこと

### ML-EDA

- **深度の決まりが説明に無い**: 版 2 の `capture` の `convention` には姿勢 (カメラから世界), カメラの軸, 世界の上の向きがあるが, 深度がカメラの前の向きの距離 (z) か光線に沿った距離か, 画素の位置を画素の中心 (+0.5) で数えるかは書かれていない. ここでは合成の見本の作り方 (`synthetic.render_view`) を読んで決めた. 深度から点に戻すのは撮影の学習でまず行う処理なので, `convention` に `depth: z` と画素の中心を足すと, 読む側が読み違えない
- **1 つの撮影を読む公開の関数が無い**: 推論のときは `normalize_capture` で新しい撮影を写すが, 写した撮影を `load_items` と同じ形の配列で読む関数 (`ml_eda.items.read_capture`) と, 点群を読む関数 (`ml_eda.media.read_cloud`) は `ml_eda` の公開の名前に無い. ここでは中のモジュールから読んだ. `normalize_capture` と組で使う `read_capture` を公開すると, 推論の段階 (ML-Inference) も同じ読み方を使える
- **説明の `source` に手元のパスが残る**: `dataset.yaml` の `source.captures` と `source.clouds` は, 渡したディレクトリの文字をそのまま書く (ここでは `prepare.py` が絶対パスを渡したので絶対パス). 版 1 の `source.path` は常に絶対パスにして書く. データセットを人に渡すと手元のディレクトリの構成が伝わる (書き出したものはここでは git の管理外). 渡すときに外せる設定か, データセットからの相対の場所で書く設定があるとよい
- **合成の見本の数**: `write_shape_captures` の既定の形は 4 つなので, 分割して学習するには形の並びを自分で作る必要があった (`make_data.py` の `make_shapes`). 数と乱数の種から形をばらつかせる引数があると, 道具の確かめが短く書ける

### ML-Found

- `trainer.valid_every_n_epochs` は ML-Found の docs/DESIGN.md の説明どおりに動いた. `on_epoch_end` を継承して `phase == "valid"` のときだけ Chamfer 距離を求めると, 検証しないエポックでは呼ばれないので, 重い評価の書き方が素直だった (`phase` が `train` のときは空の辞書を返す)
- 正解の出どころ (`target.origin`) は `task.yaml` に残る. `task_from_dataset` が `target.origin` (type, name, settings), `role: predict`, `target.kind: point_cloud`, `dataset` (`path: outputs/dataset`, `name: capture_clouds`, `format_version: 2`) を埋め, 学習のスクリプトは `output`, `what`, `method` を足すだけで済んだ
- `evaluations/test_chamfer.json` は `epoch: 81`, `baseline: input_points` で, `answer` の列 (`model` と `input_points`) の 2 行と, 比べる相手との差の列 (`chamfer_diff`, `fscore_1cm_diff`. モデルの行で 0.01575 m と -0.4252). ML-Found の `docs/FORMAT.md` は `baseline` を「差の列を持つ場合」とし, 列の中身はプロジェクトが決めるので, 比べる相手の行と差の列の両方を書いた
- 点群の見本は書き出しの正規化した座標 (メートル) のまま書き, `convention` に `units: m`, `shift`, `scale: 1.0` を添えた. ML-Found の見本の `shift` は ML-EDA の `shift` と同じ向き (正規化した座標 = 元の座標 + shift) なので, 分割の表の `shift_x`, `shift_y`, `shift_z` をそのまま写せる. 元の座標に戻すのは `original_coordinates` (値 × scale - shift) に任せられ, 符号を自分で考えなくてよい. 正解の見本を戻した点群と元の点群の点ごとのずれの最大は 2 件とも 6e-8 m ほど (PLY に書いた桁の分)
