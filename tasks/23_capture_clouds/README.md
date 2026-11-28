# 23 多視点の撮影から点群を画面の操作で学習する

## 目的

撮影 (画像, 姿勢, 深度のまとまり) と正確な点群の組 (ルートの TASK_TYPES.md の相談 16 の「単純な形から作った多視点の撮影と正確な点群」) を, 利用者が画面だけで, 問題の設定から始めて書き出し, 骨組みで学習し, 新しい撮影を推論するまでの手順を示す. ライブラリの関数を呼ぶコードは書かない. 手で用意するのは元のデータ (`make_data.py`) だけ. あわせて, 重い評価 (Chamfer 距離) を数エポックおきにだけ行う ML-Found の `trainer.valid_every_n_epochs` を, 骨組みの config のまま使えることを確かめる (相談 6).

- 元のデータは ML-EDA の合成の見本 (`write_shape_captures`). 原点の近く (水平に 0.5 から 1.5 m) に置いた球と箱 40 個の正確な点群 (2000 点) と, 1 つの撮影 8 枚 (24x32) の画像, 深度, 信頼度
- ML-Fathom の書き出しは 1 件ごとに物の中心を原点にする平行移動 (`view_center`) をかける (ML-EDA の決まり). 正解の出どころ (`target.origin`) は `synthetic`
- モデルは ML-CreateArchitecture の同梱の例 `point_cloud_predictor` (入力 512 点から 256 点を出す PointNet の形). 学習は ML-Pathfinder が書き出した骨組みを書き換えずに ML-Dashboard で起動する
- 推論で見ること: ML-Provide の推論の点群が, 正規化の平行移動を戻した元の座標 (撮影の世界の座標) にあるか. 新しい形の正解の点群の重心と比べる. 原点から離して置いた形は 28 で確かめる

**測っているもの**: 指標は正解の出どころとの一致. ここでは出どころが `synthetic` (形の式から置いた面の上の点) なので, 本当の形との一致になる. 実際の撮影で, 正解が今ある処理 (別の復元の道具など) の出力 (`process`) なら, 測れるのはその処理との一致で, モデルはその処理を超えられない.

`journey.py` は, 下の「実行方法」の画面の操作を, 画面と同じ API で順に流して確かめる (段階ごとに OK, 手, 誤り). 部品は 28 の `journey.py`, 27 の `pairs.py`, アプリの起動と記録は 25 の `apps.py`.

## ディレクトリ構成

```
23_capture_clouds/
├── make_data.py   # 元のデータ (data/capture_clouds と, 新しい撮影の data/capture_clouds_new) を ML-EDA の合成の見本で作る
├── journey.py     # 実行方法の画面の操作を API で流して確かめる
└── README.md
```

プロジェクト (骨組み, 書き出したデータセット, モデルの定義, 実験) は, ML-Pathfinder で選んだ書き出し先にできる. このディレクトリには置かない.

## データ

| 項目 | 内容 |
| --- | --- |
| 元のデータ | `write_shape_captures(count=40, views=8, size=(24, 32), points=2000, seed=0)`. 球 20 個と箱 20 個 (種類, 位置, 大きさ, 色は ML-EDA が `seed` から決める. 中心は原点から水平に 0.5 から 1.5 m, 高さ 0.05 から 0.3 m. 球の半径 0.08 から 0.18 m, 箱の辺 0.1 から 0.3 m). `captures/<名前>/` と `clouds/<名前>.ply` |
| 書き出し | ML-Fathom のファイルの組の書き出し. 版 2, `role: predict`, `target.kind: point_cloud`, `target.origin.type: synthetic`. 正規化は `view_center` (平行移動だけ) |
| 分割 | ML-Fathom の下書きの分割 (1 件ごとに無作為, 検証用とテスト用 15 %, 種 0). 学習用 28, 検証用 6, テスト用 6 |
| 新しい撮影 | `count=1`, `seed=1` の形 1 つ (`data/capture_clouds_new`. 撮影の名前は `shape_000`). 書き出しには入れず, 正解の点群は推論の結果と比べるだけに使う |

用意してもらう必要があるデータは無い. ML-EDA はルートの README の「導入」で入れる.

## モデル

ML-CreateArchitecture の同梱の例 `point_cloud_predictor`. 「データセットから入力と出力を作る」が入力 `[4, 512, 3]` と正解の点の数 2000 を返す. 点群の出す点の数は画面が当てない決まりで, 例の入力 512 点が骨組みの `data.input_points` と同じなので, 点の数は例のまま使う. 入力の名前は, 画面が例の `observed` からデータセットの項目の名前 `capture` に変える.

## 学習の設計

ML-Pathfinder が書き出した骨組みの config のまま (エポックの数 30, Adam の学習率 0.001, `trainer.valid_every_n_epochs: 2`, `valid/chamfer` を監視して Early Stopping. 待つ回数 5). 損失 (Chamfer), 深度から点に戻す処理, テスト用の Chamfer 距離の表, 元の座標に戻す値の決まり付きの点群の見本は骨組みの `train.py` が書く.

## 確かめること

- [x] 空の問題の設定から, 予測するで正解の種類が点群の受け持ちを答え, 元の撮影と点群のディレクトリを入れて骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から始め, 入力と正解のディレクトリと分割を下書きで埋め, 正解の出どころを足してプロジェクトの `outputs/dataset` に書き出せる
- [x] 正解の出どころ (`type: synthetic`, `name`, `settings`) が書き出した `dataset.yaml` と実験の `task.yaml` に残る
- [x] ML-CreateArchitecture で問題の設定を一覧から選び, 同梱の例 `point_cloud_predictor` から組んだモデルで, 骨組みの `train.py` を書き換えずに学習できる (起動の欄に何も打たない)
- [x] 骨組みの config の `trainer.valid_every_n_epochs` で, 検証 (と Chamfer 距離) は決まったエポックおきだけになり, ベストはその中から選ばれる
- [x] 学習の後に実験の中に推論の一式ができ, 一式の確かめ (`valid/chamfer`) が学習の記録と合う
- [x] ML-Dashboard で点群の見本を元の座標で見る
- [x] ML-Provide で実験の一式を一覧から登録でき, 新しい撮影の推論の点群が元の座標にある

## 期待する結果

誤りの段階 0, 手の段階 0.

## 実行方法

準備は 25 と同じ (5 つのアプリを clone して `make setup`). 画面の操作の手順.

1. 元のデータを作る. ルートのディレクトリで `.venv/bin/python tasks/23_capture_clouds/make_data.py` (`data/capture_clouds` に `captures/` と `clouds/`, `data/capture_clouds_new` に新しい撮影)
2. ML-Pathfinder: 空の問題の設定を作り, 問いに答える (予測する, 1 件は 1 つの物の多視点の撮影, 単位はまとまり (`bundle`), 出力の形は形 (`shape`), 正解の種類は点群 (`point_cloud`), 正解はファイル, 指標は Chamfer 距離). 元のデータに 1 の `data/capture_clouds` を入れ, 骨組みを書き出す
3. ML-Fathom: 問題の設定の一覧から 2 の設定を選んで始める. 書き出しの画面 (撮影と点群) に入力と正解のディレクトリ (`captures`, `clouds`) と分割が下書きで入る. 正解の出どころ (種類 `synthetic`, 名前 `ml_eda.synthetic.write_shape_captures`, 設定) と点群の単位 (メートル) を足し, プロジェクトへ書き出す
4. ML-CreateArchitecture: 問題の設定の一覧から 2 の設定を選ぶ. 同梱の例 `point_cloud_predictor` を開き, 「データセットから入力と出力を作る」を押し, プロジェクトに保存する (`models/model.yaml`)
5. ML-Dashboard: 書き出したプロジェクトを一覧から登録し, 「環境を作る」, 「学習を起動」(config の欄は骨組みのまま). 結果で Chamfer 距離の学習曲線 (検証したエポックだけ), テスト用の表, 点群の見本 (元の座標) を見る
6. ML-Provide: 実験の一式を一覧から登録し, `data/capture_clouds_new/captures` の撮影を推論して点群をダウンロードする

この手順を API で流して確かめるには, ルートのディレクトリで実行する.

```bash
.venv/bin/python tasks/23_capture_clouds/journey.py               # 1 分半ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/23_capture_clouds/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1. データは作業用の場所に作る (1 と同じ設定).

## 結果のメモ

5 つのアプリの main (ML-Pathfinder 17b542e, ML-Fathom 29f20f1, ML-CreateArchitecture fe9859d, ML-Dashboard 7c086d0, ML-Provide 5b2f4ea) を, 上の「実行方法」で動かした値. 説明は `journey.py` が書く文 (パスと長い値は略す).

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (受け持ち, 正解の種類, 元のデータ) | OK | 元のディレクトリを入れる. 確かめ: データが 40 件と少なく, 機械学習で学べることは限られます. 今のやり方やルールで足りないかも考えてください |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | 記録: データセット outputs/dataset, モデルの定義 models/model.yaml |
| ML-Fathom で問題の設定を一覧から始める | OK | 書き出しの画面 (capture_clouds) に移り, 入力と正解のディレクトリ (captures, clouds) と分割 {'method': 'random', 'valid': 0.15, 'test': 0.15, 'seed': 0} が下書きで入る |
| ML-Fathom でプロジェクトへ書き出す (書き出しの設定を書き戻す) | OK | 版 2, 分割 {'train': 28, 'valid': 6, 'test': 6}, 中身の手順 無し, 注意 無し |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 保存先が問題の設定の記録から決まる |
| ML-CreateArchitecture (同梱の例 point_cloud_predictor, 入力と出力) | OK | データセットから入力と出力: 入力 capture `[4, 512, 3]`, 出す点の数 2000 (略). 画面が当てる: 入力 observed の名前を capture にしました |
| ML-CreateArchitecture でプロジェクトに保存 | OK | 保存先 models/model.yaml |
| ML-Dashboard で書き出したプロジェクトを一覧から登録 | OK | 問題の設定「撮影から形の点群を作る (原点の近くの形)」が記録したプロジェクト |
| 学習の準備 (データセット, モデルの定義, 環境) | OK | データセット True, モデルの定義 models/model.yaml (exists True), 足りない機能 ['eda', 'modelcontrol', 'inference'] |
| 環境を作る (画面から) | OK | succeeded, 10 秒, 足りない [] |
| 学習の起動の config (エポックの数, モデルの定義) | OK | エポックの数 30 (骨組みのまま) |
| 学習 | OK | valid/chamfer 0.0159 (30 エポックまで, 15 秒) |
| テスト用の評価 (ML-Dashboard の結果) | OK | {'test_chamfer': [{'answer': 'model', 'chamfer': 0.0204}]} |
| 学習の後の推論の一式 (ML-Dashboard のログ) | OK | 推論の一式: valid の chamfer 0.0159 (記録 0.01594570279121399, 合う: True) |
| 見本を見る (ML-Dashboard の結果) | OK | 点群の見本 9 件, 元の座標 あり |
| ML-Provide で実験の一式を一覧から登録 (一式の確かめ) | OK | 問題の設定「撮影から形の点群を作る (原点の近くの形)」の実験 point_cloud_predict/(日時). 一式の確かめ: valid の chamfer が記録と合う (matches True) |
| 問題の設定をつなぐ (候補) | OK | 候補 無し. 受け持ちが列の予測でないので候補に出さない決まり |
| 新しい件の推論と出力 | OK | 1 件 (件の名前で正解と引き合わせ). 食い違い 無し. 形 [(256, 3)], 正解の重心とのずれ [0.006] m (正解の重心の原点からの距離 [1.48] m. 上限 0.1 m). 文: ['1 件のうち 1 件を推論'] |

手の段階 0, 誤りの段階 0.

- **検証の間隔**: `history.csv` の `valid/chamfer` は 30 エポックのうち偶数のエポックの 15 行だけにあり, 残りは空. ベストのエポック 30 も検証したエポックの中から選ばれた. 骨組みの config が `trainer.valid_every_n_epochs: 2` を持つので, 利用者は欄を打たずに済む
- 実験の `task.yaml` に `target.origin` (type, name, settings) が書き出しのとおり残る
- 確かめたいのは段階のつながりで, モデルの良し悪しではない. 検証用の Chamfer 距離は最後の検証 (エポック 30) でも下がっており, Early Stopping に届かずに 30 エポックで終えた

## 気付いたこと

- 深度から点に戻す式 (カメラの軸, 深度の測り方, 画素の中心), Chamfer 距離, 見本の元の座標に戻す値の決まり (`shift` の向き) は, 骨組みと ML-Fathom の書き出しが受け持つ. 利用者が打つのは正解の出どころと点群の単位だけ
- 「データセットから入力と出力を作る」の値を例に当てる所は, 28 の `journey.py` と同じく 27 の `apply_unet_info` を使う. 画面と同じく例の入力の名前 (`observed`) をデータセットの項目の名前 (`capture`) に変え, 保存したモデルの定義も `capture` になる
