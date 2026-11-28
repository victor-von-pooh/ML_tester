# 39 本物の表で表を作るを問題の設定から推論まで通す (住宅の表)

## 目的

表を作るの課題 (本物の表に似た行を作る) を, 本物の表 (California の住宅の表) で, 問題の設定から推論まで, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す (21 と同じ形). 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「総菜の作る量」から作り, 課題, 指標, 決めること, 受け持ちを表を作る課題に答え直す (架空). 地区ごとの住宅の表を外に出せないので, 似た表を作って外の人と試す. 今のやり方は手で作った表で試す
- **受け持ち**: 作るで, 1 件の単位は行, 出力の形と正解の種類は表, 正解の出どころは自分自身 (`self`), 作り方は一から学習, 最良を選ぶ指標は列ごとの分布の近さ (`column_distance`. 小さいほど良い)
- アプリの起動と段階の記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
39_table_generate/
├── journey.py   # 本物の表の一部を CSV にし, 問題の設定から推論まで通す (部品は 25 の apps.py)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ (元の表は `data/housing.csv`, 書き出したプロジェクトは `project/`).

## データ

scikit-learn の `fetch_california_housing` の表 (1990 年の California の国勢調査の区画ごとの 20,640 行) から, 種 0 で選んだ 5,000 行 (学習と検証の時間を抑えるため). 列は 9 つの数の列 `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`, `MedHouseVal` (目的の列も作る列の 1 つ). 欠損と種類の列は無い. 取得の置き場はルートの `data/sklearn/` (05 と共有. 無ければ `journey.py` が取得する). 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の雑音の予測器の雛形 `diffusion_denoiser` を, 符号化した行の幅 (数の列 9, 種類の列 0 で 9) に書き換えたもの (入力は雑音を加えた行 (8, 9) と時刻, 出力は加えた雑音の予測 `noise`). 雛形は問題の設定から決まり (`POST /api/project/target` の `example`), 幅は問題の設定を渡した `POST /api/dataset/inputs` の `table` から, `POST /api/templates/adapt` の `features` で書き換える.

## 学習の設計

ML-Pathfinder が書き出した骨組み (ML-Quest の表の生成の `train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 骨組みの既定 (100 エポック, バッチ 64, 時刻 1,000 の線形の雑音の予定, 重みの指数移動平均 0.995, 見本と一式の ddim は 50 回). 数の列は学習用の分割の平均と標準偏差で標準化する. 検証用の列ごとの分布の近さ (`valid/column_distance`. 数の列は KS 統計量の列の平均) が最も小さいエポックを選び, 検証用とテスト用の列ごとの近さを `evaluations/column_distance_<分割>.json` に書き, 雑音だけの推論の一式 (版 7. 出力の形は `table`) を実験の `inference/` に作る. 正解の出どころ `self` を `task.yaml` に書くので, ML-Found は `self` を持つ版が要る (環境づくりで git の main から入れる).

段階と API (画面のボタンに当たるもの).

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から名前を付けて作り, 課題と受け持ちを答え直し, 表を入れる | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し (入れた表を `data/` に写す), 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き, 目的の列の無いレシピ (`task: none`) のまま書き戻し, プロジェクトへ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `POST /api/sources/{id}/problem/check`, `POST /api/sources/{id}/problem/write`, `POST /api/sources/{id}/export_project` |
| 問題の設定を一覧から選び, 符号化した行の幅で雛形を書き換え, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `GET /api/examples/{name}`, `POST /api/templates/adapt`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 行の数と種で表を作る (同じ種で 2 回), 結果の行を読む | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/rows` |

推論では 1,000 行を種 7 で作り, 同じ種で作り直して列の値が一致することと, 作った表と元の表 (5,000 行) の列ごとの KS 統計量を `journey.py` が数える (学習の評価と同じ数え方を, 推論の段階の表で確かめるため). 比べる相手は元の表の 5,000 行全て (学習に使った行を含む) で, テスト用の分割と比べる学習の評価とは同じ数え方ではないので, 一致ではなく近さの平均の差で見る.

判定のしきい値: 学習の段は, テスト用の列ごとの近さの平均が 0.25 未満で OK (学ばない雑音のままの表では列ごとの KS 統計量が 1 に近く出るため). 推論の段は, 推論の表の近さの平均とテスト用の平均の差が 0.05 未満で OK (一式が学習の評価と同じ程度の表を作るかの目安). 待ちの上限と止め方は 25 と同じ.

## 確かめること

- [x] 見本から作った問題の設定を作るで表 (正解の出どころ `self`) に答え直し, 本物の表を入れると, 直すところが無く, 受け渡しの雛形 `diffusion_denoiser` と監視 `valid/column_distance` (小さいほど良い) が出て, 骨組みを書き出せる (入れた表を `data/` に写し, README が正解の出どころ `self` を持つ ML-Found を求める)
- [x] ML-Fathom が問題の設定を一覧から開き, 目的の列の無いレシピ (`task: none`, 無作為に検証用 0.15 とテスト用 0.15) で, 食い違い無くプロジェクトに書き出せる
- [x] ML-CreateArchitecture で問題の設定を一覧から選ぶと, 雛形とデータセットの欄が入り, データセットの入力から符号化した行の幅が決まる (保存の注意は 0 件)
- [x] ML-Dashboard で一覧から登録し, 画面から環境を作り (ML-Found は `self` を持つ版), 骨組みを書き換えずに学習でき, 結果に列ごとの分布の近さと作った表の見本が出る
- [x] ML-Provide で実験を一覧から登録でき, 雑音だけの一式 (版 7) の出力の形が表で, 列の名前が説明に出る
- [x] 推論で行の数と種から表ができ, 同じ種なら同じ表になり, 元の表と列ごとの分布の近さを比べられる

## 期待する結果

誤りの段階 0, 手の段階 0. テスト用の列ごとの分布の近さの平均が 0.25 未満で, 推論で作った 1,000 行の近さの平均とテスト用の平均の差が 0.05 未満.

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリをこのリポジトリと同じディレクトリに clone して `make setup` を済ませ, ルートのディレクトリで実行する (`uv` が要る). 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `39_table_generate` を入れて手で起動する.

```bash
.venv/bin/python tasks/39_table_generate/journey.py               # 2 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/39_table_generate/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境はプロジェクトの `.venv` に, ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる

## 結果のメモ

手元の Mac で, 5 つのアプリの main (ML-Quest は 2976074, 学習の環境の ML-Found は cdd1b56) を使って動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (作るで表, 正解の出どころ self, 列ごとの分布の近さ) | OK | 雛形 `diffusion_denoiser`, 監視 `valid/column_distance` (min), 分割 `random`, 誤り 0 |
| 骨組みの書き出し (表の写しと場所の記録) | OK | 記録: `dataset`, `model.yaml`. 表の写し `data/housing.csv`. README に正解の出どころ `self` を持つ ML-Found の要件 |
| ML-Fathom (目的の列の無いレシピ, プロジェクトへ書き出す) | OK | `task: none`, 分割 `random` (検証用 0.15, テスト用 0.15), 手順なし, 食い違い 0. 注意 1 件 (書き出した列を全て作るので, ID のような列は `drop_columns` で落とし, 欠損は無くす) |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `diffusion_denoiser`, データセットの欄が入る |
| ML-CreateArchitecture (雛形を行の幅に書き換え, 保存) | OK | 幅 9 (数の列 9, 種類の列 0), 入力 [8, 9], 出力 `noise`. 保存の注意 0 件 |
| ML-Dashboard で登録し環境を作る | OK | 11 秒 |
| 学習 (骨組みを書き換えない) と列ごとの分布の近さ | OK | 50 秒. ベストはエポック 99 の `valid/column_distance` 0.1458. テスト用の列の平均 0.1388 (同じ分布の行の数の違いの目安 `train_distance` 0.0278). 見本は作った表 `generated` と検証用の本物 `valid_real`. 上限 0.25 の内 |
| ML-Provide で実験を一覧から登録 | OK | 種類 `noise`, 版 7, 出力の形 `table`, 列は元の表の 9 列 |
| 推論 (行の数と種で表を作り, 元の表と比べる) | OK | 1,000 行を種 7 で作り (飛ばした行 0), 同じ種で作り直すと一致. 元の表 (5,000 行) との KS 統計量の平均 0.1258 (テスト用の平均との差 0.0130. 上限 0.05 の内) |

手の段階 0, 誤りの段階 0.

列ごとの分布の近さ (KS 統計量. テスト用は学習の評価 `column_distance_test.json`, 推論は `journey.py` が元の表と比べた値)

| 列 | テスト用 | 学習用の目安 | 推論の 1,000 行 |
| --- | --- | --- | --- |
| `MedInc` | 0.0507 | 0.0271 | 0.0476 |
| `HouseAge` | 0.1067 | 0.0504 | 0.0734 |
| `AveRooms` | 0.0653 | 0.0391 | 0.0564 |
| `AveBedrms` | 0.1067 | 0.0148 | 0.0932 |
| `Population` | 0.0773 | 0.0235 | 0.0514 |
| `AveOccup` | 0.3067 | 0.0276 | 0.3074 |
| `Latitude` | 0.2520 | 0.0208 | 0.2368 |
| `Longitude` | 0.2093 | 0.0210 | 0.1994 |
| `MedHouseVal` | 0.0747 | 0.0256 | 0.0670 |
| 平均 | 0.1388 | 0.0278 | 0.1258 |

- 推論の段階で作った表の列ごとの近さは, 学習の評価のテスト用とほぼ同じ (平均 0.1258 と 0.1388). 骨組みの評価と推論の一式が同じ表を作ることの確かめになる
- 全ての列で学習用の目安 (同じ分布でも行の数の違いから出る値) より 1.7 倍から 12 倍大きく, 元の表の代わりに使えるほど似てはいない. `MedInc`, `AveRooms`, `Population`, `MedHouseVal` は 0.1 未満で近い
- 遠い列は `AveOccup` (0.31), `Latitude` (0.25), `Longitude` (0.21). `AveOccup` は中央値 2.82 に対し最大 1,243 の外れ値で, 骨組みが標準化に使う学習用の分割の標準偏差が 20.98 に膨らみ, 標準化した値の多くが 0 の近くの狭い幅に集まる. 狭い山の形を作れていないと推定する (確かめてはいない). 緯度と経度は 2 つの都市圏に分かれた山の形で, 1 つの小さな全結合の予測器と 100 エポックでは山の分かれ方を作り切れていないと推定する
- 検証用の近さは 100 エポックの終わりの近く (エポック 99 の 0.1458) が最も良い (エポック 80 から 97 は 0.148 から 0.152 の間で上下する). 長く学ぶか (`trainer.epochs` の上書き), 外れ値の大きい列を落とすか対数にする (レシピの手順) と近づく見込みがあるが, ここでは骨組みの既定のまま通すことを確かめた
- **確かめの範囲**: 列どうしの関係 (同時の分布) は数えない (骨組みの評価も同じ). 課題は架空. 問題の設定から推論まで値の受け渡しが通ることの確かめで, 作った表が外に出してよいもの (元の行を写していない) かは確かめていない

## 気付いたこと

- 表を作るには見本が無いので, 見本「総菜の作る量」から作って課題と受け持ちを答え直す (ML-Pathfinder の表の生成のテストと同じ作り方). 正解の出どころ `self` を答えると, ML-Quest の骨組みが `task.yaml` に `origin: {type: self}` を書き, それを書ける ML-Found (ML-Dashboard の環境づくりが git の main から入れる) で学習が通る
- 本物の表の 9 列は全て数の列なので, 符号化した行の幅は列の数と同じ 9. ML-CreateArchitecture の `POST /api/dataset/inputs` は問題の設定を渡すと幅を `table` で返し, 種類の列があればワンホットの値の数を足した幅になる (ML-CreateArchitecture のテストで確かめてある). 種類の列のある本物の表では通していない
- ML-Fathom のレシピの雛形は手順が空で, そのまま書き出せる (欠損と ID のような列の無い表のため). 注意は「書き出した列を全て作る」ことの案内で, 外れ値の大きい列 (`AveOccup`) への気付きは出ない
- 骨組みの評価の `wasserstein` は Wasserstein 距離を学習用の分割の標準偏差で割った値なので, 外れ値で標準偏差が膨らむ列では小さく出る. `AveOccup` は KS 統計量が 0.31 と最も遠いのに `wasserstein` は 0.27 と列の中で最も小さい. 割る前の距離は約 5.7 (0.27 × 20.98) で, 元の表の四分位範囲 (約 0.85) よりずっと大きい. 外れ値の大きい列の近さは `wasserstein` でなく `distance` (KS 統計量) で見る
- ML-Provide の推論の 1 回の行の数の上限は 1,000 で, 結果の行は 1 回 500 行まで読める (`journey.py` は 2 回に分けて読む)
