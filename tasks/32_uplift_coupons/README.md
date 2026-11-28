# 32 効果を見積もるを問題の設定から推論まで通す (クーポン)

## 目的

効果を見積もるの課題 (無作為に配ったクーポンの記録から, 顧客ごとに買う確率がどれだけ上がるかを見積もる) を, 問題の設定から推論まで, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. 効果の決め方の 2 つ (処置の群と対照の群で 2 つの予測を別々に学ぶ T-learner と, 処置の列を入力の終わりに足して 1 つの予測で学ぶ S-learner) を, 同じデータで 1 回ずつ通す. 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「解約の引き止め」から作り, 課題, 指標, 決めること, 受け持ちをクーポンの課題に答え直す (架空). 1 回に配れるのは 300 枚までで, 誰に配るかを選ぶ. 今のやり方は全員に配る
- **受け持ち**: 効果を見積もるで, 方策でなく今ある記録から集める. 処置の列 `coupon`, 結果の列 `purchased` (起きたか. 効果は確率の差), 行の id の列 `customer_id`, 効果の見積もりに使う列 `age`, `visits`, `region`, 評価で見る上位の割合 K 0.3, 最良を選ぶ指標は Qini の係数, 判断は効果の上位から配る (配る数の上限 300)
- アプリの起動と段階の記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
32_uplift_coupons/
├── journey.py   # 合成の記録を作り, 2 つの効果の決め方で通す (部品は 25 の apps.py)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ. ただし書き出したプロジェクトは効果の決め方ごとに `project_t/` (T-learner) と `project_s/` (S-learner) に置く.

## データ

`journey.py` が作業用の場所に作る合成の記録 (3,000 行. 乱数を固定). 列は `customer_id`, `age`, `visits` (来店の回数), `region` (`north`, `south`, `west`), `coupon` (無作為に半分に配った. 1 か 0), `used` (クーポンを使ったか. 配った後に決まる列), `purchased` (買ったか. 1 か 0). 配らないときに買う確率は年齢で少し上がり, 配ったときの上がり幅 (効果) は 0.03 に, 来店が 5 回以上なら 0.25, 北の地域なら 0.1 を足したもの. 推論には, 同じ作り方で別の乱数の種から作った 1,000 行 (`coupons_trial.csv`) を使う. 処置と結果の列を持つので, ML-Provide が効果を数え直す. 学習に使った行で測ると効果の順の当たりが甘く出るため. 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の効果の雛形を, データセットの列の数 (`age`, `visits` と `region` のワンホットの 3 列で 5) に書き換えたもの. T-learner は雛形 `tabular_uplift` (入力の幅 5, 出力は処置したときの予測 `treated` としないときの予測 `control`), S-learner は雛形 `tabular_uplift_s` (処置の列を入力の終わりに足すので入力の幅 6, 出力は結果の予測 `outcome`). 雛形は問題の設定から決まり (`POST /api/project/target` の `example`), 幅は `POST /api/templates/adapt` で書き換える.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 骨組みは検証用の Qini の係数 (`valid/qini`) が最も大きいエポックを選び, 検証用とテスト用の Qini の係数と上位 K 割の効果の差を評価の表 (`evaluations/uplift_<分割>.json`) に, Qini の曲線を `evaluations/qini_curve_<分割>.json` に書き, 効果の推論の一式を実験の `inference/` に作る (T-learner は版 12, S-learner は版 18).

段階と API (画面のボタンに当たるもの). 効果の決め方ごとに, 問題の設定から別に作って通す.

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から名前を付けて作り, 課題と受け持ちを答え直し, 表を入れ, 効果の見積もりの列, 決め方, K, 指標, 配る数の上限に答える | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き, 介入の記録を要約し (群の数と結果の差, 配った後に決まる列), ワンホットを足し, 書き戻し, プロジェクトへ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `POST /api/sources/{id}/uplift`, `PUT /api/sources/{id}/recipe`, `POST /api/sources/{id}/problem/write`, `POST /api/sources/{id}/export_project` |
| 問題の設定を一覧から選び, データセットの入力から雛形の幅を書き換え, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `GET /api/examples/{name}`, `POST /api/templates/adapt`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 一式の画面の問題の設定の欄を見る, 試しの記録を推論, 配る行を読む | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `GET /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/treated` |

ML-CreateArchitecture の `POST /api/dataset/inputs` が答えない場合は, 誤りとして表に残し, 列の数を ML-Fathom の書き出しの入力の形から, 足す数を決め方 (S-learner なら 1) から手で写して続ける (後ろの段階を確かめるため). 片方の決め方が途中で止まっても, もう片方は通す. 待ちの上限と止め方は 25 と同じ.

## 確かめること

- [x] 見本から作った問題の設定を効果を見積もるに答え直すと, 直すところが無く, 受け渡しの決め方と雛形 (T-learner は `tabular_uplift`, S-learner は `tabular_uplift_s`) と配る数の上限が出て, 骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から開き, 介入の記録の要約で配った後に決まる列 (`used`) を見つけ, 雛形のレシピがその列を落とし, 層別の分割でプロジェクトに書き出せる
- [ ] ML-CreateArchitecture で問題の設定を一覧から選ぶと, 雛形とデータセットの欄が入り, データセットの入力から雛形の幅が決まる (雛形とデータセットの欄は入るが, データセットの入力が 500 で答えず, 列の数を手で写す. 「気付いたこと」の 1)
- [x] ML-Dashboard で一覧から登録し, 画面から環境を作り, 骨組みを書き換えずに学習でき, 結果に効果の評価 (Qini の係数, 上位 K 割の効果の差) と Qini の曲線が出る
- [x] ML-Provide で実験を一覧から登録でき, 一式の種類と版 (T-learner は版 12, S-learner は版 18) と決め方が説明に出る
- [x] 試しの記録の推論で, 効果の上位から配る数の上限 (300) まで配る結論が出て, 配る行が効果の高い順に並び, 処置と結果の列から効果を数え直す

## 期待する結果

誤りの段階 0, 手の段階 0. 2 つの決め方ともテスト用と試しの記録の Qini の係数が正で, 上位 3 割の効果の差が全体の差より大きい (効果の大きい顧客を上位に並べる).

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリをこのリポジトリと同じディレクトリに clone して `make setup` を済ませ, ルートのディレクトリで実行する (`uv` が要る). 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `32_uplift_coupons` を入れて手で起動する.

```bash
.venv/bin/python tasks/32_uplift_coupons/journey.py               # 3 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/32_uplift_coupons/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境は 2 つのプロジェクトの `.venv` に, それぞれ ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる (uv の手元のキャッシュがあれば十数秒)

## 結果のメモ

手元の Mac で, 5 つのアプリの main (ML-Quest は 8695ffb) を使って動かした値. 表の段階の名前の頭の [T] は T-learner, [S] は S-learner.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| [T] 問題の設定 (表, 効果の見積もりの列, 配る数の上限) | OK | 決め方 `t_learner`, 雛形 `tabular_uplift`, 判断 `{'type': 'treat', 'budget': 300}`, 監視 `valid/qini`, 誤り 0 |
| [T] 骨組みの書き出し (場所を問題の設定に記録) | OK | 記録: `dataset`, `model.yaml`. README の一式の版 12 |
| [T] ML-Fathom の介入の記録の要約 | OK | 処置の群 1,464 行 (買った割合 0.283), 対照の群 1,536 行 (0.137), 差 0.147. 配った後に決まる列 `used` を雛形が落とす |
| [T] ML-Fathom (前処理, プロジェクトへ書き出す) | OK | 分割 `stratified`, 入力 [5], 出す値 1 |
| [T] ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `tabular_uplift`, データセットの欄が入る |
| [T] ML-CreateArchitecture のデータセットの入力 | 誤り | 500 (Internal Server Error). 列の数を ML-Fathom の書き出しから写して続ける |
| [T] ML-CreateArchitecture (雛形を列の数に書き換え, 保存) | 手 | 列の数を手で写す. 列 5 (足す数 0), 入力 [8, 5], 出力 `control`, `treated`. 保存の注意 1 件 (データセットを読めないので形の最後の大きさを確かめていない) |
| [T] ML-Dashboard で登録し環境を作る | OK | 10 秒 |
| [T] 学習 (骨組みを書き換えない) と効果の評価 | OK | 14 秒. ベストはエポック 29 の `valid/qini` 0.2281. テスト用: Qini の係数 0.2081, 上位 3 割の効果の差 0.2801, 全体の差 0.1465. Qini の曲線は検証用とテスト用 |
| [T] ML-Provide で実験を一覧から登録 | OK | 種類 `uplift`, 版 12, 決め方 `t_learner`. 一式の判断 `{'type': 'treat', 'budget': 300}` |
| [T] 問題の設定の欄 (一式の画面) | OK | 問題の設定を使わない一式なので欄が無い (配る数の上限は一式が持つ) |
| [T] 推論 (効果の上位から配る数の上限まで配る) | OK | 1,000 行を推論し, 300 行に配る (効果の高い順. 最上位の効果 0.5349). 試しの記録で数え直し: 処置の群 511 行, 対照の群 489 行, Qini の係数 0.1339, 上位 3 割の効果の差 0.2201, 全体の差 0.1034 |
| [S] 問題の設定 (表, 効果の見積もりの列, 配る数の上限) | OK | 決め方 `s_learner`, 雛形 `tabular_uplift_s`, 判断と監視は [T] と同じ, 誤り 0 |
| [S] 骨組みの書き出し (場所を問題の設定に記録) | OK | README の一式の版 18 |
| [S] ML-Fathom の介入の記録の要約 | OK | [T] と同じ |
| [S] ML-Fathom (前処理, プロジェクトへ書き出す) | OK | [T] と同じ |
| [S] ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `tabular_uplift_s`, データセットの欄が入る |
| [S] ML-CreateArchitecture のデータセットの入力 | 誤り | [T] と同じ 500 |
| [S] ML-CreateArchitecture (雛形を列の数に書き換え, 保存) | 手 | 列 5 (足す数 1), 入力 [8, 6], 出力 `outcome`. 保存の注意 1 件 ([T] と同じ) |
| [S] ML-Dashboard で登録し環境を作る | OK | 10 秒 |
| [S] 学習 (骨組みを書き換えない) と効果の評価 | OK | 13 秒. ベストはエポック 15 の `valid/qini` 0.2697. テスト用: Qini の係数 0.2077, 上位 3 割の効果の差 0.3333, 全体の差 0.1465 |
| [S] ML-Provide で実験を一覧から登録 | OK | 種類 `uplift`, 版 18, 決め方 `s_learner`. 一式の判断は [T] と同じ |
| [S] 問題の設定の欄 (一式の画面) | OK | [T] と同じ |
| [S] 推論 (効果の上位から配る数の上限まで配る) | OK | 1,000 行を推論し, 300 行に配る (効果の高い順. 最上位の効果 0.0702). 試しの記録で数え直し: Qini の係数 0.1502, 上位 3 割の効果の差 0.2104, 全体の差 0.1034 |

手の段階 2, 誤りの段階 2.

- 2 つの決め方とも, テスト用と試しの記録の上位 3 割の効果の差が全体の差より大きく, 効果の大きい顧客を上位に並べた. 試しの記録の Qini の係数は S-learner (0.1502) が T-learner (0.1339) より少し大きい
- S-learner の効果の大きさは小さく出る. 学習の記録の 3,000 行の効果の平均は T-learner 0.1442, S-learner 0.0230 (処置の群と対照の群の買った割合の差は 0.147). 配る判断は効果の順と配る数の上限だけで決まるので結論は変わらないが, 効果の値を確率の差として読む (配る相手の数を効果の大きさで決めるなど) 場合は決め方で大きく違う
- **確かめの範囲**: データは作り方の分かる合成の記録で, 課題は架空. 問題の設定から推論まで値の受け渡しが通ることの確かめで, 実際の場面で効く根拠ではない

## 気付いたこと

- ML-CreateArchitecture の `POST /api/dataset/inputs` が, 効果の見積もりのデータセットで 500 を返す. ML-Fathom (ML-EDA を `ml-eda[all]` で入れる) は pyarrow があるので表を parquet で書き出すが, ML-CreateArchitecture の追加の依存 `project` は ML-EDA を `ml-eda` だけで入れるので pyarrow が無い. 効果の見積もりの入力では処置の列と行の id の列を読むために ML-EDA の `load_dataset` で表まで読み (`dataset_uplift` から `dataset_section`), parquet を読めない `ImportError` が読めないときの例外の組 (`READ_ERRORS`) に入っていないので 500 になる. 同じ理由で `PUT /api/project/model` は, 記録のデータセットを読めないのでデータセットで決まる形の最後の大きさを確かめていない, という注意を返す (保存はできる). ML-CreateArchitecture の CI のテストは同じ環境 (pyarrow が無い) で ML-EDA から書き出すので, 表が csv になり当たらない. 表を読むのは効果の見積もり, 値段を変える判断, 時間をまたぐ量の判断の説明の項目 (`uplift`, `price`, `demand`) だけで, 表の分類 (25) の入力は当たらない
- 効果の一式は ML-Provide で問題の設定を使わない (配る数の上限は ML-Quest が骨組みの config に写し, 骨組みが一式に写す). 一式の画面に問題の設定の欄が出ないので, 学習の後に配る数の上限を変えるには, 推論で配る判断 (`decision` の `budget`) を渡すか, 問題の設定を直して骨組みを書き出し直す
- 効果を見積もるには見本が無いので, 見本「解約の引き止め」から作って課題と受け持ちを答え直す (ML-Pathfinder の効果の見積もりのテストと同じ作り方). 答え直す問いは, 表を入れる前の 30 と後の 10
- ML-Fathom の介入の記録の要約 (`POST /api/sources/{id}/uplift`) は, 問題の設定から開いたレシピの処置, 結果, 結果の種類, 行の id の列をそのまま渡せる. 配った後に決まる列 `used` を気付きに出し, 雛形のレシピが同じ列を落とす (問題の設定の受け渡しの除く列)
