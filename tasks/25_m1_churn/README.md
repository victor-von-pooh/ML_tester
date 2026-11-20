# 25 表の分類を画面の API だけで通す (解約の引き止め)

## 目的

節目「表の予測を画面だけで通す」(ROADMAP.md の M1) の確かめ. 問題の設定から推論までを, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. ライブラリの関数は呼ばない (合成のデータを作るところだけ pandas を使う). 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「解約の引き止め」(架空). 月に 300 人まで引き止めの連絡ができ, 誰に連絡するかを選ぶ
- **今のやり方**: 問い合わせの多い順に連絡する. データの列 `contacted` (連絡したか) に書いてあり, 問題の設定の「今のやり方をデータで書くと」に列として答える. 予測には使わない (使ってはいけない列)
- 時系列の予測 (k 期前の今のやり方) は 26 で確かめる

## ディレクトリ構成

```
25_m1_churn/
├── apps.py      # 5 つのアプリを作業用の置き場で起動し, API で通す部品 (26 も使う)
├── journey.py   # 合成のデータを作り, 表の分類で通す
└── README.md
```

`journey.py` は作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に, アプリの置き場 (`homes/`), ログ (`logs/`), データ (`data/`), 書き出したプロジェクト (`project/`), 段階ごとの結果 (`journey.json`) を置く. 利用者の置き場 (`~/.ml-pathfinder/` など) は使わない.

## データ

`journey.py` が作業用の場所に作る合成の会員の表 (3,000 行. 乱数を固定). 列は `member_id`, `plan`, `months`, `inquiries`, `monthly_fee`, `contacted`, `churned` (目的の列. yes か no). 問い合わせが多く, 入って間もなく, 安い料金の会員ほど解約しやすい. 今のやり方の `contacted` は問い合わせの多い 2 割 (600 人) が yes. 推論とたどり直しには, 同じ作り方で別の乱数の種から作った 800 行 (`churn_past.csv`. 今のやり方は 160 人) を使う. 学習に使った行で測ると得が甘く出るため. 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の「データセットから入力と出力を作る」と同じ定義 (入力 `x` の形はデータセットの学習に使う列の数, 全結合の `head` の出す値はクラスの数. 部品の無い定義に置く). 画面の操作は画面の側の処理なので, `apps.py` の `model_from_inputs` が同じ定義を作り, `PUT /api/project/model` で保存する.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`project/train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 学習の環境は ML-Dashboard の「学習の準備」の「環境を作る」で, プロジェクトの `.venv` に入れる.

段階と API (画面のボタンに当たるもの).

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から作り, データを入れ, 今のやり方と使ってはいけない列に答える | ML-Pathfinder | `POST /api/problems`, `POST /api/problems/{id}/data`, `PUT /api/problems/{id}/answers` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き, ワンホットと標準化を足し, 書き戻し, プロジェクトへ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `PUT /api/sources/{id}/recipe`, `POST /api/sources/{id}/problem/write`, `POST /api/sources/{id}/export_project` |
| 問題の設定を一覧から選び, データセットから入力と出力を作り, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 学習の準備, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 問題の設定をつなぐ, 推論と意思決定, たどり直し | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `GET`/`PUT /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict`, `POST /api/results/{id}/backtest` |
| 学習の後に損得 (連絡 1 件の費用を 1,500 から 3,000 に) を直し, ML-Provide の次の推論の既定に入るかを見る | ML-Pathfinder, ML-Provide | `PUT /api/problems/{id}/answers`, `GET /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict` |

待ちが時間の上限 (起動は 120 秒, 環境づくりと学習は 30 分) を超えた場合は, 学習のジョブを止める API を呼んでから止まった段階を表に残す. 終わりには, まず ML-Dashboard の動いている学習のジョブを API で止める (学習はアプリとは別のプロセスのまとまりで動くため). その後, アプリは別のプロセスのまとまりで起動しているので, まとまりごと止める. 環境づくりのジョブには止める API が無いので, 途中で止まった場合は残ることがある.

## 確かめること

- [x] 問題の設定のデータと今のやり方 (列) から, 骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から開き, プロジェクトの `outputs/dataset` に書き出せる (場所を写さない)
- [x] ML-CreateArchitecture で問題の設定を一覧から選ぶと, データセットの欄が入り, 入力の形と出す値の数がデータセットから決まる
- [x] ML-Dashboard で, 問題の設定が記録したプロジェクトを一覧から登録できる
- [x] ML-Dashboard の画面から学習の環境を作り, 学習が通る
- [x] 骨組みが今のやり方の評価 (`evaluations/baseline_valid.json`, `baseline_test.json`) を書き, ML-Dashboard の結果に出る
- [x] ML-Provide で実験を一覧から登録でき, 候補の問題の設定が学習の時点の写しでなく ML-Pathfinder の置き場の元の問題の設定になる
- [x] ML-Provide で一式に問題の設定をつなぐと, 意思決定の損得と今のやり方の列が入り, 今のやり方とたどり直せる
- [x] 学習の後に ML-Pathfinder で損得を直すと, ML-Provide の次の推論の既定と推論で使う値に入る
- [x] 手の段階が 0 (場所を打つ所と値を写す所が無い)

## 期待する結果

誤りの段階 0, 手の段階 0. 骨組みの評価で, モデルの正解率が今のやり方より高い.

## 実行方法

5 つのアプリを ML_tester と同じディレクトリに clone し, それぞれの README の「導入と起動」の `make setup` を済ませておく (起動はこのスクリプトが行う. `uv` が要る). 別の場所に置いた場合は `--apps` でその親のディレクトリを渡す. ルートのディレクトリで実行する.

```bash
.venv/bin/python tasks/25_m1_churn/journey.py               # 2 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/25_m1_churn/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境はプロジェクトの `.venv` に, ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる (uv の手元のキャッシュがあれば十数秒)

## 結果のメモ

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (データ, 今のやり方) | OK | 対象を選ぶ行動なので, 今のやり方の得は求めない (試して確かめる) と出る |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | データセットとモデルの定義の場所が記録される |
| ML-Fathom (一覧から開く, 前処理, プロジェクトへ書き出す) | OK | 入力 [6], 出す値 2 |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 一覧の問題の設定が, 記録したプロジェクトとデータセットを持つ |
| ML-CreateArchitecture (入力と出力, プロジェクトに保存) | OK | データセットの欄が入る. 入力 [4, 6], 出す値 2 |
| ML-Dashboard で書き出したプロジェクトを一覧から登録 | OK | 問題の設定「解約の引き止め」が記録したプロジェクトが一覧に出る |
| 学習の準備 | OK | 足りない機能 eda, modelcontrol, inference |
| 環境を作る (画面から) | OK | 十数秒 (uv のキャッシュあり) |
| 学習 | OK | ベストの valid/loss 0.3857 |
| 比べる相手の評価 | OK | テスト用の 432 行 (450 行のうち) で, 正解率 0.852 と今のやり方 0.757 (差 0.095) |
| ML-Provide で実験を一覧から登録 | OK | 問題の設定の名前と実験の場所 (名前/日付/時刻) の行を選ぶ |
| 問題の設定をつなぐ (候補から) | OK | 候補は元の問題の設定. 損得 (価値 24,000, 費用 1,500, 防げる割合 0.2, 上限 300) と今のやり方の列 `contacted` が入る |
| 推論と意思決定 | OK | 800 件を予測し, 見込みの得が正の 21 件を選ぶ |
| 今のやり方とのたどり直し | OK | 実際の得はこの方針で 60,900, 今のやり方なら 9,600 |
| ML-Pathfinder で損得を直すと次の推論の既定に入る | OK | 連絡 1 件の費用 3,000 が, 推論の既定と推論で使った値に入る |

手の段階 0, 誤りの段階 0.

- 今のやり方との比べは, 元のデータの同じ行が分かる行だけで求める (同じ値の行で今のやり方が違う行は比べない) ので, 450 行のうち 432 行
- たどり直しの得は, 行動の効果 (防げる割合 0.2) の仮定のもとでの値. たどり直しの表の今のやり方の連絡は 160 人で, 上限 300 の中

## 気付いたこと

- 段階の間で場所と値を写す所は無い. データセットの書き出し先, モデルの定義の置き場, 入力の形と出す値の数, 今のやり方の評価と推論での今のやり方の列は問題の設定の記録と骨組みが渡し, プロジェクトと実験の一式は各アプリの一覧から選ぶ
- 利用者が場所を選ぶのは, 元のデータ (ML-Pathfinder の「データを入れる」), 骨組みの書き出し先, 推論する新しいデータ (ML-Provide) の 3 か所. どれも最初に用意するものの場所で, 他のアプリの値を写すのではないので OK とする
- ML-Provide は元の問題の設定をつなぐので, 学習の後に ML-Pathfinder で直した損得が, 一式を作り直さずに次の推論に入る
