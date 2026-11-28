# 36 異常の検知を問題の設定から推論まで通す (機械の計測)

## 目的

異常の検知の課題 (正常の行が多い機械の計測の表から, 異常の度合いの高い計測を人の確認に回す) を, 問題の設定から推論まで, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. 度合いは自己符号化器の再構成の誤差で決める. 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「解約の引き止め」から作り, 課題, 指標, 決めること, 受け持ちを機械の計測の課題に答え直す (架空). 1 日に人が確かめられるのは 20 件までで, どの計測を確かめるかを選ぶ. 今のやり方は温度が決まった値を超えたら点検する
- **受け持ち**: 予測するで, 正解の種類は異常, 正解は少し (一部の行だけに付いている). 正解の列 `is_anomaly`, 行の id の列 `reading_id`, 度合いに使う列 `temp`, `vibration`, `pressure`, `current`, 度合いの決め方は自己符号化器 (`autoencoder`), 人が確かめる件数の上限 20 (評価の上位 K 件も同じ), 最良を選ぶ指標は PR-AUC (`average_precision`). 判断は度合いの上位から件数の上限まで人の確認に回す (`{type: triage, capacity: 20}`)
- アプリの起動と段階の記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
36_anomaly_sensors/
├── journey.py   # 合成の計測の表を作り, 問題の設定から人の確認の記録まで通す (部品は 25 の apps.py)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ.

## データ

`journey.py` が作業用の場所に作る合成の計測の表 (3,000 行. 乱数を固定). 列は `reading_id`, `temp` (温度), `vibration` (振動), `pressure` (圧力), `current` (電流), `memo` (全ての行で同じ文. 度合いに使わない列), `is_anomaly` (異常は 1, 正常は 0. 3 割の行だけに付け, 残りは欠ける). 正常の行は, 見えない負荷に沿って温度と振動と電流が上がり, 圧力が下がる. 異常の行 (約 3%) は負荷との関係が崩れる (温度と振動が少し高く, 圧力が少し低い. 電流は正常のまま. 正常の揺れと重なる行もある). 推論には, 同じ作り方で別の乱数の種から作った 1,000 行 (`sensors_trial.csv`. 全ての行に正解がある) を使い, 人の確認の判断をその正解から決める (人の代わり). 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の自己符号化器の雛形 `tabular_autoencoder` を, データセットの列の数 (度合いに使う 4 列) に書き換えたもの (入力 `x` の幅 4, 出力は再構成 `recon`). 雛形は問題の設定から決まり (`POST /api/project/target` の `example`), 幅は `POST /api/templates/adapt` で書き換える.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 骨組みは正常の行で自己符号化器を学び, 検証用の PR-AUC (`valid/average_precision`) が最も大きいエポックを選び, 分割ごとに上位 K 件の適合率と PR-AUC を評価の表 (`evaluations/anomaly_<分割>.json`) に書き, 行ごとの度合い (`anomaly_scores.csv`) と異常の度合いの推論の一式を実験の `inference/` に作る.

段階と API (画面のボタンに当たるもの).

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から名前を付けて作り, 課題と受け持ちを答え直し, 表を入れ, 異常の検知の列, 度合いの決め方, 件数の上限, 指標に答える | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き, 正解の列を異常の正解として数え, 書き戻し, 分割の後の行の数を見て, プロジェクトへ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `GET /api/sources/{id}/profile?task=anomaly`, `POST /api/sources/{id}/problem/write`, `POST /api/sources/{id}/preview`, `POST /api/sources/{id}/export_project` |
| 問題の設定を一覧から選び, データセットの入力から雛形の幅を書き換え, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `GET /api/examples/{name}`, `POST /api/templates/adapt`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 試しの記録を推論, 人の確認に回した行を読み, 1 件ごとに確認の結果を記録し, 読み直す | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/queue`, `POST /api/results/{id}/reviews`, `GET /api/results/{id}/reviews` |

ML-CreateArchitecture の `POST /api/dataset/inputs` が答えない場合は, 誤りとして表に残し, 列の数を ML-Fathom の書き出しの入力の形から手で写して続ける (後ろの段階を確かめるため). 待ちの上限と止め方は 25 と同じ.

## 確かめること

- [x] 見本から作った問題の設定を異常の検知 (予測する, 正解の種類は異常, 正解は少し) に答え直すと, 直すところが無く, 受け渡しの度合いの決め方, 雛形 `tabular_autoencoder`, 件数の上限の振り分け, 監視 `valid/average_precision` が出て, 骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から開き, 正解の列を異常の正解として数え (正解の付いた行, 異常の行, 正解の欠けた行), 雛形のレシピが度合いに使わない列を落とし, 分割ごとの正常, 正常と見なす, 異常の行の数を出してプロジェクトに書き出せる
- [x] ML-CreateArchitecture で問題の設定を一覧から選ぶと, 雛形とデータセットの欄が入り, データセットの入力から雛形の幅が決まる (保存の注意は 0 件)
- [x] ML-Dashboard で一覧から登録し, 画面から環境を作り, 骨組みを書き換えずに学習でき, 結果に検証用とテスト用の上位 K 件の適合率と PR-AUC が出る
- [x] ML-Provide で実験を一覧から登録でき, 一式の種類 `anomaly` と件数の上限の振り分けが説明に出る
- [x] 試しの記録の推論で, 度合いの上位から件数の上限 (20) まで人の確認に回り, 回した行が度合いの高い順に行の id 付きで並ぶ
- [x] 回した行ごとに確認の結果 (異常か正常, 確かめた人, 時刻) を 1 件ずつ記録でき, 結果のディレクトリの `reviews.csv` に貯まり, 回した行の今の判断に出る

## 期待する結果

誤りの段階 0, 手の段階 0. テスト用の PR-AUC が異常の割合 (約 0.03) を超え (学習の段階の判定), 人の確認に回した 20 件の異常の割合が試しの記録の全体の異常の割合を超える (推論の段階の判定).

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリをこのリポジトリと同じディレクトリに clone して `make setup` を済ませ, ルートのディレクトリで実行する (`uv` が要る). 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `36_anomaly_sensors` を入れて手で起動する.

```bash
.venv/bin/python tasks/36_anomaly_sensors/journey.py               # 2 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/36_anomaly_sensors/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境はプロジェクトの `.venv` に, ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる

## 結果のメモ

手元の Mac で, 5 つのアプリの main を使って動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (表, 異常の検知の列, 人が確かめる件数の上限) | OK | 度合い `autoencoder`, 正解 `is_anomaly`, 上限 20, 雛形 `tabular_autoencoder`, 判断 `{'type': 'triage', 'capacity': 20}`, 監視 `valid/average_precision`, 除く列 `memo`, 誤り 0 |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | 記録: `dataset`, `model.yaml`. 書いたファイル 6 |
| ML-Fathom の異常の正解の数 | OK | タスク `anomaly`, 正解の付いた行 850 (異常 24, 正常 826, 異常の割合 0.0282), 正解の欠けた行 2,150. 雛形が落とす列 `memo` |
| ML-Fathom (前処理, プロジェクトへ書き出す) | OK | 分割 `random`, 入力 [4]. 分割ごとの行 (正常, 正常と見なす, 異常): 学習用 580, 1,504, 0. 検証用 128, 318, 12. テスト用 118, 328, 12 |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `tabular_autoencoder`, データセットの欄が入る |
| ML-CreateArchitecture (雛形を列の数に書き換え, 保存) | OK | データセットの入力から. 列 4, 入力 [8, 4], 出力 `recon`. 保存の注意 0 件 |
| ML-Dashboard で登録し環境を作る | OK | 11 秒 |
| 学習 (骨組みを書き換えない) と度合いの評価 | OK | 20 秒. `valid/average_precision` 0.5928. テスト用 (458 行, 異常 12 行): 上位 20 件の適合率 0.4, PR-AUC 0.482 |
| ML-Provide で実験を一覧から登録 | OK | 種類 `anomaly`, 版 11. 一式の判断 `{'type': 'triage', 'capacity': 20}`. 問題の設定を使わない一式 |
| 推論 (度合いの上位から件数の上限まで人の確認に回す) | OK | 1,000 行を推論し, 20 行を人の確認に回す (度合いの高い順, 行の id の列 `reading_id`). 回した行のうち試しの記録の正解が異常の行 15 (試しの記録の異常は 33 行) |
| 人の確認の結果を 1 件ごとに記録 | OK | 記録 20 件 (異常 15, 正常 5). 回した 20 行の全てに今の判断が出る. 記録は結果のディレクトリの `reviews.csv` |

手の段階 0, 誤りの段階 0.

- テスト用の PR-AUC 0.482 は異常の割合 (12 / 458 = 0.026) の約 18 倍で, 度合いの上位に異常が集まる. ただし評価は正解の欠けた行を正常と見なして数えるので, 正解の欠けた行に隠れた異常を上位に出すと誤りとして数え, 低く出る. 合成の表を作り直して本当の異常で数えると, テスト用の異常は 24 行で, 上位 20 件の適合率は 0.75, PR-AUC は 0.704 (試しの記録の 15 / 20 = 0.75 と合う)
- 試しの記録では回した 20 行のうち 15 行が異常で, 異常の割合 (33 / 1,000) から見ると上位に強く集まる. 残りの 18 行の異常は上位 20 に入らず, 件数の上限の分だけ見逃す
- **確かめの範囲**: データは作り方の分かる合成の表で, 課題は架空. 人の確認の判断は試しの記録の正解から決めた. 問題の設定から人の確認の記録まで値の受け渡しが通ることの確かめで, 実際の場面で効く根拠ではない

## 気付いたこと

- 異常の検知には見本が無いので, 見本「解約の引き止め」から作って課題と受け持ちを答え直す (ML-Pathfinder の異常の検知のテストと同じ作り方). 答え直す問いは, 表を入れる前の 29 と後の 7. 振り分け (`action.decision`) の問いは出ず, 件数の上限 (`anomaly.capacity`) から受け渡しの判断 `{type: triage, capacity}` が決まる
- ML-Fathom の正解の数は, 問題の設定から開いた後に分析を `task=anomaly` で読み直すと出る (`GET /api/sources/{id}/profile?task=anomaly`). 分割ごとの正常, 正常と見なす (正解の欠けた行), 異常の行の数は, 全ての手順を当てた見本 (`POST /api/sources/{id}/preview`) の `anomaly.rows` に出て, プロジェクトへの書き出しの応答には無い. 正解の付いた異常の行は学習用に入らず, 検証用とテスト用に分かれる (自己符号化器は学習用の正常の行と正常と見なす行で学ぶ. 正常と見なす行に隠れた異常 (この表では学習用の 2,084 行のうち 50 行) も正常として学ぶ)
- 異常の度合いの一式は ML-Provide で問題の設定を使わない (件数の上限は ML-Quest が骨組みの config に写し, 骨組みが一式に写す). 学習の後に件数の上限を変えるには, 推論で振り分け (`decision` の `capacity`) を渡すか, 問題の設定を直して骨組みを書き出し直す
- ML-Provide の人の確認の記録 (`POST /api/results/{id}/reviews`) は行の id の値 (人の確認に回した行の `key`) で 1 件ずつ足し, 同じ行の最後の記録が今の判断になる. 確かめた人と時刻 (ML-Inference の既定の今の時刻) が付く
