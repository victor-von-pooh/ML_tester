# 34 推薦を問題の設定から推論まで通す (会員に勧める品)

## 目的

推薦 (並べる) の課題 (会員が品物を買った記録から, 会員ごとに勧める品物を上位 K 件並べる) を, 問題の設定から推論まで, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「総菜の作る量」から作り, 課題, 決めること, 受け持ちを会員に勧める品の課題に答え直す (架空). 今のやり方は全ての会員に人気順に並べる
- **受け持ち**: 並べるで, 反応の記録から一から学ぶ. 利用者の列 `member`, 品物の列 `item_id`, 時刻の列 `day`, 反応の強さの列 `qty` (0 より大きい行だけを反応とみなす), 並べる件数 K 10, 最良を選ぶ指標は NDCG
- アプリの起動と段階の記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
34_rank_items/
├── journey.py   # 合成の買った記録を作り, 並べるの課題で通す (部品は 25 の apps.py)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ. データ (`data/`) は学習の記録 `reactions.csv`, 後の期間の記録 `reactions_later.csv`, 並べる会員の表 `members.csv`.

## データ

`journey.py` が作業用の場所に作る合成の記録 (乱数を固定). 会員 200 人と品物 60 品で, 会員は 4 つの好みの群のどれかに入り, 品物は 15 品ずつ群に分かれる. 会員の買い物の 8 割は自分の群の品物 (群の中では番号の若い品ほど売れる. 重みは順位の逆数) で, 2 割は全ての品物から無作為. 学習の記録は会員ごとに 30 行 (6,000 行. 日 `day` は 0 から 89 の無作為) で, 1 割の行は買った数 `qty` を 0 から 3 の無作為にする (0 の行は見ただけで反応でない). 後の期間の記録は同じ作り方で会員ごとに 5 行 (日 90 から 103, `qty` は 1). 推論で, 学習の記録を前に買った品物 (並べない), 後の期間の記録を正解として, ML-Provide が Recall@K と NDCG を数え直す. 学習に使った反応で測ると甘く出るため. 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の同梱の雛形 `two_tower_retrieval` (2 塔. 利用者の番号のワンホット `user` と品物の番号のワンホット `item` をそれぞれ全結合で埋め込み, 内積の定数倍を点数 `score` にする) の入力の大きさと 1 段目の `in_features` を, 学習用の反応にいる会員と品物の数に書き換えたもの. 雛形は問題の設定から決まり (`POST /api/project/target` の `example`), 数はデータセットの入力 (`POST /api/dataset/inputs`. ML-Quest の `rank_encoding`) から読み, `POST /api/templates/adapt` の `users` と `items` で書き換える.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 骨組みは反応を時刻の順に並べて古い 8 割を学習用, 新しい 1 割ずつを検証用とテスト用に分け, バッチの中の他の品物と比べる交差エントロピーで学ぶ. 検証用の NDCG (`valid/ndcg`) が最も大きいエポックを選び, 検証用とテスト用の Recall@K と NDCG と人気順との差を評価の表 (`evaluations/ranking_<分割>.json`) に書き, 並べるの推論の一式 (版 8) を実験の `inference/` に作る.

段階と API (画面のボタンに当たるもの).

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から名前を付けて作り, 課題と受け持ちを答え直し, 表を入れ, 列と並べる件数に答える | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き, 反応の表を要約し, 骨組みが読む場所へ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `GET /api/sources/{id}/reactions`, `POST /api/sources/{id}/export_reactions` |
| 問題の設定を一覧から選び, データセットの入力から雛形の会員と品物の数を書き換え, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `GET /api/examples/{name}`, `POST /api/templates/adapt`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 一式の画面の問題の設定の欄を見る, 会員の表を並べる, 並びを読む | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `GET /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/rows` |

待ちの上限と止め方は 25 と同じ.

## 確かめること

- [x] 見本から作った問題の設定を並べるに答え直し, 反応の表を入れて列と件数に答えると, 直すところが無く, 受け渡しの並べ方 (列と件数 10), 時刻の順の分割, 雛形 `two_tower_retrieval`, 監視 `valid/ndcg` が出て, 骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から開き, 反応の表の要約で会員と品物の数と反応でない行 (`qty` が 0) を出し, 前処理をかけずに骨組みが読む場所へ書き出せる
- [x] ML-CreateArchitecture で問題の設定を一覧から選ぶと, 雛形とデータセットの欄が入り, データセットの入力から雛形の会員と品物の数が決まる (保存の注意は 0 件)
- [x] ML-Dashboard で一覧から登録し, 画面から環境を作り, 骨組みを書き換えずに学習でき, 結果に検証用とテスト用の Recall@K と NDCG (人気順との差つき) が出て, テスト用は人気順より大きい
- [x] ML-Provide で実験を一覧から登録でき, 一式の種類 `rank`, 版 8, 件数 10 が説明に出る
- [x] 会員の表の推論で, 全ての会員に前に買った品物を除いた上位 10 件の品物が並び, 後の期間の記録から Recall@K と NDCG を数え直す

## 期待する結果

誤りの段階 0, 手の段階 0. テスト用の Recall@K と NDCG が人気順 (学習用の反応の多い順) より大きい (好みの群を学ぶ). 学習の段階はこの差が正であることを OK の条件にする.

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリをこのリポジトリと同じディレクトリに clone して `make setup` を済ませ, ルートのディレクトリで実行する (`uv` が要る). 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `34_rank_items` を入れて手で起動する.

```bash
.venv/bin/python tasks/34_rank_items/journey.py               # 2 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/34_rank_items/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境はプロジェクトの `.venv` に, ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる (uv の手元のキャッシュがあれば十数秒)

## 結果のメモ

手元の Mac で, 5 つのアプリの main を使って動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (反応の表, 列, 並べる件数) | OK | 並べ方 `member`, `item_id`, `day`, `qty`, 件数 10. 分割 `time`, 雛形 `two_tower_retrieval`, 監視 `valid/ndcg`, 誤り 0 |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | 記録: `reactions.csv`, `model.yaml`. README の一式の版 8 |
| ML-Fathom の反応の表の要約 | OK | 6,000 行のうち反応 5,862 行 (`qty` が 0 の 138 行を除く), 会員 200, 品物 60, 組 2,878, 密度 0.2398 |
| ML-Fathom (反応の表をプロジェクトへ書き出す) | OK | `reactions.csv` に写す. 反応 5,862 行, 除く 138 行 |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `two_tower_retrieval`, データセットの欄が入る |
| ML-CreateArchitecture (雛形を会員と品物の数に書き換え, 保存) | OK | データセットの入力から. 会員 200, 品物 60, 入力 `user` [8, 200] と `item` [8, 60], 出力 `item_embedding`, `score`, `user_embedding`. 保存の注意 0 件 |
| ML-Dashboard で登録し環境を作る | OK | 12 秒 |
| 学習 (骨組みを書き換えない) と並べ方の評価 | OK | 20 秒. ベストの `valid/ndcg` 0.3826. テスト用 (会員 104 人, K 10): Recall@K 0.5721 (人気順 0.1947), NDCG 0.3663 (人気順 0.0920) |
| ML-Provide で実験を一覧から登録 | OK | 種類 `rank`, 版 8, 件数 10. 会員 200, 品物 60 |
| 問題の設定の欄 (一式の画面) | OK | 問題の設定を使わない一式なので欄が無い (件数と列は一式が持つ) |
| 推論 (会員ごとに上位 K の品物を並べる) | OK | 会員 200 人の全てに上位 10 件 (学習の記録で買った品物を除く). 後の期間の記録で数え直し (会員 157 人): Recall@K 0.5314, NDCG 0.3542 |

手の段階 0, 誤りの段階 0.

- テスト用の Recall@K と NDCG は人気順の約 3 倍と 4 倍で, 好みの群を学んだ. 後の期間の記録で数え直した値 (Recall@K 0.5314, NDCG 0.3542) もテスト用に近い
- 後の期間の記録で数えた会員が 200 人でなく 157 人なのは, 後の期間に買った品物が全て学習の記録で買った品物 (並べず, 正解にもしない) だった会員を数えないため
- **確かめの範囲**: データは作り方の分かる合成の記録で, 課題は架空. 問題の設定から推論まで値の受け渡しが通ることの確かめで, 実際の場面で効く根拠ではない

## 気付いたこと

- 並べるはレシピを持たない. ML-Fathom は問題の設定から開くと反応の表の要約 (`GET /api/sources/{id}/reactions`) と書き出し (`POST /api/sources/{id}/export_reactions`) だけを出し, 前処理と分割は骨組みが時刻の順に行う. ML-Pathfinder が骨組みの書き出しで入れた表を骨組みの `data/` へ写すので, 書き出しは置き換え (`overwrite`) を指定する
- ML-CreateArchitecture のデータセットの入力は, 反応の表を ML-Quest の `rank_encoding` で学習用と同じに分けて, 学習用にいる会員と品物の数を返す (`rank`). 雛形 `two_tower_retrieval` の書き換えの値の名前は `users` と `items`
- 並べるの一式は ML-Provide で問題の設定を使わない (件数と列は骨組みの config から一式が写して持つ). 推論の件数は `k` で変えられ, 前に反応した品物の表 (`seen`) を渡すと並べず, 後の反応の表 (`truth`) を渡すと Recall@K と NDCG を数え直す
- 反応の記録から学ぶ並べるの見本は無い (並べるの見本は既製のモデルをそのまま使う「似た商品の写真を探す (既製の埋め込み)」だけ) ので, 見本「総菜の作る量」から作って課題と受け持ちを答え直す (ML-Pathfinder の並べるのテストと同じ作り方). 答え直す問いは, 表を入れる前の 16 と後の 5
