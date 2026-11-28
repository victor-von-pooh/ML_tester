# 35 群分けを問題の設定から推論まで通す (顧客)

## 目的

群分け (まとめる) の課題 (顧客の記録から似た顧客の群を作り, 群ごとに案内を変える) を, 問題の設定から推論まで, 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「解約の引き止め」から作り, 課題と受け持ちを顧客の群分けに答え直す (架空). 今のやり方は使う額で 3 段に分ける
- **受け持ち**: まとめるで, 出力の形は群 (`cluster`), 正解の列を使わない. 群を作る列 `spend`, `visits`, `region`, 行を見分ける列 `customer_id`, 群の数 3, 群の安定性の数え方は標本を変えて作り直す (`sample`), 最良を選ぶ指標はシルエット係数
- アプリの起動と段階の記録は 25 の `apps.py` を使う

## ディレクトリ構成

```
35_group_customers/
├── journey.py   # 合成の顧客の記録を作り, 群分けで通す (部品は 25 の apps.py)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ.

## データ

`journey.py` が作業用の場所に作る合成の顧客の記録 (1,200 行. 乱数を固定). 列は `customer_id`, `spend` (使う額), `visits` (来店の回数), `region` (`north`, `south`, `west`). 顧客は 3 つの群 (割合 0.5, 0.3, 0.2) から作り, 群ごとに使う額の平均 (3,000, 8,000, 15,000. 標準偏差 1,200), 来店の回数の平均 (2, 6, 4), 多い地域 (6 割がその群の地域) が違う. 推論には, 同じ作り方で別の乱数の種から作った 400 行 (`customers_new.csv`) を使う. 用意してもらうデータは無い.

## モデル

ML-CreateArchitecture の対照学習の雛形 `contrastive_encoder` を, 符号化した行の幅 (`spend`, `visits` と `region` のワンホットの 3 列で 5) に書き換えたもの. 雛形は問題の設定から決まり (`POST /api/project/target` の `example`), 幅は `POST /api/templates/adapt` の `features` で書き換える.

## 学習の設計

ML-Pathfinder が書き出した骨組み (`train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす. 骨組みは行の特徴の一部を他の行の値に置き換えた 2 通りの見え方で対照学習し, 検証用のシルエット係数 (`valid/silhouette`) が最も大きいエポックを選ぶ. 埋め込みを k-means で群に分け, 群の安定性 (`evaluations/group_stability.json`. 作り直した群との調整ランド指数と, シルエット係数. 列の特徴をそのまま k-means で分けた群との比べ) と群の要約 (`evaluations/group_summary.json`) を書き, 群分けの推論の一式を実験の `inference/` に作る.

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本から名前を付けて作り, 課題と受け持ちを答え直し, 表を入れ, 群を作る列, 行を見分ける列, 群の数, 安定性の数え方に答える | ML-Pathfinder | `POST /api/problems`, `PUT /api/problems/{id}/answers`, `POST /api/problems/{id}/data` |
| 骨組みを書き出し, 場所を問題の設定に記録 | ML-Pathfinder | `POST /api/problems/{id}/project` |
| 問題の設定を一覧から開き (群分けのタブ), 答えを読み, 答えの列で群の要約を作り, 表をプロジェクトへ書き出す | ML-Fathom | `GET /api/problems`, `POST /api/problems/start`, `GET /api/sources/{id}/group_problem`, `POST /api/sources/{id}/groups`, `GET /api/sources/{id}/project`, `POST /api/sources/{id}/export_group_project` |
| 問題の設定を一覧から選び, データセットの入力から雛形の幅を書き換え, プロジェクトに保存 | ML-CreateArchitecture | `GET /api/problems`, `POST /api/project/target`, `POST /api/dataset/inputs`, `GET /api/examples/{name}`, `POST /api/templates/adapt`, `PUT /api/project/model` |
| 書き出したプロジェクトを一覧から登録, 環境を作る, 学習, 結果 | ML-Dashboard | `GET /api/problems/projects`, `POST /api/projects`, `GET /api/projects/{id}/readiness`, `POST /api/projects/{id}/environment`, `POST /api/projects/{id}/jobs`, `GET <実験>/results` |
| 実験を一覧から登録, 一式の画面の問題の設定の欄を見る, 新しい顧客を推論, 群で絞った行を読む | ML-Provide | `GET /api/experiments`, `POST /api/bundles`, `GET /api/bundles/{id}/problem`, `POST /api/bundles/{id}/predict`, `GET /api/results/{id}/rows?group=` |

ML-CreateArchitecture の `POST /api/dataset/inputs` が答えない場合は, 誤りとして表に残し, 幅を ML-Fathom の群の要約 (`width`) から手で写して続ける (後ろの段階を確かめるため). 待ちの上限と止め方は 25 と同じ.

## 確かめること

- [x] 見本から作った問題の設定をまとめるに答え直すと, 直すところが無く, 受け渡しの群と雛形 (`contrastive_encoder`) とシルエット係数の監視が出て, 骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から群分けのタブで開き, 答えの列で群の要約 (群の安定性とシルエット係数) を返し, 表を骨組みが読む場所へ書き出せる
- [ ] ML-CreateArchitecture で問題の設定を一覧から選ぶと, 雛形とデータセットの欄が入り, データセットの入力から雛形の幅が決まる (雛形とデータセットの欄は入る. データセットの入力は群分けの表のファイルを読めず 404)
- [x] ML-Dashboard で一覧から登録し, 画面から環境を作り, 骨組みを書き換えずに学習でき, 結果に群の安定性と群の要約が出る
- [x] ML-Provide で実験を一覧から登録でき, 一式の種類 (`group`) と出力の形 (`cluster`) と群を作る列が説明に出る
- [x] 新しい顧客の推論で, 全ての行が群に割り当てられ, 群の要約, 代表の行, 群で絞った行が出る

## 期待する結果

誤りの段階 0, 手の段階 0. 学習の群の安定性の調整ランド指数が 0.9 以上で, 群の要約が 3 つの群と全体の 4 行になる.

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリをこのリポジトリと同じディレクトリに clone して `make setup` を済ませ, ルートのディレクトリで実行する (`uv` が要る). 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `35_group_customers` を入れて手で起動する.

```bash
.venv/bin/python tasks/35_group_customers/journey.py               # 1 分ほど (環境づくりはネットワークを使う)
.venv/bin/python tasks/35_group_customers/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

- 最後に段階ごとの表と, 手の段階と誤りの段階の数を出す. 誤りがあれば終了の番号は 1
- 学習の環境はプロジェクトの `.venv` に, ML-Found と ML-EDA, ML-ModelControl, ML-Inference を git から入れる

## 結果のメモ

手元の Mac で, 5 つのアプリの main (ML-Quest は 2976074) を使って動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (表, 群を作る列, 群の数, 安定性の数え方) | OK | 群 `{'columns': ['spend', 'visits', 'region'], 'id': 'customer_id', 'k': 3, 'stability': 'sample'}`, 雛形 `contrastive_encoder`, 監視 `valid/silhouette`, 分割 `random`, 判断 無し, 誤り 0 |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | 記録: `customers.csv`, `model.yaml`. README に `group_stability.json` と `group_summary.json` |
| ML-Fathom で問題の設定を開き, 群の要約を作る | OK | タブ `groups`, 答えの列と群の数 3 (標本を変えて作り直す). 幅 5, 安定性 ari 0.9898 (最低 0.9823), シルエット係数 0.4141. 群の行 297, 621, 282 |
| ML-Fathom (群を作る表をプロジェクトへ書き出す) | OK | 骨組みが写した表を置き換えて 1,200 行, 列 `spend`, `visits`, `region`, 行の id `customer_id` |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 雛形 `contrastive_encoder`, データセットの欄 `customers.csv` |
| ML-CreateArchitecture のデータセットの入力 | 誤り | 404 「ディレクトリ <プロジェクト>/data/customers.csv が見つかりません」 |
| ML-CreateArchitecture (雛形を行の幅に書き換え, 保存) | 手 | 幅 5 を ML-Fathom の群の要約から手で写す. 入力 [8, 5], 出力 `embedding`, `similarity`. 保存の注意 1 件 (同じ理由でデータセットを読めない) |
| ML-Dashboard で登録し環境を作る | OK | 15 秒 |
| 学習 (骨組みを書き換えない) と群の安定性, 群の要約 | OK | 23 秒. ベストはエポック 9 の `valid/silhouette` 0.4666. 群の安定性: ari 1.0 (そのままの k-means 0.9873), ari_min 1.0 (0.961), シルエット係数 0.4747 (0.4148). 群の行 455, 399, 346 |
| ML-Provide で実験を一覧から登録 | OK | 種類 `group`, 版 16, 出力の形 `cluster`, 群を作る列 `spend`, `visits`, `region` |
| 問題の設定の欄 (一式の画面) | OK | 問題の設定を使わない一式なので欄が無い |
| 推論 (新しい顧客を群に割り当て, 群の要約を出す) | OK | 400 行を全て 3 つの群に割り当て (177, 120, 103 行). 群 0 で絞ると 177 件, 代表の行は群ごとに 3 件. 2 行は群の中心から学習のときより遠い |

手の段階 1, 誤りの段階 1.

新しい顧客の群の要約 (ML-Provide):

| 群 | 行 | 割合 | 安定性 | spend の平均 | visits の平均 | region |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 177 | 0.4425 | 1.0 | 4884.6 | 2.89 | north (100 %) |
| 1 | 120 | 0.3 | 1.0 | 7319.5 | 4.19 | south (100 %) |
| 2 | 103 | 0.2575 | 1.0 | 8853.3 | 3.47 | west (100 %) |
| 全体 | 400 | 1.0 | - | 6637.0 | 3.43 | north (44 %) |

- 学習した群は地域だけで分かれた (どの群も 1 つの地域が 100 %). 合成の群は使う額で最もよく分かれる作り方で, 使う額の平均は群の間で 4,885 から 8,853 までしか離れない. ML-Fathom の列の特徴をそのまま k-means で分けた群の行の数 (297, 621, 282) も合成の群の割合 (0.5, 0.3, 0.2) と違い, どちらも作った群を取り出してはいない
- 群の安定性 (ari 1.0) とシルエット係数 (0.4747) は, 比べる相手 (0.9873, 0.4148) より高い. ただしシルエット係数はそれぞれの群を作った空間で数えるので, 埋め込みの空間で高くても, 群が課題 (案内を変える) に合うとは言えない
- **確かめの範囲**: データは作り方の分かる合成の記録で, 課題は架空. 乱数の種 1 つ. 問題の設定から推論まで値の受け渡しが通ることの確かめで, 実際の場面で効く根拠ではない

## 気付いたこと

- **ML-CreateArchitecture のデータセットの入力は群分けの表を読まない**: まとめるの骨組みのデータセット (config の `data.dataset`) は ML-Fathom が写した表のファイル 1 つで, ML-EDA の書き出しのディレクトリではない. `POST /api/dataset/inputs` (`read_dataset_inputs`) は並べるの反応の表だけをファイルで受け, それ以外は ML-EDA の `model_inputs` に渡すディレクトリを求めるので, 404 になる. 保存 (`PUT /api/project/model`) も同じ理由の注意を返す. 雛形の幅 (符号化した行の幅) は ML-Fathom の群の要約の `width` に出るので, 今は手で写す
- 学習した群が地域だけで分かれるのは, 骨組みの崩し方 (`corrupt`. 符号化した特徴ごとに他の行の値に置き換える. ワンホットの列も 1 つずつ置き換える) と関わると推定する (確かめてはいない). 骨組みの `corrupt` には, 課題に合う崩し方に置き換えられる旨の TODO がある
- 群分けの一式は ML-Provide で問題の設定を使わない. 一式の画面に問題の設定の欄が出ない
- ML-Fathom は問題の設定を一覧から開くと群分けのタブになり, 問題の設定の答え (`group_problem`) をそのまま群の要約 (`groups`) に渡せる. 骨組みの書き出しが入れた表を既に写しているので, 表の書き出しは置き換えの確かめ (`overwrite`) が要る
