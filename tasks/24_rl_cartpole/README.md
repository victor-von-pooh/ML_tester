# 24 CartPole の方策を学習し, 守りと一緒に書き出す (強化学習)

## 目的

ルートの TASK_TYPES.md の相談 8 (続けて行動を選ぶ方策) の答え「自律して行動する方策を作り, 確かめ, 守りと一緒に書き出すところまで」を, ML-Found の `ml_found.rl` で通す最初の利用者. Gymnasium の CartPole (台車を左右に押して棒を立て続ける) で, 次の流れを確かめる. モデルの良し悪しは目的ではないので, 学習は 10 秒ほどで終える.

1. **学習**: `RolloutLoader` (環境とのやり取りをバッチにする loader), `EvaluationLoader` (別の種の評価のエピソード), `ReinforceStep` (REINFORCE) を今までの `Trainer` に渡して学習する. 監視する指標は評価のエピソードの報酬の和の平均 `valid/reward` で, `mode: max`
2. **守りと一緒に書き出す**: `make_guards` で守り (選んでよい行動 `allowed_actions`, 止める条件として台車の位置と棒の角度の範囲, 外れたときの動き `fallback`) を作り, `export_policy` で実験のディレクトリの `policy/` に書き出す
3. **読み直して守り付きで動かす**: `load_policy` で新しいモデルに読み直し, ML-Found の `evaluate_policy` でシミュレータのエピソードを動かす. 1 歩ごとに `act` が返す行動と守りの結果 (`status`. `ok`, `clipped`, `fallback`, `stop`, `handover`) を数え, `stop` と `handover` は行動を返さないので, そこでエピソードを終える. あわせて, 同じ一式を推論の段階の ML-Inference の公開の `load_policy` でも読み, 止める条件の内と外の観測で ML-Found の方策と同じ行動と守りの結果を返すかを比べる (ML-Inference はこの比べだけに使う. ルートの README の「導入」で入れる. 入っていなければ比べを飛ばしたことを `train.log` に残し, 学習と守り付きの評価は続ける)
4. **守りの練習**: 止める条件を狭めた別の一式 (`drills/narrow_stop/` と `drills/narrow_fallback/`) も書き出して同じく動かし, 守りが働くこと (止まる, 決めた行動で続ける) を数で確かめる. 練習の一式は使うためのものではない
5. **守る指標の表**: `evaluate_policy` のエピソードごとの行に守りの名前を足し, `save_evaluation(..., guard=True)` で `evaluations/guarded_episodes.json` に書く
6. **守りのテスト**: 範囲の外の観測で止まること (`status: stop`, 行動なし) を `test_guards.py` で確かめる. このリポジトリは学習を pytest で判定しないが, 守りは安全の約束なので, 学習しないで済むこの確かめだけは CI で動かす
7. **記録した経験からの見積もり**: 学習した方策を ε-greedy (ε 0.2. ε の確率で選んでよい行動から一様に選ぶ) にして `evaluate_policy` に渡し, 行動を選んだ確率 `action_prob` 付きの軌跡を `save_trajectory` で `trajectories/epsilon_greedy.csv` に書く (`off_policy.py`). 軌跡を ML-Fathom の画面の API で登録して要約を読み, ML-Found の `estimate_policy_value` で, ε の違う方策の良さを環境で動かさずに見積もり, 同じ方策を環境で動かした値と比べる

**道具の外にあるもの**: 書き出した方策は, 観測から行動と守りの結果を返すだけで, 環境や機械には触れない. **実機 (ハードウェア) での実行は道具の外**で, ML-Found も推論の段階も実機を動かさない. **方策を使ってよいか (配備) と守りの範囲 (止める条件の値, 外れたときの動き) は人が決めて承認する**. ここで config に書いた範囲は確かめのための例で, 承認を経た値ではない (相談 7 の原則の当てはめ).

## ディレクトリ構成

```
24_rl_cartpole/
├── configs/
│   ├── default/train.yaml          # 基準の config (requires: [rl], monitor: valid/reward, mode: max, 守りと練習の範囲)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           └── quick.yaml          # 3 エポックで終える (学習の足りない方策で守りが働く例)
├── model.py                        # PolicyNet (観測から行動ごとのスコア. 隠れ層 1 つ)
├── train.py                        # 学習, 守りと一緒に書き出し, 読み直して守り付きで動かす
├── test_guards.py                  # 範囲の外の観測で止まる, Gymnasium が無いと入れ方を示して止まる (pytest)
├── off_policy.py                   # ε-greedy の軌跡を書き, ML-Fathom で分析し, 方策の良さを見積もる
└── outputs/                        # cartpole/<日付>/<時刻>/ (実験. policy/, drills/, evaluations/, off_policy.py の trajectories/ と fathom/ を含む. git の管理外)
```

## データ

データのファイルは無い. 学習のたびに Gymnasium の `CartPole-v1` (観測は台車の位置, 速さ, 棒の角度, 角速度の 4 つ. 行動は 0 が左, 1 が右に押す. 1 歩ごとに報酬 1. 棒の角度が約 0.2094 rad (12 度) か台車の位置が 2.4 を外れると終わり, 500 歩で打ち切り) とやり取りする.

| 用途 | 環境 | 乱数の種 |
| --- | --- | --- |
| 学習 (`RolloutLoader`) | 1 エポック 20 エピソード, 5 エピソードごとに 1 回更新. 方策の分布から選ぶ | `experiment.seed` (42) を最初の `reset` に |
| 検証 (`EvaluationLoader`) | 10 エピソード, 最もスコアの高い行動 | 10000, 10001, ... (毎エポック同じ) |
| 守り付きの評価 | 読み直した方策で 10 エピソード | 20000, 20001, ... |

## モデル

`model.py` の `PolicyNet`. 観測 (4 つ) から行動ごとのスコア (2 つ) を出す, 隠れ層 1 つ (64, tanh) の全結合のモデル.

## 学習の設計

`RolloutLoader`, `EvaluationLoader`, `ReinforceStep` を `Trainer` に渡して学習する (上の「目的」の 1). Adam (学習率 0.01), 20 エポック, `valid/reward` を `mode: max` で監視し, 同じ値の場合は `train/reward` で比べる (`monitor_tiebreak`). 学習の後に守りと一緒に書き出し, 読み直して守り付きで動かす (上の「目的」の 2 から 5).

### 守り

| 一式 | 置き場 | 選んでよい行動 | 止める条件 (絶対値の上限) | 外れたときの動き |
| --- | --- | --- | --- | --- |
| 本番の候補 | `policy/` | 0, 1 | 位置 2.0, 角度 0.18 rad (環境が終わる範囲 2.4 と 0.2094 より内側) | `stop` (行動を返さず止める) |
| 練習 `narrow_stop` | `drills/narrow_stop/` | 0, 1 | 位置 0.3, 角度 0.05 rad | `stop` |
| 練習 `narrow_fallback` | `drills/narrow_fallback/` | 0, 1 | 位置 0.3, 角度 0.05 rad | `action` (行動 0 で続ける. `status: fallback`) |

速さと角速度には止める条件を置かない (`None`). CartPole の行動は 2 つだけなので, 選んでよい行動を両方にしている (1 つに絞ると棒を立てられない). そのため `clipped` は 0 になる.

## 確かめること

- [x] `RolloutLoader`, `EvaluationLoader`, `ReinforceStep` を `Trainer` に渡して学習でき, `trainer.monitor: valid/reward`, `mode: max` でベストモデルを選ぶ
- [x] `make_guards` で選んでよい行動, 観測の範囲, 外れたときの動きを持つ守りを作り, `export_policy` で `policy/` (`policy.json` と `policy.pt`) に書き出せる
- [x] `load_policy` で新しいモデルに読み直し, `act` で守りを当てた行動と `status` を得られる
- [x] 書き出した一式を ML-Inference の公開の `load_policy` でも読め, 止める条件の内と外の観測と数でない値を含む観測 200 件で, ML-Found の `load_policy(...).act` と同じ行動, `status`, 理由 (`reasons`) を返す
- [x] `evaluate_policy` で守り付きのエピソードを動かし, エピソードごとの結果の数 (ok, clipped, fallback, stop, handover) と終えた理由 (`end`) を守る指標の表 (`guard: true`) に書ける
- [x] 範囲の外の観測 (位置 2.5, -2.01, 角度 0.19, -0.5, nan) で `status: stop` と行動なし, 範囲の中で `ok` (`test_guards.py`)
- [x] Gymnasium が無いと, config の `requires: [rl]` により出力先を作る前に入れ方を示して止まる (`test_guards.py` で Gymnasium を隠して確かめる)
- [x] `task.yaml` を書ける (受け持ちは `effect`, 出力の形は `action`, 正解の種類は `reward`, 正解の出どころは `environment`)
- [x] 確率で行動を選ぶ方策 (`act` が `action_prob` を返す) を `evaluate_policy` に渡し, `action_prob` 付きの軌跡を `save_trajectory` で書ける
- [x] 書いた軌跡を ML-Fathom の画面の API で登録でき (読み込みの候補の先頭が `trajectory`), 要約のエピソード, 歩数, 報酬の和が記録と同じで, 行動を選んだ確率の要約と守りの結果が出る
- [x] `estimate_policy_value` で, 記録と同じ ε の方策は記録した方策の平均と同じ値, ε の違う方策は見積もりと有効なサンプルの数が出て, ε 0 (最もスコアの高い行動だけ) は重みが全て 0 で決まらないと知らせる

## 期待する結果

- 1 で `valid/reward` が上限の 500 に届く. 本番の候補の守りでは, 棒が傾いて止まることはないが, 学習が 20 エポックと短いので, 長いエピソードで台車が流れて位置の止める条件に当たることがある
- 練習の一式では止める条件が狭いので, `narrow_stop` はいくつかのエピソードを途中で止め (`stop`), `narrow_fallback` は止めずに行動 0 で続ける (`fallback`)
- 2 では学習が足りず棒が傾くので, 本番の候補の守りでも止まるエピソードが出る
- 実験のディレクトリに `policy/`, `drills/narrow_stop/`, `drills/narrow_fallback/`, `evaluations/guarded_episodes.json`, `task.yaml` ができる
- 4 で `trajectories/epsilon_greedy.csv` (と説明の `.json`), `evaluations/off_policy.json` (ε ごとの見積もりと環境で動かした値) と, 段階ごとの判定の `evaluations/off_policy_journey.json` ができ, 誤りの段階が 0 (1 つでもあれば終了番号 1). 記録と ε が遠い方策ほど有効なサンプルの数が減り, 見積もりは当てにならないと知らせる

## 実行方法

### 追加の依存 (Gymnasium)

強化学習には ML-Found の追加の依存 `rl` (Gymnasium) が要る. ルートの README の「導入」で `pip install -e "./ML-Found[examples,rl]"` と入れる. config の一番上に `requires: [rl]` を書いているので, 入っていないまま `train.py` を動かすと, `start_experiment` が出力先を作る前に `MissingRequirementError` で止まり, 学習のスクリプトが次の入れ方を示して終わる.

```
機能 rl に要るものが入っていません: gymnasium
次の一手: pip install "gymnasium>=1.0"
(ML-Found の追加の依存 rl でも入る: pip install -e "./ML-Found[rl]")
説明: https://gymnasium.farama.org/introduction/basic_usage/
```

### 動かし方

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/24_rl_cartpole/train.py                                                          # 1. 学習 (20 エポック)
python tasks/24_rl_cartpole/train.py --experiment configs/experiment/examples/quick.yaml      # 2. 動作の確認 (3 エポック)
python -m pytest tasks/24_rl_cartpole                                                         # 3. 守りのテスト
python tasks/24_rl_cartpole/off_policy.py                                                     # 4. 記録した経験からの見積もり (1 の後)
```

CPU で, 1 は 10 秒ほど, 2 は 3 秒ほど, 3 は 1 秒ほど, 4 は 20 秒ほど (Python と ML-Fathom の起動を含む). 3 には pytest が要る (ルートの README の「導入」で入れる).

4 は `outputs/` の中で最も新しい実験の `policy/` を読む (`--run` で実験を選べる). ML-Fathom の画面のアプリを, 25 の `apps.py` で実験の中の `fathom/` を置き場にして起動し, 終えたら止める. ML-Fathom を clone して `make setup` したディレクトリの親を `--apps` で渡す (既定はこのリポジトリと同じ並び).

## 結果のメモ

Gymnasium 1.4.0, PyTorch 2.15.0, CPU (Apple Silicon の Mac) での値.

| 実験 | 学習したエポック | ベストのエポック | 検証の報酬の和 (ベスト) |
| --- | --- | --- | --- |
| 1 (`train.yaml`) | 20 | 19 | 500.0 |
| 2 (quick) | 3 | 3 | 63.6 |

守り付きの評価 (10 エピソード). 歩数は環境を進めた数の合計. 結果の数は `act` を呼んだ判断ごと (止めた判断を含む).

| 実験 | 一式 | 報酬の和の平均 | 歩数 | 止めたエピソード | ok | fallback | stop | clipped, handover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `policy` | 429.9 | 4299 | 7 | 4299 | 0 | 7 | 0 |
| 1 | `narrow_stop` | 173.1 | 1731 | 9 | 1731 | 0 | 9 | 0 |
| 1 | `narrow_fallback` | 195.5 | 1955 | 0 | 1864 | 91 | 0 | 0 |
| 2 | `policy` | 70.1 | 701 | 10 | 701 | 0 | 10 | 0 |
| 2 | `narrow_stop` | 24.8 | 248 | 10 | 248 | 0 | 10 | 0 |
| 2 | `narrow_fallback` | 32.9 | 329 | 0 | 249 | 80 | 0 | 0 |

- 検証の報酬の和は, エポック 6 で 500 に届いた後に下がり, 17 以降は 487 から 500 になる. 報酬が 500 で並ぶエポック (6, 17, 19) から, `trainer.monitor_tiebreak: train/reward` で学習のエピソードの報酬が最も大きいエポック 19 (371.9) をベストに選ぶ. 検証の `valid/entropy` (方策のエントロピー) も記録され, 報酬が 500 で並んでも動く
- 本番の候補の守り (位置 2.0, 角度 0.18) は, 学習した方策 (1) でも 10 エピソードのうち 7 で 305 から 473 歩目に位置の止める条件 (2.0) に当たって止め (台車が流れるため. 残り 3 は 500 歩の `truncated`), 学習の足りない方策 (2) では 10 エピソード全てを止める. 検証の 10 エピソード (シードが別) では 500 歩を続けたが, 守り付きの評価のエピソードでは長く続けるうちに止める条件の内側を外れる. 環境が終わる範囲 (角度 0.2094, 位置 2.4) より内側に置いたので, 環境が終わる前に止まる
- 練習の `narrow_stop` では 10 エピソードのうち 9 で止め, `narrow_fallback` では止めずに 91 歩で決めた行動 0 にする. 位置が 0.3 を超えて決めた行動 0 (左) を続けると棒が倒れ, 10 エピソードのうち 9 で環境が終わる (`evaluations/guarded_episodes.json` の `end` が `terminated`. 残り 1 は 500 歩の `truncated`). 外れたときの動きを `action` にするのは, その行動が安全と分かっている場合だけ (人が決める)
- `evaluations/guarded_episodes.json` は 1 行が 1 つの守りの 1 エピソード (`name`, `guards` と `evaluate_policy` の列 `episode`, `seed`, `reward`, `length`, `end`, 守りの結果ごとの数) の 30 行
- ML-Inference の `load_policy` で読んだ方策は, 3 つの一式とも 200 件全ての観測で ML-Found の方策と同じ行動, `status`, 理由を返す (`train.log` の「ML-Inference の load_policy との比べ」. 本番の候補の一式では `ok` 78 件と `stop` 122 件, 練習の 2 つでは `ok` 3 件と, `stop` か `fallback` が 197 件). 観測は止める条件の内と外にまたがる幅から乱数の種 `evaluation.guarded_seed` で選んだ 199 件と, 位置が nan の 1 件
- `test_guards.py` の 8 つのテストが通る (1 秒ほど)

記録した経験からの見積もり (4. 1 の実験の方策). 記録は ε 0.2 の 100 エピソード (種 40000 から) で, 歩数 39,330, 報酬の和の平均 393.3, 終わり方は 500 歩の打ち切り 27 と守りの止め 73. 環境で動かした値は同じ ε の方策の 100 エピソード (種 50000 から) の報酬の和の平均. 割引率は 1.

| 見積もる ε | 見積もり (重み付き) | 歩ごとの見積もり | 環境で動かした値 | 有効なサンプルの数 | 当てにならない (`low_ess`) |
| --- | --- | --- | --- | --- | --- |
| 0.2 (記録と同じ) | 393.3 | 393.3 | 387.1 | 100.0 | いいえ |
| 0.19 | 397.6 | 400.8 | 382.4 | 89.5 | いいえ |
| 0.15 | 395.2 | 446.8 | 411.5 | 6.3 | はい |
| 0.1 | 375.8 | 439.7 | 404.8 | 1.3 | はい |
| 0 | 決まらない | 60.5 | 461.8 | 0.0 | いいえ (決まらない場合) |

- ML-Fathom は軌跡の表を読み込みの候補の先頭 `trajectory` で登録し, 要約のエピソード (100), 歩数 (39,330), 報酬の和の平均 (393.3) が `evaluate_policy` の行と同じ. 行動を選んだ確率は, 欠けが守りで止めた 73 行, 範囲の外が 0, 最小 0.1, 最大 0.9 (ε 0.2 と行動 2 つから決まる 2 つの値). 守りの結果は `ok` 39,330 と `stop` 73
- 記録と同じ ε 0.2 では重みが全て 1 で, 見積もりは記録した方策の平均 393.3 と一致する. ε 0.19 では有効なサンプルの数 89.5 で, 環境で動かした値との差は 15 ほど (環境で動かした値も 100 エピソードの平均で, ε 0.2 の記録と環境の値の差 6 ほどと同じ程度のぶれを含む)
- ε 0.15 から有効なサンプルの数がエピソードの 1 割を下回り, `low_ess` で当てにならないと知らせる. ε 0.1 では最も大きい重みが全体の 87% を占める. 1 エピソードが 400 歩ほどと長いので, 1 歩の確率の比 (0.95 / 0.9 か 0.05 / 0.1) がエピソードの長さだけ掛かり, ε の小さな違いでも重みが大きくばらつく
- ε 0 は, 記録した方策が ε で替えた歩 (3,954 歩) を含む 100 エピソード全ての重みが 0 になり, 重み付きの見積もりは決まらない (`value` が None で, 「見積もる方策に近い方策で記録してください」と知らせる)

## 気付いたこと

### ML-Found

- **学習した方策でも本番の候補の守りで止まる**: 1 のベストの方策 (エポック 19) は, 検証の 10 エピソードでは 500 歩を続けるが, 守り付きの評価の 10 エピソードのうち 7 で位置の止める条件 (2.0) に当たって止まる. REINFORCE の道筋は選んだ確率の重みの比の最後の桁にも左右されるので, ML-Found の版が変わると値が変わりうる (同じ版なら毎回同じ値). REINFORCE の 20 エポックでは, 検証の報酬が 500 に届いても, 長く続けると台車が流れる方策が選ばれることがある. 本番の候補の守りが働かないことを確かめるには, 検証のエピソードを増やすか, 学習のエポックを増やす
- **`policy.json` の `model_class` が `model.PolicyNet`**: スクリプトの隣の `model.py` から読み込んだので, モジュールの名前は `model` になる. 読む側が組み立て方を探す手がかりとしては弱いので, ML-Found の例と同じく `metadata` の `model` にクラスと引数 (`class`, `args`) を書いた
- `save_evaluation` の `split` は文字なら何でもよいので `simulator` とした. 環境で評価した表の分割の名前の決まりは無い
- `config` の `requires` を確かめるのは `start_experiment` なので, その前の `load_project_config` はコマンドの引数を読む. pytest から `main()` を呼ぶと pytest の引数を読んで止まるので, `test_guards.py` は `sys.argv` をスクリプトだけにして呼ぶ
- **ε-greedy の方策は `act` を包んで作る**: `evaluate_policy` は `act` が返す `action_prob` を軌跡に書くだけなので, 確率で選ぶ方策は利用者が書く. `off_policy.py` は守りを当てた方策の `act` の行動を ε の確率で選んでよい行動から替え, 選んだ確率を添える. 守りが働いた判断 (`ok` でない) は替えない. 守りの範囲の確かめは観測だけで決まるので, 行動を替えても守りの結果は変わらない
- **最もスコアの高い行動だけの方策は見積もれない**: 重み付きの見積もりは, 見積もる方策が選ばない行動を記録した方策がとったエピソードの重みを 0 にする. ε 0 の方策では全てのエピソードの重みが 0 で `value` が None になり, `low_ess` も False (決まらない場合の決まり) になる. 歩ごとの見積もり (`per_decision`) は 60.5 と値を返すが, 環境で動かした値 461.8 から遠い. `value` が None のときは `per_decision` を使わない
- **長いエピソードでは見積もりの幅が狭い**: エピソードごとの重みは歩ごとの確率の比の積なので, 400 歩ほどのエピソードでは ε 0.2 から 0.15 に変えるだけで有効なサンプルの数が 100 から 6.3 に減る. 見積もりを当てにできるのは記録した方策にごく近い方策 (ここでは ε 0.19) だけ

### ML-Fathom と ML-EDA

- 軌跡の表は読み込みの候補の先頭に `trajectory` が出て, 登録と要約 (`GET /api/sources/{id}/trajectory`) が画面の API だけで通る. 要約の数は `evaluate_policy` の行と一致する
- **守りが働いた理由が観測の値ごとに分かれる**: 守りの要約の「働いた理由ごとの行の数」は理由の文をそのまま数える. ML-Found の理由の文は範囲の外の値を含む (「観測の要素 0 が範囲の外 (-2.00754)」) ので, 止めた 73 行が 72 種類の理由になり, どの要素で止めたかの集計にならない
