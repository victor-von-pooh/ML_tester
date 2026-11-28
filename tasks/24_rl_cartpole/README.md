# 24 CartPole の方策を学習し, 守りと一緒に書き出す (強化学習)

## 目的

ルートの TASK_TYPES.md の相談 8 (続けて行動を選ぶ方策) の答え「自律して行動する方策を作り, 確かめ, 守りと一緒に書き出すところまで」を, ML-Found の `ml_found.rl` で通す最初の利用者. Gymnasium の CartPole (台車を左右に押して棒を立て続ける) で, 次の流れを確かめる. モデルの良し悪しは目的ではないので, 学習は 10 秒ほどで終える.

1. **学習**: `RolloutLoader` (環境とのやり取りをバッチにする loader), `EvaluationLoader` (別の種の評価のエピソード), `ReinforceStep` (REINFORCE) を今までの `Trainer` に渡して学習する. 監視する指標は評価のエピソードの報酬の和の平均 `valid/reward` で, `mode: max`
2. **守りと一緒に書き出す**: `make_guards` で守り (選んでよい行動 `allowed_actions`, 止める条件として台車の位置と棒の角度の範囲, 外れたときの動き `fallback`) を作り, `export_policy` で実験のディレクトリの `policy/` に書き出す
3. **読み直して守り付きで動かす**: `load_policy` で新しいモデルに読み直し, ML-Found の `evaluate_policy` でシミュレータのエピソードを動かす. 1 歩ごとに `act` が返す行動と守りの結果 (`status`. `ok`, `clipped`, `fallback`, `stop`, `handover`) を数え, `stop` と `handover` は行動を返さないので, そこでエピソードを終える. あわせて, 同じ一式を推論の段階の ML-Inference の公開の `load_policy` でも読み, 止める条件の内と外の観測で ML-Found の方策と同じ行動と守りの結果を返すかを比べる (ML-Inference はこの比べだけに使う. ルートの README の「導入」で入れる. 入っていなければ比べを飛ばしたことを `train.log` に残し, 学習と守り付きの評価は続ける)
4. **守りの練習**: 止める条件を狭めた別の一式 (`drills/narrow_stop/` と `drills/narrow_fallback/`) も書き出して同じく動かし, 守りが働くこと (止まる, 決めた行動で続ける) を数で確かめる. 練習の一式は使うためのものではない
5. **守る指標の表**: `evaluate_policy` のエピソードごとの行に守りの名前を足し, `save_evaluation(..., guard=True)` で `evaluations/guarded_episodes.json` に書く
6. **守りのテスト**: 範囲の外の観測で止まること (`status: stop`, 行動なし) を `test_guards.py` で確かめる. このリポジトリは学習を pytest で判定しないが, 守りは安全の約束なので, 学習しないで済むこの確かめだけは CI で動かす

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
└── outputs/                        # cartpole/<日付>/<時刻>/ (実験. policy/, drills/, evaluations/ を含む. git の管理外)
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

## 期待する結果

- 1 で `valid/reward` が上限の 500 に届く. 本番の候補の守りでは, 棒が傾いて止まることはないが, 学習が 20 エポックと短いので, 長いエピソードで台車が流れて位置の止める条件に当たることがある
- 練習の一式では止める条件が狭いので, `narrow_stop` はいくつかのエピソードを途中で止め (`stop`), `narrow_fallback` は止めずに行動 0 で続ける (`fallback`)
- 2 では学習が足りず棒が傾くので, 本番の候補の守りでも止まるエピソードが出る
- 実験のディレクトリに `policy/`, `drills/narrow_stop/`, `drills/narrow_fallback/`, `evaluations/guarded_episodes.json`, `task.yaml` ができる

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
```

CPU で, 1 は 10 秒ほど, 2 は 3 秒ほど, 3 は 1 秒ほど (Python の起動を含む). 3 には pytest が要る (ルートの README の「導入」で入れる).

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

## 気付いたこと

### ML-Found

- **学習した方策でも本番の候補の守りで止まる**: 1 のベストの方策 (エポック 19) は, 検証の 10 エピソードでは 500 歩を続けるが, 守り付きの評価の 10 エピソードのうち 7 で位置の止める条件 (2.0) に当たって止まる. ML-Found の #86 で選んだ確率を対数で持つようになり, 重みの比が最後の桁で変わったことから学習の道筋が分かれた (推定. 同じ版なら毎回同じ値). REINFORCE の 20 エポックでは, 検証の報酬が 500 に届いても, 長く続けると台車が流れる方策が選ばれることがある. 本番の候補の守りが働かないことを確かめるには, 検証のエピソードを増やすか, 学習のエポックを増やす
- **`policy.json` の `model_class` が `model.PolicyNet`**: スクリプトの隣の `model.py` から読み込んだので, モジュールの名前は `model` になる. 読む側が組み立て方を探す手がかりとしては弱いので, ML-Found の例と同じく `metadata` の `model` にクラスと引数 (`class`, `args`) を書いた
- `save_evaluation` の `split` は文字なら何でもよいので `simulator` とした. 環境で評価した表の分割の名前の決まりは無い
- `config` の `requires` を確かめるのは `start_experiment` なので, その前の `load_project_config` はコマンドの引数を読む. pytest から `main()` を呼ぶと pytest の引数を読んで止まるので, `test_guards.py` は `sys.argv` をスクリプトだけにして呼ぶ
