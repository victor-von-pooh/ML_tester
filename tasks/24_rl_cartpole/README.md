# 24 CartPole の方策を学習し, 守りと一緒に書き出す (強化学習)

## 目的

相談 8 (続けて行動を選ぶ方策) の答え「自律して行動する方策を作り, 確かめ, 守りと一緒に書き出すところまで」を, ML-Found の `ml_found.rl` で通す最初の利用者. Gymnasium の CartPole (台車を左右に押して棒を立て続ける) で, 次の流れを確かめる. モデルの良し悪しは目的ではないので, 学習は 10 秒ほどで終える.

1. **学習**: `RolloutLoader` (環境とのやり取りをバッチにする loader), `EvaluationLoader` (別の種の評価のエピソード), `ReinforceStep` (REINFORCE) を今までの `Trainer` に渡して学習する. 監視する指標は評価のエピソードの報酬の和の平均 `valid/reward` で, `mode: max`
2. **守りと一緒に書き出す**: `make_guards` で守り (選んでよい行動 `allowed_actions`, 止める条件として台車の位置と棒の角度の範囲, 外れたときの動き `fallback`) を作り, `export_policy` で実験のディレクトリの `policy/` に書き出す
3. **読み直して守り付きで動かす**: `load_policy` で新しいモデルに読み直し, 1 歩ごとに `act` が返す行動と守りの結果 (`status`. `ok`, `clipped`, `fallback`, `stop`, `handover`) でシミュレータのエピソードを動かし, 結果の数を数える. `stop` と `handover` は行動を返さないので, そこでエピソードを終える
4. **守りの練習**: 止める条件を狭めた別の一式 (`drills/narrow_stop/` と `drills/narrow_fallback/`) も書き出して同じく動かし, 守りが働くこと (止まる, 決めた行動で続ける) を数で確かめる. 練習の一式は使うためのものではない
5. **守る指標の表**: 守りごとの結果の数を `save_evaluation(..., guard=True)` で `evaluations/guarded_episodes.json` に書く
6. **守りのテスト**: 範囲の外の観測で止まること (`status: stop`, 行動なし) を `test_guards.py` で確かめる. このリポジトリは学習を pytest で判定しないが, 守りは安全の約束なので, 学習しないで済むこの確かめだけは CI で動かす

**道具の外にあるもの**: 書き出した方策は, 観測から行動と守りの結果を返すだけで, 環境や機械には触れない. **実機 (ハードウェア) での実行は道具の外**で, ML-Found も推論の段階も実機を動かさない. **方策を使ってよいか (配備) と守りの範囲 (止める条件の値, 外れたときの動き) は人が決めて承認する**. ここで config に書いた範囲は確かめのための例で, 承認を経た値ではない (相談 7 の原則の当てはめ).

## 追加の依存 (Gymnasium)

強化学習には ML-Found の追加の依存 `rl` (Gymnasium) が要る. ルートの README の「導入」で `pip install -e "./ML-Found[examples,rl]"` と入れる. 入っていないまま `train.py` を動かすと, `require("rl")` が出力先を作る前に止まり, 次の入れ方を示す (ML-Found `888b8b8` での表示).

```
機能 rl に要るものが入っていません: gymnasium
次の一手: pip install "gymnasium>=1.0"
(ML-Found の追加の依存 rl でも入る: pip install -e "./ML-Found[rl]")
説明: https://gymnasium.farama.org/introduction/basic_usage/
```

## ディレクトリ構成

```
24_rl_cartpole/
├── configs/
│   ├── default/train.yaml          # 基準の config (monitor: valid/reward, mode: max, 守りと練習の範囲)
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

データのファイルは無い. 学習のたびに Gymnasium の `CartPole-v1` (観測は台車の位置, 速さ, 棒の角度, 角速度の 4 つ. 行動は 0 が左, 1 が右に押す. 1 歩ごとに報酬 1. 棒の角度が約 0.2095 rad (12 度) か台車の位置が 2.4 を外れると終わり, 500 歩で打ち切り) とやり取りする.

| 用途 | 環境 | 乱数の種 |
| --- | --- | --- |
| 学習 (`RolloutLoader`) | 1 エポック 20 エピソード, 5 エピソードごとに 1 回更新. 方策の分布から選ぶ | `experiment.seed` (42) を最初の `reset` に |
| 検証 (`EvaluationLoader`) | 10 エピソード, 最もスコアの高い行動 | 10000, 10001, ... (毎エポック同じ) |
| 守り付きの評価 | 読み直した方策で 10 エピソード | 20000, 20001, ... |

## 守り

| 一式 | 置き場 | 選んでよい行動 | 止める条件 (絶対値の上限) | 外れたときの動き |
| --- | --- | --- | --- | --- |
| 本番の候補 | `policy/` | 0, 1 | 位置 2.0, 角度 0.18 rad (環境が終わる範囲 2.4 と 0.2095 より内側) | `stop` (行動を返さず止める) |
| 練習 `narrow_stop` | `drills/narrow_stop/` | 0, 1 | 位置 0.3, 角度 0.05 rad | `stop` |
| 練習 `narrow_fallback` | `drills/narrow_fallback/` | 0, 1 | 位置 0.3, 角度 0.05 rad | `action` (行動 0 で続ける. `status: fallback`) |

速さと角速度には止める条件を置かない (`None`). CartPole の行動は 2 つだけなので, 選んでよい行動を両方にしている (1 つに絞ると棒を立てられない). そのため `clipped` は 0 になる.

## 確かめること

- [x] `RolloutLoader`, `EvaluationLoader`, `ReinforceStep` を `Trainer` に渡して学習でき, `trainer.monitor: valid/reward`, `mode: max` でベストモデルを選ぶ
- [x] `make_guards` で選んでよい行動, 観測の範囲, 外れたときの動きを持つ守りを作り, `export_policy` で `policy/` (`policy.json` と `policy.pt`) に書き出せる
- [x] `load_policy` で新しいモデルに読み直し, `act` で守りを当てた行動と `status` を得られる
- [x] 守り付きの評価で, 結果の数 (ok, clipped, fallback, stop, handover) と止めたエピソードの数を数え, 守る指標の表 (`guard: true`) に書ける
- [x] 範囲の外の観測 (位置 2.5, -2.01, 角度 0.19, -0.5, nan) で `status: stop` と行動なし, 範囲の中で `ok` (`test_guards.py`)
- [x] Gymnasium が無いと, 学習の前に `require("rl")` が入れ方を示して止まる (`test_guards.py` で Gymnasium を隠して確かめる)
- [x] `task.yaml` を書ける (ただし方策の受け持ちと出力の形の選択肢は無い. 下の気付いたこと)

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/24_rl_cartpole/train.py                                                          # 1. 学習 (20 エポック)
python tasks/24_rl_cartpole/train.py --experiment configs/experiment/examples/quick.yaml      # 2. 動作の確認 (3 エポック)
python -m pytest tasks/24_rl_cartpole                                                         # 3. 守りのテスト
```

CPU で, 1 は 10 秒ほど, 2 は 3 秒ほど, 3 は 1 秒ほど (Python の起動を含む).

## 期待する結果

- 1 で `valid/reward` が上限の 500 に届く. 本番の候補の守りでは, 棒を立て続けるので止める条件に当たらず, 全て `ok`
- 練習の一式では止める条件が狭いので, `narrow_stop` はいくつかのエピソードを途中で止め (`stop`), `narrow_fallback` は止めずに行動 0 で続ける (`fallback`)
- 2 では学習が足りず棒が傾くので, 本番の候補の守りでも止まるエピソードが出る
- 実験のディレクトリに `policy/`, `drills/narrow_stop/`, `drills/narrow_fallback/`, `evaluations/guarded_episodes.json`, `task.yaml` ができる

## 結果のメモ

ML-Found `888b8b8`, Gymnasium 1.4.0, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した (2026-11-20).

| 実験 | 学習したエポック | ベストのエポック | 検証の報酬の和 (ベスト) |
| --- | --- | --- | --- |
| 1 (`train.yaml`) | 20 | 6 | 500.0 |
| 2 (quick) | 3 | 3 | 63.6 |

守り付きの評価 (10 エピソード. 結果の数は歩ごと).

| 実験 | 一式 | 報酬の和の平均 | 歩数 | 止めたエピソード | ok | fallback | stop | clipped, handover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `policy` | 500.0 | 5000 | 0 | 5000 | 0 | 0 | 0 |
| 1 | `narrow_stop` | 257.0 | 2578 | 8 | 2570 | 0 | 8 | 0 |
| 1 | `narrow_fallback` | 268.6 | 2686 | 0 | 2603 | 83 | 0 | 0 |
| 2 | `policy` | 70.1 | 711 | 10 | 701 | 0 | 10 | 0 |
| 2 | `narrow_stop` | 24.8 | 258 | 10 | 248 | 0 | 10 | 0 |
| 2 | `narrow_fallback` | 32.9 | 329 | 0 | 249 | 80 | 0 | 0 |

- 検証の報酬の和は, エポック 6 で 500 に届いた後, 7 から 12 は 258.5 から 484.5 の間に下がり, 13 以降はまた 500 前後になった. ベストはエポック 6 のまま (同じ 500 は改善とみなさない). エポック 6 の方策でも守り付きの評価の 10 エピソードは全て 500 歩を立て続けた
- 本番の候補の守り (位置 2.0, 角度 0.18) は, 学習した方策では 1 度も働かず, 学習の足りない方策 (2) では 10 エピソード全てを止める条件で止めた. 環境が終わる範囲 (0.2095) より内側に置いたので, 環境が終わる前に止まる
- 練習の `narrow_stop` では 10 エピソードのうち 8 で止め, `narrow_fallback` では止めずに 83 歩で決めた行動 0 にした. 決めた行動で続けても, 位置が 0.3 を超えたまま行動 0 (左) を続けると, その後に環境が終わる場合がある. 外れたときの動きを `action` にするのは, その行動が安全と分かっている場合だけ (人が決める)
- `test_guards.py` の 8 つのテストが通った (0.7 秒)

### 気付いたこと (ML-Found)

- **守り付きの評価の loader が無い**: `EvaluationLoader` は学習中のモデルで最もスコアの高い行動を選び, 守りを当てない. 書き出した一式を守り付きで動かし, `status` を数え, `stop` と `handover` でエピソードを終える処理 (`run_guarded`) は学習のスクリプトで書いた. 一式と環境を渡すと結果の数を返す関数 (`evaluate_policy(policy, env, episodes, seed)` など) があると, 推論の段階 (ML-Inference) と画面でも同じ数え方になる
- **方策の `task.yaml`**: 受け持ち (`role`) にも出力の形 (`output`) にも方策の選択肢が無く (相談 1 で方策は 2.1 の扱い), 正解の種類 (`target.kind`) も無いので, どれも null で書いた. 正解の出どころも, 環境の報酬に当たる種類が無いので `unknown` と名前 (`CartPole-v1 の報酬 (シミュレータ)`) にした. `describe_origin` は「分からない」の文を返す
- **`MissingRequirementError` が `ml_found` の直下に無い**: README は `require("rl")` が `MissingRequirementError` で止まると書くが, `from ml_found import MissingRequirementError` はできず, `ml_found.utils` から読み込んだ (`ImportError` で受けてもよい)
- **`policy.json` の `model_class` が `model.PolicyNet`**: スクリプトの隣の `model.py` から読み込んだので, モジュールの名前は `model` になる. 読む側が組み立て方を探す手がかりとしては弱いので, `metadata` に隠れ層の次元数などを書いた
- `save_evaluation` の `split` は文字なら何でもよいので `simulator` とした. 環境で評価した表の分割の名前の決まりは無い
- 報酬が上限 (500) で頭打ちになるので, 最初に 500 に届いたエポックがベストになり, その後の同じ 500 は改善とみなさない. 長く安定した方策を選びたい場合は, 評価のエピソードを増やすか, 長さで打ち切らない指標が要る
