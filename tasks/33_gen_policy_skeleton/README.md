# 33 画像の生成と方策を問題の設定の骨組みから通す

## 目的

11 (拡散) と 24 (CartPole) と同じ課題を, 問題の設定の段階 (ML-Quest) から始めて, 書き出した骨組みを書き換えずに学習し, 推論の段階 (ML-Inference) で読むところまで通す (21 と同じ形). 11 と 24 は学習のスクリプトを手で書いた. ここでは学習のスクリプトも config も ML-Quest が問題の設定から書き出したものをそのまま使い, 利用者が置くのは骨組みの README が案内するもの (データセットとモデルの定義) だけにする.

- **画像の生成** (`generate/`): 課題は「試験に使う手書きの数字の画像が足りない」(架空). 受け持ちは作るで, 正解の種類と出力の形が画像 (条件の無い画像の生成). データは 11 と同じ MNIST の手書きの数字. 骨組みが拡散のモデルを学び, 学習の後に ML-Inference の雑音だけの一式 (版 5) を作る. 推論の段階で件数と種を渡して画像を作る
- **方策** (`policy/`): 課題は「台車の上の棒を立て続けるのに, 人が付きっきりで押す」(架空). 受け持ちは効果を見積もるで方策, 環境は 24 と同じ Gymnasium の `CartPole-v1`. 骨組みが方策を学んで守りと一緒に方策の一式を書き出す. 推論の段階で ML-Inference の `load_policy` でモデルを渡さずに読み, 守りを当てたまま環境で動かす
- **方策を画面で通す** (`provide.py`): 方策の問題の設定を ML-Pathfinder に取り込んで骨組みを書き出し, ML-Dashboard で骨組みを書き換えずに学習し, ML-Provide の方策の実験の一覧から一式を登録して, 観測から行動と守りの結果を出すまでを, 3 つの画面のアプリの API だけで通す (25 から 32 と同じ形. 段階ごとに OK, 手, 誤りを表にする)

**道具の外にあるもの**: 方策の一式は観測から行動と守りの結果を返すだけで, 実機は動かさない. 方策を使ってよいかと守りの範囲は人が決めて承認する (ここで渡した範囲は確かめのための例).

## ディレクトリ構成

```
33_gen_policy_skeleton/
├── setup_problem.py    # 2 つの問題の設定を確かめ, それぞれの problem.yaml と骨組み (project/) を書き出す
├── prepare.py          # MNIST の一部を PNG にし, レシピの雛形で generate/project/ にデータセットを書き出し, 雑音を当てるモデルの定義を置く
├── infer.py            # 2 つの一式を ML-Inference で読む (雑音だけの一式で画像を作る, load_policy で CartPole を守り付きで動かす)
├── provide.py          # 方策の問題の設定から, ML-Provide の一覧からの登録と観測からの行動までを画面の API で通す
├── generate/
│   ├── problem.yaml    # 画像の生成の問題の設定 (setup_problem.py が書く)
│   └── project/        # ML-Quest が書き出した骨組み (README.md, problem.yaml, configs, train.py) と, prepare.py が置く models/model.yaml
├── policy/
│   ├── problem.yaml    # 方策の問題の設定 (setup_problem.py が書く)
│   └── project/        # ML-Quest が書き出した骨組み (README.md, problem.yaml, configs, train.py, model.py) と, 1 回目の train.py が書いた既定の定義 models/policy.yaml (git に入れてあり, 消すと次の学習で書き直す)
└── outputs/            # infer.py が書く並べた画像 generated.png (git の管理外)
```

## データ

- 画像の生成: MNIST の学習用の画像の先頭の 10,000 枚 (`prepare.py` が torchvision で `data/torchvision/` に取得し, 08 と 11 と共有. `data/mnist_png/` に 28 × 28 の灰色の PNG で書く). レシピは ML-Quest の `recipe_from` の雛形のまま (目的の列の無い `task: none`, 無作為に検証用 0.15 とテスト用 0.15). 学習用 7,000 枚, 検証用 1,500 枚, テスト用 1,500 枚
- 方策: データの分析の段階は無い (行動の結果は環境とのやり取りで集める)

用意してもらう必要があるデータは無い.

## モデル

- 画像の生成: `prepare.py` が骨組みの config の `model.spec` の場所 (`generate/project/models/model.yaml`) に置く, ML-ModelControl の小さな U-Net の形の定義 (126,945 個のパラメータ). 入力は雑音を加えた画像 (1, 28, 28) と int64 の時刻. 時刻は `op.SinusoidalEmbedding` (64 次元) と全結合で埋め込み, 各段のチャンネルの数にして足す. 28 × 28 で 2 段, 14 × 14 で 1 段の畳み込みと, 28 × 28 の特徴を飛ばしてつなぐ. 出力は画像と同じ形の加えた雑音の予測
- 方策: 骨組みの `train.py` が 1 回目の学習で観測と行動の空間から書いた既定の定義 (git に入れてある. 消すと次の学習で書き直す) (`policy/project/models/policy.yaml`. 全結合と Tanh を 2 段 (幅 64) と, 行動の数 2 の全結合). 価値の関数は骨組みの `model.py` の `PolicyNet` (REINFORCE では使わない)

## 学習の設計

2 つとも, ML-Quest が書き出した骨組みの `train.py` と config を書き換えずに使う. 変える値はコマンドの「項目=値」の上書きで渡す (`diff.yaml` に残る).

- 画像の生成: 骨組みの既定 (30 エポック, AdamW, 学習率 0.0002, バッチ 32, 時刻 1,000 の線形の雑音の予定, 重みの指数移動平均 0.995, 見本と一式の ddim は 50 回). 長く学ぶ実験 2 は `trainer.epochs=150` と実験の名前だけを上書きする
- 方策: 骨組みの既定 (20 エポック, 1 エポック 20 エピソード, REINFORCE, 検証 10 エピソード, 守り付きの評価 10 エピソード). 問題の設定の守りは文 (「左右に押すだけ」「台車が中心から 2.4 を出たら」) なので, 骨組みの config の `guards` は数が空のまま. 実験 2 は, 選んでよい行動と台車の位置の範囲を上書きで数にして渡す

## 確かめること

- [x] 2 つの問題の設定に直すところ (error) が無く, ML-Quest が骨組みを書き出す
- [x] 画像の生成の骨組みを書き換えずに, レシピの雛形で書き出したデータセットと置いた定義から学習でき, 雑音だけの一式 (版 5) ができる
- [x] 推論の段階 (ML-Inference の `load_bundle` と `predict`) で, 雑音だけの一式から件数と種で画像ができ, 骨組みの見本と同じ画像になる
- [x] 方策の骨組みを書き換えずに学習でき, 守りと一緒に方策の一式ができる
- [x] 推論の段階 (ML-Inference の `load_policy`) で, モデルを渡さずに方策の一式を読み, 守りを当てたまま環境で動かせる
- [x] 方策の問題の設定を ML-Pathfinder に取り込んで骨組みを書き出し, ML-Dashboard で一覧から登録して, 守りを experiment の config で数にして骨組みを書き換えずに学習できる
- [x] 学習した一式が ML-Provide の方策の実験の一覧 (`GET /api/experiments/policies`) に出て, 一覧から登録でき, 登録の後の行が登録した id を持つ
- [x] ML-Provide の `act` で, 守りの範囲の中の観測から行動と `ok`, 範囲の外と数でない値の観測から外れたときの動きの行動 0 と `fallback` と理由が出て, `act` で CartPole の 1 エピソードを動かせる

## 期待する結果

- `setup_problem.py` の確かめは 2 つとも 0 件 (「確かめ: 0 件」と出る)
- 画像の生成: `train.log` の最後に「推論の一式 (雑音だけ): inference/」が出る. `infer.py` の結果が全て `ok` で, 骨組みの見本との差が 2.5 / 255 以下 (見本は PNG に丸めた値. 超えると終了番号 1 で止まる)
- 方策: `train.log` の最後に守り付きの評価が出て, 実験の中に `policy/` ができる. `infer.py` で余った重みが無く, 守りの結果が全て `ok` で, 報酬の和が出る (余った重みがあるか, ok でない歩があると終了番号 1 で止まる)
- `provide.py`: 段階ごとの表の誤りの段階が 0 (1 つでもあれば終了番号 1). 作業用の場所の `journey.json` に同じ表を書く

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する (他の場所から実行してもよい).

```bash
python tasks/33_gen_policy_skeleton/setup_problem.py           # 2 つの problem.yaml と骨組み
python tasks/33_gen_policy_skeleton/prepare.py                 # MNIST の PNG, generate/project/outputs/dataset と models/model.yaml
python tasks/33_gen_policy_skeleton/generate/project/train.py  # 1. 画像の生成の学習と雑音だけの一式 (MPS で 1 分半ほど)
python tasks/33_gen_policy_skeleton/policy/project/train.py    # 1. 方策の学習と方策の一式 (CPU で 20 秒ほど)
python tasks/33_gen_policy_skeleton/infer.py                   # 推論の段階で 2 つの一式を読む
```

実験 2 (骨組みは書き換えず, コマンドの上書きだけを変える):

```bash
python tasks/33_gen_policy_skeleton/generate/project/train.py trainer.epochs=150 experiment.name=image_generate_long
python tasks/33_gen_policy_skeleton/policy/project/train.py "guards.allowed_actions=[0,1]" "guards.observation_low=[-2.4,null,null,null]" "guards.observation_high=[2.4,null,null,null]"
python tasks/33_gen_policy_skeleton/infer.py
```

- `infer.py` は, それぞれの骨組みの `outputs/` の中で最も新しい実験の一式を読む

方策を画面で通す (3 つのアプリを clone して `make setup` したディレクトリの親を `--apps` で渡す. 既定はこのリポジトリと同じ並び. 作業用の場所は `--work` で, 既定は一時ディレクトリ):

```bash
python tasks/33_gen_policy_skeleton/provide.py   # ML-Pathfinder, ML-Dashboard, ML-Provide を 25 の apps.py で起動して通す (1 分ほど)
```

- `provide.py` は `policy/problem.yaml` を取り込むので, 先に `setup_problem.py` を動かす. 骨組みは作業用の場所の `project/` に書き出し, このディレクトリの `policy/project/` は使わない
- 守りの数 (選んでよい行動 0 と 1, 台車の位置が -2.4 から 2.4) は, 画面の config の編集と同じく ML-Dashboard の学習の依頼の experiment の config で渡す (上の実験 2 と同じ数. `diff.yaml` に残る)
- `act` で動かすエピソードの環境は手元の Gymnasium で, ML-Provide は観測から行動と守りの結果を返すだけ
- `setup_problem.py` は, 答えが前の `problem.yaml` と同じなら時刻を含めて前のものを使う (実行のたびに差分を出さない)
- `generate/project/train.py` と `policy/project/train.py` が, データセットや定義が足りないときに「README の『次にすること』」と案内するのは, それぞれの骨組みの `project/README.md` のこと

## 結果のメモ

PyTorch 2.15.0, Gymnasium 1.4.0, Apple Silicon の Mac での値 (画像の生成は device が auto で MPS, 方策は骨組みの config のとおり CPU).

- `setup_problem.py` の確かめは 2 つとも 0 件
- 画像の生成

| 実験 | エポック | ベストのエポック | 検証用の損失 (ベスト) | テスト用の損失 | 学習時間 |
| --- | --- | --- | --- | --- | --- |
| 1 (骨組みの既定) | 30 | 30 | 0.0245 | 0.0251 | 79 秒 |
| 2 (`trainer.epochs=150`) | 150 | 150 | 0.0180 | 0.0186 | 402 秒 |

- 2 つとも学習の後に雑音だけの一式 (版 5. 手順は ddim の 50 回, 学習の時刻 1,000, 種 0) が実験の `inference/` にでき, `infer.py` で 16 件が全て `ok`. 骨組みの見本 8 件との最も大きい差は 1 で 1.50 / 255, 2 で 1.47 / 255 (見本は PNG に丸めた値. 丸めの誤差は最大 0.5 / 255 なので, 残りは数値の誤差). 値の範囲は 1 で -0.188 から 1.226, 2 で -0.041 から 1.056 (骨組みの README のとおり 0 から 1 の外に出ることもある)
- 並べた画像は, 1 では太さのそろわない点と線の塊で, 2 では滑らかな手書きの線になる. ただし 2 でも数字として読めるのは一部 (6 に見えるものなど) だけ
- 方策

| 実験 | ベストのエポック | 検証の報酬の和 (ベスト) | 守りを数にしたか | 守り付きの評価 (10 エピソード) | `infer.py` (10 エピソード, 種 30000 から) |
| --- | --- | --- | --- | --- | --- |
| 1 (骨組みの既定) | 7 | 500.0 | いいえ (`guards_set` が false) | 平均 500.0, ok 5000 | 平均 500.0, ok 5000 |
| 2 (守りを上書きで数に) | 7 | 500.0 | はい (選んでよい行動 0 と 1, 台車の位置が -2.4 から 2.4) | 平均 500.0, ok 5000 | 平均 500.0, ok 5000 |

- 検証の報酬の和はエポック 3 から上限の 500 に届き, ベストは同じ値が並んだときの学習のエピソードの報酬 (`monitor_tiebreak`) でエポック 7. 学習時間は 18 秒ほど. 方策の頭は, 骨組みが観測と行動の空間から書いて git に入れた既定の定義 `policy/project/models/policy.yaml` で組み立てる (2 つの実験で同じ)
- `infer.py` が `load_policy` で読んだ一式は, 2 つとも余った重みが無い. 2 の位置の範囲 (2.4) は CartPole がエピソードを終える位置と同じなので, 守りの結果は全て `ok` で, 守りが働く場面は無い (守りが働くことの確かめは 24 の練習の一式)

- 方策を画面で通す (`provide.py`): 誤りの段階 0, 手の段階 0

| 段階 | 判定 | 結果 |
| --- | --- | --- |
| 問題の設定を取り込む (ML-Pathfinder) | OK | 「台車の棒を立て続ける」, 直すところ 0 件 |
| 骨組みの書き出し | OK | 6 つのファイル. 場所を問題の設定に記録 |
| ML-Dashboard で一覧から登録 | OK | 問題の設定が記録したプロジェクト |
| 環境を作る | OK | 足りない機能なし (全てがそろったかは False. 下の「気付いたこと」) |
| 学習 (守りを experiment の config で数に) | OK | ベストのエポック 7, `valid/reward` 500.0, 守り付きの評価 平均 500.0, ok 5000 |
| ML-Provide で方策の実験を一覧から登録 | OK | 行が登録した id を持つ. 一式の守りは選んでよい行動 0 と 1, 台車の位置の上限 2.4 |
| 観測から行動と守りの結果 (範囲の中 `[0, 0, 0.02, 0]`) | OK | 行動 1, `ok` |
| 同 (位置が範囲の外 `[3, 0, 0, 0]`) | OK | 行動 0, `fallback`, 理由「観測の要素 0 が範囲の外 (3)」 |
| 同 (位置が nan) | OK | 行動 0, `fallback`, 理由「観測に有限でない値がある (float32 での丸めも含む)」 |
| `act` で CartPole の 1 エピソード (種 40000) | OK | 報酬の和 500, `ok` 500 |

## 気付いたこと

### ML-Quest

- 2 つの問題の設定とも, 問いに順に答えるだけで直すところ (error) が無くなり, 骨組みの `train.py` と config を書き換えずに学習から一式まで通った. 利用者が置いたのは, 骨組みの README の「次にすること」が案内するデータセット (レシピの雛形のまま) と画像の生成のモデルの定義だけ. 方策は定義も骨組みが書く
- **画像の生成の `task.yaml` に正解の出どころとデータセットが入らない**: 骨組みの `train.py` は `write_task` に受け持ち, 出力の形, 作るもの, 作り方だけを渡す. そのため `task.yaml` の `target` は種類が null で出どころが `unknown`, `dataset` は場所が null になり, `train.log` に「正解の出どころは分からない. 指標が何との一致かを確かめる」と出る. 問題の設定の答えは `target.source: self` (データそのもの) と `target.kind: image` で, データセットは config の `data.dataset` から読んでいる. 方策の骨組みは出どころ (`environment`) を渡している
- **方策の守りの文が数にならない**: 問題の設定の守り (「左右に押すだけ」「台車が中心から 2.4 を出たら」) は文なので, 骨組みの config の `guards` は数が空 (`null`) のまま書き出され, 既定のままの学習では方策の一式の `guards_set` が false (範囲の守りが入らない) になる. 骨組みの README の「次にすること」の 2 と config の印 (TODO) がそれを案内する. 骨組みを書き換えずに守りを入れるには, コマンドの上書きで数を渡す (実験 2). 上書きは `diff.yaml` に残る
- **外れたときの動きの選択肢の例が CartPole に合わない**: 選択肢の名前は「決めた行動にする (例: 動かさない)」. ただし CartPole の行動は「左に押す (0)」と「右に押す (1)」だけで, 動かさない行動は無い. 決めた行動 0 は左に押すことで, 例の文のとおりにはならない

### ML-Dashboard

- **方策の骨組みは最初の学習の前に「全てがそろった」にならない**: 方策の骨組みの config の `model.spec` (`models/policy.yaml`) は, 骨組みの `train.py` が最初の学習で観測と行動の空間から書く. そのため書き出した直後のプロジェクトでは, 環境を作り終えても学習の準備 (`GET /api/projects/{id}/readiness`) の `ready` が False (モデルの定義 False) のままで, 画面の準備のカードに「学習を始める準備ができています」が出ない. 学習の依頼はそのまま通り, 学習の後に定義ができる. 異常の検知の骨組みのように config に `model.placeholder: true` を書けば, 定義が無くても準備ができたと数える

### ML-Provide

- 骨組みの学習の方策の一式は, 方策の実験の一覧 (`GET /api/experiments/policies`) に問題の設定の名前と実験の場所で出て, 一覧から登録できる. 登録した一式は `act` で観測から行動, 守りの結果, 理由を返す. 守りの数は学習の時の experiment の config の値が一式に入る

### ML-Inference

- 雑音だけの一式は, 骨組みの見本と同じ種 (一式の手順の種) で, 見本と同じ画像を作る. 差は PNG への丸めと数値の誤差 (計算の機器の違いなど) の分. `load_policy` は, 骨組みが一式に写した定義 (`policy/model.yaml`) から, モデルを渡さずに方策を組み立てる (余った重みは無い)

### 画像の質

- 骨組みの既定 (30 エポック) で学んだ画像は, 手書きの線の太さと明るさは似る. ただし数字の形にはならない (実験 1 の並べた画像). 小さな定義 (126,945 個のパラメータ) と MNIST の 10,000 枚で, 骨組みの既定は通しの確かめには足りるが, 質の目安にはならない. 5 倍のエポックの実験 2 でも, 線は滑らかになるが数字として読めるのは一部だけ. 画像の生成の骨組みの評価は損失と見本だけなので, 質を数で比べるには, 最良を選ぶ指標に FID を選んで書き出し直す (ML-Found の追加の依存 `fid`)
