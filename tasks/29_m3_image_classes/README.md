# 29 画像の分類を画面の API だけで通す (図形の画像)

## 目的

節目「画像と文の分類を画面で通す」(ROADMAP.md の 4 の節の「済んだこと」) の確かめ. 画像の分類を, 問題の設定から推論まで 5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Provide) の API だけで, 画面と同じ順に通す. 段階ごとに, 画面だけでできたか (OK), 値や場所を手で写すか打つか (手), 誤りかを表にする.

- **課題**: ML-Pathfinder の見本「解約の引き止め」(架空) の問いの答えを, 画像の表 (ML-EDA の `load_data` の列) に合わせたもの. 当たりのクラスは `circle`
- 通しは 25 の `apps.py` の `run_journey` をそのまま使う. 足りない所 (同梱の例からのモデルの作り方と, 中身の手順) だけを設定の関数 (`make_model`, `item_steps_from`) で渡す
- 文の分類は 30 で確かめる
- 学習しない既製のモデルの画面の通しも確かめる. 本物の ResNet18 (torchvision の ImageNet で学習した重み `IMAGENET1K_V1`) の一式を ML-Provide の画面 (ブラウザ) から下見して作って登録し, 新しい画像を推論して種類の名前が画像の分類の画面に出ることを, ブラウザの操作で確かめる (`resnet_ui.py`)

## ディレクトリ構成

```
29_m3_image_classes/
├── classes.py   # 同梱の例に画面と同じく入力と出力を当てる部品 (30 も使う)
├── journey.py   # 合成の画像を作り, 画像の分類で通す
├── resnet_ui.py # 本物の ResNet18 の一式を画面から登録して推論する通し (画面の配信と引き合わせ)
├── resnet_ui.mjs # resnet_ui.py が動かすブラウザの操作 (ML-Provide の画面の Playwright を使う)
└── README.md
```

作業用の場所 (既定は一時ディレクトリ. `--work` で変えられる) に置くものは 25 と同じ.

## データ

`journey.py` が作業用の場所に作る合成の図形の画像. `<クラス>/<画像>.png` のディレクトリで, クラスは `circle` と `square` (各 60 枚). 28x28 の RGB で, 暗い雑音の背景に, 色と位置と大きさの違う図形を 1 つ描く. 大きさが全て同じなので, 中身の手順 (`resize`) は要らない. 推論とたどり直しには, 同じ作り方で別の乱数の種から作った各 10 枚 (`shapes_past`) を使う. 用意してもらうデータは無い.

`resnet_ui.py` は, 作業用の場所の `data/patterns` に 256x256 の RGB の模様の画像を 6 枚書く (無地の赤, 緑, 青, 白黒の市松, 黄と黒の縞, 赤と白の同心円). ImageNet の種類のどれかに当たるかは決まっておらず, 画面に出る種類の名前を, 同じ重みを torchvision で直に読んだ予測と引き合わせる.

クラスを 2 つにするのは, 見本の意思決定 (対象を選ぶ, `select`) が 2 クラスの分類の一式にだけ使えるため (ML-Inference の `decision.py`).

## モデル

ML-CreateArchitecture の同梱の例 `cnn_classifier` (入力 `image` [4, 1, 28, 28], 畳み込み 2 つ, 全体の平均のプーリング, 全結合の `head` の出す値 10). 画面の「データセットから入力と出力を作る」(ML-CreateArchitecture の `frontend/src/lib/dataset.ts` の `applyDatasetInputs`) と同じ当て方を `classes.py` の `apply_dataset_inputs` に写す. 入力の名前の付け替え, 形と型, 入力に直につながる部品の `in_features` と `in_channels` と `num_embeddings` (入力が語彙の大きさを持つときだけ. 30 で使う), 出力のつなぎ元の `out_features` と `out_channels`. 部品が引数を持つかは, 画面と同じく ML-CreateArchitecture の部品の一覧 (`GET /api/units`) で見る. 当てた後と手で直した後に, 画面の「確かめる」(`POST /api/check`) で形を確かめる. 当てた後に残る直しを「手」と数える (この例では無し).

`resnet_ui.py` は学習しない. ML-Provide の画面の「既製のモデルから作る」で出どころ `torchvision`, モデル `resnet18`, 重み `IMAGENET1K_V1` を選ぶ (種類は 1000. 前処理は重みの説明から一式に書く). 重みは PyTorch の置き場 (`torch.hub.get_dir()` の `checkpoints`) に無ければ, 作るときに出どころから取りに行く (約 45 MB).

## 学習の設計

ML-Pathfinder が書き出した骨組み (`project/train.py`) を書き換えずに, ML-Dashboard の「学習を起動」で動かす (エポックの数は骨組みの既定の 50, 監視する指標は問題の設定の「学習で見る指標」の `valid/loss`). 段階と API は 25 の README の「学習の設計」と同じで, 次が加わる.

| 段階 | アプリ | API |
| --- | --- | --- |
| 見本の答えを画像の表に合わせる (データの列, 目的の列 `label`, 当たりのクラス, 1 件の単位, 予測のときに分かる列, 使ってはいけない列 `digest`) | ML-Pathfinder | `PUT /api/problems/{id}/answers` |
| 分析の結果の画像の大きさを見て中身の手順を決める (1 種類なら無し) | ML-Fathom | `GET /api/sources/{id}/profile` |
| 同梱の例を開き, 入力と出力を当て, 形を確かめる | ML-CreateArchitecture | `GET /api/examples/cnn_classifier`, `GET /api/units`, `POST /api/check` |

## 確かめること

- [x] 見本の問題の設定に画像のディレクトリを入れ, 答えを画像の表に合わせると, 確かめに誤りが無く骨組みを書き出せる
- [x] ML-Fathom が問題の設定を一覧から開き, 大きさのそろった画像を中身の手順なしでプロジェクトの `outputs/dataset` に書き出せる
- [x] ML-CreateArchitecture で同梱の例 `cnn_classifier` に画面と同じく入力と出力を当てると, 手で直す所なく形の確かめが通る
- [x] ML-Dashboard の画面から学習の環境を作り, 骨組みの `train.py` のまま学習と推論の一式の作成が通る
- [x] 骨組みが決まりで今のやり方と比べない理由を README と `train.py` に書き, 比べる相手の評価とたどり直しの今のやり方が無いことがその理由で分かる
- [x] ML-Provide で実験を一覧から登録し, 元の問題の設定をつないで新しい画像を推論と意思決定できる
- [x] 学習の後に ML-Pathfinder で損得を直すと, ML-Provide の次の推論の既定と推論で使う値に入る
- [x] 手の段階 0, 誤りの段階 0
- [x] 本物の ResNet18 の一式を ML-Provide の画面 (ブラウザ) から下見して作って登録し, 画像のディレクトリを推論すると, 予測の表に ImageNet の種類の名前が出て, 同じ重みを torchvision で直に読んだ予測と同じになる (`resnet_ui.py`. 手の段階 0, 誤りの段階 0)

## 期待する結果

手の段階 0, 誤りの段階 0. 比べる相手の評価と今のやり方とのたどり直しは, 骨組みが決まりで今のやり方と比べない (理由を骨組みの README に書く) ので, 今のやり方が無いのを OK とする (25 の `apps.py` の `run_journey` の `compare`).

`resnet_ui.py` も手の段階 0, 誤りの段階 0. 下見に 1000 種類が出て, 推論した全ての画像の種類の名前が ImageNet の種類で, torchvision で直に読んだ予測と同じ.

## 実行方法

25 の README の「実行方法」と同じく, 5 つのアプリを clone して `make setup` を済ませ, ルートのディレクトリで実行する. 手元で動かせなければ, CI の「通しを測る」(`.github/workflows/journeys.yml`) を `tasks` に `29_m3_image_classes` を入れて手で起動する.

```bash
.venv/bin/python tasks/29_m3_image_classes/journey.py
.venv/bin/python tasks/29_m3_image_classes/journey.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

`resnet_ui.py` は ML-Provide だけを起動し (25 の `apps.py`), ML-Provide の画面を作業用の場所の `dist` に作って (`vite build`. clone の中の配信の場所は変えない) `vite preview` で配信し, `node` で `resnet_ui.mjs` を動かしてブラウザを操作する. ML-Provide の `make setup` (画面の依存の `pnpm install`) と, Playwright のブラウザ (ML-Provide の `frontend` で `pnpm exec playwright install chromium`) が要る. 重みは PyTorch の置き場に無ければ初めにネットワークから取りに行く. 画面の写し (下見, 一式の画面, 推論の結果) は作業用の場所の `shots` に, ブラウザの操作の記録は `logs/browser.log` に書く.

```bash
.venv/bin/python tasks/29_m3_image_classes/resnet_ui.py
.venv/bin/python tasks/29_m3_image_classes/resnet_ui.py --work <作業用の場所> --apps <アプリの親のディレクトリ>
```

## 結果のメモ

CI の「通しを測る」で, 5 つのアプリの main を使って動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 問題の設定 (データ, 今のやり方) | OK | 確かめに誤りが無く書き出せる. 今のやり方の文は無し |
| 骨組みの書き出し (場所を問題の設定に記録) | OK | データセットとモデルの定義の場所が記録される |
| ML-Fathom の中身の手順 (画像と文) | OK | 大きさが 1 種類 (28x28, 120 枚) なので, 中身の手順は無し |
| ML-Fathom (一覧から開く, 前処理, プロジェクトへ書き出す) | OK | 入力 [3, 28, 28], 出す値 2. 分割は学習用 84, 検証用 18, テスト用 18 (層別) |
| ML-CreateArchitecture で問題の設定を一覧から選ぶ | OK | 一覧の問題の設定が, 記録したプロジェクトとデータセットを持つ |
| ML-CreateArchitecture (入力と出力, プロジェクトに保存) | OK | 画面が当てる: 入力 `image` の名前を `x` に, `conv1` の `in_channels` を 3 に, `head` の `out_features` を 2 に. 形の確かめが通る. 入力 [4, 3, 28, 28] |
| ML-Dashboard で書き出したプロジェクトを一覧から登録 | OK | 問題の設定「解約の引き止め」が記録したプロジェクト |
| 学習の準備 | OK | 足りない機能 eda, modelcontrol, inference |
| 環境を作る (画面から) | OK | 7 秒 (uv のキャッシュあり) |
| 学習 | OK | ベストの valid/loss 0.4654 (50 エポック目). テスト用の正解率 0.6111. 推論の一式: valid の acc 0.8333 (監視した指標が accuracy か mae でないので, 学習の記録とは比べない) |
| 比べる相手の評価 (ML-Dashboard の結果) | OK | 決まりで比べない (骨組みの README の理由: 入れたデータが 1 件の中身で学ぶ画像か文で, データセットに学習に使う列が無く, 元のデータの同じ行が分からない) |
| ML-Provide で実験を一覧から登録 | OK | 問題の設定の名前と実験の場所の行を選ぶ |
| 問題の設定をつなぐ (候補から) | OK | 候補は元の問題の設定. 損得 (価値 24,000, 費用 1,500, 防げる割合 0.2, 上限 300, 当たりのクラス `circle`) が入る |
| 推論と意思決定 | OK | 新しい 20 枚 (各クラス 10 枚) を予測 `circle` 15 件, `square` 5 件 |
| 今のやり方とのたどり直し (問題の設定から作る) | OK | 方針は「この方針」22,500 と「確率の高い順に同じ件数」22,500. 決まりで今のやり方は無い |
| ML-Pathfinder で損得を直すと次の推論の既定に入る | OK | 連絡 1 件の費用 3,000 が, 推論の既定と推論で使った値に入る |

手の段階 0, 誤りの段階 0.

`resnet_ui.py` を手元で, ML-Provide の main と, 重みが PyTorch の置き場にある状態で動かした値.

| 段階 | 判定 | 説明 |
| --- | --- | --- |
| 画面を作って配信 | OK | ML-Provide の main の画面 (`vite build` と `vite preview`) |
| 画面を開く (トークンの URL) | OK | 一式の一覧に「既製のモデルから作る」 |
| 既製のモデルの下見 | OK | 種類 1,000 (tench, goldfish, great white shark ほか), 重みの版 `IMAGENET1K_V1`, 前処理は短い辺を 256 |
| 作って登録 (一式の画面) | OK | 出どころ torchvision の resnet18 (重み `IMAGENET1K_V1`). 重みは出どころから読んで置き場にある |
| 画像のディレクトリを推論 | OK | 6 件のうち 6 件を予測 |
| 種類の名前が表に出る | OK | 予測の表の `prediction` の列に 6 件の種類の名前 (列 `#`, `status`, `reason`, `prediction`, `confidence`, `decision` と画像の表の列) |
| 画面の種類の名前が ImageNet の種類 | OK | 赤 nematode, 緑 jellyfish, 青 jellyfish, 市松 crossword puzzle, 縞 picket fence, 同心円 maze |
| torchvision で直に読んだ予測と同じ | OK | 6 / 6 枚が同じ |

手の段階 0, 誤りの段階 0.

## 気付いたこと

- 画像の分類は画面だけで通る. 骨組みの `train.py` は画像の配列 (件数, 3, 28, 28) で学習と推論の一式の作成まで通り, 学習の記録 (`train.log`) に 1 件の形「1 件の画像の形 (チャンネル, 高さ, 幅): (3, 28, 28)」を書く
- 骨組みは画像と文の課題では決まりで今のやり方と比べず, 理由 (入れたデータが 1 件の中身で学ぶ画像か文で, データセットに学習に使う列が無く, 元のデータの同じ行が分からない) を README と `train.py` に書く. 対象を選ぶ意思決定の今のやり方 (ML-Quest の問い `action.current`) は「行動したかの列」で書くが, 画像のディレクトリの表 (ML-EDA の `load_data`) の列はファイルから作る列 (`path`, `label`, `width`, `height`, `channels`, `mode`, `format`, `bytes`, `digest`) だけで, 書く列が無い. ML-Dashboard の結果に今のやり方の評価は出ず, ML-Provide のたどり直しにも今のやり方は出ない. 比べる道は TODO.md の「画像と文の課題で今のやり方と比べる道」
- 目的の列が 2 種類でない (3 クラスの画像など) と, ML-Quest のデータの確かめ (`inspect_data`) が, 分類で答える意思決定 (対象を選ぶ, 振り分ける) では誤りを出す (意思決定しないは見ない. 書き出しは止めない). 推論の段階の ML-Inference の `decision.py` がこれらの意思決定を 2 クラスの分類の一式にだけ使い, 学習を終えて一式を作るまで分からないため. この通しは 2 クラスなので当たらない
- 骨組みの既定の 50 エポックでは検証用の損失がまだ下がり続けていて, テスト用の正解率 0.6111 と低い. 推論の 20 枚も `circle` に寄る. 画面だけで通るかの確かめなのでエポックの数は変えない
- 本物の ResNet18 の一式は画面だけで作って登録し, 推論できる. 種類が 1000 あるので, 予測の表に種類ごとの確率の列は出さず (CSV に入れる), 結論に種類の名前ごとの件数が出る. 既製のモデルの一式はデータセットと外れ方の記録を持たないので, 意思決定とたどり直しは出ない (画面がその理由を書く)
- 模様の画像は ImageNet の種類に当たらないので, 確信度は低い (赤の nematode で 0.038). 確かめは種類の名前が出ることと直の予測と同じことで, 当たり外れは見ない
- ML-Provide の画面の開発用の転送 (`frontend/vite.config.ts` の `server.proxy` の `/api`) は転送先を文字だけで書くので, vite が `changeOrigin: true` にして Host を転送先に変える. ML-Provide は Origin と Host が違う書き込み (POST など) を 403 で断るので, `make dev-frontend` の画面や同じ転送の `vite preview` では書き込みが通らない. `resnet_ui.py` は Host を変えない転送の設定を作業用の場所に書いて配信する. ML-Pathfinder, ML-Fathom, ML-CreateArchitecture, ML-Dashboard, ML-Studio の画面も同じ書き方の転送と同じ Origin の確かめを持つ
