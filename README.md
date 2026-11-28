# ML_tester
機械学習の整理と試しの場.

「機械学習を全て GUI 上で行う」ための 11 のリポジトリ (問題の設定の ML-Quest と ML-Pathfinder, データの分析の ML-EDA と ML-Fathom, モデルの作成の ML-ModelControl と ML-CreateArchitecture, 学習の [ML-Found](https://github.com/victor-von-pooh/ML-Found) と ML-Dashboard, 推論の ML-Inference と ML-Provide, 5 つの段階のアプリを 1 つにまとめる殻の ML-Studio) を, 利用者の立場でまとめて使い, いろいろな種類のタスクで機能するかを確かめる. 各タスクのスクリプトを手で動かして結果を見て, 気付いたことを各タスクの README に書く. 結果は ML-Dashboard の画面でも見られる (下の「ML-Dashboard で見る」).

ML_tester は他のリポジトリの依存ではない. 各リポジトリはこのリポジトリが無くても単独で使え, 手順書, 例, テストも各リポジトリだけで完結する.

- [VISION.md](VISION.md): 全体の方針と各リポジトリの役割
- [TASK_TYPES.md](TASK_TYPES.md): 課題の型と全体の広げ方, 利用者の決めたこと
- [ROADMAP.md](ROADMAP.md): 理想と今の差, 差を埋める節目 (節目「まだ扱っていない型の最初の 1 つを画面で通す」から先) の順
- [TODO.md](TODO.md): まだやっていないこと. 各リポジトリの TODO.md の項目を, 節目ごとに, リポジトリをまたいで進める順に並べる

## 導入

```bash
git clone --recursive https://github.com/victor-von-pooh/ML_tester.git
cd ML_tester
python3.13 -m venv .venv    # Python 3.13 で動作確認
source .venv/bin/activate
pip install -e "./ML-Found[examples,rl]"    # scikit-learn と, 24 で使う Gymnasium (追加の依存 rl) も入る
pip install torchvision==0.29.2           # 02, 08 から 11 で使う (torch 2.15.0 と組み合わせて動作確認)
pip install qiskit==2.6.2 qiskit-machine-learning==0.9.1   # 12 で使う
pip install "ml-modelcontrol @ git+https://github.com/victor-von-pooh/ML-ModelControl.git@76c6bfc"   # 13, 17, 18, 21 で使う
pip install "ml-eda[all] @ git+https://github.com/victor-von-pooh/ML-EDA.git@660ea3b"   # 14 から 23 と, 27, 28 の合成のデータで使う (all に部分語の tokenizers も入る. 22 と 23 は書き出しの版 2 を使う)
pip install --no-deps "ml-inference @ git+https://github.com/victor-von-pooh/ML-Inference.git@1ed123a"   # 17 から 21 と 24 で使う (依存は上の 2 つ. 版をそろえるため --no-deps)
pip install --no-deps "ml-quest @ git+https://github.com/victor-von-pooh/ML-Quest.git@ae3d60c"   # 21 で使う (依存の pandas と PyYAML は上で入る)
pip install transformers==5.22.2          # 04 の段階 2 で使う
pip install pytest                        # 24 の守りのテスト (tasks/24_rl_cartpole/test_guards.py) で使う
```

## ML-Dashboard で見る

実験の一覧, 学習曲線の比較, 画面からの学習の起動には [ML-Dashboard](https://github.com/victor-von-pooh/ML-Dashboard) を使える. 各タスクは ML-Found のプロジェクトの形なので, そのまま読み込める.

1. 上の「導入」で `.venv` を作っておく. 学習に使う Python は, 登録したときにタスクのディレクトリから上に向かって仮想環境を探して決まるので, 先に作ると `.venv` が自動で選ばれる
2. ML-Dashboard を ML_tester とは別のディレクトリに clone し, その README の「導入と起動」の手順で起動して, 表示されたトークン付きの URL を開く
3. 「プロジェクトを追加」で ML_tester を clone したディレクトリを絶対パスで入れて「探す」を押すと, `tasks/` の 22 のプロジェクト (01 から 18 と, 21 の `project/`, 22 から 24. 19, 20, 25 から 31 は学習のスクリプトを持たないのでプロジェクトではない) が候補に出る (submodule の ML-Found の例は対象外). そのまま「22 件を追加」で登録する

clone した直後は実験が無いので, 各タスクの README の手順で学習するか, 画面から起動すると一覧に出る.

画面から学習を起動する場合の注意.

- 04 の `train_hf.py` と 06 の `train_pinn.py` を選ぶと, experiment の config が `configs/experiment/hf.yaml` (06 は `pinn.yaml`) に切り替わり, 起動の前の確認は `configs/default/hf.yaml` (06 は `pinn.yaml`) を元にする. `examples/hf_smoke.yaml` と `examples/pinn_force_grad.yaml` も, それぞれのスクリプトを選んでから使う. 06 の `train_pinn.py` の既定 (`pinn.yaml`) は検証で失敗することを確かめる例なので, エポック 1 の検証でエラーになる (起動の前の確認では止まらない)
- 04 と 07 はデータの用意が要るので, 先に各タスクの README の手順で用意する. 02, 05, 08 から 11 と 04 の `train_hf.py` は, 初回にデータやモデルをダウンロードする (ネットワークが必要)
- 01 の `invalid_key.yaml` と 05 の `plateau_max.yaml`, `plateau_step.yaml` はエラーになることを確かめる例なので, 起動の前の確認で止まる
- 手早く試すなら 01 の `examples/cpu.yaml` (Python の起動を含めて数秒で終わる)

## タスク

| # | ディレクトリ | タスク | データ | 主に確かめる機能 |
| --- | --- | --- | --- | --- |
| 1 | `tasks/01_tabular_cls` | 表形式の多クラス分類 | Iris など | 基本の流れ, monitor と mode, Early Stopping, experiment の差分 |
| 2 | `tasks/02_image_cls` | 画像分類 | Fashion-MNIST | ミニバッチ, デバイス, step ごとの Scheduler, 勾配クリッピング |
| 3 | `tasks/03_imbalanced_cls` | 不均衡な二値分類 | 合成データ | `on_epoch_end` での AUC, `batch_size` の指定 |
| 4 | `tasks/04_text_cls` | テキスト分類 | arXiv の論文リスト | 長さの違う系列, 辞書のバッチ |
| 5 | `tasks/05_tabular_reg` | 表形式の回帰 | California Housing | ReduceLROnPlateau, 自作の Optimizer |
| 6 | `tasks/06_func_approx` | 関数近似 | 合成データ | フルバッチ, LBFGS |
| 7 | `tasks/07_weather_forecast` | 時系列予測 | 気象庁の東京の天気 | 系列データ, 学習の再開 |
| 8 | `tasks/08_autoencoder` | オートエンコーダ | MNIST | 教師なし学習, 再構成画像の保存 |
| 9 | `tasks/09_vae` | VAE | MNIST | 損失の内訳の記録, サンプルの生成 |
| 10 | `tasks/10_gan` | GAN | MNIST | 2 つの Optimizer の交互の更新 |
| 11 | `tasks/11_diffusion` | 拡散モデル | MNIST | timestep の乱数, EMA |
| 12 | `tasks/12_quantum_nn` | 量子ニューラルネット | make_moons, Iris | CPU だけで動くモデル, 外部ライブラリの乱数, チェックポイント |
| 13 | `tasks/13_model_spec` | モデルの定義から組み立てたモデルの学習 | 手書き数字 (8x8) | ML-ModelControl の定義を config で選んで学習, 定義を実験に残して作り直す (モデルの作成と学習の交点) |
| 14 | `tasks/14_eda_dataset` | ML-EDA で前処理したデータセットの学習 | 合成した顧客の表 | 気付きからレシピを書き, 書き出したデータセットを load_dataset で学習, 新しいデータに同じ前処理 (データの分析と学習の交点) |
| 15 | `tasks/15_eda_images` | ML-EDA で配列にした画像のデータセットの学習 | 合成した図形の画像 | 大きさ, 色の形式, 読めないファイルの混ざる画像のフォルダを中身の手順で配列にし, 中身を読まずにレシピを確かめ, 新しい画像に同じ手順 (データの分析と学習の交点) |
| 16 | `tasks/16_eda_pointcloud` | ML-EDA で配列にした点群のデータセットの学習 | 合成した形の点群 | PLY, PCD, xyz の混在, 点の数の違い, 座標が数でない点を中身の手順でそろえ, 新しい点群に同じ手順 (データの分析と学習の交点) |
| 17 | `tasks/17_ja_reviews_bert` | 日本語のレビューを画面で組んだ BERT で分類 | 合成した日本語のレビュー (汚れた csv とラベル無しの文) | ML-Fathom で文の列を前処理, ML-CreateArchitecture で組んだ BERT を事前学習 (MLM) と微調整, 文字の平均のモデルと比べる (3 つの段階を通しで). ML-Fathom が勧めた部分語の区切りでも比べる |
| 18 | `tasks/18_store_sales` | 店ごとの日々の売上の予測 | 合成した 12 店の売上 (円の付いた数, 混在した日付, 重複) | ML-Fathom で店ごとの時系列として前処理 (店ごとのずらし), ML-CreateArchitecture で組んだモデルで回帰, 店をまたいだずらしと比べる |
| 19 | `tasks/19_inference` | 学習したモデルで推論し, 結論を出す | 17 と 18 のデータ | 17 と 18 の学習の結果から ML-Inference で推論の一式を作り, 学習のときのテスト用の評価と同じ値になるかを確かめる. 結論 (保留, 値の帯, 店ごと) と ML-Provide の画面 (推論の段階) |
| 20 | `tasks/20_decision` | 予測から意思決定へ | 17 と 18 のデータ | 18 の売上の予測から置く量 (損の比と予測の外れ方), 17 のレビューから人が確かめるもの (損の比のしきい値, 人手の上限) を決め, 予測どおりと今のやり方と過去のデータで比べる (ML-Inference の意思決定) |
| 21 | `tasks/21_problem_setting` | 問題の設定から推論まで | 合成した総菜の売れた数 | ML-Quest の問題の設定から骨組みを書き出し, 受け渡しの値でデータセットとモデルの定義を置いて学習し, 問題の設定の損得と比べる相手と成功の基準でたどり直す (5 つの段階の通し) |
| 22 | `tasks/22_image_masks` | 画像と領域の対の学習 (書き出しの版 2) | ML-EDA の合成の見本の画像と領域 | `export_image_masks` で版 2 に書き出し (正解の出どころ synthetic), `load_items` を torch の Dataset に包んで領域の分割を学習, IoU (データの分析と学習の交点) |
| 23 | `tasks/23_capture_clouds` | 多視点の撮影から点群を作る (書き出しの版 2) | ML-EDA の合成の見本の撮影と正確な点群 | `export_capture_clouds` の正規化, `load_items` の撮影の深度から点に戻して点群を予測, Chamfer 距離を `trainer.valid_every_n_epochs` で数エポックおきに測ってベストを選ぶ, 新しい撮影に `normalize_capture` |
| 24 | `tasks/24_rl_cartpole` | CartPole の方策の強化学習と守り | Gymnasium の CartPole (シミュレータ) | `ml_found.rl` の `RolloutLoader` と `ReinforceStep` で学習 (`valid/reward`, max), 守り (選んでよい行動, 止める条件, 外れたときの動き) と一緒に書き出し, 読み直して守り付きで動かし結果の数を守る指標の表に. 実機での実行は道具の外で, 配備と守りの範囲は人が承認する |
| 25 | `tasks/25_m1_churn` | 表の分類を画面の API だけで通す | 合成した会員の表 (今のやり方で連絡したかの列) | 5 つの画面のアプリを作業用の置き場で起動し, 問題の設定から推論までを画面と同じ API で通して, 段階ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを表にする (節目「表の予測を画面だけで通す」の確かめ). 今のやり方は列 |
| 26 | `tasks/26_m1_sozai` | 時系列の予測を画面の API だけで通す | 合成した 2 店 × 3 品の売れた数 | 25 と同じ通しを時系列の予測で. 今のやり方は k 期前 (前の週の同じ曜日) の決まりで, 骨組みの評価と推論のたどり直しが同じ決まりから作る |
| 27 | `tasks/27_m2_image_masks` | 画像と領域を画面の API だけで通す | 合成した大きさの混ざる画像と領域 | 25 と同じ通しを画像の領域で (節目「点群と領域を画面で通す」の確かめ). ファイルの組の書き出しで大きさをそろえる中身の手順, U-Net の同梱の例, 推論の領域の値 |
| 28 | `tasks/28_m2_capture_clouds` | 多視点の撮影から点群を画面の API だけで通す | 合成した形の多視点の撮影と正確な点群 | 27 と同じ通しを撮影と点群で. 点群の同梱の例, 推論の点群が元の座標に戻るか |
| 29 | `tasks/29_m3_image_classes` | 画像の分類を画面の API だけで通す | 合成した図形の画像 (円と四角. 大きさはそろう) | 25 と同じ通しを画像の分類で (節目「画像と文の分類を画面で通す」の確かめ). 見本「解約の引き止め」の答えを画像の表に合わせ, 畳み込みの同梱の例に画面と同じく入力と出力を当てる. 今のやり方は骨組みが決まりで比べない |
| 30 | `tasks/30_m3_text_classes` | 文の分類を画面の API だけで通す | 合成した日本語の問い合わせの文 (苦情と質問) | 29 と同じ通しを文の分類で. ML-Fathom の分析が勧める区切り方, LSTM の同梱の例と, 画面が当てる埋め込みの語彙の数 |
| 31 | `tasks/31_search_photos` | 似た写真を探すを問題の設定から推論まで通す | 合成した種類ごとのディレクトリの写真 (円, 四角, 三角, 縞) | 25 と同じ表で, ML-Pathfinder の見本「似た商品の写真を探す (既製の埋め込み)」から骨組みを書き出し, 骨組みの `bundle.py` で既製の埋め込み (resnet18) の索引と評価 (Recall@K, NDCG@K, 写真を縮めた画素との差) と検索の一式を作り, ML-Provide で問いの写真ごとに上位 K 件と待ち時間を出す (学習しない. 一式づくりはコマンド) |

各タスクのディレクトリは次の構成にする.

```
tasks/XX_name/
├── configs/
│   ├── default/train.yaml
│   └── experiment/
│       ├── train.yaml
│       └── examples/   # README の実験で使う差分, --experiment で切り替える
├── model.py        # モデルのクラス
├── train.py        # データ, step, 学習の実行
└── README.md       # 目的, ディレクトリ構成, データ, モデル, 学習の設計, 確かめること, 期待する結果, 実行方法, 結果のメモ, 気付いたこと
```

必要に応じて `data.py` (データの用意) や `evaluate.py` (学習後の評価) などを足す.

`train.py` は ML-Found の `load_project_config` で, `--experiment` (experiment の config のパス, 相対パスはタスクのディレクトリから辿り, 無い場合は実行した場所から辿る) と「項目=値」の上書きを受け取る. 例: `.venv/bin/python tasks/01_tabular_cls/train.py --experiment configs/experiment/examples/cpu.yaml trainer.epochs=5`

## モデルの書き方

モデルは各タスクの `model.py` に自前のクラスとして書く (13, 17, 18 は ML-ModelControl のモデルの定義から組み立てるので `model.py` を持たない). 書き方の要点.

- 部品ごとにクラスを分け, 全体のクラスで組み合わせる (例: `Encoder`, `Decoder` と, それらを持つ `VAE`)
- `__init__` では `super(クラス名, self).__init__()` を呼び, 引数を `self` に保存してから層と活性化関数を属性として定義する (`self.fc1 = nn.Linear(...)`, `self.relu = nn.ReLU()`)
- `forward` の docstring は「順伝播を行う関数」とし, 複数の値を返す場合はタプルで返す (例: VAE は `outputs, mu, log_var, z`)
- 一部だけを使う処理は `encode` や `decode` のようなメソッドに分ける
- docstring とコメントの細かい書き方は ML-Found の [CODING_STYLE.md](ML-Found/CODING_STYLE.md) に従う (`Returns` の下線は 10 本, コメントは体言止め)

## データ

外部から取得したデータは `data/` に置く (git の管理外).
