# ML_tester
機械学習のテストフィールド.

[ML-Found](https://github.com/victor-von-pooh/ML-Found) を利用者の立場で使い, いろいろな種類のタスクで機能するかを確かめる.
各タスクの `train.py` を手で動かして結果を見て, 書きにくい点や足りない機能を [FEEDBACK.md](FEEDBACK.md) にまとめる. 結果は ML-Dashboard の画面でも見られる (下の「ML-Dashboard で見る」).

## 導入

```bash
git clone --recursive https://github.com/victor-von-pooh/ML_tester.git
cd ML_tester
python3.13 -m venv .venv    # Python 3.13 で動作確認
source .venv/bin/activate
pip install -e "./ML-Found[examples]"    # scikit-learn も入る
pip install torchvision==0.29.2           # 02, 08 から 11 で使う (torch 2.15.0 と組み合わせて動作確認)
pip install qiskit==2.6.2 qiskit-machine-learning==0.9.1   # 12 で使う
pip install transformers==5.22.2          # 04 の段階 2 で使う
```

## ML-Dashboard で見る

実験の一覧, 学習曲線の比較, 画面からの学習の起動には [ML-Dashboard](https://github.com/victor-von-pooh/ML-Dashboard) を使える. 各タスクは ML-Found のプロジェクトの形なので, そのまま読み込める.

1. 上の「導入」で `.venv` を作っておく. 学習に使う Python は, 登録したときにタスクのディレクトリから上に向かって仮想環境を探して決まるので, 先に作ると `.venv` が自動で選ばれる
2. ML-Dashboard を ML_tester とは別のディレクトリに clone し, その README の「導入と起動」の手順で起動して, 表示されたトークン付きの URL を開く
3. 「プロジェクトを追加」で ML_tester を clone したディレクトリを絶対パスで入れて「探す」を押すと, `tasks/` の 12 のタスクが候補に出る (submodule の ML-Found の例は対象外). そのまま「12 件を追加」で登録する

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
└── README.md       # 目的, 確かめる機能, 実行方法, 期待する結果, 結果のメモ
```

必要に応じて `data.py` (データの用意) や `evaluate.py` (学習後の評価) などを足す.

## モデルの書き方

モデルは各タスクの `model.py` に自前のクラスとして書く. 書き方の要点.

- 部品ごとにクラスを分け, 全体のクラスで組み合わせる (例: `Encoder`, `Decoder` と, それらを持つ `VAE`)
- `__init__` では `super(クラス名, self).__init__()` を呼び, 引数を `self` に保存してから層と活性化関数を属性として定義する (`self.fc1 = nn.Linear(...)`, `self.relu = nn.ReLU()`)
- `forward` の docstring は「順伝播を行う関数」とし, 複数の値を返す場合はタプルで返す (例: VAE は `outputs, mu, log_var, z`)
- 一部だけを使う処理は `encode` や `decode` のようなメソッドに分ける
- docstring とコメントの細かい書き方は ML-Found の [CODING_STYLE.md](ML-Found/CODING_STYLE.md) に従う (`Returns` の下線は 10 本, コメントは体言止め)

## データ

外部から取得したデータは `data/` に置く (git の管理外).
