# ML_tester
機械学習のテストフィールド.

[ML-Found](https://github.com/victor-von-pooh/ML-Found) を利用者の立場で使い, いろいろな種類のタスクで機能するかを確かめる.
各タスクの `train.py` を手で動かして結果を見て, 書きにくい点や足りない機能を [FEEDBACK.md](FEEDBACK.md) にまとめる.

## 導入

```bash
git clone --recursive https://github.com/victor-von-pooh/ML_tester.git
cd ML_tester
python3.13 -m venv .venv    # Python 3.13 で動作確認
source .venv/bin/activate
pip install -e "./ML-Found[examples]"    # scikit-learn も入る
pip install torchvision                   # 02, 08 から 11 で使う (torch 2.15.0 と torchvision 0.29.2 で動作確認)
```

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

モデルは各タスクの `model.py` に自前のクラスとして書き, 次のリポジトリの書き方に合わせる.

- `~/myprograms/Pooh/src/gnn/models.py` (VGAE)
- `~/myprograms/MIRAGE-GS/src/segment/model.py` (Swin Transformer など)
- [OddNN の models](https://github.com/victor-von-pooh/OddNN/tree/main/models) (VAE, 自作の LSTM, 量子ニューラルネット)

書き方の要点.

- 部品ごとにクラスを分け, 全体のクラスで組み合わせる (例: `Encoder`, `Decoder` と, それらを持つ `VAE`)
- `__init__` では `super(クラス名, self).__init__()` を呼び, 引数を `self` に保存してから層と活性化関数を属性として定義する (`self.fc1 = nn.Linear(...)`, `self.relu = nn.ReLU()`)
- `forward` の docstring は「順伝播を行う関数」とし, 複数の値を返す場合はタプルで返す (例: VAE は `outputs, mu, log_var, z`)
- 一部だけを使う処理は `encode` や `decode` のようなメソッドに分ける
- docstring とコメントの細かい書き方は ML-Found の [CODING_STYLE.md](ML-Found/CODING_STYLE.md) に従う (`Returns` の下線は 10 本, コメントは体言止め)

## データ

外部から取得したデータは `data/` に置く (git の管理外).
