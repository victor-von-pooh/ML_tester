# ML_tester
機械学習のテストフィールド.

[ML-Found](https://github.com/victor-von-pooh/ML-Found) を利用者の立場で使い, いろいろな種類のタスクで機能するかを確かめる.
各タスクの `train.py` を手で動かして結果を見て, 書きにくい点や足りない機能を [FEEDBACK.md](FEEDBACK.md) にまとめる.

## 導入

```bash
git clone --recursive https://github.com/victor-von-pooh/ML_tester.git
cd ML_tester
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./ML-Found
```

## タスク

| # | ディレクトリ | タスク | データ | 主に確かめる機能 |
| --- | --- | --- | --- | --- |
| 1 | `tasks/01_tabular_cls` | 表形式の多クラス分類 | Iris など | 基本の流れ, monitor と mode, Early Stopping, experiment の差分 |
| 2 | `tasks/02_image_cls` | 画像分類 | MNIST / FashionMNIST | ミニバッチ, デバイス, step ごとの Scheduler, 勾配クリッピング |
| 3 | `tasks/03_imbalanced_cls` | 不均衡な二値分類 | 合成データ | `on_epoch_end` での AUC, `batch_size` の指定 |
| 4 | `tasks/04_text_cls` | テキスト分類 | arXiv の論文リスト | 長さの違う系列, 辞書のバッチ |
| 5 | `tasks/05_tabular_reg` | 表形式の回帰 | California Housing | ReduceLROnPlateau, 自作の Optimizer |
| 6 | `tasks/06_func_approx` | 関数近似 | 合成データ | フルバッチ, LBFGS |
| 7 | `tasks/07_weather_forecast` | 時系列予測 | 気象庁の東京の天気 | 系列データ, 学習の再開 |
| 8 | `tasks/08_autoencoder` | オートエンコーダ | MNIST | 教師なし学習, 再構成画像の保存 |
| 9 | `tasks/09_vae` | VAE | MNIST | 損失の内訳の記録, サンプルの生成 |
| 10 | `tasks/10_gan` | GAN | MNIST | 2 つの Optimizer の交互の更新 |
| 11 | `tasks/11_diffusion` | 拡散モデル | MNIST | timestep の乱数, EMA |

各タスクのディレクトリは次の構成にする.

```
tasks/XX_name/
├── configs/
│   ├── default/train.yaml
│   └── experiment/train.yaml
├── train.py
└── README.md       # 目的, 確かめる機能, 実行方法, 期待する結果, 結果のメモ
```

## データ

外部から取得したデータは `data/` に置く (git の管理外).
