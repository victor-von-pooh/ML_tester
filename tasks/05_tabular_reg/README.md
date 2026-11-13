# 05 表形式の回帰

## 目的

回帰タスクで, ReduceLROnPlateau と, `.` を含む名前で自作の Optimizer を読み込む機能を確かめる.
RMSE のように, バッチの平均をとると値が変わってしまう指標の扱いも確かめる.

## ディレクトリ構成

```
05_tabular_reg/
├── configs/
│   ├── default/train.yaml          # 基準の config
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 4 の差分
│           ├── lion.yaml
│           ├── plateau_max.yaml
│           └── plateau_step.yaml
├── custom_optim.py                 # 自作の Optimizer (Lion)
├── model.py                        # HousingRegressor
├── train.py                        # データ, RegressionTrainer (Trainer を継承), 学習の実行
└── outputs/california/<日付>/<時刻>/   # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | California Housing (`sklearn.datasets.fetch_california_housing`) |
| 取得方法 | scikit-learn が初回に自動でダウンロード (リポジトリの `data/sklearn/` に保存, 約 360 KB) |
| 規模 | 20,640 サンプル, 特徴量 8 (所得の中央値, 築年数, 世帯あたりの平均部屋数, 緯度経度など) |
| 目的変数 | 地区の住宅価格の中央値 (単位: 10 万ドル) |
| 分割 | 学習用 80 % と検証用 20 % (`experiment.seed` で分割) |
| 前処理 | 特徴量は `StandardScaler`, 目的変数も学習用データの平均と標準偏差 (1.156) で標準化して学習し, 指標は元の単位に戻して求める |

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `HousingRegressor`: `Linear → ReLU` を 2 回と出力層の 3 層の MLP (隠れ層 128), 出力は 1 次元
- 自作の Optimizer として, Lion (勾配の移動平均の符号だけで更新する Optimizer) を `custom_optim.py` に `optim.Optimizer` の子クラスで実装し, config で `optimizer.name: custom_optim.Lion` と指定する

## step と Trainer の設計

`Trainer` を継承した `RegressionTrainer` で, 次のメソッドを上書きする.

- `step`: 標準化した空間で `nn.MSELoss()` の損失を求める. 元の単位に戻した誤差から, MAE (`mae`) とバッチごとの RMSE (`rmse_batch`) を返す. エポック全体の RMSE のために二乗誤差の合計とサンプル数も貯める
- `on_epoch_start`: 二乗誤差の合計とサンプル数を 0 にする
- `on_epoch_end`: 二乗誤差の合計から, エポック全体の RMSE (`rmse`) を求めて返す

MAE はバッチの平均の重み付き平均がエポック全体の平均と一致するので `step` で返してよい.
RMSE はバッチごとの RMSE の平均がエポック全体の RMSE と一致しないので, `on_epoch_end` で求めた `rmse` と, `step` で返した `rmse_batch` を両方記録して比べる.

## config

```yaml
# default
experiment:
  name: california
  seed: 42
  device: cpu        # 小さな MLP なので cpu の方が速い (01 で確認)
optimizer:
  name: Adam
  params:
    lr: 0.001
scheduler:
  name: ReduceLROnPlateau
  params:            # mode は省略して trainer.mode (min) に合わせる
    factor: 0.5
    patience: 5
trainer:
  epochs: 200
  early_stopping:
    use: true
    patience: 20
data:
  root: ../../data/sklearn   # 相対パスはタスクのディレクトリから辿る
  valid_size: 0.2
  batch_size: 128
model:
  hidden_dim: 128
```

experiment で次の差分を試す. 2 から 4 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `optimizer.name: custom_optim.Lion`, `params: {lr: 0.0001, weight_decay: 0.01}` (`lion.yaml`)
3. `scheduler.params.mode: max`. `trainer.mode: min` と食い違うのでエラーになることの確認 (`plateau_max.yaml`)
4. `scheduler.interval: step`. ReduceLROnPlateau ではエラーになることの確認 (`plateau_step.yaml`)

## 確かめる ML-Found の機能

- [x] ReduceLROnPlateau が `valid/loss` を見て学習率を下げ, `lr_curve.png` に階段状に出る
- [x] ReduceLROnPlateau の `mode` を省略すると `trainer.mode` に合わせられる (`training/optim.py` の `build_scheduler` で確認. 実験 1 は `trainer.mode` も torch の既定値も `min` なので, 実験では区別できない)
- [x] `mode` や `interval` が食い違うと分かりやすいエラーになる. ML-Found `3aca743` ではエラーになるのが出力先を作った後で, 学習結果の無い出力先が残った (下の「気付いたこと」). その後 ML-Found の PR #1 で, 出力先を作る前にエラーになるようになった
- [x] `custom_optim.Lion` のように `.` を含む名前で自作の Optimizer を読み込める. `train.py` のあるディレクトリは Python が自動で import の探索先に入れるので, 別のディレクトリから実行しても読み込めた
- [x] `on_epoch_end` で求めた RMSE と, `step` で求めた RMSE の違い (下の結果を参照)

## 期待する結果

- 検証用データの RMSE が 0.55 (5.5 万ドル) 以下
- Lion でも Adam と同程度まで下がる

## 予想される問題

- 指標の平均の扱い (RMSE, R² など) は利用者が気付きにくいので, ドキュメントでの注意か, 合計と件数を返す仕組みがあると良さそう → 差は小さかったが, バッチの RMSE の平均は全てのエポックで全体の RMSE 以下だった
- `train.py` を別のディレクトリから実行すると, `custom_optim` が import の探索先に無くて読み込めない可能性がある → 問題なかった. Python はスクリプトのディレクトリを探索先の先頭に入れるため

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく (有効にしない場合は `python` を `.venv/bin/python` に読み替える).
`train.py` はどこから実行してもよく, 出力先は常に `tasks/05_tabular_reg/outputs/california/<日付>/<時刻>/` になる.
初回はデータのダウンロードが入る (ネットワークが必要, リポジトリの `data/sklearn/` に保存).

```bash
# リポジトリのルートから
python tasks/05_tabular_reg/train.py                                                           # 実験 1
python tasks/05_tabular_reg/train.py --experiment configs/experiment/examples/lion.yaml          # 実験 2
python tasks/05_tabular_reg/train.py --experiment configs/experiment/examples/plateau_max.yaml   # 実験 3 (エラー)
python tasks/05_tabular_reg/train.py --experiment configs/experiment/examples/plateau_step.yaml  # 実験 4 (エラー)
```

- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 実験 1 と 2 の出力先には, `config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線 (`loss`, `mae`, `rmse`, `rmse_batch`, `lr`), `best.pt`, `last.pt`, `final_model.pt` (未コミットの変更があれば `git_diff.patch`) が入る
- 実験 3 と 4 は, 出力先を作る前にエラーで止まる (出力先は作られない. ML-Found `3aca743` では `config.yaml`, `diff.yaml`, `train.log` だけを含む出力先が残った)
- 学習時間は `train.log` の最後の「学習時間」に出る

## 結果のメモ

2026-11-08, ML-Found `3aca743`, PyTorch 2.15.0, scikit-learn 1.9.1, macOS (Apple Silicon), Python 3.13, cpu で実行.
同じ環境なら同じ結果になる (実験 2 を別のディレクトリから実行し直して `history.csv` が一致した). OS やライブラリの版が違う場合は一致しないことがある.

| 実験 | 止まったエポック | ベストのエポック | 検証用 RMSE (`rmse`) | 検証用 RMSE (`rmse_batch`) | 検証用 MAE | 学習時間 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Adam | 111 | 91 | **0.5161** | 0.5133 | 0.3499 | 6.8 秒 |
| 2 `lion` (Lion) | 123 | 103 | **0.5132** | 0.5107 | 0.3466 | 6.5 秒 |
| 3 `plateau_max` | - | - | エラー (想定通り) | - | - | - |
| 4 `plateau_step` | - | - | エラー (想定通り) | - | - | - |

- 実験 1 と 2 はどちらも目標の RMSE 0.55 (5.5 万ドル) を下回った. Lion は Adam と同程度で, わずかに良かった
- 実験 1 の学習率は, ReduceLROnPlateau で 0.001 から 0.00000098 まで 10 回半分になった (`lr_curve.png` は階段状). エポック 22 から 27 まで `valid/loss` が改善しなかったので, エポック 28 から 0.0005 になった
- 実験 3 は `ValueError: ReduceLROnPlateau の mode (max) が trainer.mode (min) と違います`, 実験 4 は `ValueError: ReduceLROnPlateau は scheduler.interval を epoch にしてください` で止まった
- 実験 1 と 2 の全てのエポックで, `rmse_batch` (バッチごとの RMSE のサンプル数での重み付き平均) は `rmse` (エポック全体の RMSE) 以下だった. 平方根は上に凸なので, バッチごとに平方根をとってから平均すると全体の RMSE 以下になる (全てのバッチの二乗誤差の平均が等しい場合だけ一致). 検証用データでの差はベストのエポックで約 0.003, 最大で 0.005 だった

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した.

- ReduceLROnPlateau の `mode` や `interval` の食い違いは, `Trainer` を作るとき (`start_experiment` の後) に確かめられるので, エラーで止まっても `config.yaml`, `diff.yaml`, `train.log` (と `git_diff.patch`) だけを含む出力先が残る. 01 の「default に無い項目」のエラーは `load_config` で起きるので出力先が作られなかったのと対照的
- エラーで止まった場合, `train.log` にはエラーの内容が書かれず, 最後の行は途中の記録 (このタスクでは「目的変数の標準偏差」) のままになる. 後から出力先を見ても, 失敗した実験だと分かりにくい
- `step` で RMSE を返すと, ML-Found はバッチごとの値をサンプル数で重み付けして平均するので, エポック全体の RMSE 以下になる. README の `step` の説明に, 平均してよい指標とそうでない指標の注意があると良い

### 最新の ML-Found での再実行 (2026-11-13)

上の結果は ML-Found `3aca743` で得たもの. 2026-11-13 に, 全ての実験をその後の ML-Found (`24f18c2`) で流し直し, 学習時間は `e2df46e` でも測り直した. 実験 1 と 2 の `history.csv` は一致した. 実験 3 と 4 は同じエラーで, 今は出力先を作る前に止まる.

- 学習時間: 実験 1 は 6.32 秒 (`e2df46e`), 実験 2 は 6.98 秒 (`24f18c2`)
