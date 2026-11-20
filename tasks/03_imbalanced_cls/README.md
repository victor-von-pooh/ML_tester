# 03 不均衡な二値分類

## 目的

正解率では評価できない不均衡なデータで, バッチの平均では求められない指標 (ROC-AUC, PR-AUC) を `on_epoch_end` で記録できるかを確かめる.
`Trainer` を継承する使い方の確認がこのタスクの中心になる.

## ディレクトリ構成

```
03_imbalanced_cls/
├── configs/
│   ├── default/train.yaml          # 基準の config
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/               # 下の「config」の実験 2 から 4 の差分
│           ├── eval_train.yaml
│           ├── no_pos_weight.yaml
│           └── pos1.yaml
├── model.py                        # ImbalancedClassifier
├── train.py                        # データ, AUCTrainer (Trainer を継承), 学習の実行
└── outputs/imbalanced/<日付>/<時刻>/   # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 合成データ (`sklearn.datasets.make_classification`) |
| 取得方法 | その場で生成, ダウンロード不要 |
| 規模 | 20,000 サンプル, 特徴量 20 (有効 8, 冗長 4, 残りは雑音), 正例の割合は `data.pos_ratio` (既定 5 %) |
| 分割 | 学習用 80 % と検証用 20 %, `stratify` で正例の割合を保つ |
| 前処理 | 学習用データで `StandardScaler` を当てはめ, 両方に適用 |

`make_classification` は既定で 1 % のサンプル (約 200 件) のラベルをランダムなクラスに置き換える (`flip_y=0.01`).
その約半分 (約 100 件) は特徴量と無関係に正例になるので, 実際の正例の割合は `pos_ratio` より少し多くなる (5 % で 5.49 %, 1 % で 1.54 %).
この雑音の正例は, 5 % の設定では正例約 1,100 件のうち 1 割程度だが, 1 % の設定では正例約 310 件のうち 3 分の 1 程度を占める.
正例の割合やサンプル数は config の `data` で変えられる.
実データで試したい場合は, クレジットカードの不正検知 (Kaggle の Credit Card Fraud Detection, 正例 0.17 %) に差し替えられる.

用意してもらう必要があるデータは無い.

## モデル

自前で用意する. 事前学習済みモデルは使わない.

- `ImbalancedClassifier`: `Linear → ReLU → Dropout` を 2 回と出力層の 3 層の MLP, 出力は 1 つのロジット (隠れ層 64, ドロップアウト 0.2)
- 損失は `nn.BCEWithLogitsLoss(pos_weight=...)`. `pos_weight` は学習用データの (負例の数 / 正例の数) で, 5 % のとき 17.2

## step と Trainer の設計

`Trainer` を継承した `AUCTrainer` で, `step_fn` を渡さずに次のメソッドを上書きする.

- `__init__`: 親クラスを `step_fn=None` で作った後, `self.device` に置いた `pos_weight` で損失関数を作る
- `on_epoch_start(phase)`: 予測確率と正解ラベルを集めるリストを空にする. あわせて `self.epoch` が「記録済みのエポック数 + 1」と一致することを確かめる (継承したクラスから `self.epoch` を参照できることの確認)
- `step(batch, phase)`: 損失を求め, 予測確率を `detach().cpu()` してリストに追加し, `{"loss": loss, "batch_size": len(y)}` を返す
- `on_epoch_end(phase)`: 集めた予測から `roc_auc_score` と `average_precision_score` を求めて `{"auc": ..., "ap": ...}` を返す

## config

```yaml
# default
experiment:
  name: imbalanced
  seed: 42
  device: cpu        # 小さな MLP なので cpu の方が速い (01 で確認)
optimizer:
  name: Adam
  params:
    lr: 0.001
trainer:
  epochs: 50
  monitor: valid/ap  # on_epoch_end で求める平均適合率
  mode: max
  early_stopping:
    use: true
    patience: 10
data:
  n_samples: 20000
  n_features: 20
  n_informative: 8
  n_redundant: 4
  pos_ratio: 0.05
  valid_size: 0.2
  batch_size: 256
model:
  hidden_dim: 64
  dropout: 0.2
loss:
  use_pos_weight: true
```

experiment で次の差分を試す. 2 から 4 は `configs/experiment/examples/` に置き, `--experiment` で切り替える.

1. 差分なし (`configs/experiment/train.yaml`, 空のファイル)
2. `loss.use_pos_weight: false` で AP がどう変わるか (`no_pos_weight.yaml`)
3. `data.pos_ratio: 0.01` (`pos1.yaml`)
4. `trainer.eval_train: true` (`eval_train.yaml`, 学習用データを推論モードで評価し直す)

## 確かめる ML-Found の機能

- [x] `Trainer` を継承し, `step_fn` を渡さずに `step` を上書きできる
- [x] `on_epoch_end` の戻り値が `train/auc`, `valid/auc` として記録され, 学習曲線 (`auc_curve.png`, `ap_curve.png`) も描かれる
- [x] `on_epoch_end` で返した指標 (`valid/ap`) を `monitor` にできる
- [x] `self.epoch` を継承したクラスから参照できる (`on_epoch_start` での確認が全てのエポックで通った)
- [x] `step` の戻り値に `batch_size` を入れると指標の重みに使われ, `train/batch_size` のような指標としては記録されない

## 期待する結果

- 検証用データの ROC-AUC が 0.90 以上, AP が 0.6 前後

## 予想される問題

- `pos_weight` は学習用データの正例の割合から決まるので, config の値だけでは書けず, train.py で計算して損失関数に渡す形になる → その通り. ML-Found の問題ではない
- 学習時の予測は Dropout が効いた状態なので `train/auc` は参考程度にしかならない → その通り (下の結果を参照). `trainer.eval_train: true` で学習用データを推論モードで評価し直せる (実験 4)

## 実行方法

事前にルートの README の「導入」を済ませ, `.venv` を有効にしておく.
`train.py` はどこから実行してもよく, 出力先は常に `tasks/03_imbalanced_cls/outputs/imbalanced/<日付>/<時刻>/` になる.

```bash
# リポジトリのルートから
python tasks/03_imbalanced_cls/train.py                                                            # 実験 1
python tasks/03_imbalanced_cls/train.py --experiment configs/experiment/examples/no_pos_weight.yaml  # 実験 2
python tasks/03_imbalanced_cls/train.py --experiment configs/experiment/examples/pos1.yaml           # 実験 3
python tasks/03_imbalanced_cls/train.py --experiment configs/experiment/examples/eval_train.yaml     # 実験 4
```

- `.venv` を有効にしない場合は `python` を `.venv/bin/python` に読み替える
- `--experiment` のパスは, 絶対パス, タスクのディレクトリから, 実行した場所から, の順に探す
- 1 回の実行で出力先が 1 つでき, `config.yaml`, `diff.yaml`, `train.log`, `history.csv`, 学習曲線 (`ap_curve.png`, `auc_curve.png`, `loss_curve.png`, `lr_curve.png`), `best.pt`, `last.pt`, `final_model.pt` (未コミットの変更があれば `git_diff.patch`) が入る
- 下の表の指標は `history.csv`, 正例の割合と `pos_weight` は `train.log` で確かめられる. 学習時間は `train.log` の最後の「学習時間」に出る

## 結果のメモ

PyTorch 2.15.0, scikit-learn 1.9.1, macOS (Apple Silicon), Python 3.13, cpu での値.
同じ環境なら同じ結果になる (流し直すと `history.csv` が一致する). OS やライブラリの版が違う場合は一致しないことがある.

| 実験 | 正例の割合 (学習用) | pos_weight | 止まったエポック | ベストのエポック | 検証用 ROC-AUC | 検証用 AP |
| --- | --- | --- | --- | --- | --- | --- |
| 1 差分なし | 5.49 % | 17.2 | 50 (最後まで) | 46 | 0.9468 | 0.8331 |
| 2 `no_pos_weight` | 5.49 % | なし | 50 (最後まで) | 50 | 0.9523 | 0.8423 |
| 3 `pos1` | 1.54 % | 64.0 | 23 | 13 | 0.7749 | 0.4527 |
| 4 `eval_train` | 5.49 % | 17.2 | 50 (最後まで) | 46 | 0.9468 | 0.8331 |

- 実験 1 は目標 (ROC-AUC 0.90 以上) を満たし, AP は予想 (0.6 前後) より高い 0.83 になる
- 実験 2 (重みなし) と実験 1 の AP の差は 0.009 で, 1 つのシードでは差があるとは言えない. ROC-AUC と AP は予測確率の並び順だけで決まるので, 十分に学習できれば `pos_weight` はロジットを一様にずらすだけで並び順を変えないと予想される. ただし学習の途中では勾配の重み付けが変わるので, 並び順にも影響しうる
- 実験 2 はエポック 50 でもまだ AP が上がっている. 実験 1 はエポック 30 以降ほぼ横ばい (0.819 から 0.833) になる
- 実験 3 (正例 1 %) は検証用の AP が 0.45 前後で頭打ちになって Early Stopping で止まる. 主な原因は, 上の「データ」に書いた雑音の正例が正例の 3 分の 1 程度を占めることで, 雑音の正例は特徴量では見分けられないので ROC-AUC にも 0.8 前後の上限がある. 学習用の ROC-AUC はベストのエポック 13 で 0.858 (検証用 0.775), 最後のエポック 23 で 0.902 (検証用 0.756) と差が開き, 雑音の正例を覚えていると考えられる
- `train/auc` が `valid/auc` より低いエポックが多い. 実験 2 では 50 エポックのうち 35 エポック, 実験 1 でもエポック 1 から 7 と 11 で低い (エポック 50 では実験 1 が 0.970 と 0.944, 実験 2 が 0.954 と 0.952). `train/auc` は Dropout が効いた状態の予測から求めるうえ, パラメータを更新しながら集めた予測なので, 推論モードで学習用データを評価した値とは異なる

### 学習用データの評価し直し (実験 4)

`trainer.eval_train: true` にすると, 各エポックの学習の後に学習用データを推論モードで評価し直し, `train_eval/` を付けて記録する. このタスクの `AUCTrainer` は書き換えずに済み, `on_epoch_start`, `step`, `on_epoch_end` は `phase` が `train_eval` でもそのまま AUC と AP を求める.

- 実験 4 の `train/`, `valid/`, `lr` の列は実験 1 と完全に一致し, 評価し直しは学習に影響しない
- `train/auc` が `valid/auc` より低いエポックは 50 のうち 8 (エポック 1 から 7 と 11) だが, `train_eval/auc` は全てのエポックで `valid/auc` 以上になる

| エポック | `train/auc` | `train_eval/auc` | `valid/auc` | `train/ap` | `train_eval/ap` | `valid/ap` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7135 | 0.8235 | 0.8028 | 0.2273 | 0.4419 | 0.3769 |
| 10 | 0.9368 | 0.9487 | 0.9337 | 0.7922 | 0.8286 | 0.7793 |
| 46 (ベスト) | 0.9701 | 0.9834 | 0.9468 | 0.8657 | 0.9140 | 0.8331 |
| 50 | 0.9702 | 0.9847 | 0.9436 | 0.8698 | 0.9131 | 0.8218 |

- エポック 1 の差が最も大きい. `train/` はパラメータを更新しながら集めた予測なので, 学習の始めの悪い予測を含む. Dropout が効いた状態の予測であることと合わせて, 学習用の値が低く出る
- 学習用データの推論 1 周分が増えるので, 学習時間は実験 1 より長くなる

### ML-Found について気付いたこと

[FEEDBACK.md](../../FEEDBACK.md) に転記した (ML-Found で解消したものは FEEDBACK.md から消し, 残っているものだけを載せる).

- `Trainer` を継承する書き方は README の説明通りで, `__init__`, `step`, `on_epoch_start`, `on_epoch_end` の上書きと `self.epoch` の参照に問題は無い
- `on_epoch_end` の戻り値に `step` の戻り値と同じ名前の指標があると, エラーで止まる (黙って上書きされない)
