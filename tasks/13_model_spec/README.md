# 13 モデルの定義から組み立てたモデルの学習

## 目的

ML-ModelControl のモデルの定義 (YAML) から組み立てたモデルを, ML-Found で学習できることを確かめる.
モデルの作成 (ML-ModelControl, ML-CreateArchitecture) と学習 (ML-Found, ML-Dashboard) の 2 つの段階の交点を, ML-ModelControl の README の「ML-Found で学習する」の手順どおりに使う.

- モデルは `model.py` に書かず, `models/` の定義だけで決める (ML-CreateArchitecture で保存した定義と同じ形)
- 使う定義を config の `model.spec` で選び, experiment で切り替える
- 実験の出力先に定義を写し, 出力先の定義と重みだけで同じモデルを作り直せるかを確かめる

## ディレクトリ構成

```
13_model_spec/
├── configs/
│   ├── default/train.yaml          # 基準の config (model.spec: models/cnn.yaml)
│   └── experiment/
│       ├── train.yaml              # 普段使う差分 (空)
│       └── examples/
│           ├── wide.yaml           # チャンネル数を 32 にした定義に切り替える
│           ├── broken.yaml         # 誤りのある定義 (出力先を作る前に止まることの確認)
│           └── quick.yaml          # 2 エポックで終える (動作の確認用)
├── models/                         # ML-ModelControl のモデルの定義
│   ├── cnn.yaml                    # 残差のブロック (部分モデル) を 2 つ重ねた CNN
│   ├── cnn_wide.yaml               # cnn.yaml のチャンネル数を 32 にしたもの
│   └── broken.yaml                 # 最後の層の out_features を書き忘れたもの
├── train.py                        # データ, step, 定義からの組み立て, 学習, 作り直しの確認
└── outputs/digits_spec/<日付>/<時刻>/   # ML-Found が実験ごとに作る (git の管理外)
```

## データ

| 項目 | 内容 |
| --- | --- |
| データセット | 手書き数字 (`sklearn.datasets.load_digits`) |
| 取得方法 | scikit-learn に同梱, ダウンロード不要 |
| 規模 | 1797 枚, 8x8 の灰色の画像, 10 クラス |
| 分割 | 学習用 80 % と検証用 20 %, クラスの比率を保って分割 (`stratify`) |
| 前処理 | 画素 (0 から 16) を 16 で割って 0 から 1 にし, `(枚数, 1, 8, 8)` の形にする |

用意してもらう必要があるデータは無い. ML-ModelControl はルートの README の「導入」で入れる.

## モデル

`models/cnn.yaml` (ML-ModelControl の定義).

- `nn.Conv2d(1, 16, 3, padding=1)` と `nn.ReLU` の後に, 部分モデル `sub.block` を 2 つ重ね, `nn.MaxPool2d(2)`, `nn.Flatten`, `nn.Linear(256, 10)`
- 部分モデル `block` は `nn.Conv2d(16, 16, 3, padding=1)`, `nn.ReLU` と, 入力を足す残差のつなぎ (`op.Add`). 定義の `subgraphs` に写してあるので, 1 つのファイルだけで組み立てられる (2 つのブロックの重みは別)
- パラメータ数は 7,370 (`cnn_wide.yaml` は 23,946)

## train.py の流れ

1. `load_project_config` で config を作り, `model.spec` の定義から `build_model` で組み立てる. 定義に誤りがあれば `SpecError` の誤りの一覧を表示し, **実験の出力先を作る前に**終える
2. `infer` で出力の形を推定し, 最後の次元がクラスの数 (10) と合うかを確かめる
3. `start_experiment` で実験を始め, 使った定義を出力先に `model_spec.yaml` として写す
4. 学習する (`step` は交差エントロピー損失と正解率. 出力が 1 つの定義なので, モデルはテンソルを返す)
5. 出力先の `model_spec.yaml` と `final_model.pt` から組み立て直したモデルと, `export_python` で書き出したコード (`model_export.py`) のモデルの出力が, 学習の後のモデルと一致するかを表示する

## 確かめること

- [x] 定義から組み立てたモデルを, `Trainer` にそのまま渡して学習できる
- [x] 誤りのある定義では, 実験の出力先を作る前に理由を表示して止まる
- [x] experiment で `model.spec` を変えると, 違う定義で学習でき, `diff.yaml` に `model.spec` の差分が残る
- [x] 出力先の定義と重みだけで同じモデルを作り直せる (出力の差が 0)
- [x] 書き出したコード (ML-ModelControl 無しで動く) のモデルも同じ出力になる

## 実行方法

ルートのディレクトリで, `.venv` を有効にして実行する.

```bash
python tasks/13_model_spec/train.py                                                      # 1. cnn.yaml
python tasks/13_model_spec/train.py --experiment configs/experiment/examples/wide.yaml   # 2. cnn_wide.yaml
python tasks/13_model_spec/train.py --experiment configs/experiment/examples/broken.yaml # 3. 誤りのある定義
```

CPU で, 1 と 2 はそれぞれ 3 秒ほど.

## 期待する結果

- 1 と 2 は検証用データの正解率が 0.95 を超え, 作り直したモデルと書き出したコードのモデルの出力の差が 0
- 3 は「head: nn.Linear の必須の引数 out_features がありません」と表示して終了コード 1 で終わり, 出力先 (`outputs/digits_spec/...`) を作らない

## 結果のメモ

ML-Found `c1fde7e`, ML-ModelControl `5cfabe4`, PyTorch 2.15.0, CPU (Apple Silicon の Mac) で実行した.

| 実験 | 定義 | パラメータ数 | 学習したエポック | ベストのエポック | 検証用データの正解率 (ベスト) | 作り直したモデルとの差 | 書き出したコードとの差 | 所要時間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `cnn.yaml` | 7,370 | 16 (Early Stopping) | 11 | 0.9722 | 0 | 0 | 2.3 秒 |
| 2 | `cnn_wide.yaml` | 23,946 | 14 (Early Stopping) | 9 | 0.9778 | 0 | 0 | 2.7 秒 |
| 3 | `broken.yaml` | - | - | - | - | - | - | - |

- 1 と 2 の実験の出力先には, ML-Found のファイルに加えて, 写した `model_spec.yaml` と書き出した `model_export.py` が残る. 2 の `diff.yaml` には `model.spec` (`models/cnn.yaml` から `models/cnn_wide.yaml`) が残る
- 3 は出力先を作らずに終わった (`outputs/digits_spec/` の下は 1 と 2 の 2 つだけ)
- 定義を実験の出力先に写すのは学習のスクリプトの仕事で, ML-Found は定義を知らない. そのため ML-Dashboard の「この設定でもう一度起動」は, 起動した時点の `models/` の定義から組み立てる (前の実験の後に `models/cnn.yaml` を書き換えていれば, 違うモデルで学習する). 前の実験と同じモデルにしたい場合は, その実験の `model_spec.yaml` を `models/` に写してから起動する. 定義の置き場とプロジェクトの結び方は, 全体の方針 (ルートの VISION.md) で保留中の項目
