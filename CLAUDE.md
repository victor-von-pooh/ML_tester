# CLAUDE.md

ML_tester で作業するときの方針をまとめる.

## このリポジトリの目的

[ML-Found](ML-Found/) (PyTorch の学習の基盤, submodule) を利用者の立場で使い, いろいろな種類のタスクで機能するかを確かめるテストフィールド.

- 各タスクの `train.py` を手で動かして結果を見る. pytest による自動判定はしない (例外は, 学習しないで済む安全の約束の確かめだけ. 今は 24 の守りのテスト `tasks/24_rl_cartpole/test_guards.py` を CI で動かす)
- 動かないことが予想されるタスク (LBFGS, GAN など) も実装し, どこでどう詰まったかを記録する. それが ML-Found への FB になる
- ML-Found 側の修正はこのリポジトリでは行わない. 見つけた問題は [FEEDBACK.md](FEEDBACK.md) に書く
- 「機械学習を全て GUI 上で行う」という全体の方針と, 各リポジトリの役割は [VISION.md](VISION.md) にまとめる. このリポジトリは動作確認の場で, 各システムの機能をここに集約しない (機能は各リポジトリに足す). 各リポジトリもこのリポジトリを前提にしない

## 構成

- `tasks/XX_name/`: 1 タスク 1 ディレクトリ. ML-Found のプロジェクトのディレクトリとして扱う
    - `configs/default/train.yaml`, `configs/experiment/train.yaml`: ML-Found の 2 段の config
    - `configs/experiment/examples/`: README の実験で使う差分. `--experiment` で切り替える
    - `model.py`: モデルのクラス (13, 17, 18 は代わりに `models/` の ML-ModelControl のモデルの定義)
    - `train.py`: データ, step, 学習の実行
    - `README.md`: 目的, データ, モデル, 確かめる機能, 期待する結果, 予想される問題, 実行方法, 結果のメモ
    - `outputs/`: ML-Found が作る (git の管理外)
    - 19 と 20 は学習しない (推論と意思決定のスクリプトだけ). 21 は `project/` が ML-Found のプロジェクト
- `data/`: 外部から取得したデータ (git の管理外). ユーザーに用意してもらうデータは各タスクの README に条件を書く
- `FEEDBACK.md`: ML-Found への FB のうち残っているものを「タスク, 症状, 回避策, 提案」の表で集約. ML-Found で解消したものは消す

## 環境

- 仮想環境は `.venv` (Python 3.13). 他の版の `python3` で動かさない
- 実行はリポジトリのルートから `.venv/bin/python tasks/XX_name/train.py`
- 商用利用できるライセンスのライブラリなら, 必要に応じて入れてよい (scikit-learn, torchvision, PyG, qiskit, matplotlib, transformers など). 入れたものはルートの README の「導入」に追記する

## コードの書き方

- [ML-Found/CODING_STYLE.md](ML-Found/CODING_STYLE.md) に従う (日本語の NumPy 形式の docstring, 下線 10 本, コメントは体言止め, `nn.functional` を使わない, 79 文字以内など)
- モデルはルートの README の「モデルの書き方」に合わせる (部品ごとのクラス, 引数を `self` に保存してから層を定義, など)
- 他の実装を参考にする場合も全部真似せず, エラーや誤りがあれば適切に直して使う
- このリポジトリのドキュメント (README, FEEDBACK.md, この CLAUDE.md) とコードのコメントでは, 「機械学習を全て GUI 上で行う」という目的のために開発しているリポジトリ (今は ML-Found, ML-Dashboard, ML-ModelControl, ML-CreateArchitecture, ML-EDA, ML-Fathom, ML-Inference, ML-Provide, ML-Quest, ML-Pathfinder, ML_tester) 以外に触れない. ローカルのディレクトリ構成やパスも書かない. 第三者が読んで分からない情報になるため

## タスクを進める流れ

1. タスクの README の方針に沿って, `configs/`, `model.py`, `train.py` を作る
2. README の「実行方法」の実験を全て実行する
3. 結果を README の「結果のメモ」に書き, 確かめた項目のチェックボックスを埋める. 数値は実際の `outputs/` の値だけを書く. 文書には今の状態だけを書き, 日付, 過去の版の commit, 直した経緯は書かない (経緯は git の履歴)
4. ML-Found について気付いたことを README と `FEEDBACK.md` に書く. ML-Found の挙動についての主張は, ソースコードで原因を確かめてから書く
5. **セルフレビュー**: 複数のエージェントを並行して動かし, おかしなところが無いかを確認する. 少なくとも次の 3 つの観点で行う
    - コードのレビュー: バグ, ML-Found の API の使い方, パスの扱い, CODING_STYLE.md への準拠
    - 記述の検証: README の結果と FEEDBACK.md の主張が `outputs/` と ML-Found のソースコードに合っているか
    - 新しい利用者としての再現: README の手順だけで, 別のディレクトリからも動かせるか
    - レビューを頼むときは, 自分の結論ではなく対象のファイルを渡す
6. レビューの指摘を直し, 直した内容と残した指摘をユーザーに報告する

## git

- 変更は意味のまとまりごとにブランチを切り, PR を通して `main` に入れる. `main` に直接プッシュしない (ML-Found, ML-Dashboard も同じ)
- PR を作るときに, 上の「タスクを進める流れ」の 5 と 6 のとおり, 複数のエージェントで並行してセルフレビューを行い, 指摘を同じブランチで直す
- CI が通ったことを確かめてからマージする. 自分の PR は GitHub の仕様で Approve できないので, 確かめた内容をレビューのコメントに残してからマージする
- 誰が行うか
    - 通常: コミット, プッシュ, PR の作成とマージはユーザーが行う. Claude はブランチ名, コミットメッセージ, PR のタイトルと説明を提案する
    - ユーザーが `[Commit] <日時>までなら...` の prefix を付けて指示した場合: その日時まで, Claude の判断でコミット, プッシュ, PR の作成とマージを行ってよい
        - これは「期限まで確認を待たずに自律的に進めてよい」という意味. 1 つの PR をマージしたら, ユーザーに確認せずに次の作業へ進む
        - ユーザーの判断が要ること (用意してもらうデータが無い, 方針を変える必要がある, など) に当たった場合は, その作業を飛ばして次に進む
        - 最後に, マージした PR, 直した指摘と残した指摘, 飛ばした作業をまとめてユーザーに報告する
        - 期限を過ぎたら通常の運用に戻る
- コミットメッセージと PR のタイトルは ML-Found に合わせて日本語の 1 行 (例: 「表形式の多クラス分類のタスクを実装し実験結果とFBを記録」)
