# ML-Found への FB

各タスクで ML-Found を使って見つかった, 書きにくい点や足りない機能のうち, まだ残っているものをまとめる. ML-Found で解消したものは消した (経緯は git の履歴と各タスクの README の「結果のメモ」にある).

## 残っているもの

| タスク | 症状 | 回避策 | ML-Found への提案 |
| --- | --- | --- | --- |
| 04, 06 | チェックポイントの保存は `trainer.checkpoint.every_n_epochs`, `best_weights_only`, `last_with_best`, 記録を列ごとの数の配列で持つ形で軽くなったが, `best.pt` は改善のたびに保存する. 06 のように毎エポック改善するフルバッチ学習で, これらの設定を合わせたときの時間は確かめていない. 記録を `history.csv` に追記する形は ML-Found で見送り (再開の扱いが変わるため) | `every_n_epochs` と `best_weights_only: true` (または `save_best: false`) を合わせる | 06 で時間を測り, 足りなければ `best.pt` の保存の間隔を設定にする |
| 02, 04 | 自分で作って `Trainer` に渡した Scheduler は, `config.yaml` の `scheduler` が `name: null` のままになる (クラスの名前は `run.json` と `train.log` に残る). warmup 以外で複数の Scheduler を組み合わせる設定は, ML-Found で対応しないと決めた | warmup は `scheduler.warmup_steps` か `warmup_ratio` で config に書く. それ以外の組み合わせは自分で作り, 引数を自分の config の項目にする | 自分で渡した Scheduler の主な引数も出力先に残す |
| 11 | EMA の更新と保存は利用者が書く (`update_parameters`, `extra_state_dict`, `load_extra_state_dict` を上書きする. 検証に使うモデルは `eval_model` で差し替えられる) | このタスクの `DiffusionTrainer` | `trainer.ema_decay` のような設定で, 更新, 保存, 再開, 検証に使うモデルの切り替えまで行う |
| 13 | ML-Found はモデルの定義 (ML-ModelControl) を知らないので, 定義を実験の出力先に写すのは学習のスクリプトの仕事になり, ML-Dashboard の「この設定でもう一度起動」も起動した時点の `models/` の定義から組み立てる (前の実験の後に定義を書き換えると違うモデルになる) | 学習のスクリプトで定義を `model_spec.yaml` として出力先に写す. 同じモデルで起動し直す場合は, その `model_spec.yaml` を `models/` に写してから起動する | ML-Found だけの問題ではなく, 定義の置き場とプロジェクトの結び方 (VISION.md で保留中) と合わせて決める. 決まるまでは今の回避策で足りる |
| 22, 23 | 領域の見本 (`kind: mask`) と見本の値の決まり (`convention`. 点群の単位, 正規化で引いた中心) は ML-Found に入ったが, ML-Dashboard で凡例と元の座標を出すのは残り (ML-Found の TODO にある) | 22 は正解と予測を `kind: mask` (`classes` に名前), 23 は点群に `convention` (単位と元の座標に戻す `shift`) を添えて書いた. 画面では凡例と元の座標がまだ出ない | ML-Dashboard で領域の色分けの凡例と, 正規化の前の座標を出す |
| 23 | 見本の `convention` の推奨の `shift` は「元の座標 = 値 × scale + shift」の量で, ML-EDA の書き出しの `shift` (正規化した座標 = 元の座標 + shift) と同じ名前で向きが逆 | ML-EDA の `shift` の符号を反転して書く | 推奨の名前を ML-EDA とそろえる (向きを合わせるか, 別の名前にする) |

## うまく動いたこと

- **学習の再開**: モデル, Optimizer, Scheduler, 記録, Early Stopping, 乱数 (DataLoader の並べ替えを含む) が全て戻り, 途中で止めて再開した学習と通しの学習の結果が完全に一致した (07, 09, 11, 12). `persistent_workers=True` の DataLoader (09) は一致しない (ML-Found の README に書かれ, 警告が出る). `T_max: ${trainer.epochs}` のように再開で変わる Scheduler の引数 (11) は `trainer.resume_scheduler: config` で一致する
- **再現性**: 同じ環境で学習し直すと `history.csv` が完全に一致した (01, 03, 05, 06, 07, 08, 09, 11, 12). 別のディレクトリから実行しても, mps でも一致した. 12 は qiskit の乱数を `register_rng` で固定した場合
- **config の 2 段構え**: default に無い項目のエラー, クラスを変えたときの引数の切り替え, `diff.yaml` の差分の記録, `${trainer.epochs}` の参照が README の通りに動いた (01, 02)
- **継承のフック**: `step`, `on_epoch_start`, `on_epoch_end`, `update_parameters`, `extra_state_dict`, `load_extra_state_dict`, `eval_model` を上書きして, AUC (03), 複数ラベルの F1 (04), RMSE (05), KL annealing (09), GAN の書き分け (10), EMA (11) を書けた
- **外部のモデル**: Hugging Face の DistilBERT, qiskit の量子回路 (`TorchConnector`) を `nn.Module` としてそのまま載せられた
- **検証の間隔**: 23 で `trainer.valid_every_n_epochs: 3` にすると, 検証と `on_epoch_end` の重い評価 (Chamfer 距離) はエポック 3 の倍数と予定の最後のエポックだけになり, ベストモデルは検証したエポックから選ばれ, Early Stopping は検証の回数で数えた. 学習曲線は検証したエポックだけを点で描いた
- **出力先**: 同じ秒に始めた実験も連番 (`_1`) の別の出力先になり, 並行して動かしても上書きされなかった (06)
