# ML-Found への FB

各タスクで ML-Found を使って見つかった, 書きにくい点や足りない機能をまとめる.

| タスク | 症状 | 回避策 | ML-Found への提案 |
| --- | --- | --- | --- |
| 01 | 正解率のように同じ値を取りやすい指標を `monitor` にすると, 最初にその値に達したエポックがベストのまま固定され, その後 `valid/loss` が改善しても反映されない. Early Stopping も同じ値が続くと止まるので, `valid/loss` が下がり続けている学習の途中で止まった | 検証用データを増やす, または `monitor` を `valid/loss` にする | 同じ値の場合に別の指標で比べる設定 (例: `trainer.monitor_tiebreak: valid/loss`) |
| 01 | Early Stopping で止まると, 進捗バーの最後の表示が 1 エポック少ない (エポック 39 で止まったのに `38/200`) | なし (表示だけの問題) | `break` の前に `pbar.update()` するなど, 止めたエポックも数える |
| 01 | `device: auto` では小さなモデルでも mps が選ばれ, cpu より 6 倍以上遅い | `experiment.device: cpu` を書く | README で, 小さなモデルでは cpu を明示した方が速いことを注意する (自動で判定するのは難しいので, 文書での対応で十分) |
| 01 | experiment の config のパスを `train.py` に固定で書くので, 複数の実験を切り替えにくい | `train.py` に `--experiment` の引数を自分で用意 | experiment のパスやコマンドラインからの上書き (`trainer.epochs=10` のような dotlist) を受け取る補助関数を ML-Found に用意する |
