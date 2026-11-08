# ML-Found への FB

各タスクで ML-Found を使って見つかった, 書きにくい点や足りない機能をまとめる.

| タスク | 症状 | 回避策 | ML-Found への提案 |
| --- | --- | --- | --- |
| 01 | 指標が最良値と同じ値になっても改善とみなされない (`training/callbacks.py` の `is_improved` が `>` と `<` で比べ, `Checkpoint` と `EarlyStopping` の両方が使う). 正解率のように同じ値を取りやすい指標を `monitor` にすると, ベストモデルは最初にその値に達したエポックのままになり, Early Stopping も最良値に並んだだけのエポックを「改善なし」と数えて, `valid/loss` が下がり続けている学習の途中で止めた | `monitor` を `valid/loss` にする. `early_stopping.min_delta` は負の値にしない限り同じ値を改善にできず, `Checkpoint` は `min_delta` を使わないので回避策にならない | 同じ値の場合に別の指標で比べる設定 (例: `trainer.monitor_tiebreak: valid/loss`) を用意し, `Checkpoint` と `EarlyStopping` の両方で使う. または同じ値を改善とみなすかどうかの設定 |
| 01 | Early Stopping で止まると, 進捗バーの最後の表示が 1 エポック少ない (エポック 39 で止まったのに `38/200`). `for epoch in pbar` の tqdm は次の要素を取り出すときに数えるので, `break` した最後のエポックが数えられない | なし (表示だけの問題) | `tqdm(total=len(epochs))` を普通のループと組み合わせ, 各エポックの最後 (止めるかどうかの判定の前) に `pbar.update(1)` する. `break` の前に `pbar.update()` を足すだけでは, ループの終了時に tqdm が自分の数で上書きするので直らない |
| 01 | `device: auto` では mps が選ばれるが, このタスク (パラメータ 131 個, 150 サンプル) では cpu の方が 6 倍以上速かった | `experiment.device: cpu` を書く | README で, 小さなモデルでは cpu を明示した方が速い場合があることを注意する (自動で判定するのは難しいので, 文書での対応で十分) |
| 01 | experiment の config のパスを `train.py` に固定で書くので, 複数の実験を切り替えにくい | `train.py` に `--experiment` の引数を自分で用意 | experiment のパスやコマンドラインからの上書き (`trainer.epochs=10` のような dotlist) を受け取る補助関数を ML-Found に用意する |
| 01 | `train.log` の「未コミットの変更: N ファイル」が `git status --porcelain` の行数なので, 未追跡のディレクトリが 1 行にまとめられ, `git_diff.patch` のファイル数より少なく出る (3 ファイルと出たが patch には 7 ファイル) | なし (`git_diff.patch` 自体は正しい) | `git status --porcelain --untracked-files=all` で数える (`experiment/env_info.py` の `get_git_info`) |
