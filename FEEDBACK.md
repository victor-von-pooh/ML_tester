# ML-Found への FB

各タスクで ML-Found を使って見つかった, 書きにくい点や足りない機能をまとめる.

## まとめ

12 のタスク (ML-Found `3aca743`) を通して見つかったことを, 優先度の高い順にまとめる. 詳細と根拠は下の表と各タスクの README にある.

### 優先度: 高

1. **ベストモデルの判定と学習後の巻き戻しを止められない** (09, 10). GAN では生成器の損失が最小のエポック 1 がベストになり, `final_model.pt` が崩れた画像を出すモデルになった. β-VAE でも学習の途中のモデルが残った. `trainer.restore_best: false` のような設定が要る
2. **Optimizer が 1 つで更新手順が固定** (06, 10). LBFGS は closure を渡せずに最初の更新で失敗し, GAN は 2 つ目の Optimizer の作成, 更新, 保存, 再開の全てを自分で書く必要があった. ML-Found の TODO の項目 2 (closure への対応) に加え, 複数の Optimizer と自動の更新を止める設定 (`manual_optimization`) があると, この 2 つのタスクはどちらも素直に書ける
3. **チェックポイントの保存が重い** (04, 06). 1 エポックが短いフルバッチ学習では保存が学習時間の 8 割以上を占め (`last.pt` を毎エポック, `best.pt` を改善のたびに, 記録の全てを含めて保存するため), 事前学習済みモデルでは `best.pt` と `last.pt` がそれぞれ 1 GB を超えた. 保存の間隔と, `last.pt` にベストモデルの重みを入れない設定が要る
4. **検証時の扱いを変えられない** (06, 08, 11). 検証では勾配が止まるので PINN のように入力の微分を使う損失が失敗し, 損失に乱数を含む拡散モデルでは検証用の損失が揺れてベストモデルの判定が不安定になった. `valid_grad` や `valid_seed` のような設定が要る. 逆に `on_epoch_end` は検証の後でも勾配が有効なまま呼ばれる
5. **データの量で決まる Scheduler の引数を config で書けず, 自分で作った Scheduler は記録に残らない** (02, 04). `OneCycleLR` や warmup 付きの Scheduler は `load_config` と `start_experiment` の間で config を書き換えるか, 自分で作って `Trainer` に渡す必要がある. 後者では `config.yaml` の `scheduler` が `name: null` のままになり, ML-Found の記録はクラス名だけになる
6. **最良値と同じ値を改善とみなさない** (01). 正解率のように同じ値を取りやすい指標を監視すると, `valid/loss` が下がり続けている途中で Early Stopping が止めた

### 優先度: 中

7. **指標の平均の扱い** (04, 05, 09). `step` の戻り値はサンプル数で重み付けして平均されるので, RMSE のように平均してはいけない指標は小さめに出る. `BatchEncoding` のような `dict` の子クラスでない Mapping のバッチはサンプル数が 1 と推定される. 記録だけしたい値 (スケジュールの値など) も平均されて学習曲線が描かれる
8. **失敗したときの後始末** (05, 06, 07). Scheduler の組み合わせの誤り, closure の不足, 存在しない `resume_from` は, 出力先を作った後に失敗し, 学習結果の無い出力先が残る. エラーの内容は `train.log` に書かれない
9. **再開で通しの学習と一致しない場合がある** (09, 11). `persistent_workers=True` の DataLoader はワーカーの乱数の種を引く時点がずれ, Scheduler の引数は再開元の値が使われて config の変更が効かない. どちらも黙って起きる
10. **ML-Found の外の状態** (10, 11, 12). 2 つ目の Optimizer, EMA の重み, 外部ライブラリ (qiskit) の乱数は, チェックポイントと乱数の固定の対象外. `resume` が読み込んだ状態を外に出さないので, 継承して足すにもパスの解決を写す必要があった
11. **進捗の見え方** (01, 04). 進捗バーがエポック単位なので, 1 エポックが 17.5 分かかる学習では中の進み具合が分からない. Early Stopping で止まると最後の表示が 1 エポック少ない. ML-Found の TODO の項目 3 (進捗を渡すコールバック) を作るときに合わせて見直すと良い

### 優先度: 低

12. ドキュメントで補えるもの: 継承したクラスの `__init__` の例 (03), `on_epoch_end` で指標以外を保存する例 (08), `step_fn` にエポックが渡らないこと (09), 小さなモデルでは `device: cpu` を明示した方が速いこと (01, 12), `final_model.pt` を読み込む例の `map_location` (02)
13. その他: コマンドラインから experiment の config や `resume_from` を渡す補助 (01, 07), 学習率をエポックの開始時点しか記録しない (02), 勾配のノルムを記録しない (02), 学習用データを推論モードで評価し直せない (03), `on_epoch_end` と `step` の同じ名前の指標の上書き (03), 未コミットの変更の数え方 (01), モデルの構造のログ (12)

### うまく動いたこと

- **学習の再開**: モデル, Optimizer, Scheduler, 記録, Early Stopping, 乱数 (DataLoader の並べ替えを含む) が全て戻り, 途中で止めて再開した学習と通しの学習の結果が完全に一致した (07, 09, 11, 12). ただし `persistent_workers=True` の DataLoader (09) と, `T_max: ${trainer.epochs}` のように再開で変わる Scheduler の引数 (11) では一致しない
- **再現性**: 同じ環境で学習し直すと `history.csv` が完全に一致した (01, 03, 05, 06, 07, 08, 09, 11, 12). 別のディレクトリから実行しても一致した (03, 04, 05, 06, 07, 08, 09, 10, 12, 11 は短い動作確認の設定). mps でも一致した (01, 08, 09, 10, 11 は短い動作確認の設定). 12 は qiskit の乱数を自分で固定した場合
- **config の 2 段構え**: default に無い項目のエラー, クラスを変えたときの引数の切り替え, `diff.yaml` の差分の記録, `${trainer.epochs}` の参照が README の通りに動いた (01, 02)
- **継承のフック**: `step`, `on_epoch_start`, `on_epoch_end`, `update_parameters`, `state_dict`, `resume` を上書きして, AUC (03), 複数ラベルの F1 (04), RMSE (05), closure による更新 (06), KL annealing (09), GAN の 3 通りの書き分け (10), EMA (11) を書けた
- **外部のモデル**: Hugging Face の DistilBERT, qiskit の量子回路 (`TorchConnector`) を `nn.Module` としてそのまま載せられた
- **出力先**: 同じ秒に始めた実験も連番 (`_1`) の別の出力先になり, 並行して動かしても上書きされなかった (06 のレビューで同じ秒に始めた 2 つの実験が別の出力先になったことを確認)

## ML-Found での対応状況

ML-Found の PR #1 (`main` の `8a6a69c` にマージ済み) で, 次の項目に対応した. 各タスクの README の結果は, 更新前の `3aca743` で得たもの. その後 submodule は, 実験の状態と進み具合を書き出す PR #2 と, Optimizer と Scheduler の引数の仕様を返し, 引数の名前の誤りと必須の引数の不足を出力先を作る前に確かめる PR #4, 引数の種類と Scheduler の範囲を整えた PR #5 と #6, 引数の値の範囲を出力先を作る前に確かめる PR #7, step ごとの学習率を記録する PR #8, `OneCycleLR` の総ステップ数を学習用データから補う PR #9, チェックポイントを軽くする設定を加えた PR #10, エポックの途中で止める PR #11, 文書と TODO を整えた PR #12 から #15, `Trainer` の作成のエラーを記録する PR #16, 下の流し直しで見つけた不具合を直した PR #17 と #18 を含む `e2df46e` に更新した.

| まとめの項目 | 対応 | 内容 |
| --- | --- | --- |
| 1 ベストモデルへの巻き戻し | 対応 | `trainer.restore_best: false` で最後のエポックの重みを残す. ベストモデルの判定 (`best.pt`) 自体を止める設定は無い (ML-Found の TODO に記載) |
| 2 Optimizer が 1 つで更新手順が固定 | 一部対応 | closure が必要な Optimizer (LBFGS) を自動で見分けて対応. `trainer.manual_optimization` で `step` の中で更新できる. 複数の Optimizer を config で扱う仕組みは未対応 (ML-Found の TODO に記載) |
| 3 チェックポイントの保存が重い | 一部対応 | `trainer.checkpoint.every_n_epochs` で `last.pt` の間隔を空けられる. PR #10 で, `best.pt` がベストモデルの重みを重ねて持たなくなり, `trainer.checkpoint.best_weights_only` で `best.pt` を重みだけに, `last_with_best: false` で `last.pt` からベストモデルの重みを外せるようになった (04 の `hf_smoke.yaml` で, 以前は両方 1071 MB だったものが, 既定で `best.pt` 804 MB (`last.pt` は 1071 MB のまま), 両方の設定で `best.pt` 268 MB と `last.pt` 804 MB. 重みだけの `best.pt` は再開に使えないので, 再開は `last.pt` から). `best.pt` を改善のたびに保存すること自体は変わらないので, 06 のように毎エポック改善するフルバッチ学習では `every_n_epochs` と `best_weights_only: true` (または `save_best: false`) を合わせる必要がある (時間は未確認). チェックポイントが記録の全てを含むこと (06 の改善案の `history.csv` への追記) は未対応 (ML-Found の TODO に記載) |
| 4 検証時の扱い | 対応 | `trainer.valid_grad` と `trainer.valid_seed`. `on_epoch_end` の勾配は README に注意を記載 |
| 5 データの量で決まる Scheduler の引数 | 一部対応 | `load_config` と `start_experiment` の間で書き込む方法を README に記載. 自分で渡した Scheduler の状態を `train.log` に記録. PR #9 で, `OneCycleLR` の総ステップ数を書かなければ `fit` で学習用データのバッチ数から補い, `config.yaml` と `train.log` に残すようになった (02 の `train.py` の書き込みはやめた). 総ステップ数が `trainer.epochs` とバッチ数から求めた予定と合わない場合 (再開で `trainer.epochs` を変えた場合など) は学習の前にエラーか警告. `OneCycleLR` 以外 (warmup 付きの Scheduler, step ごとに使う `CosineAnnealingLR` の `T_max` など) は補わず (ML-Found の TODO に記載), 自分で作った Scheduler は `config.yaml` に残らないまま |
| 6 最良値と同じ値を改善とみなさない | 対応 | `trainer.monitor_tiebreak` で, 同じ値の場合に別の指標 (例: `valid/loss`) で比べる. ベストモデルの判定と Early Stopping の両方に使う |
| 7 指標の平均の扱い | 一部対応 | `dict` 以外の Mapping のバッチのサンプル数を推定. 平均してよい指標の注意を README に記載. 記録だけする値と, サンプル数を推定できずに 1 とした場合の警告は ML-Found の TODO に記載 |
| 8 失敗したときの後始末 | 一部対応 | Optimizer と Scheduler のクラス名と組み合わせ, 学習の設定の組み合わせ, 存在しない `resume_from` を出力先を作る前に確認. `fit` の中のエラーを `train.log` に記録. PR #2 で, `fit` の中で失敗した実験は `run.json` の状態が `failed` になり, エラーの内容も残るようになった (05 の「`FAILED` のような印」). PR #4 で, Optimizer と Scheduler の引数の名前の誤りと必須の引数の不足も出力先を作る前に確かめるようになった. PR #7 で, Optimizer と Scheduler の引数の値が登録簿の範囲の外 (負の学習率など) の場合も出力先を作る前にエラーになった. 配列の値 (Adam の `betas` など) と自作のクラスの値は確かめないので, 誤りは `Trainer` の作成でエラーになり出力先が残る. `Trainer` の作成で起きたエラーは `train.log` に書かれない (どちらも ML-Found の TODO に記載) |
| 9 再開で通しの学習と一致しない場合 | 警告で対応 | Scheduler の引数の食い違いと `persistent_workers` の DataLoader を再開時に警告. config の値で上書きするかを選べる設定は ML-Found の TODO に記載 |
| 10 ML-Found の外の状態 | 一部対応 | `extra_state_dict` と `load_extra_state_dict` で継承したクラスがチェックポイントに状態を加えられる. 勾配クリッピングは `Trainer` の Optimizer が更新するパラメータだけに掛けるようにした (10). 外部ライブラリの乱数と, 検証に使うモデルを差し替えるフック (11 の EMA) は未対応 (ML-Found の TODO に記載) |
| 11 進捗の見え方 | 対応 | Early Stopping で止めたエポックも進捗バーで数える. `trainer.progress_bar_batches` でエポックの中のバッチ単位の進捗バーも出せる. PR #2 で, 学習中の進み具合を `progress.json` に書き出し, 外から読めるようにした (ML-Dashboard で表示). PR #11 で, 実験のディレクトリの `STOP` に `now` と書くと, 1 エポックが長い学習 (04 の 17.5 分など) でもエポックの途中で止め, 途中のエポックを捨てて最後に終えたエポックの状態に戻すようになった (ML-Dashboard の「エポックの途中で止める」). Python のコードからエポックごとの指標を受け取るコールバックは ML-Found の TODO に記載 |
| 12 ドキュメント | 対応 | README に継承の `__init__` の例, `on_epoch_end` の使い方, cpu の明示, `final_model.pt` (cpu で保存するよう変更) などを記載 |
| 13 その他 | 一部対応 | 勾配のノルムの記録 (`trainer.log_grad_norm`), `on_epoch_end` と `step` の同じ名前の指標をエラーに, 未コミットの変更をファイルごとに数える. モデルの構造のログは README で `extra_repr` を勧め, 残す仕組みは TODO に記載. PR #8 で, `scheduler.interval: step` の場合に更新ごとの学習率を `lr_steps.csv` と `lr_steps.png` に残すようになった. 他 (コマンドラインからの上書き, 学習用データの推論モードでの評価) は ML-Found の TODO に記載 |

PR #1 の途中の内容 (`monitor_tiebreak` と `progress_bar_batches` を加える前) で, このリポジトリのタスクを流し, 次を確かめた. 01 (cpu), 03, 05 (Lion), 06 (`ClosureTrainer`), 07 (再開を含む), 12 (古典の MLP) の `history.csv` が `3aca743` の結果と完全に一致した (回帰が無い). 10 の書き方 B と C と, 04 の段階 2 の動作確認の設定はエラーなく動いた. 06 の LBFGS は回避策なしで学習でき, 結果は回避策の `ClosureTrainer` と完全に一致した. 06 の PINN は `valid_grad: true` で回避策なしで学習でき, 結果は回避策の `torch.enable_grad()` と完全に一致した. 05 の ReduceLROnPlateau の設定の誤りは, 出力先を作る前にエラーになった.

PR #1 の最終的な内容で, 01 (cpu) に `trainer.monitor_tiebreak: valid/loss` を加えて流すと, 正解率が同じエポックを検証用データの損失で比べるようになり, Early Stopping のエポックは 39 から 89 に, ベストモデルのエポックは 69 (正解率 1.0000) になった.

submodule を `8a6a69c` に更新した後に, 全タスクの全ての config (`configs/experiment/` の下, 04 の DistilBERT の本番の設定を除く) を 2 エポックだけ流し, エラーと警告が無いことを確かめた. エラーになったのは, エラーになることを確かめるための 3 つ (05 の `plateau_max` と `plateau_step`, 06 の `pinn`) だけで, どれも各タスクの README に書いた通りのエラーだった. 05 の 2 つは, 学習を始める前の確認で止まるようになった.

`f0c26ab` に更新した後に, 全タスクの全ての config を新しい引数の確認に通し, 結果が更新前と同じ (新たに拒否される config は無い) ことを確かめた. 02 の OneCycleLR の `steps_per_epoch` は, `train.py` が `start_experiment` の前に書き込むので必須の引数の確認を通る. (PR #9 以降は `OneCycleLR` の総ステップ数が必須の引数の確認の対象外になり, `train.py` も書き込まない)

## 最新の ML-Found での再実行 (2026-11-13)

全てのタスクの全ての実験を, その後の ML-Found (`24f18c2` から `e2df46e`) で流し直した. 乱数を固定しない実験 (12 の実験 7) と mps の計算の非決定性による小さな差 (04 の段階 2) を除き, `history.csv` は `3aca743` の結果とバイト単位で一致した. 対応した項目 (06 の LBFGS が `Trainer` のままで学習できる, エラーで出力先が残らない, など) も実際の学習で確かめた. 各タスクの README の「最新の ML-Found での再実行」に詳細を書いた.

流し直しで, ML-Found の不具合が 2 つ見つかり, どちらも直した.

| タスク | 症状 | ML-Found の修正 |
| --- | --- | --- |
| 10 | mps で `load_checkpoint` (と再開) が `TypeError: Cannot convert a MPS Tensor to float64` で失敗した. PR #8 からチェックポイントに更新ごとの学習率の記録 (float64) を入れていたため | PR #17: cpu に読み込んでから, 移せるテンソルだけをデバイスに移す |
| 01, 03, 06 | 1 エポックが短い学習が遅くなった (06 の PINN: 3.2 秒 → 12.8 秒). PR #2 からエポックごとに `run.json` と `progress.json` をディスクまで書き出していたため | PR #18: 1 秒以上の間隔で書き, ディスクまで書き出すのは開始と終了だけ (3.0 秒に戻った) |

## 一覧

症状と回避策は ML-Found `3aca743` で確かめたもの. 今の対応状況は上の「ML-Found での対応状況」を参照.

| タスク | 症状 | 回避策 | ML-Found への提案 |
| --- | --- | --- | --- |
| 01 | 指標が最良値と同じ値になっても改善とみなされない (`training/callbacks.py` の `is_improved` が `>` と `<` で比べ, `Checkpoint` と `EarlyStopping` の両方が使う). 正解率のように同じ値を取りやすい指標を `monitor` にすると, ベストモデルは最初にその値に達したエポックのままになり, Early Stopping も最良値に並んだだけのエポックを「改善なし」と数えて, `valid/loss` が下がり続けている学習の途中で止めた | `monitor` を `valid/loss` にする. `early_stopping.min_delta` は負の値にしない限り同じ値を改善にできず, `Checkpoint` は `min_delta` を使わないので回避策にならない | 同じ値の場合に別の指標で比べる設定 (例: `trainer.monitor_tiebreak: valid/loss`) を用意し, `Checkpoint` と `EarlyStopping` の両方で使う. または同じ値を改善とみなすかどうかの設定 |
| 01 | Early Stopping で止まると, 進捗バーの最後の表示が 1 エポック少ない (エポック 39 で止まったのに `38/200`). `for epoch in pbar` の tqdm は次の要素を取り出すときに数えるので, `break` した最後のエポックが数えられない | なし (表示だけの問題) | `tqdm(total=len(epochs))` を普通のループと組み合わせ, 各エポックの最後 (止めるかどうかの判定の前) に `pbar.update(1)` する. `break` の前に `pbar.update()` を足すだけでは, ループの終了時に tqdm が自分の数で上書きするので直らない |
| 01 | `device: auto` では mps が選ばれるが, このタスク (パラメータ 131 個, 150 サンプル) では cpu の方が 6 倍以上速かった | `experiment.device: cpu` を書く | README で, 小さなモデルでは cpu を明示した方が速い場合があることを注意する (自動で判定するのは難しいので, 文書での対応で十分) |
| 01 | experiment の config のパスを `train.py` に固定で書くので, 複数の実験を切り替えにくい | `train.py` に `--experiment` の引数を自分で用意 | experiment のパスやコマンドラインからの上書き (`trainer.epochs=10` のような dotlist) を受け取る補助関数を ML-Found に用意する |
| 01 | `train.log` の「未コミットの変更: N ファイル」が `git status --porcelain` の行数なので, 未追跡のディレクトリが 1 行にまとめられ, `git_diff.patch` のファイル数より少なく出る (3 ファイルと出たが patch には 7 ファイル) | なし (`git_diff.patch` 自体は正しい) | `git status --porcelain --untracked-files=all` で数える (`experiment/env_info.py` の `get_git_info`) |
| 02 | データの量で決まる Scheduler の引数 (`OneCycleLR` の `steps_per_epoch` など) を config から渡す仕組みが無い (Scheduler を自分で作って `Trainer(..., scheduler=...)` に渡せるが, `config.yaml` に残らない). `load_config` と `start_experiment` の間で `cfg` を書き換えれば動き `config.yaml` にも残るが, この順番は利用者が知っている必要があり, `start_experiment` の後に書き換えると `config.yaml` と実際の設定がずれる. 書き込んだ値は `diff.yaml` には出ない | `train.py` で `start_experiment` の前に DataLoader を作り, `cfg.scheduler.params.steps_per_epoch = len(train_loader)` と書き込む | README にこの書き方を載せる. または `Trainer.fit` で `train_loader` の長さが分かる時点で Scheduler を作り, `steps_per_epoch` を自動で補う. ただし `fit` は `len()` を持たない Iterable も受け取り, `resume` も `fit` の中で Scheduler の状態を戻すので, 長さが無い場合の扱いと, 復元より前に作る順番に注意が要る |
| 02 | `lr` はエポックの開始時点の値だけが記録されるので, `scheduler.interval: step` の場合はエポックの中の学習率の変化が `lr_curve.png` に出ず, 最後のステップの後の学習率も残らない. 04 の warmup 付きの Scheduler では, エポック 1 の `lr` が 0.0 と記録された | `Trainer` を継承し, `on_epoch_end` で `optimizer.param_groups[0]["lr"]` を返せばエポックの終了時点の学習率は残せる. ステップごとの値は自分で記録する必要がある | `interval: step` の場合はステップごとの学習率を別のファイル (例: `lr_steps.csv`) に記録して描く, またはエポックの終了時点の学習率も記録する |
| 02 | 勾配のノルムが記録されないので, 勾配クリッピングが効いているか (どのくらいの頻度で切られているか) を確かめられない. `nn.utils.clip_grad_norm_` はクリップ前のノルムを返すが, `Trainer.update_parameters` は捨てている | `Trainer` を継承し, `update_parameters` を上書きしてノルムを貯め, `on_epoch_end` でエポック平均を返す (このタスクでは未実装) | クリップ前の勾配のノルムのエポック平均を `train/grad_norm` として記録する (設定で切り替え) |
| 02 | `final_model.pt` を学習したデバイスのまま保存するので (mps で学習すると `mps:0` のテンソル), ML-Found の README の読み込み例 `torch.load("outputs/.../final_model.pt")` は mps や cuda の無い環境ではそのままでは動かない | `torch.load(..., map_location="cpu")` で読み込む | 保存する前に cpu に移す (`best.pt` などの重みも同じ), または README の例に `map_location="cpu"` を加える |
| 03 | ML-Found の README に, `Trainer` を継承したクラスの `__init__` の例が無い. `step_fn` を使わない場合に親クラスへ何を渡すか (`None` を位置で渡すか, `optimizer=` などをキーワードで渡すか) を利用者が調べる必要がある | `super().__init__(model, experiment, None, optimizer, scheduler)` またはキーワード引数で呼ぶ | 小さな問題. README に継承時の `__init__` の例を載せる |
| 03 | `on_epoch_end` の戻り値と `step` の戻り値に同じ名前の指標があると, `on_epoch_end` の値で黙って上書きされる (`training/trainer.py` の `run_epoch`) | 名前を変える | 同じ名前があればエラーにするか, 警告を出す |
| 03 | `train/` の指標は学習中 (Dropout が効いた状態, パラメータ更新の途中) の予測から求めるので, エポック全体で求める AUC のような指標は `valid/` と比べにくい | `on_epoch_end` の `phase == "train"` のときに, 並べ替えない学習用の loader を `model.eval()` と `torch.no_grad()` で回し, `eval_auc` のような別名で返す (`run_epoch` は次の検証の前に推論モードへ戻すので安全. このタスクでは未実装) | 学習用データを推論モードで評価し直す設定 (例: `trainer.eval_train: true`) を用意する |
| 04 | Hugging Face のトークナイザの出力 (`BatchEncoding`) は `dict` の子クラスではない (`UserDict`) ので, `get_batch_size` がサンプル数を見つけられず 1 として扱う. `step` で返す指標がサンプル数でなくバッチ数で平均され, 小さなバッチがサンプル数の割に重く効く (`move_to_device` は `BatchEncoding` が `.to` を持つので動く) | `step` の戻り値に `batch_size` を入れる | `find_batch_size` で `collections.abc.Mapping` を辞書として扱う. 見つからずに 1 とした場合は一度だけ警告する |
| 04 | warmup 付きの Scheduler のように総ステップ数がバッチ数で決まる Scheduler を自分で作って `Trainer` に渡すと, `config.yaml` の `scheduler` は `name: null` のままで, ML-Found が残す記録は `train.log` の `Scheduler: LambdaLR` だけになる | Scheduler の引数を自分の config の項目にする (このタスクでは `warmup_ratio`), または自分で `train.log` に書く | 02 の「データの量で決まる Scheduler の引数」と合わせて, `Trainer` に渡した Optimizer と Scheduler の設定 (クラス名と主な引数) もログと出力先に記録する |
| 04 | 事前学習済みモデル (DistilBERT, 6,700 万パラメータ) では `best.pt` と `last.pt` がそれぞれ 1.07 GB になる. `last.pt` にもベストモデルの重みを入れるので, 2 つで 2 GB を超え, 毎エポック書き込む | `trainer.checkpoint.save_best: false` にする (再開には `last.pt` だけで足りる) | ベストモデルの重みを `last.pt` に入れるかどうかの設定. 大きなモデル向けに `best.pt` を重みだけにする設定 |
| 04 | 進捗バーがエポック単位なので, 1 エポックが 17.5 分かかる事前学習済みモデルの微調整では, エポックの中の進み具合が全く分からない (02 で予想した問題が実際に起きた) | なし | バッチ単位の進捗バー (エポックの中の内側のバー) を設定で出せるようにする |
| 05 | ReduceLROnPlateau の `mode` や `interval` の食い違いは `Trainer` を作るとき (`start_experiment` の後) に確かめられるので, エラーで止まっても `config.yaml`, `diff.yaml`, `train.log` (と `git_diff.patch`) だけを含む出力先が残る (01 の「default に無い項目」のエラーは `load_config` で起きるので残らない) | 残った出力先を手で消す | config だけで確かめられる部分 (`scheduler.params.mode` と `trainer.mode` の一致, `interval` と Scheduler のクラスの組み合わせ, クラス名の解決など) を, `load_config` か `start_experiment` の出力先を作る前に確かめる |
| 05 | エラーで止まった場合, `train.log` にエラーの内容が書かれない. 最後の行は途中の記録のままなので, 後から出力先を見ても失敗した実験だと分かりにくい | `train.py` で `start_experiment` の後の処理を `try` で囲み, `experiment.logger.exception(...)` で書いてから投げ直す (`Trainer` の作成で失敗する場合もあるので, `fit` だけを囲んでも足りない) | `Trainer.fit` (と `Trainer` の作成) で例外を受けたらロガーに書いてから投げ直す. 出力先に `FAILED` のような印を残す |
| 05 | `step` で RMSE を返すと, バッチごとの値をサンプル数で重み付けして平均するので, エポック全体の RMSE 以下になる (このタスクの検証用データではベストのエポックで約 0.003, 最大で 0.005) | `Trainer` を継承し, 二乗誤差の合計を貯めて `on_epoch_end` で求める | README の `step` の説明に, 平均してよい指標 (MAE, 正解率など) とそうでない指標 (RMSE, AUC, F1 など) の注意を書く. または `step` から合計と件数を返せる仕組みを用意する |
| 06 | closure を渡す更新手順に対応していないので, LBFGS は最初の更新で `TypeError: LBFGS.step() missing 1 required positional argument: 'closure'` になる. torch のエラーがそのまま出るので, ML-Found が未対応だとは分からず, 出力先も作られた後 | `Trainer` を継承し, `step` でバッチを `self` に保存して, `update_parameters` で closure を作って `optimizer.step(closure)` を呼ぶ (このタスクの `ClosureTrainer`, Adam でも同じ結果になることを確認) | TODO の項目 2 の通り, `update_parameters` を closure に対応させる. 対応するまでは, closure が必要な Optimizer を `build_optimizer` で検出して分かりやすいエラーにする |
| 06 | `update_parameters` は損失しか受け取らないので, 損失を計算し直す closure を作るにはバッチを別の経路 (`self`) で渡す必要がある. 継承して上書きすると, 元の勾配クリッピングと `interval: step` の Scheduler の処理も自分で写す必要がある | `step` でバッチを保存し, クリッピングと Scheduler の処理を写す (このタスクの `ClosureTrainer`). 1 回の更新ごとに順伝播が 1 回余分に増える | `update_parameters(loss, batch)` のようにバッチも渡す. 勾配の計算と `optimizer.step` の呼び方を別のメソッドに分け, クリッピングと Scheduler は共通の処理として残す |
| 06 | `last.pt` を毎エポック, `best.pt` を改善したエポックごとに保存し, どちらにも記録の全て, 乱数の状態, ベストモデルの重みが入るので, 1 エポックが短いフルバッチ学習では保存が学習時間の大部分を占める (5000 エポックで 44.9 秒, 保存をやめると 6.1 秒). 記録が増えるほど 1 回の保存も重くなる | `trainer.checkpoint.save_last: false` と `save_best: false` で止める (記録は同じで 6.1 秒になることを確認. 再開はできなくなる) | `trainer.checkpoint.every_n_epochs` のような保存の間隔の設定. 記録は `history.csv` に追記する形にして, チェックポイントに毎回全てを入れない |
| 06 | 検証時は `run_epoch` が `step` を `torch.set_grad_enabled(False)` で囲むので, PINN のように損失の中で入力に対する微分 (`torch.autograd.grad`) を使う学習は, エポック 1 の検証で `RuntimeError: element 0 of tensors does not require grad` になる | `step` の中で順伝播と微分を `torch.enable_grad()` で囲む (このタスクの `train_pinn.py`, ベストのエポックで厳密解との最大誤差 2.5e-4 まで学習できた) | 検証時に勾配の計算を止めるかどうかの設定 (例: `trainer.valid_grad: true`), または README に `torch.enable_grad()` の回避策を書く |
| 07 | `trainer.resume_from` のパスが存在しなくても, 出力先を作った後の `fit` で初めて読み込むので, torch の `FileNotFoundError` で止まって学習結果の無い出力先が残る. 相対パスはプロジェクトのディレクトリから辿るので, リポジトリのルートからのパスを書くとパスが二重になる | プロジェクトのディレクトリからのパスか, 絶対パスを書く | `load_config` か `start_experiment` で `resume_from` の存在を確かめ, 解決したパスを含む分かりやすいエラーにする |
| 08 | `on_epoch_end` は勾配の計算を止める範囲 (`torch.set_grad_enabled(train)`) の外で呼ばれるので, 検証の後でも勾配の計算が有効になっている. 中で推論すると `torch.no_grad()` を書かない限り計算グラフが作られる (このタスクで `torch.is_grad_enabled()` が `True` だった) | `on_epoch_end` の中で `torch.no_grad()` を書く | README の `on_epoch_end` の説明に書く. または検証の `on_epoch_end` を `torch.no_grad()` の中で呼ぶ (学習と検証で `on_epoch_end` の中の勾配の扱いが変わることも README に書く) |
| 08 | 学習中の画像の保存のように, 指標以外の出力を残すフックが無いので, 指標を返すための `on_epoch_end` を流用することになる (空の辞書を返す) | `on_epoch_end` で保存し `{}` を返す. 学習後の処理は `fit` の後に書く | 小さな問題. 指標を返さない `on_epoch_end` の使い方を README に例として載せる |
| 09 | `step_fn(model, batch, phase)` には引数としてエポックが渡らないので, KL annealing のようにエポックに依存する損失を関数の形で書くには, closure の中から `trainer.epoch` を参照するしかない | `Trainer` を継承して `step` を上書きし `self.epoch` を使う | `step_fn` にエポックなどの状態を渡す (例: `step_fn(model, batch, phase, trainer)` の形を選べるようにする) |
| 09 | 学習の途中で損失の定義が変わる場合 (annealing, β のスケジュール), `monitor` の選び方で残るモデルが大きく変わる. β を 4 に上げる実験では, 重み 1 の負の ELBO で判定したベストモデルが重み 1.6 のエポック 4 になり, 学習後にその重みに戻された | 最後の重みが要る場合は `last.pt` の `model` から読み込む | ベストモデルに戻さずに最後の重みを残す設定 (例: `trainer.restore_best: false`). `monitor: null` は既に「`valid/loss` を使う」意味なので, 判定しない設定には別の値が要る |
| 09 | 指標でない値 (`beta` のようなスケジュールの値) も `step` で返すと, `train/` と `valid/` の両方に記録され学習曲線も描かれる. サンプル数で重み付けして平均するので `1.2000000000000008` のような誤差も出る | そのまま記録する | 平均せずに記録だけする値を返す仕組み (例: 戻り値の `extra` という辞書). `on_epoch_end` で返しても `train/` か `valid/` が付いて学習曲線が描かれるので, 小数の誤差が消えるだけで記録だけにはならない |
| 09 | DataLoader が `persistent_workers=True` の場合, 再開した学習が通しの学習と一致しない (再開した直後のエポックから異なる). ワーカーの乱数の種をイテレータを最初に作るときに 1 回だけ全体の乱数から引くので, 通しの学習 (エポック 1 で引く) と再開した学習 (再開後の最初のエポックで DataLoader ごとに引く) で乱数の流れがずれる. `num_workers: 0` では完全に一致した | 再開を通しの学習と一致させたい場合は `num_workers: 0` にする, または `persistent_workers` を使わない (後者は未確認) | README の再開の説明に注意を書く. `fit` で `persistent_workers=True` の DataLoader を受け取って再開する場合に警告する. DataLoader に専用の `generator` を渡す場合は, その状態もチェックポイントに保存して戻す必要がある (渡すだけでは再開後に最初の状態からやり直しになる) |
| 10 | Optimizer, Scheduler, チェックポイントが 1 つの Optimizer を前提にしているので, GAN の 2 つ目の Optimizer は config (プロジェクト固有の項目), 作成 (`build_optimizer`), 更新, 保存 (`state_dict` の上書き), 再開 (`resume` の上書き) の全てを利用者が書く必要がある | このタスクの `GANTrainer` の B と C. 再開で 2 つ目の Optimizer の状態が続くことを確認 | 複数の Optimizer を名前付きで config に書ける仕組み (例: `optimizers: {generator: ..., discriminator: ...}`) と, それらを全て保存, 再開する仕組み |
| 10 | `update_parameters` は損失を 1 つしか受け取らないので, 2 つの損失を交互に更新するには `step` から `self` 経由で渡す必要があり, 識別器を先に更新すると生成器の損失の逆伝播が in-place の変更の検出でエラーになるので, 一般的な順と逆に生成器を先に更新する必要があった | 生成器を先に更新する (B), または `step` の中で全て更新する (C) | `step` が複数の損失を返し, 損失ごとに Optimizer を対応させて順に更新する仕組み. または自動の更新を止める設定 (C の書き方を公式にする) |
| 10 | 更新を全て `step` の中で行う場合も, 元の `update_parameters` が `loss` で必ず `backward` するので, 勾配の無い `loss` では失敗する (`check_outputs` は通る) | `g_loss.detach().requires_grad_(True)` のような勾配がパラメータに流れない損失を返す, または `update_parameters` を上書きして逆伝播と更新を飛ばす | 自動の更新を止める設定 (例: `trainer.manual_optimization: true`). その場合は `loss` の `backward` と `optimizer.step` を行わない |
| 10 | ベストモデルの判定と学習後の巻き戻しを止める設定が無い. GAN では生成器の損失が学習とともに大きくなるので `train/loss` のベストがエポック 1 になり, `final_model.pt` と学習後のモデルが画像の崩れたエポック 1 の重みになった | 最後の重みを `last.pt` の `model` から読み込み直す. または `on_epoch_end` で毎エポック増える値を返して `mode: max` で監視する (見せかけの指標) | 09 と同じく `trainer.restore_best: false` のような設定. GAN のように良し悪しの指標が無い場合のために, `monitor` を使わない設定も用意する |
| 10 | `Trainer.resume` は読み込んだチェックポイントを外に出さないので, 継承して追加の状態を戻すには, パスの解決の規則を写してもう一度読み込む必要がある | `resume` を上書きし, 同じ規則でパスを解決して `load_checkpoint` で読み込み直す | `resume` が読み込んだ状態を返す, または `load_state_dict(state)` のようなフックを用意して, 継承したクラスが追加の状態を戻せるようにする |
| 10 | 勾配クリッピング (`trainer.grad_clip_norm`) は `model.parameters()` 全体に掛かり, config の Scheduler は `Trainer` に渡した 1 つの Optimizer にしか付かないので, Optimizer を分けた場合に Optimizer ごとにクリップや学習率の調整ができない | クリップや Scheduler を自分で書く (このタスクでは使わなかった) | クリップを `self.optimizer` のパラメータグループに対して掛ける. 複数の Optimizer に対応する場合は, Optimizer ごとにクリップと Scheduler を設定できるようにする |
| 11 | 拡散モデルのように損失に乱数 (時刻とノイズ) を含む学習では, 検証用の損失がエポックごとに乱数で揺れ, ベストモデルの判定と Early Stopping が乱数に左右される. 毎回違う乱数の検証用の損失は 29 回のエポック間の変化のうち 10 回で上がり, ベストはエポック 24 になった (固定した乱数では 3 回, ベストはエポック 30) | 専用の乱数生成器を用意し, `on_epoch_start("valid")` で毎エポック同じ状態に戻して検証の乱数を固定する (このタスクの `DiffusionTrainer`) | 検証の前に乱数の状態を毎回同じに戻す設定 (例: `trainer.valid_seed`). 検証の後には学習の乱数の状態を元に戻す |
| 11 | EMA の重みがチェックポイントにも `final_model.pt` にも入らず, 検証やベストモデルの判定に EMA の重みを使うフックも無い | `update_parameters` を上書きして EMA を更新し, 学習後に自分で保存する. 再開に対応するには 10 と同じく `state_dict` と `resume` も上書きする | `trainer.ema_decay` のような設定で, 更新, 保存, 再開, 検証に使うモデルの切り替えまで行う |
| 11 | 再開すると Scheduler の状態が引数 (`T_max` など) ごとチェックポイントから戻るので, config で引数を変えても効かない. `T_max: ${trainer.epochs}` で 1 エポックで止めてから `epochs: 2` で再開すると, `T_max = 1` のまま動いてエポック 2 の学習率が 0 になり, 通しの学習 (0.0001) と異なった. 再開した学習の `config.yaml` には `T_max: 2` と記録され, 実際の設定と食い違う | `T_max` を最後まで学習する値に固定しておく (固定すると再開した学習が通しの学習と完全に一致) | 再開時に Scheduler の引数が config と違う場合は警告する, または config の値で上書きするかを選べるようにする |
| 12 | 外部ライブラリの乱数 (qiskit の `algorithm_globals`, Estimator の雑音や Sampler の測定に使う `seed`) は `fix_seed` と再開時の乱数の保存 (`get_rng_state`) の対象外. 固定しないと同じ config でも結果がずれる (固定しない 2 回で検証用の正解率が最大 0.033 違った) | `train.py` で `algorithm_globals.random_seed` などに `experiment.seed` を自分で渡す (このタスクで 2 回の結果が完全に一致することを確認) | 外部の乱数の固定と状態の保存, 復元を登録できる仕組み (例: `start_experiment(..., extra_seeders=[...])`) |
| 12 | `train.log` のモデルの構造は `print(model)` なので, 量子層のように中身を `nn.Module` の子として持たないモデルは `TorchConnector()` とだけ出て, 回路の構成が記録されない | モデルに `extra_repr` を書く, または自分でログに書く | 小さな問題. README で, 再現に必要なモデルの情報は `extra_repr` で出すよう勧める |
