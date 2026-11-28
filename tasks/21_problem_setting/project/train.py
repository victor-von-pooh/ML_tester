"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

骨組みなので, 問題に合わせて書き足す
"""
from pathlib import Path
import shutil
import sys
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from ml_inference import make_bundle
from ml_modelcontrol import build_model, validate

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測すること
PROBLEM_NAME = "総菜の作る量"
PREDICT = "翌日の品ごとの売れる数"

# 推論の一式の意思決定と結論の既定値 (問題の損得と制約から)
DECISION = {"type": "quantity", "under_cost": 200, "over_cost": 150}
CONCLUSION = {"group_by": "store"}


class Step:
    def __init__(self, scale: float):
        # 目的の値の標準化の幅を保存
        self.scale = scale

    def __call__(self, model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失と, 元の単位の誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            特徴量と標準化した目的の値
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            二乗誤差 (損失) と, 元の単位の平均絶対誤差 (mae)
        """
        # 標準化した値で学び, 元の単位に直した誤差の記録 (バッチの平均で
        # 書ける指標だけ. RMSE はエポック全体の二乗誤差から求める)
        x, y = batch
        pred = model(x).squeeze(-1)
        mse = nn.MSELoss()
        loss = mse(pred, y)
        error = (pred - y) * self.scale
        outputs = {"loss": loss, "mae": error.abs().mean()}

        return outputs


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment,
    center: float, scale: float
) -> None:
    """
    テスト用の分割の誤差を記録する関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)
    center: float
        目的の値の中心 (学習用の平均)
    scale: float
        目的の値の幅 (学習用の標準偏差)

    Returns
    ----------
    None
    """
    # テスト用の分割が無い場合は知らせだけ
    if "test" not in dataset.frames:
        experiment.logger.info("テスト用の分割が無いので, 評価しません")
        return
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        pred = model(x).squeeze(-1) * scale + center
    error = pred - y
    value = error.abs().mean()
    experiment.logger.info(f"テスト用の平均絶対誤差: {value.item():,.4f}")

# 比べる相手 (前の週の同じ曜日と同じ数): 問題に今のやり方をデータで書いていない (今のやり方をデータで書く問い,
# action.current)ので, 今のやり方とは比べない. 答えて骨組みを書き出し直すと, 検証用とテスト用で今のやり方と比べた評価の表
# (evaluations/baseline_<分割>.json) を書く


def spec_errors(spec: Path) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義

    Returns
    ----------
    errors: list[str]
        誤りの文 (無ければ空. 外部の重みのファイルが読めるかは, 組み立てる
        ときに分かる)
    """
    # 読めない定義も誤りとして返る (例外にならない)
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]

    return errors


def model_file(
    spec: Path, experiment: Experiment
) -> tuple[Path, Optional[Path]]:
    """
    モデルを組み立てる ML-ModelControl の定義のファイルと, 定義の中の相対
    の参照 (外部の重みなど) の起点を返す関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義
    experiment: Experiment
        実験 (ML-Found が実験に写した定義の場所 model_spec_path と, 写した
        元の定義 model_spec_source)

    Returns
    ----------
    path: Path
        ML-Found が実験に写した定義 (学習を始めたときの定義で, source.yaml
        の sha256 と同じ中身). 写せなければプロジェクトの定義
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. プロジェクトの
        定義なら None で, 定義のディレクトリ)
    """
    # 写しの相対の参照は元の定義の場所から (写しの隣には重みが無い)
    if experiment.model_spec_path is not None:
        path = Path(experiment.model_spec_path)
        base_dir = Path(experiment.model_spec_source).parent
        return path, base_dir

    # 写しが無ければ理由を知らせて, プロジェクトの定義
    experiment.logger.warning(
        "モデルの定義を実験に写せなかったので, 学習のプロジェクトの定義から"
        "組み立てます (理由は source.yaml の model_spec の status)"
    )
    path = spec
    base_dir = None

    return path, base_dir


def main() -> None:
    """
    データセットとモデルの定義から学習し, 推論の一式を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ作り方の案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "データセットを書き出してください")
        sys.exit(1)
    if not cfg.model.spec:
        print("config の model.spec に定義の場所を書いてください (既定は "
              "models/model.yaml)")
        sys.exit(1)
    spec = PROJECT_DIR / cfg.model.spec
    if not spec.is_file():
        print(f"{spec} がありません. README の「次にすること」の 2 で, "
              "モデルの定義を置いてください")
        sys.exit(1)
    errors = spec_errors(spec)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\n".join(f"- {message}" for message in errors))
        sys.exit(1)
    dataset = load_dataset(path)
    if "valid" not in dataset.frames:
        print("データセットに検証用の分割がありません. 最も良いモデルを選ぶ"
              "ために, 検証用の割合を 0 より大きくして書き出してください")
        sys.exit(1)

    # 学習用と検証用 (目的の値は学習用の平均と標準偏差で標準化)
    x_train, y_train = dataset.tensors("train")
    x_valid, y_valid = dataset.tensors("valid")
    center, scale = y_train.mean().item(), y_train.std().item()
    train_loader = DataLoader(
        TensorDataset(x_train, (y_train - center) / scale),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, (y_valid - center) / scale),
        batch_size=cfg.data.batch_size
    )

    # 実験に写したモデルの定義からの組み立てと学習
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    spec, base_dir = model_file(spec, experiment)
    model = build_model(spec, seed=cfg.experiment.seed, base_dir=base_dir)
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 予測すること: {PREDICT}. 学習に使う列 "
        f"{len(dataset.features)}: {dataset.features}"
    )
    trainer = Trainer(model, experiment, Step(scale))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    best_value = trainer.checkpoint.best_value
    print(f"保存先: {run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの {cfg.trainer.monitor}: {best_value}")
    evaluate_test(trainer.model, dataset, experiment, center, scale)

    # 実験の中に推論の一式 (問題の設定の意思決定と結論を既定値に)
    info = make_bundle(
        run_dir / "inference", spec, path, run_dir,
        target_scale={"center": center, "scale": scale}, conclusion=CONCLUSION,
        decision=DECISION, name=cfg.experiment.name
    )
    check = info["check"]
    if check and check["recorded"] is None:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (監視した指標が accuracy か mae でない"
            "ので, 学習の記録とは比べない)"
        )
    elif check:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (記録 {check['recorded']}, 合う: "
            f"{check['matches']})"
        )


if __name__ == "__main__":
    main()
