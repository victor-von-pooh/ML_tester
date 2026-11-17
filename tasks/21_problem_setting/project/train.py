"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

骨組みなので, 印 (TODO) の所を書き足す
"""
from pathlib import Path
import shutil
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from ml_inference import make_bundle
from ml_modelcontrol import build_model

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測すること
PROBLEM_NAME = "総菜の作る量"
PREDICT = "翌日の品ごとの売れる数"

# 推論の一式の意思決定と結論の既定値 (問題の設定の損得と制約から)
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

    # TODO: 比べる相手 (前の週の同じ曜日と同じ数) の値も記録する


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
    spec = PROJECT_DIR / cfg.model.spec
    if not spec.is_file():
        print(f"{spec} がありません. README の「次にすること」の 2 で, "
              "モデルの定義を置いてください")
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

    # モデルの定義からの組み立てと学習
    model = build_model(spec, seed=cfg.experiment.seed)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 予測すること: {PREDICT}. 学習に使う列 "
        f"{len(dataset.features)}: {dataset.features}"
    )
    trainer = Trainer(model, experiment, Step(scale))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定とモデルの定義の写し
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    shutil.copy(spec, run_dir / "model_spec.yaml")

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
