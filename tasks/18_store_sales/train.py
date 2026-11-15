from pathlib import Path
import sys
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from ml_modelcontrol import build_model

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


def unscale(dataset: Dataset, name: str, values: torch.Tensor) -> torch.Tensor:
    """
    レシピで標準化した列の値を, 元の大きさ (円) に戻す関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット
    name: str
        列の名前
    values: torch.Tensor
        標準化した値

    Returns
    ----------
    original: torch.Tensor
        元の大きさの値 (標準化していない列はそのまま)
    """
    # 標準化の手順で当てはめた中心と幅での復元 (後の手順から順に)
    original = values
    for step in reversed(dataset.recipe["steps"]):
        if step["op"] == "standardize":
            center, scale = step["state"]["values"].get(name, [0.0, 1.0])
            original = original * scale + center

    return original


def make_step(
    scale: float
) -> Callable[[nn.Module, tuple, str], dict]:
    """
    標準化した売上の二乗誤差と, 円での平均絶対誤差を求める関数を作る関数

    Parameters
    ----------
    scale: float
        売上の標準化の幅 (学習用の標準偏差)

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        ML-Found の step
    """
    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失と平均絶対誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            特徴量と標準化した売上
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            二乗誤差 (損失) と円での平均絶対誤差 (mae)
        """
        # 標準化した売上で学び, 円に直した誤差を記録
        x, y = batch
        pred = model(x).squeeze(-1)
        mse = nn.MSELoss()
        loss = mse(pred, y)
        mae = (pred - y).abs().mean() * scale
        outputs = {"loss": loss, "mae": mae}

        return outputs

    return step


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment,
    center: float, scale: float
) -> None:
    """
    テスト用の期間の誤差と, 7 つ前の行の売上 (多くは前の週の同じ曜日) を
    答えた場合の誤差を比べる関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)
    center: float
        売上の中心 (学習用の平均)
    scale: float
        売上の幅 (学習用の標準偏差)

    Returns
    ----------
    None
    """
    # モデルの円での平均絶対誤差と平均の割合の誤差
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        pred = model(x).squeeze(-1) * scale + center
    mae = (pred - y).abs().mean().item()
    mape = ((pred - y).abs() / y.clamp(min=1)).mean().item()
    experiment.logger.info(
        f"テスト用の平均絶対誤差: {mae:,.0f} 円, 平均の割合の誤差: {mape:.3%}"
    )

    # 7 つ前の行の売上 (sales_lag7. 多くは前の週の同じ曜日) を答えた場合
    if "sales_lag7" in dataset.features:
        column = dataset.features.index("sales_lag7")
        last_week = unscale(dataset, "sales_lag7", x[:, column])
        base = (last_week - y).abs().mean().item()
        base_pct = ((last_week - y).abs() / y.clamp(min=1)).mean().item()
        experiment.logger.info(
            f"7 つ前の行 (sales_lag7) を答えた場合: {base:,.0f} 円, "
            f"{base_pct:.3%}"
        )


def main() -> None:
    """
    ML-Fathom で作った店ごとの売上のデータセットで, ML-CreateArchitecture
    で組んだモデルを学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ作り方を案内)
    cfg, diff = load_project_config(str(TASK_DIR))
    path = TASK_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. 先に prepare.py を実行してください")
        sys.exit(1)
    dataset = load_dataset(path)

    # 学習用の平均と標準偏差での売上の標準化
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

    # ML-CreateArchitecture で組んだ定義 (入力の列の数を確かめる)
    model = build_model(TASK_DIR / cfg.model.spec, seed=cfg.experiment.seed)
    first = next(m for m in model.modules() if isinstance(m, nn.Linear))
    if first.in_features != x_train.shape[1]:
        print(f"定義の入力 {first.in_features} がデータセットの列の数 "
              f"{x_train.shape[1]} と合いません. ML-CreateArchitecture で "
              "最初の nn.Linear の in_features を直してください")
        sys.exit(1)
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    experiment.logger.info(
        f"データセット: {cfg.data.dataset}, 学習に使う列 "
        f"{len(dataset.features)}: {dataset.features}"
    )
    trainer = Trainer(model, experiment, make_step(scale))
    history = trainer.fit(train_loader, valid_loader)

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証用の平均絶対誤差: "
          f"{history.get('valid/mae')[best_epoch - 1]:,.0f} 円")
    evaluate_test(trainer.model, dataset, experiment, center, scale)


if __name__ == "__main__":
    main()
