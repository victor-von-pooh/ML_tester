import argparse
import math
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Experiment, Trainer, load_config, start_experiment
from model import HousingRegressor

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class RegressionTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, y_std: float,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(RegressionTrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # 標準化した目的変数を元の単位に戻すための標準偏差を保存
        self.y_std = y_std

        # 損失関数を定義
        self.mse = nn.MSELoss()

        # エポック全体の二乗誤差の合計とサンプル数の入れ物を用意
        self.sq_sum = 0.0
        self.count = 0

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの学習または検証を始める前に入れ物を空にする関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        None
        """
        # 二乗誤差の合計とサンプル数を初期化
        self.sq_sum = 0.0
        self.count = 0

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失と誤差を求める関数

        Parameters
        ----------
        batch: Any
            標準化した特徴量と標準化した目的変数
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            標準化した空間での平均二乗誤差, 元の単位での MAE と
            バッチごとの RMSE
        """
        # 予測して標準化した空間で損失の計算
        x, y = batch
        pred = self.model(x)
        loss = self.mse(pred, y)

        # 元の単位 (10 万ドル) に戻した誤差の計算
        err = (pred.detach() - y) * self.y_std
        mae = err.abs().mean()
        rmse_batch = err.pow(2).mean().sqrt()

        # エポック全体の RMSE のために二乗誤差の合計を保存
        self.sq_sum += err.pow(2).sum().item()
        self.count += len(y)

        # バッチごとの RMSE も, 平均との違いを見るために記録
        outputs = {"loss": loss, "mae": mae, "rmse_batch": rmse_batch}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の二乗誤差から RMSE を求める関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            元の単位での RMSE
        """
        # 二乗誤差の合計をサンプル数で割ってから平方根を計算
        metrics = {"rmse": math.sqrt(self.sq_sum / self.count)}

        return metrics


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    args: argparse.Namespace
        コマンドライン引数, experiment の config のパスを持つ
    """
    # README の実験を順に試せるように experiment の config を切り替え可能に設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", default="configs/experiment/train.yaml",
        help=(
            "experiment の config のパス, 相対パスはタスクのディレクトリから辿り, "
            "無い場合は実行した場所から辿る"
        )
    )
    args = parser.parse_args()

    return args


def resolve_experiment_path(path: str) -> Path:
    """
    experiment の config のパスを解決する関数

    Parameters
    ----------
    path: str
        コマンドライン引数で指定されたパス

    Returns
    ----------
    resolved: Path
        experiment の config の絶対パス
    """
    # 絶対パスの場合はそのまま使用
    given = Path(path).expanduser()
    if given.is_absolute():
        resolved = given
        return resolved

    # タスクのディレクトリから辿り, 無い場合は実行した場所から探索
    resolved = TASK_DIR / given
    if not resolved.exists() and given.exists():
        resolved = given.resolve()

    return resolved


def make_data(
    cfg: DictConfig
) -> tuple[TensorDataset, TensorDataset, float]:
    """
    California Housing を読み込んで分割と標準化を行う関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: TensorDataset
        学習用のデータセット
    valid_dataset: TensorDataset
        検証用のデータセット
    y_std: float
        学習用データの目的変数の標準偏差 (10 万ドル単位)
    """
    # データを読み込み, 学習用と検証用に分割
    # 相対パスの保存先はタスクのディレクトリを基準に解決
    data_home = Path(cfg.data.root).expanduser()
    if not data_home.is_absolute():
        data_home = (TASK_DIR / data_home).resolve()
    x, y = fetch_california_housing(
        data_home=str(data_home), return_X_y=True
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, random_state=cfg.experiment.seed
    )

    # 学習用データで特徴量と目的変数の標準化の基準を決めて両方に適用
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_valid = x_scaler.transform(x_valid)
    y_mean = y_train.mean()
    y_std = float(y_train.std())
    y_train = (y_train - y_mean) / y_std
    y_valid = (y_valid - y_mean) / y_std

    # テンソルにしてデータセットに集約
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).float()
    )

    return train_dataset, valid_dataset, y_std


def main() -> None:
    """
    config を読み込んで California Housing の回帰を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    args = parse_args()
    cfg, diff = load_config(
        str(TASK_DIR / "configs/default/train.yaml"),
        str(resolve_experiment_path(args.experiment))
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データを用意し, 目的変数の標準偏差を記録
    train_dataset, valid_dataset, y_std = make_data(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )
    experiment.logger.info(f"目的変数の標準偏差: {y_std:.4f}")

    # モデルを用意して学習
    input_dim = train_dataset.tensors[0].shape[1]
    model = HousingRegressor(
        input_dim=input_dim, hidden_dim=cfg.model.hidden_dim
    )
    trainer = RegressionTrainer(model, experiment, y_std)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/rmse", "valid/rmse_batch", "valid/mae"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.4f}")


if __name__ == "__main__":
    main()
