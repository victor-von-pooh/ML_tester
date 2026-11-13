import math
from pathlib import Path
from typing import Any, Callable, Optional

from omegaconf import DictConfig
import torch
from torch import nn, optim

from ml_found import Experiment, Trainer, load_project_config, start_experiment
from ml_found.training.optim import step_scheduler
from model import FunctionMLP

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class ClosureTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        step_fn: Callable[[nn.Module, Any, str], dict],
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(ClosureTrainer, self).__init__(
            model, experiment, step_fn, optimizer, scheduler
        )

        # closure で損失を計算し直すための, 学習時のバッチの入れ物を用意
        self.current_batch: Any = None

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失を求め, closure で使うバッチを保存する関数

        Parameters
        ----------
        batch: Any
            デバイスに移したバッチ
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            平均二乗誤差と最大誤差
        """
        # update_parameters は損失しか受け取らないので, 学習時のバッチを保存
        if phase == "train":
            self.current_batch = batch

        # 元の step で計算
        outputs = super(ClosureTrainer, self).step(batch, phase)

        return outputs

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        損失を計算し直す closure を Optimizer に渡して更新する関数

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失, closure の中で計算し直すので使わない

        Returns
        ----------
        None
        """
        # LBFGS は直線探索のために損失と勾配を何度も計算し直すので,
        # 保存したバッチから計算し直す関数を用意
        # 勾配クリッピングは元の update_parameters と同じく逆伝播の後に適用
        def closure() -> torch.Tensor:
            self.optimizer.zero_grad()
            new_loss = self.step_fn(
                self.model, self.current_batch, "train"
            )["loss"]
            new_loss.backward()
            if self.trainer_cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_cfg.grad_clip_norm
                )
            return new_loss

        # closure を渡して更新し, 回数のカウント
        self.optimizer.step(closure)
        self.global_step += 1

        # 元の update_parameters と同じく, step ごとの場合は学習率を更新
        if self.scheduler_interval == "step":
            step_scheduler(self.scheduler)


def target_function(x: torch.Tensor) -> torch.Tensor:
    """
    近似する関数の値を求める関数

    Parameters
    ----------
    x: torch.Tensor
        入力

    Returns
    ----------
    y: torch.Tensor
        sin(2πx) + 0.5 sin(6πx) の値
    """
    # 周期の違う 2 つの正弦波の和を計算
    y = torch.sin(2 * math.pi * x) + 0.5 * torch.sin(6 * math.pi * x)

    return y


def make_data(cfg: DictConfig) -> tuple[dict, dict]:
    """
    学習用と検証用のデータを作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_data: dict
        一様乱数の入力と, 雑音を加えた関数の値
    valid_data: dict
        等間隔の入力と, 雑音の無い関数の値
    """
    # 全体の乱数に影響しないように専用の乱数生成器を用意
    generator = torch.Generator().manual_seed(cfg.experiment.seed)

    # 学習用は 0 から 1 の一様乱数の入力に, 小さな雑音を加えた値
    x_train = torch.rand(cfg.data.n_train, 1, generator=generator)
    noise = torch.randn(cfg.data.n_train, 1, generator=generator)
    y_train = target_function(x_train) + cfg.data.noise * noise
    train_data = {"x": x_train, "y": y_train}

    # 検証用は等間隔の入力に, 雑音の無い値
    x_valid = torch.linspace(0.0, 1.0, cfg.data.n_valid).unsqueeze(1)
    valid_data = {"x": x_valid, "y": target_function(x_valid)}

    return train_data, valid_data


def step(model: nn.Module, batch: dict, phase: str) -> dict:
    """
    全データの平均二乗誤差と最大誤差を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: dict
        入力 x と関数の値 y
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        平均二乗誤差と, 絶対誤差の最大値
    """
    # 予測して損失の計算
    pred = model(batch["x"])
    mse = nn.MSELoss()
    loss = mse(pred, batch["y"])

    # 最も外れた点の誤差を求め, 損失と集約
    max_err = (pred.detach() - batch["y"]).abs().max()
    outputs = {"loss": loss, "max_err": max_err}

    return outputs


def main() -> None:
    """
    config を読み込んで関数近似を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    cfg, diff = load_project_config(str(TASK_DIR))
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データとモデルを用意
    train_data, valid_data = make_data(cfg)
    model = FunctionMLP(
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers
    )

    # closure を渡す更新手順を使うかどうかで Trainer を切り替え
    trainer_cls = ClosureTrainer if cfg.train.use_closure else Trainer
    trainer = trainer_cls(model, experiment, step)

    # 全データを 1 つのバッチとして学習
    history = trainer.fit([train_data], [valid_data])

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/loss", "valid/max_err"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.3e}")


if __name__ == "__main__":
    main()
