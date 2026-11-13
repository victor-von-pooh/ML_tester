import contextlib
import math
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig
import torch
from torch import nn

from ml_found import Trainer, load_project_config, start_experiment
from model import FunctionMLP

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


def make_data(cfg: DictConfig) -> tuple[dict, dict]:
    """
    微分方程式の残差を評価する点を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_data: dict
        一様乱数の内部の点 x と, 両端の点 x_bc
    valid_data: dict
        両端を含む等間隔の点 x と, 両端の点 x_bc, 厳密解 u
    """
    # 全体の乱数に影響しないように専用の乱数生成器を用意
    generator = torch.Generator().manual_seed(cfg.experiment.seed)

    # 学習用は 0 から 1 の一様乱数の点
    x_bc = torch.tensor([[0.0], [1.0]])
    x_train = torch.rand(cfg.data.n_train, 1, generator=generator)
    train_data = {"x": x_train, "x_bc": x_bc}

    # 検証用は等間隔の点と厳密解 sin(πx)
    x_valid = torch.linspace(0.0, 1.0, cfg.data.n_valid).unsqueeze(1)
    valid_data = {
        "x": x_valid, "x_bc": x_bc, "u": torch.sin(math.pi * x_valid)
    }

    return train_data, valid_data


def make_step(force_grad: bool) -> Callable[[nn.Module, dict, str], dict]:
    """
    微分方程式の残差を損失にする step を作る関数

    Parameters
    ----------
    force_grad: bool
        検証時にも勾配の計算を有効にするかどうか

    Returns
    ----------
    step: Callable[[nn.Module, dict, str], dict]
        ML-Found に渡す, 残差と境界条件の損失を求める関数
    """
    # 損失関数の定義
    mse = nn.MSELoss()

    def step(model: nn.Module, batch: dict, phase: str) -> dict:
        """
        u'' = -π² sin(πx), u(0) = u(1) = 0 の残差を求める関数

        Parameters
        ----------
        model: nn.Module
            解 u(x) を近似するモデル
        batch: dict
            内部の点 x, 両端の点 x_bc, 検証時は厳密解 u
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            残差と境界条件の損失, 検証時は厳密解との誤差も含む
        """
        # ML-Found は検証時に勾配の計算を止めるので, 必要なら自分で有効化
        grad_context = (
            torch.enable_grad() if force_grad else contextlib.nullcontext()
        )
        with grad_context:
            # 入力に対する 2 階微分を自動微分で計算
            x = batch["x"].clone().requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]

            # 方程式の残差と境界条件の誤差を損失に集約
            source = -(math.pi ** 2) * torch.sin(math.pi * x)
            residual = mse(d2u, source)
            boundary = mse(
                model(batch["x_bc"]), torch.zeros_like(batch["x_bc"])
            )
            loss = residual + boundary

        # 検証時は厳密解との誤差も記録
        outputs = {
            "loss": loss, "residual": residual.detach(),
            "boundary": boundary.detach()
        }
        if phase == "valid":
            outputs["max_err"] = (u.detach() - batch["u"]).abs().max()

        return outputs

    return step


def main() -> None:
    """
    config を読み込んで微分方程式を PINN で解く関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    cfg, diff = load_project_config(
        str(TASK_DIR), default="configs/default/pinn.yaml",
        experiment="configs/experiment/pinn.yaml"
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データとモデルを用意し, 全ての点を 1 つのバッチとして学習
    train_data, valid_data = make_data(cfg)
    model = FunctionMLP(
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers
    )
    trainer = Trainer(model, experiment, make_step(cfg.pinn.force_grad))
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
