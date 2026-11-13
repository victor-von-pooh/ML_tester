from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn, optim
from torchvision.utils import save_image

from data import make_loader, make_train_valid
from ml_found import Experiment, Trainer, load_project_config, start_experiment
from model import VAE

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class VAETrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, beta: float,
        kl_warmup_epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(VAETrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # KL の重みの最終値と, 重みを 0 から上げるエポック数を保存
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs

        # 画素ごとの二値交差エントロピーを画像ごとに合計する損失関数を定義
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def get_beta(self) -> float:
        """
        今のエポックの KL の重みを求める関数

        Parameters
        ----------
        None

        Returns
        ----------
        beta: float
            KL の重み, warmup の間はエポックに比例して上昇
        """
        # warmup が無い場合は最初から最終値
        if self.kl_warmup_epochs <= 0:
            beta = self.beta
            return beta

        # エポックに比例して最終値まで上昇
        beta = self.beta * min(1.0, self.epoch / self.kl_warmup_epochs)

        return beta

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の負の ELBO とその内訳を求める関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            学習に使う損失, 再構成誤差, KL, 負の ELBO (いずれも画像 1 枚
            あたり), 記録だけする KL の重み (extra)
        """
        # 正解ラベルは使わずに画像だけを再構成
        x, _ = batch
        logits, mu, log_var, _ = self.model(x)
        n = len(x)

        # 再構成誤差と, 標準正規分布との KL ダイバージェンスを画像 1 枚あたりで計算
        recon = self.bce(logits, x) / n
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / n

        # 学習には KL の重みを掛けた損失を使い, 比較には重み 1 の負の ELBO を使用
        beta = self.get_beta()
        loss = recon + beta * kl
        nelbo = (recon + kl).detach()
        # KL の重みは指標ではないので, 平均せずに記録だけする extra で返却
        outputs = {
            "loss": loss, "recon": recon.detach(), "kl": kl.detach(),
            "nelbo": nelbo, "extra": {"beta": beta}
        }

        return outputs


def save_samples(
    model: VAE, n_images: int, device: torch.device, path: Path
) -> None:
    """
    標準正規分布から潜在変数をサンプルして画像を生成し保存する関数

    Parameters
    ----------
    model: VAE
        学習したモデル
    n_images: int
        生成する枚数
    device: torch.device
        生成に使うデバイス
    path: Path
        保存先のパス

    Returns
    ----------
    None
    """
    # 事前分布から潜在変数をサンプルしてデコード
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_images, model.latent_dim, device=device)
        images = model.decode(z).cpu()

    # 8 枚ずつ並べて保存
    save_image(images, path, nrow=8)


def save_traversal(
    model: VAE, n_steps: int, device: torch.device, path: Path
) -> None:
    """
    2 次元の潜在空間を格子状にたどって生成した画像を保存する関数

    Parameters
    ----------
    model: VAE
        潜在変数が 2 次元のモデル
    n_steps: int
        縦横それぞれの格子の数
    device: torch.device
        生成に使うデバイス
    path: Path
        保存先のパス

    Returns
    ----------
    None
    """
    # -3 から 3 の範囲を格子状に区切った潜在変数を作成
    values = torch.linspace(-3.0, 3.0, n_steps)
    grid_y, grid_x = torch.meshgrid(values.flip(0), values, indexing="ij")
    z = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # デコードして格子の順に並べて保存
    model.eval()
    with torch.no_grad():
        images = model.decode(z.to(device)).cpu()
    save_image(images, path, nrow=n_steps)


def main() -> None:
    """
    config を読み込んで MNIST の VAE を学習する関数

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

    # データを用意
    train_dataset, valid_dataset = make_train_valid(cfg)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = make_loader(valid_dataset, cfg, shuffle=False)

    # モデルを用意して学習
    model = VAE(
        model_type=cfg.model.type, hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim
    )
    trainer = VAETrainer(
        model, experiment, cfg.loss.beta, cfg.loss.kl_warmup_epochs
    )
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルから画像を生成して保存
    run_dir = experiment.run_dir
    save_samples(
        trainer.model, cfg.visualize.n_images, trainer.device,
        run_dir / "samples.png"
    )
    if cfg.model.latent_dim == 2:
        save_traversal(
            trainer.model, cfg.visualize.n_steps, trainer.device,
            run_dir / "traversal.png"
        )

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/nelbo", "valid/recon", "valid/kl"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.2f}")


if __name__ == "__main__":
    main()
