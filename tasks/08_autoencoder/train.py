from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import make_loader, make_train_valid
from ml_found import Experiment, Trainer, load_project_config, start_experiment
from model import AutoEncoder

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class AETrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        fixed_images: torch.Tensor, every_n_epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(AETrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # 毎回同じ画像の再構成を比べられるように, 検証用の画像を保存
        self.fixed_images = fixed_images.to(self.device)

        # 再構成画像を保存する間隔と保存先を保存
        self.every_n_epochs = every_n_epochs
        self.recon_dir = experiment.run_dir / "recon"
        self.recon_dir.mkdir()

        # 損失関数の定義
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の再構成の損失を求める関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            画素ごとの二値交差エントロピー損失と平均二乗誤差
        """
        # 正解ラベルは使わずに画像だけを再構成
        x, _ = batch
        recon, _ = self.model(x)

        # 損失と, 比較用の平均二乗誤差の計算
        loss = self.bce(recon, x)
        mse = self.mse(recon.detach(), x)
        outputs = {"loss": loss, "mse": mse}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        決まった間隔で再構成画像を保存する関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            記録する指標は無いので空の辞書
        """
        # 検証の後に, 最初と決まった間隔のエポックと最後のエポックだけ保存
        metrics = {}
        is_last = self.epoch == self.trainer_cfg.epochs
        is_interval = self.epoch % self.every_n_epochs == 0
        if phase != "valid" or not (
            self.epoch == 1 or is_interval or is_last
        ):
            return metrics

        # on_epoch_end は勾配の計算を止める範囲の外で呼ばれるので,
        # 最初の 1 回だけ勾配の計算が有効かを記録してから自分で停止
        if self.epoch == 1:
            self.logger.info(
                f"on_epoch_end での勾配の計算: {torch.is_grad_enabled()}"
            )
        path = self.recon_dir / f"epoch_{self.epoch:03d}.png"
        self.save_recon(path)

        return metrics

    def save_recon(self, path: Path) -> None:
        """
        固定した画像と再構成画像を並べて保存する関数

        Parameters
        ----------
        path: Path
            保存先のパス

        Returns
        ----------
        None
        """
        # 推論モードで勾配を止めて再構成
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(self.fixed_images)

        # 上の段に元の画像, 下の段に再構成画像を並べて保存
        grid = torch.cat([self.fixed_images, recon]).cpu()
        save_image(grid, path, nrow=len(self.fixed_images))
        self.logger.info(f"再構成画像を保存: {path.name}")


def encode_all(
    model: AutoEncoder, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    データ全体を潜在変数に変換する関数

    Parameters
    ----------
    model: AutoEncoder
        学習したモデル
    loader: DataLoader
        変換するデータの DataLoader
    device: torch.device
        推論に使うデバイス

    Returns
    ----------
    z: torch.Tensor
        潜在変数, 形状は (サンプル数, latent_dim)
    labels: torch.Tensor
        正解ラベル, 色分けにだけ使う
    """
    # 推論モードで潜在変数を収集
    model.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            zs.append(model.encode(x.to(device)).cpu())
            labels.append(y)
    z = torch.cat(zs)
    labels = torch.cat(labels)

    return z, labels


def plot_latent(z: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """
    潜在変数を 2 次元に落として正解ラベルで色分けした図を保存する関数

    Parameters
    ----------
    z: torch.Tensor
        潜在変数
    labels: torch.Tensor
        正解ラベル
    path: Path
        保存先のパス

    Returns
    ----------
    None
    """
    # 2 次元の図にできるのは 2 次元以上の場合だけ
    if z.shape[1] < 2:
        raise ValueError(
            f"潜在変数は 2 次元以上にしてください: {z.shape[1]} 次元"
        )

    # 2 次元より大きい場合は, 厳密な特異値分解による主成分分析で 2 次元に縮約
    # torch.pca_lowrank は乱数を使う近似なので, 図が実行ごとに変わりうる
    if z.shape[1] > 2:
        centered = z - z.mean(dim=0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        points = centered @ vh[:2].T
        title = "Latent space (PCA)"
    else:
        points = z
        title = "Latent space"

    # 数字ごとに色を変えて散布図を描画
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    scatter = ax.scatter(
        points[:, 0], points[:, 1], c=labels, cmap="tab10", s=3
    )
    ax.legend(*scatter.legend_elements(), title="digit", loc="upper right")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """
    config を読み込んで MNIST のオートエンコーダを学習する関数

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

    # データを用意し, 再構成画像に使う検証用の画像を固定
    train_dataset, valid_dataset = make_train_valid(cfg)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = make_loader(valid_dataset, cfg, shuffle=False)
    fixed_images = torch.stack(
        [valid_dataset[i][0] for i in range(cfg.visualize.n_images)]
    )

    # モデルを用意して学習
    model = AutoEncoder(latent_dim=cfg.model.latent_dim)
    trainer = AETrainer(
        model, experiment, fixed_images, cfg.visualize.every_n_epochs
    )
    history = trainer.fit(train_loader, valid_loader)

    # Early Stopping で止まった場合も残るように, ベストモデルの再構成画像を保存
    trainer.save_recon(trainer.recon_dir / "best.png")

    # ベストモデルで検証用データの潜在空間を図にして保存
    z, labels = encode_all(trainer.model, valid_loader, trainer.device)
    plot_latent(z, labels, experiment.run_dir / "latent.png")

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/loss", "valid/mse"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.4f}")


if __name__ == "__main__":
    main()
