from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.utils import save_image

from data import make_loader, make_train
from ml_found import Experiment, Trainer, load_project_config, start_experiment
from model import GAN

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class NamedGANTrainer(Trainer):
    def __init__(
        self, model: GAN, experiment: Experiment,
        fixed_noise: torch.Tensor, sample_every: int
    ):
        # 親クラスのコンストラクタを呼び出す
        # 生成器と識別器の Optimizer は, ML-Found が config の optimizers から作成
        super(NamedGANTrainer, self).__init__(model, experiment)

        # 毎回同じノイズから生成した画像を比べられるように, ノイズを保存
        self.fixed_noise = fixed_noise.to(self.device)

        # 生成画像を保存する間隔と保存先を保存
        self.sample_every = sample_every
        self.sample_dir = experiment.run_dir / "samples"
        self.sample_dir.mkdir()

        # 損失関数の定義
        self.bce = nn.BCEWithLogitsLoss()

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の識別器と生成器の損失を, Optimizer の名前ごとに返す関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            Optimizer の名前ごとの損失と, 識別器の判定の確率
        """
        # バッチと同じ数のノイズから画像を生成
        x, _ = batch
        z = torch.randn(len(x), self.model.noise_dim, device=self.device)
        fake = self.model(z)

        # 識別器は本物を 1, 生成画像を 0 と判定させる損失
        # 生成画像は detach して, 識別器の損失から生成器に勾配を流さない
        real_logits = self.model.discriminator(x)
        fake_logits = self.model.discriminator(fake.detach())
        d_loss = (
            self.bce(real_logits, torch.ones_like(real_logits))
            + self.bce(fake_logits, torch.zeros_like(fake_logits))
        )

        # 生成器は生成画像を 1 と判定させる損失
        # (ML-Found は生成器の損失の勾配を生成器のパラメータだけに求める)
        g_logits = self.model.discriminator(fake)
        g_loss = self.bce(g_logits, torch.ones_like(g_logits))

        # Optimizer の名前ごとの損失と, 判定の確率
        outputs = {
            "loss": {"discriminator": d_loss, "generator": g_loss},
            "d_real": torch.sigmoid(real_logits).mean().detach(),
            "d_fake": torch.sigmoid(fake_logits).mean().detach()
        }

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        決まった間隔で固定したノイズから画像を生成して保存する関数

        Parameters
        ----------
        phase: str
            学習時は train

        Returns
        ----------
        metrics: dict
            記録する指標は無いので空の辞書
        """
        # 最初と決まった間隔のエポックと最後のエポックだけ保存
        metrics: dict = {}
        is_last = self.epoch == self.end_epoch
        is_interval = self.epoch % self.sample_every == 0
        if not (self.epoch == 1 or is_interval or is_last):
            return metrics

        # 推論モードで生成し, 画素値を 0 から 1 に戻して保存
        # 学習モードへは次のエポックの run_epoch で復帰
        self.model.eval()
        with torch.no_grad():
            images = self.model(self.fixed_noise).cpu()
        path = self.sample_dir / f"epoch_{self.epoch:03d}.png"
        save_image(images * 0.5 + 0.5, path, nrow=8)

        return metrics


def main() -> None:
    """
    config の optimizers に名前付きで書いた 2 つの Optimizer で GAN を学習する
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始し, データを用意
    cfg, diff = load_project_config(
        str(TASK_DIR), default="configs/default/named.yaml",
        experiment="configs/experiment/named.yaml"
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    train_loader = make_loader(make_train(cfg), cfg, shuffle=True)

    # モデルと, 生成画像に使う固定のノイズを用意
    model = GAN(
        noise_dim=cfg.model.noise_dim, g_channels=cfg.model.g_channels,
        d_channels=cfg.model.d_channels
    )
    fixed_noise = torch.randn(cfg.visualize.n_images, cfg.model.noise_dim)

    # 検証用データは使わずに学習
    trainer = NamedGANTrainer(
        model, experiment, fixed_noise, cfg.visualize.sample_every
    )
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()
