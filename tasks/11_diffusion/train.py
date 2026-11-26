from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn, optim
from torchvision.utils import save_image

from data import make_loader, make_train_valid
from ml_found import Experiment, Trainer, load_project_config, start_experiment
from model import Diffusion, UNet

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class DiffusionTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, diffusion: Diffusion,
        ema_eval: bool, image_size: int, n_images: int, sample_every: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(DiffusionTrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # ノイズの加え方と生成の手順を保存
        self.diffusion = diffusion

        # 検証とベストモデルの判定に EMA の重みを使うかを保存
        # (EMA は trainer.ema.decay の設定で ML-Found が持つ)
        self.ema_eval = ema_eval
        if ema_eval and self.ema_model is None:
            self.logger.warning(
                "trainer.ema.decay が null なので, ema.eval: true は無視します"
            )

        # 生成画像の大きさ, 枚数, 保存する間隔, 保存先を保存
        self.image_size = image_size
        self.n_images = n_images
        self.sample_every = sample_every
        self.sample_dir = experiment.run_dir / "samples"
        self.sample_dir.mkdir()

        # 損失関数の定義
        self.mse = nn.MSELoss()

        # 検証では毎エポック同じ時刻とノイズを使うための乱数生成器を用意
        self.valid_seed = experiment.cfg.experiment.seed
        self.valid_generator = torch.Generator()

    def on_epoch_start(self, phase: str) -> None:
        """
        検証の前に, 検証用の乱数生成器を同じ状態に戻す関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        None
        """
        # 検証の損失を毎エポック同じ時刻とノイズで比べられるように初期化
        if phase == "valid":
            self.valid_generator.manual_seed(self.valid_seed)

    def noise_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズを加えた画像から加えたノイズを予測した誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            ノイズを予測するモデル
        x0: torch.Tensor
            元の画像
        t: torch.Tensor
            時刻
        noise: torch.Tensor
            加えるノイズ

        Returns
        ----------
        loss: torch.Tensor
            予測したノイズと加えたノイズの平均二乗誤差
        """
        # ノイズを加えてから予測
        x_t = self.diffusion.add_noise(x0, t, noise)
        loss = self.mse(model(x_t, t), noise)

        return loss

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分のノイズの予測誤差を求める関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train, valid, 学習用データの評価し直しでは train_eval

        Returns
        ----------
        outputs: dict
            学習に使う損失と, 毎回違う乱数で求めた損失
        """
        # 学習時は学習するモデル, 検証時は評価に使うモデル (EMA など) で予測
        model = self.model if phase == "train" else self.eval_model()

        # 毎回違う乱数で時刻とノイズを選んだ損失
        x0, _ = batch
        n = len(x0)
        t = torch.randint(
            0, self.diffusion.timesteps, (n,), device=self.device
        )
        noise = torch.randn_like(x0)
        loss_random = self.noise_loss(model, x0, t, noise)

        # 検証時だけ毎エポック同じ乱数で選び直した損失を使用
        # (学習時と, 学習用データの評価し直しではそのまま)
        if phase != "valid":
            loss = loss_random
        else:
            t = torch.randint(
                0, self.diffusion.timesteps, (n,),
                generator=self.valid_generator
            ).to(self.device)
            noise = torch.randn(
                x0.shape, generator=self.valid_generator
            ).to(self.device)
            loss = self.noise_loss(model, x0, t, noise)

        outputs = {"loss": loss, "loss_random": loss_random.detach()}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        決まった間隔で画像を生成して保存する関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            記録する指標は無いので空の辞書
        """
        # 検証の後に, 決まった間隔のエポックと最後のエポックだけ生成
        metrics = {}
        is_last = self.epoch == self.trainer_cfg.epochs
        is_interval = self.epoch % self.sample_every == 0
        if phase != "valid" or not (is_interval or is_last):
            return metrics

        # EMA がある場合は EMA の重みで生成
        sampler = (
            self.model if self.ema_model is None else self.ema_model.module
        )
        sampler.eval()
        images = self.diffusion.sample(
            sampler, self.n_images, self.image_size
        )
        path = self.sample_dir / f"epoch_{self.epoch:03d}.png"
        save_image(images.clamp(-1, 1) * 0.5 + 0.5, path, nrow=8)
        self.logger.info(f"生成画像を保存: {path.name}")

        return metrics

    def eval_model(self) -> nn.Module:
        """
        検証, ベストモデルの判定, final_model.pt に使うモデルを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            ema.eval が true なら ML-Found の既定 (EMA を使う場合は EMA の
            重みのモデル), そうでなければ学習するモデル
        """
        # ema.eval が false なら EMA を持っていても学習するモデルを使用
        if self.ema_eval:
            model = super(DiffusionTrainer, self).eval_model()
        else:
            model = self.model

        return model


def main() -> None:
    """
    config を読み込んで MNIST の拡散モデルを学習する関数

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

    # モデルとノイズの加え方を用意して学習
    model = UNet(
        base_channels=cfg.model.base_channels,
        channel_mults=list(cfg.model.channel_mults)
    )
    diffusion = Diffusion(
        cfg.diffusion.timesteps, cfg.diffusion.beta_start,
        cfg.diffusion.beta_end, experiment.device
    )
    trainer = DiffusionTrainer(
        model, experiment, diffusion, cfg.ema.eval,
        cfg.data.image_size, cfg.sampling.n_images,
        cfg.sampling.every_n_epochs
    )
    history = trainer.fit(train_loader, valid_loader)

    # ML-Found が戻したベストモデルの重みで生成して保存
    # ema.eval が true の場合は EMA のベストモデルの重み
    run_dir = experiment.run_dir
    trainer.model.eval()
    images = diffusion.sample(
        trainer.model, cfg.sampling.n_images, cfg.data.image_size
    )
    save_image(
        images.clamp(-1, 1) * 0.5 + 0.5, run_dir / "best_model.png", nrow=8
    )

    # EMA を使った場合は, 最後の EMA の重みを保存して生成
    # 一度も更新されていない場合は初期の重みなので保存しない
    ema = trainer.ema_model
    if ema is not None and ema.n_averaged.item() > 0:
        ema_model = ema.module
        torch.save(ema_model.state_dict(), run_dir / "ema_model.pt")
        ema_model.eval()
        images = diffusion.sample(
            ema_model, cfg.sampling.n_images, cfg.data.image_size
        )
        save_image(
            images.clamp(-1, 1) * 0.5 + 0.5, run_dir / "ema_model.png",
            nrow=8
        )

    # 結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/loss", "valid/loss_random"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.5f}")


if __name__ == "__main__":
    main()
