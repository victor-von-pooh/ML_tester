from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.utils import save_image

from data import make_loader, make_train_valid
from ml_found import Experiment, Trainer, load_project_config, start_experiment
from ml_found.training import load_checkpoint
from model import Diffusion, UNet

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class DiffusionTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, diffusion: Diffusion,
        ema_decay: Optional[float], image_size: int, n_images: int,
        sample_every: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(DiffusionTrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # ノイズの加え方と生成の手順を保存
        self.diffusion = diffusion

        # EMA を使う場合は重みの指数移動平均を持つモデルを作成
        self.ema = None
        if ema_decay is not None:
            self.ema = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
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
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズを加えた画像から加えたノイズを予測した誤差を求める関数

        Parameters
        ----------
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
        loss = self.mse(self.model(x_t, t), noise)

        return loss

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分のノイズの予測誤差を求める関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            学習に使う損失と, 毎回違う乱数で求めた損失
        """
        # 毎回違う乱数で時刻とノイズを選んだ損失
        x0, _ = batch
        n = len(x0)
        t = torch.randint(
            0, self.diffusion.timesteps, (n,), device=self.device
        )
        noise = torch.randn_like(x0)
        loss_random = self.noise_loss(x0, t, noise)

        # 学習時はそのまま使い, 検証時は毎エポック同じ乱数で選び直した損失を使用
        if phase == "train":
            loss = loss_random
        else:
            t = torch.randint(
                0, self.diffusion.timesteps, (n,),
                generator=self.valid_generator
            ).to(self.device)
            noise = torch.randn(
                x0.shape, generator=self.valid_generator
            ).to(self.device)
            loss = self.noise_loss(x0, t, noise)

        outputs = {"loss": loss, "loss_random": loss_random.detach()}

        return outputs

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        パラメータを更新した後に EMA の重みも更新する関数

        Parameters
        ----------
        loss: torch.Tensor
            1 バッチ分の損失

        Returns
        ----------
        None
        """
        # ML-Found の手順で更新した後に, 重みの指数移動平均を更新
        super(DiffusionTrainer, self).update_parameters(loss)
        if self.ema is not None:
            self.ema.update_parameters(self.model)

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
        sampler = self.model if self.ema is None else self.ema.module
        sampler.eval()
        images = self.diffusion.sample(
            sampler, self.n_images, self.image_size
        )
        path = self.sample_dir / f"epoch_{self.epoch:03d}.png"
        save_image(images.clamp(-1, 1) * 0.5 + 0.5, path, nrow=8)
        self.logger.info(f"生成画像を保存: {path.name}")

        return metrics


    def state_dict(self, epoch: int, monitor: str) -> dict:
        """
        チェックポイントに EMA の状態も加える関数

        Parameters
        ----------
        epoch: int
            今回のエポック
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        state: dict
            ML-Found の状態と, EMA の重みと更新回数
        """
        # ML-Found が保存する状態に EMA の状態を追加
        state = super(DiffusionTrainer, self).state_dict(epoch, monitor)
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()

        return state

    def resume(self, path: str, monitor: str) -> None:
        """
        チェックポイントから EMA の状態も戻す関数

        Parameters
        ----------
        path: str
            チェックポイントのパス, 相対パスはプロジェクトのディレクトリから辿る
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        None
        """
        # ML-Found の状態の復元
        super(DiffusionTrainer, self).resume(path, monitor)

        # ML-Found は読み込んだ状態を外に出さないので,
        # 同じ規則でパスを解決して再読み込み
        if self.ema is None:
            return
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.experiment.project_dir / checkpoint_path
        state = load_checkpoint(str(checkpoint_path), self.device)
        self.ema.load_state_dict(state["ema"])
        self.logger.info("EMA の状態を復元")


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
    ema_decay = cfg.ema.decay if cfg.ema.use else None
    trainer = DiffusionTrainer(
        model, experiment, diffusion, ema_decay, cfg.data.image_size,
        cfg.sampling.n_images, cfg.sampling.every_n_epochs
    )
    history = trainer.fit(train_loader, valid_loader)

    # ML-Found が戻したベストモデルの重みで生成して保存
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
    if trainer.ema is not None and trainer.ema.n_averaged.item() > 0:
        ema_model = trainer.ema.module
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
