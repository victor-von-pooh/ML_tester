import argparse
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn, optim
from torchvision.utils import save_image

from data import make_loader, make_train
from ml_found import Experiment, Trainer, load_config, start_experiment
from ml_found.training import build_optimizer, load_checkpoint
from model import GAN

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 試す書き方
VARIANTS = ("A", "B", "C")


class GANTrainer(Trainer):
    def __init__(
        self, model: GAN, experiment: Experiment, variant: str,
        fixed_noise: torch.Tensor, sample_every: int,
        optimizer: Optional[optim.Optimizer] = None,
        d_optimizer: Optional[optim.Optimizer] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(GANTrainer, self).__init__(model, experiment, None, optimizer)

        # 書き方と, 識別器用の 2 つ目の Optimizer を保存
        self.variant = variant
        self.d_optimizer = d_optimizer

        # 毎回同じノイズから生成した画像を比べられるように, ノイズを保存
        self.fixed_noise = fixed_noise.to(self.device)

        # 生成画像を保存する間隔と保存先を保存
        self.sample_every = sample_every
        self.sample_dir = experiment.run_dir / "samples"
        self.sample_dir.mkdir()

        # 損失関数の定義
        self.bce = nn.BCEWithLogitsLoss()

        # B で update_parameters に渡す識別器の損失の入れ物を用意
        self.d_loss: Optional[torch.Tensor] = None

    def discriminator_loss(
        self, x: torch.Tensor, fake: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        識別器の損失を求める関数

        Parameters
        ----------
        x: torch.Tensor
            本物の画像
        fake: torch.Tensor
            生成した画像

        Returns
        ----------
        d_loss: torch.Tensor
            本物を 1, 生成画像を 0 と判定させる損失
        d_real: torch.Tensor
            本物の画像を本物と判定した確率の平均
        d_fake: torch.Tensor
            生成画像を本物と判定した確率の平均
        """
        # 本物と生成画像のロジットの計算
        real_logits = self.model.discriminator(x)
        fake_logits = self.model.discriminator(fake)

        # 本物は 1, 生成画像は 0 を正解として損失を計算
        d_loss = (
            self.bce(real_logits, torch.ones_like(real_logits))
            + self.bce(fake_logits, torch.zeros_like(fake_logits))
        )
        d_real = torch.sigmoid(real_logits).mean().detach()
        d_fake = torch.sigmoid(fake_logits).mean().detach()

        return d_loss, d_real, d_fake

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """
        生成器の損失を求める関数

        Parameters
        ----------
        fake: torch.Tensor
            生成した画像

        Returns
        ----------
        g_loss: torch.Tensor
            生成画像を識別器に本物と判定させる損失
        """
        # 生成画像を 1 と判定させる損失を計算
        fake_logits = self.model.discriminator(fake)
        g_loss = self.bce(fake_logits, torch.ones_like(fake_logits))

        return g_loss

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の生成器と識別器の損失を求める関数

        Parameters
        ----------
        batch: Any
            画像と正解ラベル, 正解ラベルは使わない
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            ML-Found が逆伝播する損失と, 識別器と生成器の損失,
            識別器の判定の確率
        """
        # バッチと同じ数のノイズから画像を生成
        x, _ = batch
        z = torch.randn(len(x), self.model.noise_dim, device=self.device)
        fake = self.model(z)

        # A: 2 つの損失の和を 1 つの Optimizer で最小化 (敵対的にならない)
        if self.variant == "A":
            d_loss, d_real, d_fake = self.discriminator_loss(x, fake)
            g_loss = self.generator_loss(fake)
            loss = d_loss + g_loss

        # B: 識別器の損失を保存し, update_parameters で 2 つを順に更新
        elif self.variant == "B":
            d_loss, d_real, d_fake = self.discriminator_loss(
                x, fake.detach()
            )
            g_loss = self.generator_loss(fake)
            self.d_loss = d_loss
            loss = g_loss

        # C: step の中で識別器と生成器を手動で更新し, ML-Found には
        # 勾配がパラメータに流れない損失を返却
        # ML-Found の update_parameters の zero_grad で勾配が None になるので,
        # その後の optimizer.step では何も更新されない
        else:
            d_loss, d_real, d_fake = self.discriminator_loss(
                x, fake.detach()
            )
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
            g_loss = self.generator_loss(fake)
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()
            loss = g_loss.detach().requires_grad_(True)

        # 損失と判定の確率を集約
        outputs = {
            "loss": loss, "d_loss": d_loss.detach(),
            "g_loss": g_loss.detach(), "d_real": d_real, "d_fake": d_fake
        }

        return outputs

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        生成器と識別器のパラメータを更新する関数

        Parameters
        ----------
        loss: torch.Tensor
            step が返した損失, B の場合は生成器の損失

        Returns
        ----------
        None
        """
        # A と C は ML-Found の更新手順のまま
        if self.variant != "B":
            super(GANTrainer, self).update_parameters(loss)
            return

        # B: 生成器を ML-Found の手順で更新
        # 識別器を先に更新すると, 生成器の損失の計算グラフの中の識別器の重みが
        # 書き換わって逆伝播できないので, 生成器を先に更新
        super(GANTrainer, self).update_parameters(loss)

        # 生成器の逆伝播で識別器にも溜まった勾配を消してから識別器を更新
        self.d_optimizer.zero_grad()
        self.d_loss.backward()
        self.d_optimizer.step()

        # 次のバッチまで識別器の計算グラフを残さないように解放
        self.d_loss = None

    def on_epoch_end(self, phase: str) -> dict:
        """
        決まった間隔で固定したノイズから画像を生成して保存する関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            記録する指標は無いので空の辞書
        """
        # 最初と決まった間隔のエポックと最後のエポックだけ保存
        metrics = {}
        is_last = self.epoch == self.trainer_cfg.epochs
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

    def state_dict(self, epoch: int, monitor: str) -> dict:
        """
        チェックポイントに識別器の Optimizer の状態も加える関数

        Parameters
        ----------
        epoch: int
            今回のエポック
        monitor: str
            ベストモデルの判定に使う指標の名前

        Returns
        ----------
        state: dict
            ML-Found の状態と, 識別器の Optimizer の状態
        """
        # ML-Found が保存する状態に 2 つ目の Optimizer の状態を追加
        state = super(GANTrainer, self).state_dict(epoch, monitor)
        if self.d_optimizer is not None:
            state["d_optimizer"] = self.d_optimizer.state_dict()

        return state

    def resume(self, path: str, monitor: str) -> None:
        """
        チェックポイントから識別器の Optimizer の状態も戻す関数

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
        super(GANTrainer, self).resume(path, monitor)

        # ML-Found は読み込んだ状態を外に出さないので,
        # 同じ規則でパスを解決して再読み込み
        if self.d_optimizer is None:
            return
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.experiment.project_dir / checkpoint_path
        state = load_checkpoint(str(checkpoint_path), self.device)
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.logger.info("識別器の Optimizer の状態を復元")


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


def main() -> None:
    """
    config を読み込んで MNIST の GAN を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで書き方の確認
    args = parse_args()
    cfg, diff = load_config(
        str(TASK_DIR / "configs/default/train.yaml"),
        str(resolve_experiment_path(args.experiment))
    )
    variant = cfg.train.variant
    if variant not in VARIANTS:
        raise ValueError(
            f"train.variant は A, B, C のどれかにしてください: {variant}"
        )

    # 実験を開始してデータを用意
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    train_loader = make_loader(make_train(cfg), cfg, shuffle=True)

    # モデルと, 生成画像に使う固定のノイズを用意
    model = GAN(
        noise_dim=cfg.model.noise_dim, g_channels=cfg.model.g_channels,
        d_channels=cfg.model.d_channels
    )
    fixed_noise = torch.randn(cfg.visualize.n_images, cfg.model.noise_dim)

    # A は全てのパラメータを ML-Found の Optimizer 1 つで更新
    # B と C は生成器を ML-Found の Optimizer で, 識別器を 2 つ目で更新
    optimizer = None
    d_optimizer = None
    if variant != "A":
        optimizer = build_optimizer(
            cfg.optimizer, model.generator.parameters()
        )
        d_optimizer = build_optimizer(
            cfg.discriminator_optimizer, model.discriminator.parameters()
        )

    # 検証用データは使わずに学習
    trainer = GANTrainer(
        model, experiment, variant, fixed_noise,
        cfg.visualize.sample_every, optimizer, d_optimizer
    )
    history = trainer.fit(train_loader)

    # ML-Found が戻したベストモデルで生成
    run_dir = experiment.run_dir
    trainer.model.eval()
    with torch.no_grad():
        best_images = trainer.model(trainer.fixed_noise).cpu()
    save_image(best_images * 0.5 + 0.5, run_dir / "best_model.png", nrow=8)

    # 最後のエポックの重みは last.pt にしか残らないので, ある場合は読み込んで生成
    if cfg.trainer.checkpoint.save_last:
        last_state = load_checkpoint(
            str(run_dir / "last.pt"), trainer.device
        )
        trainer.model.load_state_dict(last_state["model"])
        with torch.no_grad():
            last_images = trainer.model(trainer.fixed_noise).cpu()
        save_image(
            last_images * 0.5 + 0.5, run_dir / "last_model.png", nrow=8
        )

    # 結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {run_dir}")
    print(f"書き方: {variant}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    keys = ("train/d_loss", "train/g_loss", "train/d_real", "train/d_fake")
    for key in keys:
        print(f"最後のエポックの {key}: {history.get(key)[-1]:.4f}")


if __name__ == "__main__":
    main()
