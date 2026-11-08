import argparse
from pathlib import Path

from omegaconf import DictConfig
from torch import nn

from data import make_loader, make_train_valid
from ml_found import Trainer, load_config, start_experiment
from model import FashionCNN

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


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


def fill_scheduler_steps(cfg: DictConfig, n_batches: int) -> DictConfig:
    """
    データの量で決まる Scheduler の引数を config に書き込む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    n_batches: int
        1 エポックあたりのバッチ数

    Returns
    ----------
    cfg: DictConfig
        OneCycleLR の場合は steps_per_epoch を書き込んだ config
    """
    # torch.optim.lr_scheduler.OneCycleLR のような完全な名前も対象
    name = cfg.scheduler.name
    if name is None or name.rsplit(".", 1)[-1] != "OneCycleLR":
        return cfg

    # 学習率の山の位置を決めるのに全体のステップ数が必要
    # エポックごとに更新する場合は 1 エポックを 1 ステップとして数え,
    # config に書かれた値があっても実際の更新回数で上書き
    if cfg.scheduler.interval == "step":
        cfg.scheduler.params.steps_per_epoch = n_batches
    else:
        cfg.scheduler.params.steps_per_epoch = 1

    return cfg


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        標準化した画像と正解ラベル
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # 予測して損失の計算
    x, y = batch
    logits = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)

    # 最もスコアの高いクラスを予測として正解率を求め, 損失と集約
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


def main() -> None:
    """
    config を読み込んで Fashion-MNIST の分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config の読み込み
    args = parse_args()
    cfg, diff = load_config(
        str(TASK_DIR / "configs/default/train.yaml"),
        str(resolve_experiment_path(args.experiment))
    )

    # バッチ数が決まらないと Scheduler の引数が決まらないので, 実験の開始前にデータを用意
    train_dataset, valid_dataset, stats = make_train_valid(cfg)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = make_loader(valid_dataset, cfg, shuffle=False)
    cfg = fill_scheduler_steps(cfg, len(train_loader))

    # 実験を開始し, 標準化の基準を記録
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    experiment.logger.info(f"画素値の平均と標準偏差: {stats}")

    # モデルを用意して学習
    model = FashionCNN(
        in_channels=1, image_size=28, channels=list(cfg.model.channels),
        hidden_dim=cfg.model.hidden_dim, num_classes=10
    )
    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    best_acc = history.get("valid/acc")[best_epoch - 1]
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストモデルの検証用データの正解率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
