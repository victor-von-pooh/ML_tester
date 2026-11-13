from pathlib import Path

from torch import nn

from data import make_loader, make_train_valid
from ml_found import Trainer, load_project_config, start_experiment
from model import FashionCNN

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


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
    cfg, diff = load_project_config(str(TASK_DIR))

    # データを用意 (OneCycleLR の steps_per_epoch は, ML-Found が学習を始める
    # ときにバッチ数から補う)
    train_dataset, valid_dataset, stats = make_train_valid(cfg)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = make_loader(valid_dataset, cfg, shuffle=False)

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
