import argparse
from pathlib import Path

from omegaconf import DictConfig
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Trainer, load_config, start_experiment
from model import IrisClassifier

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


def make_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Iris データセットから学習用と検証用の DataLoader を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_loader: DataLoader
        学習用の DataLoader
    valid_loader: DataLoader
        検証用の DataLoader
    """
    # データを読み込み, クラスの比率を保って学習用と検証用に分割
    x, y = load_iris(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, stratify=y,
        random_state=cfg.experiment.seed
    )

    # 学習用データで標準化の基準を決めて両方に適用
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_valid = ss.transform(x_valid)

    # テンソルにして DataLoader に集約
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )

    return train_loader, valid_loader


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        標準化した特徴量と正解ラベル
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
    config を読み込んで Iris の分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    args = parse_args()
    cfg, diff = load_config(
        str(TASK_DIR / "configs/default/train.yaml"),
        str(resolve_experiment_path(args.experiment))
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データとモデルを用意
    train_loader, valid_loader = make_loaders(cfg)
    model = IrisClassifier(
        input_dim=4, hidden_dim=cfg.model.hidden_dim, num_classes=3
    )

    # 学習
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
