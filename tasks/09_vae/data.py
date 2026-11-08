from pathlib import Path

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# このタスクのディレクトリ (データの保存先の相対パスの基準)
TASK_DIR = Path(__file__).resolve().parent


def get_root(cfg: DictConfig) -> Path:
    """
    データの保存先のディレクトリを取得する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    root: Path
        データの保存先, 相対パスの場合はタスクのディレクトリから辿る
    """
    # 相対パスはタスクのディレクトリを基準に解決
    root = Path(cfg.data.root).expanduser()
    if not root.is_absolute():
        root = (TASK_DIR / root).resolve()

    return root


def make_train_valid(cfg: DictConfig) -> tuple[Subset, Subset]:
    """
    MNIST の学習用データを学習用と検証用に分ける関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: Subset
        学習用のデータセット
    valid_dataset: Subset
        検証用のデータセット
    """
    # 画素値を 0 から 1 のテンソルにして読み込み
    full = datasets.MNIST(
        get_root(cfg), train=True, download=True,
        transform=transforms.ToTensor()
    )

    # 全体の乱数に影響しないように専用の乱数生成器で並べ替えて分割
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    order = torch.randperm(len(full), generator=generator).tolist()
    valid_dataset = Subset(full, order[:cfg.data.valid_size])
    train_dataset = Subset(full, order[cfg.data.valid_size:])

    return train_dataset, valid_dataset


def make_loader(
    dataset: Dataset, cfg: DictConfig, shuffle: bool
) -> DataLoader:
    """
    データセットから DataLoader を作る関数

    Parameters
    ----------
    dataset: Dataset
        バッチにするデータセット
    cfg: DictConfig
        学習に使う config
    shuffle: bool
        エポックごとに並べ替えるかどうか

    Returns
    ----------
    loader: DataLoader
        作成した DataLoader
    """
    # ワーカーを使う場合はエポックごとに作り直さないように保持
    num_workers = cfg.data.num_workers
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=shuffle,
        num_workers=num_workers, persistent_workers=num_workers > 0
    )

    return loader
