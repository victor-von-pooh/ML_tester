from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
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


def make_train(cfg: DictConfig) -> datasets.MNIST:
    """
    MNIST の学習用データを読み込む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: datasets.MNIST
        画素値を -1 から 1 にした学習用データ全体
    """
    # 生成器の出力 (Tanh) に合わせて画素値を -1 から 1 に変換
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        get_root(cfg), train=True, download=True, transform=transform
    )

    return train_dataset


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
