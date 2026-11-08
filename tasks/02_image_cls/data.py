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


def split_indices(
    n: int, valid_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """
    学習用と検証用に分ける番号を決める関数

    Parameters
    ----------
    n: int
        学習用データ全体のサンプル数
    valid_size: int
        検証用に回すサンプル数
    seed: int
        分割に使う乱数シード

    Returns
    ----------
    train_indices: list[int]
        学習用のサンプルの番号
    valid_indices: list[int]
        検証用のサンプルの番号
    """
    # 全体の乱数に影響しないように専用の乱数生成器で並べ替え
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()

    # 先頭を検証用, 残りを学習用に分割
    valid_indices = order[:valid_size]
    train_indices = order[valid_size:]

    return train_indices, valid_indices


def compute_stats(
    dataset: datasets.FashionMNIST, indices: list[int]
) -> tuple[float, float]:
    """
    学習用データの画素値の平均と標準偏差を求める関数

    Parameters
    ----------
    dataset: datasets.FashionMNIST
        学習用データ全体
    indices: list[int]
        学習用のサンプルの番号

    Returns
    ----------
    mean: float
        画素値の平均, 0 から 1 の範囲での値
    std: float
        画素値の標準偏差, 0 から 1 の範囲での値
    """
    # 検証用の情報が混ざらないように学習用のサンプルだけで計算
    pixels = dataset.data[indices].float() / 255.0
    mean = pixels.mean().item()
    std = pixels.std().item()

    return mean, std


def make_transform(mean: float, std: float) -> transforms.Compose:
    """
    画像をテンソルにして標準化する変換を作る関数

    Parameters
    ----------
    mean: float
        画素値の平均
    std: float
        画素値の標準偏差

    Returns
    ----------
    transform: transforms.Compose
        テンソルへの変換と標準化
    """
    # 0 から 1 のテンソルにしてから標準化
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )

    return transform


def make_train_valid(
    cfg: DictConfig
) -> tuple[Subset, Subset, tuple[float, float]]:
    """
    学習用と検証用のデータセットを作る関数

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
    stats: tuple[float, float]
        学習用データの画素値の平均と標準偏差
    """
    # 学習用データ全体を読み込み, 学習用と検証用に分割
    full = datasets.FashionMNIST(get_root(cfg), train=True, download=True)
    train_indices, valid_indices = split_indices(
        len(full), cfg.data.valid_size, cfg.experiment.seed
    )

    # 学習用のサンプルで標準化の基準を決め, 両方に同じ変換を適用
    stats = compute_stats(full, train_indices)
    full.transform = make_transform(*stats)
    train_dataset = Subset(full, train_indices)
    valid_dataset = Subset(full, valid_indices)

    return train_dataset, valid_dataset, stats


def make_test(
    cfg: DictConfig, stats: tuple[float, float]
) -> datasets.FashionMNIST:
    """
    テスト用のデータセットを作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使った config
    stats: tuple[float, float]
        学習用データの画素値の平均と標準偏差

    Returns
    ----------
    test_dataset: datasets.FashionMNIST
        学習用データの基準で標準化したテスト用のデータセット
    """
    # 学習用データと同じ基準で標準化
    test_dataset = datasets.FashionMNIST(
        get_root(cfg), train=False, download=True,
        transform=make_transform(*stats)
    )

    return test_dataset


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
