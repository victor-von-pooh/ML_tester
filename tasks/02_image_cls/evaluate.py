import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import make_loader, make_test, make_train_valid
from ml_found.utils import get_device, move_to_device
from model import FashionCNN

# このタスクのディレクトリ (出力先の相対パスの基準)
TASK_DIR = Path(__file__).resolve().parent

# Fashion-MNIST のクラス名
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
    "Sneaker", "Bag", "Ankle boot"
]


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    args: argparse.Namespace
        コマンドライン引数, 評価する実験の出力先を持つ
    """
    # 評価する実験の出力先の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        help=(
            "実験の出力先 (outputs/fashion_mnist/<日付>/<時刻>), "
            "相対パスはタスクのディレクトリから辿り, 無い場合は実行した場所から辿る"
        )
    )
    args = parser.parse_args()

    return args


def resolve_run_dir(path: str) -> Path:
    """
    実験の出力先のパスを解決する関数

    Parameters
    ----------
    path: str
        コマンドライン引数で指定されたパス

    Returns
    ----------
    run_dir: Path
        実験の出力先の絶対パス
    """
    # train.py の --experiment と同じく, 絶対パス, タスクのディレクトリ,
    # 実行した場所の順に探索
    given = Path(path).expanduser()
    if given.is_absolute():
        run_dir = given
    else:
        run_dir = (TASK_DIR / given).resolve()
        if not run_dir.exists() and given.exists():
            run_dir = given.resolve()

    # 出力先に必要なファイルがあることの確認
    for name in ("config.yaml", "final_model.pt"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{name} がありません: {run_dir}")

    return run_dir


def load_model(cfg: DictConfig, run_dir: Path) -> nn.Module:
    """
    学習したモデルを読み込む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使った config
    run_dir: Path
        実験の出力先

    Returns
    ----------
    model: nn.Module
        ベストモデルの重みを読み込んだモデル
    """
    # 学習時と同じ構造のモデルを作り, ベストモデルの重みを読み込み
    # 学習時のデバイス (mps など) に関わらず読めるように cpu に展開
    model = FashionCNN(
        in_channels=1, image_size=28, channels=list(cfg.model.channels),
        hidden_dim=cfg.model.hidden_dim, num_classes=10
    )
    state = torch.load(
        run_dir / "final_model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)

    return model


def evaluate(
    model: nn.Module, loader: DataLoader,
    device: torch.device
) -> tuple[float, list[float]]:
    """
    テスト用データで正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        評価するモデル
    loader: DataLoader
        テスト用の DataLoader
    device: torch.device
        推論に使うデバイス

    Returns
    ----------
    acc: float
        全体の正解率
    class_acc: list[float]
        クラスごとの正解率
    """
    # 推論モードにして予測を収集
    model = model.to(device)
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x, y = move_to_device(batch, device)
            preds.append(model(x).argmax(dim=1).cpu())
            labels.append(y.cpu())
    pred = torch.cat(preds)
    label = torch.cat(labels)

    # 全体とクラスごとの正解率の計算
    correct = (pred == label).float()
    acc = correct.mean().item()
    class_acc = [
        correct[label == c].mean().item() for c in range(len(CLASS_NAMES))
    ]

    return acc, class_acc


def main() -> None:
    """
    学習したモデルをテスト用データで評価する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 学習に使った config とモデルの読み込み
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    model = load_model(cfg, run_dir)

    # 学習時と同じ基準で標準化したテスト用データを用意
    _, _, stats = make_train_valid(cfg)
    test_loader = make_loader(make_test(cfg, stats), cfg, shuffle=False)

    # 評価して表示
    device = get_device(cfg.experiment.device)
    acc, class_acc = evaluate(model, test_loader, device)
    print(f"評価した実験: {run_dir}")
    print(f"テスト用データの正解率: {acc:.4f}")
    for name, value in zip(CLASS_NAMES, class_acc):
        print(f"\t{name}: {value:.4f}")


if __name__ == "__main__":
    main()
