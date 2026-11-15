from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from make_data import place, surface
from model import PointClassifier

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
        正規化した点群とクラスの番号
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # 予測して損失と正解率の計算
    x, y = batch
    logits = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


def new_cloud() -> dict:
    """
    学習に無い点の数 (3000) で, 欠けた点 (NaN) を含む円錐の点群を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    cloud: dict
        points ((3000, 3) の座標) と colors (None). ML-EDA の読み込みと
        同じ形
    """
    # 学習用と同じ作り方 (回転, 拡大, 平行移動) の円錐に欠けを入れる
    rng = np.random.default_rng(123)
    points = place(surface("cone", 3000, rng), rng)
    points[rng.random(3000) < 0.1] = np.nan
    cloud = {"points": points, "colors": None}

    return cloud


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment
) -> None:
    """
    テスト用の分割の正解率と, 新しい点群への前処理と予測を確かめる関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # テスト用の分割の正解率
    model = model.cpu().eval()
    x_test, y_test = dataset.tensors("test")
    with torch.no_grad():
        accuracy = (model(x_test).argmax(dim=1) == y_test).float().mean()
    experiment.logger.info(f"テスト用の正解率: {accuracy.item():.4f}")

    # 点の数も欠けも違う新しい点群に, 書き出したときと同じ中身の手順
    # (有限の点からの sample_points, normalize_cloud) をかけて予測
    array = dataset.transform_item(new_cloud())
    x = torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probability = model(x).softmax(dim=1)[0]
    best = int(probability.argmax())
    experiment.logger.info(
        f"新しい点群の予測: {dataset.info['classes'][best]} "
        f"({probability[best].item():.3f}), 形 {tuple(array.shape)}"
    )


def main() -> None:
    """
    ML-EDA で書き出した点群のデータセットで形の分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config と, 書き出したデータセット (無ければ作り方を案内)
    cfg, diff = load_project_config(str(TASK_DIR))
    path = TASK_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. 先に prepare.py を実行してください")
        sys.exit(1)
    dataset = load_dataset(path)

    # 点群は sample_points で (件数, 点の数, 3) の配列になる前提
    shape = dataset.info["splits"]["train"]["shape"]
    if len(shape) != 3 or shape[2] != 3:
        print(
            f"学習用の形 {shape} は (件数, 点の数, 3) ではありません. "
            "レシピに sample_points を入れて書き出し直してください"
        )
        sys.exit(1)
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    experiment.logger.info(
        f"データセット: {dataset.info['name']}, 学習用の形 {shape}, "
        f"クラス {dataset.info['classes']}"
    )

    # 分割ごとのテンソルの DataLoader
    train_loader = DataLoader(
        TensorDataset(*dataset.tensors("train")),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(*dataset.tensors("valid")),
        batch_size=cfg.data.batch_size
    )

    # クラスの数はデータセットの説明から
    model = PointClassifier(
        channels=cfg.model.channels,
        num_classes=len(dataset.info["classes"])
    )
    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証用の正解率: "
          f"{history.get('valid/acc')[best_epoch - 1]:.4f}")
    evaluate_test(trainer.model, dataset, experiment)


if __name__ == "__main__":
    main()
