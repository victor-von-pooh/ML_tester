from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from model import ShapeCNN

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
        正規化した画像とクラスの番号
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


def new_image() -> np.ndarray:
    """
    学習に使っていない大きさ (100x80) の灰色の三角の画像を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    image: np.ndarray
        (高さ, 幅) の uint8 の配列 (ML-EDA の読み込みと同じ形)
    """
    # 暗い背景に明るい三角
    image = Image.new("L", (100, 80), 30)
    ImageDraw.Draw(image).polygon([(50, 15), (22, 65), (78, 65)], fill=220)
    array = np.asarray(image)

    return array


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment
) -> None:
    """
    テスト用の分割の正解率と, 新しい画像への前処理と予測を確かめる関数

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

    # 大きさも色の形式も違う新しい画像に, 書き出したときと同じ中身の手順
    # (rgb, resize, 学習用の平均と標準偏差での正規化) をかけて予測
    x = torch.from_numpy(dataset.transform_item(new_image())).unsqueeze(0)
    with torch.no_grad():
        probability = model(x).softmax(dim=1)[0]
    best = int(probability.argmax())
    experiment.logger.info(
        f"新しい画像の予測: {dataset.info['classes'][best]} "
        f"({probability[best].item():.3f})"
    )


def main() -> None:
    """
    ML-EDA で書き出した画像のデータセットで図形の分類を学習する関数

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
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    shape = dataset.info["splits"]["train"]["shape"]
    experiment.logger.info(
        f"データセット: {dataset.info['name']}, 学習用の形 {shape}, "
        f"クラス {dataset.info['classes']}"
    )

    # 分割ごとのテンソルを DataLoader にする
    train_loader = DataLoader(
        TensorDataset(*dataset.tensors("train")),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(*dataset.tensors("valid")),
        batch_size=cfg.data.batch_size
    )

    # 入力のチャンネル数とクラスの数はデータセットの説明から
    model = ShapeCNN(
        in_channels=shape[1], channels=cfg.model.channels,
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
