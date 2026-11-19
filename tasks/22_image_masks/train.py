from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_eda import ItemSplit, load_items
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from model import TinyUNet

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class MaskDataset(Dataset):
    def __init__(self, items: ItemSplit, lookup: np.ndarray):
        # 親クラスのコンストラクタを呼び出す
        super(MaskDataset, self).__init__()

        # ML-EDA の分割の件の並びと, 領域の値からクラスの番号の表を保存
        self.items = items
        self.lookup = lookup

    def __len__(self) -> int:
        """
        件数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        count: int
            分割の件数
        """
        # ML-EDA の並びの件数
        count = len(self.items)

        return count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1 件の画像と領域をテンソルにする関数

        Parameters
        ----------
        index: int
            分割の中の位置

        Returns
        ----------
        pair: tuple[torch.Tensor, torch.Tensor]
            (チャンネル, 高さ, 幅) の 0 から 1 の画像と, (高さ, 幅) の
            クラスの番号
        """
        # load_items の 1 件は (高さ, 幅, チャンネル) の uint8 と
        # (高さ, 幅) の整数. 画像は書き込めない配列なので写したもの
        item = self.items[index]
        image = torch.from_numpy(np.array(item["image"])).permute(2, 0, 1)
        mask = torch.from_numpy(self.lookup[item["mask"]])
        pair = (image.float() / 255.0, mask.long())

        return pair


class SegmentationTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, names: list[str]
    ):
        # step を上書きするので step_fn は None
        super(SegmentationTrainer, self).__init__(model, experiment, None)

        # クラスの名前, 損失関数と, エポックごとの重なりと和の画素数
        self.names = names
        self.ce = nn.CrossEntropyLoss()
        self.intersection = torch.zeros(len(names))
        self.union = torch.zeros(len(names))

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの始めに画素数の入れ物を空にする関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # クラスごとの重なりと和を 0 に
        self.intersection = torch.zeros(len(self.names))
        self.union = torch.zeros(len(self.names))

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失と画素の正解率を求め, 重なりと和を貯める関数

        Parameters
        ----------
        batch: Any
            画像とクラスの番号
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            画素ごとの交差エントロピー損失と画素の正解率
        """
        # 画素ごとの分類の損失
        x, y = batch
        logits = self.model(x)
        loss = self.ce(logits, y)
        predicted = logits.argmax(dim=1)
        acc = (predicted == y).float().mean()

        # IoU はエポック全体の画素数から求めるので, その貯め置き
        counts = confusion_counts(predicted, y, len(self.names))
        self.intersection += counts[0]
        self.union += counts[1]
        outputs = {"loss": loss, "pixel_acc": acc}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の画素数からクラスごとの IoU を求める関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            背景を除いた IoU の平均 (miou) と, クラスごとの IoU
        """
        # 背景 (番号 0) を除いた平均と, クラスごとの値
        metrics = iou_metrics(self.intersection, self.union, self.names)

        return metrics


def confusion_counts(
    predicted: torch.Tensor, target: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    クラスごとの重なりと和の画素数を数える関数

    Parameters
    ----------
    predicted: torch.Tensor
        予測したクラスの番号
    target: torch.Tensor
        正解のクラスの番号
    count: int
        クラス数

    Returns
    ----------
    result: tuple[torch.Tensor, torch.Tensor]
        クラスごとの重なり (予測と正解の両方) と和 (どちらか) の画素数
    """
    # クラスごとの予測と正解の画素の集合の比較
    intersection = torch.zeros(count)
    union = torch.zeros(count)
    for index in range(count):
        guess = predicted == index
        truth = target == index
        intersection[index] = (guess & truth).sum().item()
        union[index] = (guess | truth).sum().item()
    result = (intersection, union)

    return result


def iou_metrics(
    intersection: torch.Tensor, union: torch.Tensor, names: list[str]
) -> dict:
    """
    重なりと和の画素数から IoU の指標を作る関数

    Parameters
    ----------
    intersection: torch.Tensor
        クラスごとの重なりの画素数
    union: torch.Tensor
        クラスごとの和の画素数
    names: list[str]
        クラスの名前 (番号 0 が背景)

    Returns
    ----------
    metrics: dict
        miou (背景を除き, 和が 0 でないクラスの平均) と iou_<名前>
    """
    # 和が 0 のクラス (予測にも正解にも無い) は平均の対象外
    metrics = {}
    values = []
    for index, name in enumerate(names):
        if union[index] == 0:
            continue
        value = (intersection[index] / union[index]).item()
        metrics[f"iou_{name}"] = value
        if index > 0:
            values.append(value)
    metrics["miou"] = float(np.mean(values)) if values else float("nan")

    return metrics


def class_lookup(classes: Optional[list]) -> tuple[np.ndarray, list[str]]:
    """
    説明の classes から, 領域の値をクラスの番号にする表を作る関数

    Parameters
    ----------
    classes: Optional[list]
        dataset.yaml の entries.mask.classes (value と name の並び)

    Returns
    ----------
    result: tuple[np.ndarray, list[str]]
        値 (0 から 255) から番号の表 (知らない値は -1) と, 番号の順の名前
    """
    # 値の小さい順の番号 (背景の 0 が番号 0)
    if not classes:
        raise ValueError("領域の値の名前 (classes) がありません")
    ordered = sorted(classes, key=class_value)
    lookup = np.full(256, -1, dtype=np.int64)
    names = []
    for index, entry in enumerate(ordered):
        lookup[int(entry["value"])] = index
        names.append(str(entry["name"]))
    result = (lookup, names)

    return result


def class_value(entry: dict) -> int:
    """
    classes の 1 つの値を返す関数 (並べ替えの鍵)

    Parameters
    ----------
    entry: dict
        value と name

    Returns
    ----------
    value: int
        領域の値
    """
    # 整数の値
    value = int(entry["value"])

    return value


def evaluate_test(
    trainer: SegmentationTrainer, loader: DataLoader,
    experiment: Experiment
) -> None:
    """
    テスト用の分割の IoU を記録する関数

    Parameters
    ----------
    trainer: SegmentationTrainer
        学習した後の Trainer (ベストモデルの重み)
    loader: DataLoader
        テスト用の DataLoader
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # 学習と同じ数え方で, ベストモデルの予測の重なりと和
    model = trainer.model.cpu().eval()
    count = len(trainer.names)
    intersection, union = torch.zeros(count), torch.zeros(count)
    with torch.no_grad():
        for x, y in loader:
            counts = confusion_counts(model(x).argmax(dim=1), y, count)
            intersection += counts[0]
            union += counts[1]
    metrics = iou_metrics(intersection, union, trainer.names)
    text = ", ".join(f"{key} {value:.4f}" for key, value in metrics.items())
    experiment.logger.info(f"テスト用の IoU: {text}")
    print(f"テスト用の IoU: {text}")


def main() -> None:
    """
    ML-EDA の版 2 で書き出した画像と領域の対で, 領域の分割を学習する関数

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
    dataset = load_items(path)
    info = dataset.info
    if info["target"]["kind"] != "mask":
        print(f"正解の種類が mask ではありません: {info['target']['kind']}")
        sys.exit(1)
    lookup, names = class_lookup(dataset.entries["mask"].get("classes"))
    channels = dataset.entries["image"]["shape"][2]

    # 正解の出どころを実験に残す (指標はこの出どころとの一致)
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    origin = info["target"]["origin"]
    experiment.logger.info(
        f"データセット: {info['name']} (版 {info['format_version']}), "
        f"分割 {info['splits']}, クラス {names}"
    )
    experiment.logger.info(
        f"正解の出どころ: type={origin['type']}, name={origin['name']}, "
        f"settings={origin['settings']}. 指標はこの正解との一致"
    )

    # load_items の分割を torch の Dataset に包む
    loaders = {}
    for name in ("train", "valid", "test"):
        loaders[name] = DataLoader(
            MaskDataset(dataset.split(name), lookup),
            batch_size=cfg.data.batch_size, shuffle=name == "train"
        )

    # チャンネル数とクラス数はデータセットの説明から
    model = TinyUNet(
        in_channels=channels, channels=cfg.model.channels,
        num_classes=len(names)
    )
    trainer = SegmentationTrainer(model, experiment, names)
    history = trainer.fit(loaders["train"], loaders["valid"])

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"正解の出どころ: {origin['type']} ({origin['name']})")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証用の mIoU: "
          f"{history.get('valid/miou')[best_epoch - 1]:.4f}")
    evaluate_test(trainer, loaders["test"], experiment)


if __name__ == "__main__":
    main()
