import argparse
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn, optim

from data import CATEGORIES, make_datasets, make_loader
from ml_found import Experiment, Trainer, load_config, start_experiment
from model import FastTextClassifier

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 指標の名前に使う分野の短い名前 (CATEGORIES と同じ順)
SHORT_NAMES = ["qph", "ml", "oc"]


class MultiLabelTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(MultiLabelTrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # 分野ごとに独立した 2 値分類の損失関数を定義
        self.bce = nn.BCEWithLogitsLoss()

        # エポックごとの予測と正解の入れ物を用意
        self.preds: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの学習または検証を始める前に入れ物を空にする関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        None
        """
        # 予測と正解の入れ物を空に初期化
        self.preds = []
        self.labels = []

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失と完全一致率を求め, 予測を貯める関数

        Parameters
        ----------
        batch: Any
            input_ids, attention_mask, labels の辞書
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            二値交差エントロピー損失と, 全ての分野が当たった割合
        """
        # 予測して損失の計算
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.bce(logits, batch["labels"])

        # 0.5 を境に分野ごとの予測を決め, エポック全体の指標のために保存
        pred = (logits.detach() > 0).float()
        self.preds.append(pred.cpu())
        self.labels.append(batch["labels"].cpu())

        # 全ての分野の予測が当たった割合を損失と集約
        exact = (pred == batch["labels"]).all(dim=1).float().mean()
        outputs = {"loss": loss, "exact": exact}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の予測から分野ごとと平均の F1 を求める関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            分野ごとの F1, その単純平均 (macro_f1), 全体で数えた F1 (micro_f1)
        """
        # 貯めた予測と正解を結合し, 分野ごとに真陽性, 偽陽性, 偽陰性を集計
        pred = torch.cat(self.preds)
        label = torch.cat(self.labels)
        tp = (pred * label).sum(dim=0)
        fp = (pred * (1 - label)).sum(dim=0)
        fn = ((1 - pred) * label).sum(dim=0)

        # 分野ごとの F1 と, その単純平均, 全体で数えた F1 を計算
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
        metrics = {
            f"f1_{name}": f1[i].item() for i, name in enumerate(SHORT_NAMES)
        }
        metrics["macro_f1"] = f1.mean().item()
        metrics["micro_f1"] = (
            2 * tp.sum()
            / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1.0)
        ).item()

        return metrics


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


def main() -> None:
    """
    config を読み込んで arXiv の論文の分野の分類を学習する関数

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

    # データを用意し, 件数と分野ごとの割合を記録
    train_dataset, valid_dataset, vocab = make_datasets(cfg)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    valid_loader = make_loader(valid_dataset, cfg, shuffle=False)
    for name, dataset in (("学習用", train_dataset), ("検証用", valid_dataset)):
        ratio = torch.tensor(dataset.targets).mean(dim=0).tolist()
        ratio_str = ", ".join(
            f"{c} {r:.3f}" for c, r in zip(CATEGORIES, ratio)
        )
        experiment.logger.info(
            f"{name}データ: {len(dataset)} 件 (分野の割合: {ratio_str})"
        )
    experiment.logger.info(f"語彙の大きさ: {len(vocab)}")

    # モデルを用意して学習
    model = FastTextClassifier(
        vocab_size=len(vocab), embed_dim=cfg.model.embed_dim,
        num_labels=len(CATEGORIES), dropout=cfg.model.dropout
    )
    trainer = MultiLabelTrainer(model, experiment)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    keys = ["valid/macro_f1", "valid/micro_f1", "valid/exact"]
    keys += [f"valid/f1_{name}" for name in SHORT_NAMES]
    for key in keys:
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.4f}")


if __name__ == "__main__":
    main()
