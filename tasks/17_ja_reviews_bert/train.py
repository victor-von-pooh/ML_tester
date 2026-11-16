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
from ml_inference import make_bundle
from tokens import add_cls, load_model, special_ids

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 文字は全く同じで, どの点が良い (悪い) かだけが違うレビューの組. 文字を
# 数えるだけのモデルは, 組の 2 つに同じ答えしか出せない
PROBES = [
    ("本体は良い、配送は悪い。", "良い"),
    ("配送は良い、本体は悪い。", "悪い"),
    ("本体は悪くない、梱包は満足できない。", "良い"),
    ("梱包は悪くない、本体は満足できない。", "悪い"),
    ("説明書は問題があった、本体は期待通りだ。", "良い"),
    ("説明書は期待通りだ、本体は問題があった。", "悪い"),
]


def step(model: nn.Module, batch: tuple, phase: str) -> dict:
    """
    1 バッチ分の損失と正解率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    batch: tuple
        [CLS] を足した番号の並びとクラスの番号
    phase: str
        train または valid

    Returns
    ----------
    outputs: dict
        交差エントロピー損失と正解率
    """
    # 分類の出力 (logits) で損失と正解率を計算
    x, y = batch
    logits = model(x)
    if isinstance(logits, dict):
        logits = logits["logits"]
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    outputs = {"loss": loss, "acc": acc}

    return outputs


def predict(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """
    番号の並びからクラスごとの確率を求める関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    ids: torch.Tensor
        [CLS] を足した番号の並び

    Returns
    ----------
    probability: torch.Tensor
        (件数, クラス数) の確率
    """
    # 推論のモードでの, 分類の出力だけの確率
    model.eval()
    with torch.no_grad():
        logits = model(ids)
    if isinstance(logits, dict):
        logits = logits["logits"]
    probability = logits.softmax(dim=1)

    return probability


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment, cls_id: int
) -> None:
    """
    テスト用の分割の正解率と, 文字が同じで意味の違う組の予測を確かめる
    関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)
    cls_id: int
        [CLS] の番号

    Returns
    ----------
    None
    """
    # テスト用の分割の正解率
    model = model.cpu()
    x_test, y_test = dataset.tensors("test")
    probability = predict(model, add_cls(x_test, cls_id))
    accuracy = (probability.argmax(dim=1) == y_test).float().mean()
    experiment.logger.info(f"テスト用の正解率: {accuracy.item():.4f}")

    # 新しい文に書き出したときと同じ中身の手順 (語彙) をかけて予測
    classes = dataset.info["classes"]
    ids = torch.from_numpy(
        np.stack([dataset.transform_item(text) for text, _ in PROBES])
    ).long()
    probability = predict(model, add_cls(ids, cls_id))
    hits = 0
    for (text, answer), row in zip(PROBES, probability):
        guess = classes[int(row.argmax())]
        hits += guess == answer
        experiment.logger.info(
            f"見分けの確かめ: {text} → {guess} ({row.max().item():.3f}), "
            f"正解 {answer}"
        )
    experiment.logger.info(f"見分けの確かめの正解: {hits}/{len(PROBES)}")


def main() -> None:
    """
    ML-Fathom で作ったデータセットで, ML-CreateArchitecture で組んだモデル
    を学習 (事前学習した重みがあれば微調整) する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ作り方を案内)
    cfg, diff = load_project_config(str(TASK_DIR))
    path = TASK_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. 先に prepare.py を実行してください")
        sys.exit(1)
    dataset = load_dataset(path)
    vocab, _, cls_id = special_ids(dataset)

    # 定義からモデルを作る (事前学習の重みがあれば読み込む)
    try:
        model = load_model(
            TASK_DIR, cfg.model.spec, cfg.model.init, cfg.experiment.seed,
            vocab, dataset.info["splits"]["train"]["shape"][1] + 1
        )
    except ValueError as error:
        print(error)
        sys.exit(1)
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    experiment.logger.info(
        f"モデル: {cfg.model.spec}, 読み込んだ重み: {cfg.model.init}, "
        f"クラス {dataset.info['classes']}"
    )

    # 分割ごとのテンソルの DataLoader ([CLS] を先頭に足す)
    x_train, y_train = dataset.tensors("train")
    x_valid, y_valid = dataset.tensors("valid")
    train_loader = DataLoader(
        TensorDataset(add_cls(x_train, cls_id), y_train),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(add_cls(x_valid, cls_id), y_valid),
        batch_size=cfg.data.batch_size
    )
    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証用の正解率: "
          f"{history.get('valid/acc')[best_epoch - 1]:.4f}")
    evaluate_test(trainer.model, dataset, experiment, cls_id)

    # 推論の一式を実験の中に (前処理の外で先頭に足した [CLS] も書く)
    info = make_bundle(
        Path(experiment.run_dir) / "inference", TASK_DIR / cfg.model.spec,
        path, experiment.run_dir,
        inputs={"ids": {"source": "items", "prepend": [cls_id]}},
        output="logits", name=cfg.experiment.name
    )
    check = info["check"]
    experiment.logger.info(
        f"推論の一式: {check['split']} の {check['metric']} "
        f"{check['value']:.4f} (記録 {check['recorded']}, 合う: "
        f"{check['matches']})"
    )


if __name__ == "__main__":
    main()
