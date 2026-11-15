from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import Dataset, load_data, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, start_experiment
)
from model import ChurnClassifier

# 推論の確かめに使う新しい顧客 (元のデータと同じ形の csv. ID, 欠損,
# カンマ付きの数を含む. 答えから作った列は予測のときには無い)
NEW_CUSTOMER = (
    "customer_id,tenure_months,monthly_fee,total_paid,plan,contract,"
    "support_calls,data_gb\n"
    'C999999,3,,"1,210",premium,monthly,6,12.0\n'
)

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
        前処理した特徴量とクラスの番号
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


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment
) -> None:
    """
    テスト用の分割の正解率と, 新しいデータへの前処理を確かめる関数

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

    # 元のデータと同じ形の新しい顧客を load_data で読み (カンマ付きの数は
    # ここで数になる), 書き出したときの前処理をかける. 予測のときに無い列
    # (答えから作った列など) を学習に使っていると, 前処理で誤りになる
    path = experiment.run_dir / "new_customer.csv"
    path.write_text(NEW_CUSTOMER, encoding="utf-8")
    try:
        raw = load_data(path, kind="table").frame
        features = torch.from_numpy(
            dataset.transform(raw).to_numpy(dtype="float32")
        )
    except ValueError as error:
        experiment.logger.warning(f"新しい顧客を予測できません: {error}")
        return
    with torch.no_grad():
        probability = model(features).softmax(dim=1)[0, 1].item()
    experiment.logger.info(f"新しい顧客の解約の確率: {probability:.3f}")


def main() -> None:
    """
    ML-EDA で書き出したデータセットで解約の予測を学習する関数

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
    experiment.logger.info(
        f"データセット: {dataset.info['name']}, 列 {len(dataset.features)}, "
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

    # 入力の次元とクラスの数はデータセットの説明から
    model = ChurnClassifier(
        input_dim=len(dataset.features), hidden_dim=cfg.model.hidden_dim,
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
