from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Experiment, Trainer, load_project_config, start_experiment
from model import ImbalancedClassifier

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class AUCTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        pos_weight: Optional[torch.Tensor] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(AUCTrainer, self).__init__(
            model, experiment, None, optimizer, scheduler
        )

        # 正例の重みを付けた損失関数を定義
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # エポックごとの予測確率と正解ラベルの入れ物を用意
        self.probs: list[torch.Tensor] = []
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
        # 継承したクラスから今のエポックを参照できることの確認
        # 記録は検証の後に追加されるので, fit の中では常に記録の数 + 1
        if self.epoch != len(self.history.records) + 1:
            raise RuntimeError(
                f"self.epoch ({self.epoch}) と記録の数が合いません"
            )

        # 予測確率と正解ラベルの入れ物を空に初期化
        self.probs = []
        self.labels = []

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の損失を求め, 予測確率を貯める関数

        Parameters
        ----------
        batch: Any
            標準化した特徴量と正解ラベル
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            重み付きの二値交差エントロピー損失とバッチのサンプル数
        """
        # 予測して損失の計算
        x, y = batch
        logits = self.model(x)
        loss = self.bce(logits, y)

        # エポック全体の指標を求めるために予測確率と正解ラベルを保存
        self.probs.append(torch.sigmoid(logits).detach().cpu())
        self.labels.append(y.detach().cpu())

        # サンプル数は明示して受け渡し
        outputs = {"loss": loss, "batch_size": len(y)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の予測から ROC-AUC と AP を求める関数

        Parameters
        ----------
        phase: str
            学習時は train, 検証時は valid

        Returns
        ----------
        metrics: dict
            ROC-AUC (auc) と平均適合率 (ap)
        """
        # 貯めた予測確率と正解ラベルを 1 つの配列に結合
        prob = torch.cat(self.probs).numpy()
        label = torch.cat(self.labels).numpy()

        # バッチの平均では求められない指標をエポック全体で計算
        metrics = {
            "auc": roc_auc_score(label, prob),
            "ap": average_precision_score(label, prob)
        }

        return metrics


def make_data(cfg: DictConfig) -> tuple[TensorDataset, TensorDataset, float]:
    """
    不均衡な二値分類のデータを作って分割する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: TensorDataset
        学習用のデータセット
    valid_dataset: TensorDataset
        検証用のデータセット
    pos_ratio: float
        学習用データの正例の割合
    """
    # 正例が少ないデータを生成
    x, y = make_classification(
        n_samples=cfg.data.n_samples, n_features=cfg.data.n_features,
        n_informative=cfg.data.n_informative,
        n_redundant=cfg.data.n_redundant,
        weights=[1.0 - cfg.data.pos_ratio], random_state=cfg.experiment.seed
    )

    # 正例の割合を保って学習用と検証用に分割
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, stratify=y,
        random_state=cfg.experiment.seed
    )

    # 学習用データで標準化の基準を決めて両方に適用
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_valid = ss.transform(x_valid)

    # テンソルにしてデータセットに集約, ラベルは損失関数に合わせて float
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).float()
    )
    pos_ratio = float(np.mean(y_train))

    return train_dataset, valid_dataset, pos_ratio


def main() -> None:
    """
    config を読み込んで不均衡な二値分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    cfg, diff = load_project_config(str(TASK_DIR))
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データを用意し, 正例の割合を記録
    train_dataset, valid_dataset, pos_ratio = make_data(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )
    experiment.logger.info(f"学習用データの正例の割合: {pos_ratio:.4f}")

    # 正例の重みは学習用データの負例と正例の比から決定
    pos_weight = None
    if cfg.loss.use_pos_weight:
        pos_weight = torch.tensor((1.0 - pos_ratio) / pos_ratio)
        experiment.logger.info(f"pos_weight: {pos_weight.item():.4f}")

    # モデルを用意して学習
    model = ImbalancedClassifier(
        input_dim=cfg.data.n_features, hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout
    )
    trainer = AUCTrainer(model, experiment, pos_weight)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/auc", "valid/ap"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.4f}")


if __name__ == "__main__":
    main()
