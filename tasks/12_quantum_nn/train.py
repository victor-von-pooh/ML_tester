import math
from pathlib import Path
from typing import Callable, Optional

from omegaconf import DictConfig
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import (
    Trainer, load_project_config, register_rng, start_experiment
)
from model import ClassicalMLP, HybridQNN, QuantumNeuralNetwork

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


def make_data(cfg: DictConfig) -> tuple[TensorDataset, TensorDataset]:
    """
    2 値分類のデータを作り, 回転角の範囲に変換して分割する関数

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
    """
    # moons は 2 つの半月, iris は setosa を除いた 2 クラス
    if cfg.data.dataset == "moons":
        x, y = make_moons(
            n_samples=cfg.data.n_samples, noise=cfg.data.noise,
            random_state=cfg.experiment.seed
        )
    elif cfg.data.dataset == "iris":
        x, y = load_iris(return_X_y=True)
        keep = y > 0
        x, y = x[keep], y[keep] - 1
    else:
        raise ValueError(
            f"data.dataset は moons か iris にしてください: {cfg.data.dataset}"
        )

    # クラスの比率を保って学習用と検証用に分割
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=cfg.data.valid_size, stratify=y,
        random_state=cfg.experiment.seed
    )

    # 量子回路の回転角として入れるので, 学習用データの基準で 0 から π に変換
    scaler = MinMaxScaler(feature_range=(0.0, math.pi))
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    # テンソルにしてデータセットに集約
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long()
    )

    return train_dataset, valid_dataset


def seed_qiskit(seed: int) -> None:
    """
    qiskit の乱数 (algorithm_globals) を固定する関数

    Parameters
    ----------
    seed: int
        乱数シードの値

    Returns
    ----------
    None
    """
    # qiskit の乱数生成器を作り直して固定
    algorithm_globals.random_seed = seed


def get_qiskit_state() -> dict:
    """
    qiskit の乱数の状態を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    state: dict
        乱数生成器 (PCG64) の状態
    """
    # 乱数生成器の状態を取得 (取り出すたびに新しい辞書)
    state = algorithm_globals.random.bit_generator.state

    return state


def set_qiskit_state(state: dict) -> None:
    """
    qiskit の乱数の状態を戻す関数

    Parameters
    ----------
    state: dict
        get_qiskit_state が返した状態

    Returns
    ----------
    None
    """
    # 乱数生成器の状態を上書き
    algorithm_globals.random.bit_generator.state = state


def register_qiskit_rng() -> None:
    """
    qiskit の乱数を ML-Found の固定と再開時の保存の対象に加える関数

    start_experiment の前に呼ぶ

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ML-Found の乱数の登録簿に追加
    register_rng("qiskit", seed_qiskit, get_qiskit_state, set_qiskit_state)


def build_model(cfg: DictConfig, input_dim: int) -> nn.Module:
    """
    config に応じて量子, ハイブリッド, 古典のモデルを作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    input_dim: int
        特徴量の数

    Returns
    ----------
    model: nn.Module
        作成したモデル
    """
    # 古典の MLP は量子のライブラリを使わずに作成
    if cfg.model.kind == "mlp":
        model = ClassicalMLP(input_dim, cfg.model.hidden_dim, 1)
        return model

    # qiskit の Estimator と Sampler には, 固定する場合だけシードを指定
    # (algorithm_globals は register_qiskit_rng で ML-Found が固定)
    seed: Optional[int] = None
    if cfg.train.seed_qiskit:
        seed = cfg.experiment.seed

    # 量子ビットの数を特徴量の数に合わせて量子層を作成
    # ハイブリッドは前の古典の層を学習するために入力に対する勾配も計算
    qnn = QuantumNeuralNetwork(
        input_dim=input_dim, num_reuploads=cfg.model.num_reuploads,
        net_type=cfg.model.net_type, seed=seed,
        feature_map_reps=cfg.model.feature_map_reps,
        ansatz_reps=cfg.model.ansatz_reps,
        input_gradients=cfg.model.kind == "hybrid"
    )

    # ハイブリッドは古典の層で挟み, それ以外は量子層だけ
    if cfg.model.kind == "hybrid":
        model = HybridQNN(input_dim, qnn)
    elif cfg.model.kind == "qnn":
        model = qnn
    else:
        raise ValueError(
            f"model.kind は qnn, hybrid, mlp のどれかにしてください: "
            f"{cfg.model.kind}"
        )

    return model


def make_step(
    probs_output: bool
) -> Callable[[nn.Module, tuple, str], dict]:
    """
    モデルの出力の種類に合わせた step を作る関数

    Parameters
    ----------
    probs_output: bool
        モデルがクラスごとの確率を出力する場合は True,
        2 クラス目のロジットまたは期待値を 1 つ出力する場合は False

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        ML-Found に渡す 1 バッチ分の損失と正解率を求める関数
    """
    # 確率の対数を使う損失と, 1 つのロジットを使う損失の定義
    nll = nn.NLLLoss()
    bce = nn.BCEWithLogitsLoss()

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失と正解率を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            回転角に変換した特徴量と正解ラベル
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            損失と正解率
        """
        # Sampler は確率の対数, それ以外は 1 つのロジットで損失を計算
        x, y = batch
        out = model(x)
        if probs_output:
            loss = nll(torch.log(out.clamp_min(1e-8)), y)
            pred = out.argmax(dim=1)
        else:
            logit = out.squeeze(-1)
            loss = bce(logit, y.float())
            pred = (logit > 0).long()

        # 正解率を求めて損失と集約
        acc = (pred == y).float().mean()
        outputs = {"loss": loss, "acc": acc}

        return outputs

    return step


def main() -> None:
    """
    config を読み込んで量子ニューラルネットの 2 値分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    cfg, diff = load_project_config(str(TASK_DIR))
    if cfg.train.seed_qiskit:
        register_qiskit_rng()
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データを用意
    train_dataset, valid_dataset = make_data(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )

    # モデルを用意し, 出力の種類に合わせた step で学習
    input_dim = train_dataset.tensors[0].shape[1]
    model = build_model(cfg, input_dim)
    probs_output = (
        cfg.model.kind == "qnn" and cfg.model.net_type == "Sampler"
    )
    trainer = Trainer(model, experiment, make_step(probs_output))
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    best_acc = history.get("valid/acc")[best_epoch - 1]
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストモデルの検証用データの正解率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
