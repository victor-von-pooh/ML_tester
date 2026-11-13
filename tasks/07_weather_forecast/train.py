import csv
from datetime import date
import math
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_found import Trainer, load_project_config, start_experiment
from model import WeatherForecaster

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# 入力にそのまま使う気象要素の列
RAW_FEATURES = [
    "temp_mean", "temp_max", "temp_min", "sunshine", "wind_mean",
    "wind_max", "vapor_pressure", "humidity_mean", "humidity_min",
    "pressure_sea", "cloud_mean"
]

# 予測する翌日の値の列
TARGETS = ["temp_max", "temp_min"]


def load_daily(cfg: DictConfig) -> tuple[list[date], torch.Tensor]:
    """
    整形した日別値を読み込んで特徴量にする関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    dates: list[date]
        日付
    features: torch.Tensor
        日ごとの特徴量, 形状は (日数, 特徴量の数)
    """
    # 相対パスはタスクのディレクトリを基準に解決して読み込み
    path = Path(cfg.data.path).expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 気象要素に, 降水量の対数, 最多風向と季節の sin と cos を加えて特徴量に変換
    dates = []
    features = []
    for row in rows:
        day = date.fromisoformat(row["date"])
        wind = math.radians(float(row["wind_mode_dir"]))
        season = 2 * math.pi * day.timetuple().tm_yday / 365.25
        values = [float(row[name]) for name in RAW_FEATURES]
        values += [
            math.log1p(float(row["precip"])), math.sin(wind),
            math.cos(wind), math.sin(season), math.cos(season)
        ]
        dates.append(day)
        features.append(values)
    features = torch.tensor(features, dtype=torch.float32)

    return dates, features


def make_datasets(
    cfg: DictConfig
) -> tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor]:
    """
    過去の日数分の特徴量と翌日の気温の組を作り, 日付で分ける関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: TensorDataset
        予測する日が data.valid_from より前の組
    valid_dataset: TensorDataset
        予測する日が data.valid_from 以降の組
    y_mean: torch.Tensor
        学習用の期間の最高気温と最低気温の平均
    y_std: torch.Tensor
        学習用の期間の最高気温と最低気温の標準偏差
    """
    # 学習用の期間の日だけで標準化の基準を決定
    dates, features = load_daily(cfg)
    valid_from = date.fromisoformat(cfg.data.valid_from)
    is_train_day = torch.tensor([d < valid_from for d in dates])
    x_mean = features[is_train_day].mean(dim=0)
    x_std = features[is_train_day].std(dim=0).clamp_min(1e-6)
    x = (features - x_mean) / x_std

    # 予測する翌日の最高気温と最低気温も同じく標準化
    target_idx = [RAW_FEATURES.index(name) for name in TARGETS]
    y = features[:, target_idx]
    y_mean = y[is_train_day].mean(dim=0)
    y_std = y[is_train_day].std(dim=0)
    y = (y - y_mean) / y_std

    # 過去 input_days 日分と翌日の組を, 予測する日の日付で学習用と検証用に分割
    n_in = cfg.data.input_days
    groups = {"train": ([], []), "valid": ([], [])}
    for t in range(n_in - 1, len(dates) - 1):
        key = "train" if dates[t + 1] < valid_from else "valid"
        groups[key][0].append(x[t - n_in + 1:t + 1])
        groups[key][1].append(y[t + 1])
    train_dataset = TensorDataset(
        torch.stack(groups["train"][0]), torch.stack(groups["train"][1])
    )
    valid_dataset = TensorDataset(
        torch.stack(groups["valid"][0]), torch.stack(groups["valid"][1])
    )

    return train_dataset, valid_dataset, y_mean, y_std


def persistence_mae(
    dataset: TensorDataset, y_std: torch.Tensor
) -> list[float]:
    """
    翌日も今日と同じ気温とする単純な予測の誤差を求める関数

    Parameters
    ----------
    dataset: TensorDataset
        過去の特徴量と翌日の気温の組
    y_std: torch.Tensor
        最高気温と最低気温の標準偏差

    Returns
    ----------
    mae: list[float]
        最高気温と最低気温の平均絶対誤差 (℃)
    """
    # 入力の最後の日の気温を, そのまま翌日の予測として誤差を計算
    x, y = dataset.tensors
    target_idx = [RAW_FEATURES.index(name) for name in TARGETS]
    last = x[:, -1, target_idx]

    # 入力の気温は特徴量の基準で標準化しているので, 気温の基準に直して比較
    # (どちらも学習用の期間の同じ列から求めた平均と標準偏差なので一致)
    mae = ((last - y).abs().mean(dim=0) * y_std).tolist()

    return mae


def make_step(
    y_std: torch.Tensor
) -> Callable[[nn.Module, tuple, str], dict]:
    """
    気温の誤差を ℃ で記録する step を作る関数

    Parameters
    ----------
    y_std: torch.Tensor
        最高気温と最低気温の標準偏差

    Returns
    ----------
    step: Callable[[nn.Module, tuple, str], dict]
        ML-Found に渡す, 標準化した空間の損失と ℃ の誤差を求める関数
    """
    # 損失関数の定義
    mse = nn.MSELoss()

    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失と, 最高気温と最低気温の誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            過去の特徴量と翌日の気温 (標準化した値)
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            標準化した空間の平均二乗誤差と, ℃ に戻した平均絶対誤差
        """
        # 予測して標準化した空間で損失の計算
        x, y = batch
        pred = model(x)
        loss = mse(pred, y)

        # ℃ に戻した平均絶対誤差を集約
        err = (pred.detach() - y).abs() * y_std.to(y.device)
        outputs = {
            "loss": loss, "mae_max": err[:, 0].mean(),
            "mae_min": err[:, 1].mean()
        }

        return outputs

    return step


def main() -> None:
    """
    config を読み込んで東京の翌日の気温の予測を学習する関数

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

    # データを用意し, 件数と単純な予測の誤差を記録
    train_dataset, valid_dataset, _, y_std = make_datasets(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )
    base = persistence_mae(valid_dataset, y_std)
    experiment.logger.info(
        f"学習用: {len(train_dataset)} 組, 検証用: {len(valid_dataset)} 組"
    )
    experiment.logger.info(
        f"前日と同じとする予測の検証用 MAE: 最高気温 {base[0]:.3f} ℃, "
        f"最低気温 {base[1]:.3f} ℃"
    )

    # モデルを用意して学習
    n_features = train_dataset.tensors[0].shape[2]
    model = WeatherForecaster(
        input_dim=n_features, hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers, output_dim=len(TARGETS),
        lstm_type=cfg.model.lstm_type
    )
    trainer = Trainer(model, experiment, make_step(y_std))
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    for key in ("valid/mae_max", "valid/mae_min"):
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.3f} ℃")
    print(f"前日と同じとする予測の MAE: {base[0]:.3f} ℃, {base[1]:.3f} ℃")


if __name__ == "__main__":
    main()
