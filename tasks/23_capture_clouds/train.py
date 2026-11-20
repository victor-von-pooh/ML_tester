from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import ItemSplit, capture_shift, load_items, normalize_capture
from ml_eda.items import read_capture  # 1 つの撮影を配列で読む (公開外)
from ml_eda.media import read_cloud  # 1 つの点群を読む (公開外)
from ml_found import (
    Experiment, Trainer, describe_origin, load_project_config, load_samples,
    original_coordinates, save_evaluation, save_sample, start_experiment,
    task_from_dataset, write_task
)
from make_data import DATA_DIR, NEW_DIR
from model import CloudPredictor

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent

# F 値の距離のしきい値 (メートル)
FSCORE_THRESHOLD = 0.01

# 見本に残すテスト用の件の数
SAMPLE_COUNT = 2


class CaptureTrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment):
        # step を上書きするので step_fn は None
        super(CaptureTrainer, self).__init__(model, experiment, None)

        # 検証の予測と正解の入れ物と, 検証したエポックの並び
        self.predictions: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.judged: list[int] = []

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの始めに予測と正解の入れ物を空にする関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # 入れ物を空に
        self.predictions = []
        self.targets = []

    def step(self, batch: Any, phase: str) -> dict:
        """
        1 バッチ分の Chamfer の損失を求め, 検証では予測を貯める関数

        Parameters
        ----------
        batch: Any
            深度から戻した点と正解の点群
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            2 乗の距離の Chamfer の損失とバッチの件数
        """
        # 予測と, 2 乗の距離の Chamfer (学習の損失)
        x, y = batch
        predicted = self.model(x)
        squared = chamfer(predicted, y, True)
        if phase == "valid":
            self.predictions.append(predicted.detach().cpu())
            self.targets.append(y.detach().cpu())
        outputs = {"loss": squared.mean(), "batch_size": len(y)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証したエポックだけ, Chamfer 距離と F 値を求める関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証では chamfer (メートル) と fscore_1cm, 他は空
        """
        # 学習の段階は毎エポック空. 検証の段階は検証するエポックだけの呼び出し
        if phase != "valid":
            metrics = {}
            return metrics
        self.judged.append(self.epoch)
        metrics = cloud_metrics(
            torch.cat(self.predictions), torch.cat(self.targets)
        )

        return metrics


def chamfer(
    predicted: torch.Tensor, target: torch.Tensor, squared: bool
) -> torch.Tensor:
    """
    1 件ごとの対称な Chamfer 距離を求める関数

    Parameters
    ----------
    predicted: torch.Tensor
        (バッチ, 予測の点の数, 3)
    target: torch.Tensor
        (バッチ, 正解の点の数, 3)
    squared: bool
        最も近い点までの距離を 2 乗するか (学習の損失は 2 乗)

    Returns
    ----------
    distance: torch.Tensor
        (バッチ,) の, 両向きの最も近い点までの距離の平均の和の半分
    """
    # 全ての組の距離から, 両向きに最も近い点
    pairs = torch.cdist(predicted, target)
    if squared:
        pairs = pairs ** 2
    forward = pairs.min(dim=2).values.mean(dim=1)
    backward = pairs.min(dim=1).values.mean(dim=1)
    distance = (forward + backward) / 2

    return distance


def cloud_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    """
    予測した点群の Chamfer 距離と F 値を求める関数

    Parameters
    ----------
    predicted: torch.Tensor
        (件数, 予測の点の数, 3)
    target: torch.Tensor
        (件数, 正解の点の数, 3)

    Returns
    ----------
    metrics: dict
        chamfer (距離の平均. メートル) と fscore_1cm (1 cm 以内の点の
        適合率と再現率の調和平均の平均)
    """
    # 勾配の計算は不要 (on_epoch_end は no_grad の外での呼び出し)
    with torch.no_grad():
        pairs = torch.cdist(predicted, target)
        to_target = pairs.min(dim=2).values
        to_predicted = pairs.min(dim=1).values
        distance = (to_target.mean(dim=1) + to_predicted.mean(dim=1)) / 2
        precision = (to_target < FSCORE_THRESHOLD).float().mean(dim=1)
        recall = (to_predicted < FSCORE_THRESHOLD).float().mean(dim=1)
        fscore = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall).clamp(min=1e-12),
            torch.zeros_like(precision),
        )
    metrics = {
        "chamfer": distance.mean().item(), "fscore_1cm": fscore.mean().item()
    }

    return metrics


def backproject(
    capture: dict, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    撮影の深度を世界の座標の点に戻し, 決まった数を選ぶ関数

    Parameters
    ----------
    capture: dict
        load_items の撮影 (intrinsics, poses, depth, convention)
    count: int
        選ぶ点の数 (足りなければ重ねて選ぶ)
    rng: np.random.Generator
        選ぶ乱数

    Returns
    ----------
    points: np.ndarray
        (count, 3) の float32 の点 (撮影の世界の座標. メートル)

    Raises
    ----------
    ValueError
        カメラの軸が OpenGL でない場合, 深度が無い場合, 点が無い場合
    """
    # 扱うカメラの軸は OpenGL (+X 右, +Y 上, -Z 前) だけ
    axes = str(capture["convention"].get("camera_axes", "OpenGL"))
    if not axes.startswith("OpenGL"):
        raise ValueError(f"カメラの軸が OpenGL ではありません: {axes}")
    if capture["depth"] is None:
        raise ValueError("深度の無い枚があります")
    intr = capture["intrinsics"]
    clouds = []
    for pose, depth in zip(capture["poses"], capture["depth"]):
        if not np.all(np.isfinite(pose)):
            continue
        rows, columns = np.nonzero(depth > 0)
        value = depth[rows, columns].astype(np.float64)

        # 画素の中心を通る光線. 深度はカメラの前 (-Z) の向きの距離
        camera = np.stack([
            (columns + 0.5 - intr["cx"]) / intr["fl_x"] * value,
            -(rows + 0.5 - intr["cy"]) / intr["fl_y"] * value,
            -value,
        ], axis=1)
        clouds.append(camera @ pose[:3, :3].T + pose[:3, 3])
    merged = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    if len(merged) == 0:
        raise ValueError("深度から戻せる点がありません")
    index = rng.choice(len(merged), count, replace=len(merged) < count)
    points = merged[index].astype(np.float32)

    return points


def split_tensors(
    items: ItemSplit, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    分割の全ての件を読み, 深度から戻した点と正解の点群のテンソルにする
    関数

    件が少ないので最初に全て読んで持つ

    Parameters
    ----------
    items: ItemSplit
        load_items の分割の件の並び
    count: int
        入力の点の数
    seed: int
        点を選ぶ乱数の種

    Returns
    ----------
    tensors: tuple[torch.Tensor, torch.Tensor]
        (件数, count, 3) の入力と (件数, 正解の点の数, 3) の正解
    """
    # 1 件ずつ, 撮影から入力の点と正解の点群
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for index in range(len(items)):
        item = items[index]
        inputs.append(backproject(item["capture"], count, rng))
        targets.append(np.asarray(item["cloud"], dtype=np.float32))
    tensors = (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets)),
    )

    return tensors


def evaluate_test(
    model: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor],
    experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割の Chamfer 距離を, 入力の点そのものと比べて記録する関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル (ベストモデルの重み)
    tensors: tuple[torch.Tensor, torch.Tensor]
        テスト用の入力と正解
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        answer (model か input_points), chamfer, fscore_1cm と, 入力の点
        との差 (chamfer_diff, fscore_1cm_diff. 比べる相手の行は 0) の行
    """
    # モデルの予測と, 深度から戻した点をそのまま答えにした場合
    model = model.cpu().eval()
    x, y = tensors
    with torch.no_grad():
        predicted = cloud_metrics(model(x), y)
    reference = cloud_metrics(x, y)
    rows = []
    for answer, label, metrics in (
        ("model", "モデル", predicted),
        ("input_points", "入力の点", reference),
    ):
        text = (f"テスト用 ({label}): chamfer {metrics['chamfer']:.5f} m, "
                f"fscore_1cm {metrics['fscore_1cm']:.4f}")
        experiment.logger.info(text)
        print(text)
        rows.append({"answer": answer, **metrics})
    for row in rows:
        row["chamfer_diff"] = row["chamfer"] - reference["chamfer"]
        row["fscore_1cm_diff"] = row["fscore_1cm"] - reference["fscore_1cm"]

    return rows


def cloud_convention(row: Any) -> dict:
    """
    1 件の正規化の平行移動から, 点群の見本の値の決まりを作る関数

    Parameters
    ----------
    row: Any
        分割の表の 1 行 (shift_x, shift_y, shift_z. 正規化した座標 =
        元の座標 + shift)

    Returns
    ----------
    convention: dict
        units (m), shift (ML-EDA の shift のまま. ML-Found の見本の
        shift も同じ向き), scale (1.0. 平行移動だけ)
    """
    # ML-EDA と ML-Found の shift は同じ向き (正規化した座標 = 元の座標 +
    # shift) なので, 符号を変えずに写す
    shift = [
        float(row[column]) for column in ("shift_x", "shift_y", "shift_z")
    ]
    convention = {"units": "m", "shift": shift, "scale": 1.0}

    return convention


def save_cloud_samples(
    model: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor],
    rows: Any, experiment: Experiment, epoch: int
) -> None:
    """
    テスト用の最初の数件の予測と正解の点群を見本に書く関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル (ベストモデルの重み)
    tensors: tuple[torch.Tensor, torch.Tensor]
        テスト用の入力と正解
    rows: Any
        テスト用の分割の表 (id と正規化の平行移動 shift_x, shift_y,
        shift_z の列. tensors と同じ順)
    experiment: Experiment
        実験 (samples/ に書く)
    epoch: int
        見本を作った重みのエポック (ベストモデルのエポック)

    Returns
    ----------
    None
    """
    # 1 件ごとに予測と正解の 2 つの点群 (書き出しの正規化した座標)
    model = model.cpu().eval()
    x, y = tensors
    with torch.no_grad():
        predicted = model(x[:SAMPLE_COUNT])
    for index in range(len(predicted)):
        row = rows.iloc[index]
        convention = cloud_convention(row)
        pairs = (
            ("pred", predicted[index], "予測した点群"),
            ("truth", y[index], "正解の点群"),
        )
        for suffix, points, label in pairs:
            save_sample(
                experiment, f"test_{row['id']}_{suffix}", points,
                epoch=epoch, kind="point_cloud", split="test",
                caption=f"{label} (書き出しの正規化した座標)",
                convention=convention
            )

    # 正解の見本を値の決まりで元の座標に戻し, 元の点群と重なるかの確認
    for item in load_samples(experiment.run_dir):
        if not item["name"].endswith("_truth"):
            continue
        name = item["name"][len("test_"):-len("_truth")]
        points = read_cloud(experiment.run_dir / "samples" / item["file"])
        original = read_cloud(DATA_DIR / "clouds" / f"{name}.ply")
        restored = original_coordinates(
            points["points"], item["convention"]
        )
        # 書き出しは点の順を保つので, 同じ番号の点どうしのずれの最大
        distance = float(
            np.abs(restored - np.asarray(original["points"])).max()
        )
        text = (
            f"正解の見本 {name} を元の座標に戻した点群と元の点群の"
            f"点ごとのずれの最大: {distance:.2e} m"
        )
        experiment.logger.info(text)
        print(text)


def predict_new(
    model: nn.Module, cfg: Any, experiment: Experiment
) -> None:
    """
    書き出しに無い撮影に同じ正規化をかけて予測し, 元の座標に戻す関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    cfg: Any
        config (入力の点の数と乱数の種)
    experiment: Experiment
        実験 (写した撮影の置き場と train.log)

    Returns
    ----------
    None
    """
    # 新しい撮影を実験のディレクトリに写して正規化 (書き出しと同じ決まり)
    place = NEW_DIR / "captures"
    sources = sorted(
        path for path in place.iterdir() if path.is_dir()
    ) if place.is_dir() else []
    if not sources:
        print("新しい撮影がありません. make_data.py を実行してください")
        return
    source = sources[0]
    copied = Path(experiment.run_dir) / "new_capture"
    summary = normalize_capture(source, copied)
    shift = summary["shift"]
    same = np.allclose(shift, capture_shift(source))

    # 写した撮影の読み込みと, 元の座標に戻した予測 (- shift) と元の点群の比較
    rng = np.random.default_rng(cfg.data.seed)
    x = backproject(read_capture(copied), cfg.data.input_points, rng)
    with torch.no_grad():
        predicted = model.cpu().eval()(torch.from_numpy(x)[None])[0]
    original = predicted.numpy() - shift
    truth = read_cloud(NEW_DIR / "clouds" / f"{source.name}.ply")["points"]
    metrics = cloud_metrics(
        torch.from_numpy(original.astype(np.float32))[None],
        torch.from_numpy(np.asarray(truth, dtype=np.float32))[None],
    )
    text = (
        f"新しい撮影 {source.name}: shift {np.round(shift, 4).tolist()} "
        f"(capture_shift と一致: {same}), 元の座標で chamfer "
        f"{metrics['chamfer']:.5f} m, fscore_1cm "
        f"{metrics['fscore_1cm']:.4f}"
    )
    experiment.logger.info(text)
    print(text)


def main() -> None:
    """
    ML-EDA の版 2 で書き出した撮影と点群の対で, 点群の予測を学習する関数

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
    if info["target"]["kind"] != "point_cloud":
        print(f"正解の種類が point_cloud ではありません: "
              f"{info['target']['kind']}")
        sys.exit(1)

    # 何の実験か (受け持ち, 正解の種類と出どころ) を dataset.yaml から
    # 埋める. 誤りは出力先を作る前に見つける
    task = task_from_dataset(cfg.data.dataset, project_dir=str(TASK_DIR))
    task.update({
        "output": "shape", "what": "撮影ごとの物の表面の点群",
        "method": "scratch"
    })

    # 正解の出どころを実験に残す (指標はこの出どころとの一致)
    experiment = start_experiment(cfg, diff, str(TASK_DIR))
    task = write_task(experiment, task)
    origin = info["target"]["origin"]
    experiment.logger.info(
        f"データセット: {info['name']} (版 {info['format_version']}), "
        f"分割 {info['splits']}, 正規化 {info['normalization']['method']}"
    )
    experiment.logger.info(
        f"正解の出どころ: type={origin['type']}, name={origin['name']}, "
        f"settings={origin['settings']}. 指標はこの正解との一致"
    )

    # 分割ごとに読み, 深度から戻した点と正解のテンソル
    tensors = {}
    for offset, name in enumerate(("train", "valid", "test")):
        tensors[name] = split_tensors(
            dataset.split(name), cfg.data.input_points,
            cfg.data.seed + offset
        )
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]), batch_size=cfg.data.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(*tensors["valid"]), batch_size=cfg.data.batch_size
    )

    # 学習 (検証は trainer.valid_every_n_epochs ごとと最後のエポック)
    model = CloudPredictor(
        channels=cfg.model.channels, hidden=cfg.model.hidden,
        num_points=cfg.model.output_points
    )
    trainer = CaptureTrainer(model, experiment)
    history = trainer.fit(train_loader, valid_loader)

    # 結果の表示, テスト用の評価と新しい撮影の予測
    best_epoch = trainer.checkpoint.best_epoch
    values = history.get("valid/chamfer")
    print(f"保存先: {experiment.run_dir}")
    print(f"正解の出どころ: {origin['type']} ({origin['name']}). "
          f"{describe_origin(task['target']['origin'])}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"検証したエポック: {trainer.judged}")
    print(f"valid/chamfer の無いエポックの数: "
          f"{sum(value is None for value in values)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの検証用の Chamfer 距離: {values[best_epoch - 1]:.5f} m")
    rows = evaluate_test(trainer.model, tensors["test"], experiment)

    # テスト用の評価の表 (入力の点との比べ) と見本を実験のディレクトリに
    # id と正規化の平行移動は分割の表から (件の中身を読み直さない)
    test_rows = dataset.split("test").rows
    save_evaluation(
        experiment, "test_chamfer", rows, "test",
        description="テスト用の分割の Chamfer 距離 (メートル) と 1 cm の "
        "F 値. input_points は深度から戻した点をそのまま答えにした場合. "
        "_diff はその値との差",
        baseline="input_points", epoch=best_epoch
    )
    save_cloud_samples(
        trainer.model, tensors["test"], test_rows, experiment, best_epoch
    )
    print(f"評価と見本: {experiment.run_dir}/evaluations, samples")
    predict_new(trainer.model, cfg, experiment)


if __name__ == "__main__":
    main()
