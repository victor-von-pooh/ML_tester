from pathlib import Path
import shutil

import numpy as np

# 書き出す先 (リポジトリの data, git の管理外)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "objects"

# クラス (形) の名前
SHAPES = ("sphere", "cube", "cylinder", "cone")


def surface(shape: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """
    形の表面から点を取る関数

    Parameters
    ----------
    shape: str
        sphere, cube, cylinder, cone のどれか
    count: int
        点の数
    rng: np.random.Generator
        乱数

    Returns
    ----------
    points: np.ndarray
        (count, 3) の座標
    """
    # 球は正規分布を半径 1 に, 立方体は 1 つの軸を ±1 に, 円柱は側面,
    # 円錐は高さに比例した半径の側面
    if shape == "sphere":
        points = rng.normal(0, 1, (count, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    elif shape == "cube":
        points = rng.uniform(-1, 1, (count, 3))
        axis = rng.integers(0, 3, count)
        points[np.arange(count), axis] = rng.choice([-1.0, 1.0], count)
    else:
        angle = rng.uniform(0, 2 * np.pi, count)
        height = rng.uniform(-1, 1, count)
        radius = (1 - height) / 2 if shape == "cone" else np.ones(count)
        points = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle), height], axis=1
        )

    return points


def place(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    点群を z 軸の周りに回し, 大きさと位置と雑音をばらつかせる関数

    Parameters
    ----------
    points: np.ndarray
        (N, 3) の座標
    rng: np.random.Generator
        乱数

    Returns
    ----------
    placed: np.ndarray
        (N, 3) の座標
    """
    # 回転, 拡大, 平行移動, 雑音
    angle = rng.uniform(0, 2 * np.pi)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    placed = points @ rotation.T * rng.uniform(0.5, 3.0)
    placed = placed + rng.uniform(-5, 5, 3) + rng.normal(0, 0.02, points.shape)

    return placed


def write_ply(path: Path, points: np.ndarray) -> None:
    """
    点群をバイナリの PLY で書く関数

    Parameters
    ----------
    path: Path
        書く場所
    points: np.ndarray
        (N, 3) の座標

    Returns
    ----------
    None
    """
    # ヘッダと float32 の座標
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    body = points.astype("<f4").tobytes()
    path.write_bytes(header.encode("ascii") + body)


def write_pcd(path: Path, points: np.ndarray) -> None:
    """
    点群を ASCII の PCD で書く関数 (距離のセンサーの欠けとして NaN を含む)

    Parameters
    ----------
    path: Path
        書く場所
    points: np.ndarray
        (N, 3) の座標 (NaN の点を含んでよい)

    Returns
    ----------
    None
    """
    # ヘッダと 1 行 1 点
    header = (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA ascii\n"
    )
    rows = "\n".join(" ".join(f"{v:.5f}" for v in p) for p in points)
    path.write_text(header + rows + "\n", encoding="ascii")


def main(per_class: int = 60, seed: int = 0) -> None:
    """
    形ごとの点群を, PLY, PCD, xyz を混ぜてクラスのフォルダに書く関数

    点の数の違い, 欠けた点 (NaN), 読めないファイルを含める

    Parameters
    ----------
    per_class: int = 60
        クラスごとの件数
    seed: int = 0
        乱数のシード

    Returns
    ----------
    None
    """
    # 前の点群を消してからの, 形ごとの書き出し (3 件に 1 件ずつ形式を変更)
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        folder = DATA_DIR / shape
        folder.mkdir(parents=True)
        for index in range(per_class):
            count = int(rng.integers(300, 2000))
            points = place(surface(shape, count, rng), rng)
            name = folder / f"{index:03d}"
            if index % 3 == 0:
                write_ply(name.with_suffix(".ply"), points)
            elif index % 3 == 1:
                points[rng.random(count) < 0.05] = np.nan
                write_pcd(name.with_suffix(".pcd"), points)
            else:
                np.savetxt(name.with_suffix(".xyz"), points, fmt="%.5f")

    # 途中で切れたファイル
    (DATA_DIR / "cone" / "broken.ply").write_bytes(b"ply\nformat ascii")
    print(f"保存: {DATA_DIR} ({per_class * len(SHAPES)} 件と読めない 1 件)")


if __name__ == "__main__":
    main()
