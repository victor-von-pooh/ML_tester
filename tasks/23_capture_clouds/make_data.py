from pathlib import Path
import shutil

import numpy as np

from ml_eda.synthetic import write_shape_captures

# 書き出す先 (リポジトリの data, git の管理外). new は学習に使わない撮影
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "capture_clouds"
NEW_DIR = DATA_DIR.parent / "capture_clouds_new"

# 見本の作り方 (journey.py が正解の出どころの設定に写す. 形の種類, 位置,
# 大きさ, 色は count と seed から make_shapes が決める)
SETTINGS = {
    "count": 40, "views": 8, "size": [24, 32], "points": 2000, "seed": 0,
}


def make_shapes(count: int, seed: int, prefix: str) -> list[dict]:
    """
    原点の近くに置いた球と箱の並びを作る関数

    Parameters
    ----------
    count: int
        形の数 (球と箱を交互に)
    seed: int
        乱数の種
    prefix: str
        名前の頭 (撮影のディレクトリと点群のファイルの名前になる)

    Returns
    ----------
    shapes: list[dict]
        ml_eda.synthetic.write_shape_captures に渡す形の並び
    """
    # 中心は原点から水平に 0.5 から 1.5 m, 高さ 0.05 から 0.3 m
    rng = np.random.default_rng(seed)
    shapes = []
    for index in range(count):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        reach = float(rng.uniform(0.5, 1.5))
        shape = {
            "name": f"{prefix}_{index:03d}",
            "center": [reach * np.cos(angle), float(rng.uniform(0.05, 0.3)),
                       reach * np.sin(angle)],
            "color": [int(value) for value in rng.integers(60, 240, 3)]
        }
        if index % 2 == 0:
            shape.update(type="sphere",
                         radius=float(rng.uniform(0.08, 0.18)))
        else:
            shape.update(type="box", size=[
                float(value) for value in rng.uniform(0.1, 0.3, 3)
            ])
        shapes.append(shape)

    return shapes


def write_data(data: Path, new: Path) -> list[str]:
    """
    学習の撮影と点群と, 推論の確かめに使う新しい撮影を書く関数

    Parameters
    ----------
    data: Path
        学習の撮影と点群を書くディレクトリ
    new: Path
        新しい撮影を書くディレクトリ (書き出しに入れない)

    Returns
    ----------
    names: list[str]
        新しい撮影の名前
    """
    # 前に作ったものの削除 (名前の違う撮影が残ると書き出しに入るため)
    for place in (data, new):
        shutil.rmtree(place, ignore_errors=True)

    # 撮影 (captures/<名前>/) と点群 (clouds/<名前>.ply)
    common = {
        "views": SETTINGS["views"], "size": tuple(SETTINGS["size"]),
        "points": SETTINGS["points"], "seed": SETTINGS["seed"]
    }
    seed = SETTINGS["seed"]
    write_shape_captures(
        data, shapes=make_shapes(SETTINGS["count"], seed, "shape"), **common
    )
    shapes = make_shapes(1, seed + 1, "new")
    write_shape_captures(new, shapes=shapes, **common)
    names = [shape["name"] for shape in shapes]

    return names


def main() -> None:
    """
    形の正確な点群と多視点の撮影を ML-EDA で作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # リポジトリの data に書く
    names = write_data(DATA_DIR, NEW_DIR)
    print(f"書き出し: {DATA_DIR} ({SETTINGS['count']} 件)")
    print(f"書き出し: {NEW_DIR} ({', '.join(names)})")


if __name__ == "__main__":
    main()
