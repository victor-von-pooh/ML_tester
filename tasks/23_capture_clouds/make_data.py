from pathlib import Path

import numpy as np

from ml_eda.synthetic import write_shape_captures

# 書き出す先 (リポジトリの data, git の管理外). new は学習に使わない撮影
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "capture_clouds"
NEW_DIR = DATA_DIR.parent / "capture_clouds_new"

# 見本の作り方 (prepare.py が正解の出どころの設定に写す)
SETTINGS = {
    "count": 40, "views": 8, "size": [24, 32], "points": 2000, "seed": 0,
}


def make_shapes(count: int, seed: int, prefix: str) -> list[dict]:
    """
    位置, 大きさ, 色をばらつかせた球と箱の並びを作る関数

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
    # 位置は世界の原点から離し (書き出しの正規化を確かめるため), 大きさは
    # 周回の半径 (0.6 m) に収まる範囲
    rng = np.random.default_rng(seed)
    shapes = []
    for index in range(count):
        shape = {
            "name": f"{prefix}_{index:03d}",
            "center": [
                float(rng.uniform(-2.0, 2.0)), float(rng.uniform(0.0, 0.3)),
                float(rng.uniform(-2.0, 2.0)),
            ],
            "color": [int(value) for value in rng.integers(60, 240, 3)],
        }
        if index % 2 == 0:
            shape.update(type="sphere", radius=float(rng.uniform(0.08, 0.2)))
        else:
            shape.update(
                type="box",
                size=[float(value) for value in rng.uniform(0.1, 0.35, 3)],
            )
        shapes.append(shape)

    return shapes


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
    # 学習と評価の撮影 (captures/<名前>/) と点群 (clouds/<名前>.ply)
    common = {
        "views": SETTINGS["views"], "size": tuple(SETTINGS["size"]),
        "points": SETTINGS["points"],
    }
    shapes = make_shapes(SETTINGS["count"], SETTINGS["seed"], "shape")
    write_shape_captures(
        DATA_DIR, shapes=shapes, seed=SETTINGS["seed"], **common
    )
    print(f"書き出し: {DATA_DIR} ({len(shapes)} 件)")

    # 推論の確かめに使う, 書き出しに入れない撮影 (違う乱数の箱 1 つ)
    new = make_shapes(2, SETTINGS["seed"] + 1, "new")[1:]
    write_shape_captures(
        NEW_DIR, shapes=new, seed=SETTINGS["seed"] + 1, **common
    )
    print(f"書き出し: {NEW_DIR} ({new[0]['name']}, {new[0]['type']})")


if __name__ == "__main__":
    main()
