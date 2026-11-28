from pathlib import Path
import shutil

from ml_eda.synthetic import write_shape_captures

# 書き出す先 (リポジトリの data, git の管理外). new は学習に使わない撮影
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "capture_clouds"
NEW_DIR = DATA_DIR.parent / "capture_clouds_new"

# 見本の作り方 (journey.py が正解の出どころの設定に写す. 形の種類, 位置,
# 大きさ, 色は count と seed から ML-EDA が決める)
SETTINGS = {
    "count": 40, "views": 8, "size": [24, 32], "points": 2000, "seed": 0,
}


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

    # 撮影 (captures/<名前>/) と点群 (clouds/<名前>.ply). 新しい撮影は
    # 違う乱数の種の形 1 つ
    common = {
        "views": SETTINGS["views"], "size": tuple(SETTINGS["size"]),
        "points": SETTINGS["points"],
    }
    write_shape_captures(
        data, count=SETTINGS["count"], seed=SETTINGS["seed"], **common
    )
    info = write_shape_captures(
        new, count=1, seed=SETTINGS["seed"] + 1, **common
    )
    names = [shape["name"] for shape in info["shapes"]]

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
    # リポジトリの data への書き込み
    names = write_data(DATA_DIR, NEW_DIR)
    print(f"書き出し: {DATA_DIR} ({SETTINGS['count']} 件)")
    print(f"書き出し: {NEW_DIR} ({', '.join(names)})")


if __name__ == "__main__":
    main()
