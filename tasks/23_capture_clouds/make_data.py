from pathlib import Path

from ml_eda.synthetic import write_shape_captures

# 書き出す先 (リポジトリの data, git の管理外). new は学習に使わない撮影
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "capture_clouds"
NEW_DIR = DATA_DIR.parent / "capture_clouds_new"

# 見本の作り方 (prepare.py が正解の出どころの設定に写す. 形の種類, 位置,
# 大きさ, 色は count と seed から ML-EDA が決める)
SETTINGS = {
    "count": 40, "views": 8, "size": [24, 32], "points": 2000, "seed": 0,
}


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
    info = write_shape_captures(
        DATA_DIR, count=SETTINGS["count"], seed=SETTINGS["seed"], **common
    )
    print(f"書き出し: {DATA_DIR} ({len(info['shapes'])} 件)")

    # 推論の確かめに使う, 書き出しに入れない撮影 (違う乱数の形 1 つ)
    info = write_shape_captures(
        NEW_DIR, count=1, seed=SETTINGS["seed"] + 1, **common
    )
    shape = info["shapes"][0]
    print(f"書き出し: {NEW_DIR} ({shape['name']}, {shape['type']})")


if __name__ == "__main__":
    main()
