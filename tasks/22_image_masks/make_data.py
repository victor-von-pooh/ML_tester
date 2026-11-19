from pathlib import Path

from ml_eda.synthetic import write_image_masks

# 書き出す先 (リポジトリの data, git の管理外)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "image_masks"

# 見本の作り方 (prepare.py が正解の出どころの設定に写す)
SETTINGS = {"count": 60, "size": [32, 32], "seed": 0}


def main() -> None:
    """
    円か四角を 1 つ置いた画像と領域の見本を ML-EDA で作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # images/ と masks/ に同じ stem の PNG の組
    result = write_image_masks(
        DATA_DIR, count=SETTINGS["count"], size=tuple(SETTINGS["size"]),
        seed=SETTINGS["seed"]
    )
    print(f"書き出し: {DATA_DIR}")
    print(f"組の数: {len(result['names'])}, 値の名前: {result['classes']}")


if __name__ == "__main__":
    main()
