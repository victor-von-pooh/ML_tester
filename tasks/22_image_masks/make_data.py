from pathlib import Path

from ml_eda.synthetic import write_image_masks

# 書き出す先 (リポジトリの data, git の管理外). new は学習に使わない画像
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "image_masks"
NEW_DIR = DATA_DIR.parent / "image_masks_new"

# 見本の作り方 (journey.py が正解の出どころの設定に写す). 新しい画像は
# 同じ大きさで違う乱数の種
SETTINGS = {"count": 60, "size": [32, 32], "seed": 0}
NEW_SETTINGS = {"count": 4, "seed": 1}


def write_data(data: Path, new: Path) -> list[str]:
    """
    学習の画像と領域の組と, 推論の確かめに使う新しい組を書く関数

    Parameters
    ----------
    data: Path
        学習の組を書くディレクトリ (images と masks を作る)
    new: Path
        新しい組を書くディレクトリ (書き出しに入れない)

    Returns
    ----------
    names: list[str]
        新しい画像のファイルの名前
    """
    # images/ と masks/ に同じ stem の PNG の組
    size = tuple(SETTINGS["size"])
    write_image_masks(data, count=SETTINGS["count"], size=size,
                      seed=SETTINGS["seed"])
    write_image_masks(new, count=NEW_SETTINGS["count"], size=size,
                      seed=NEW_SETTINGS["seed"])
    names = sorted(path.name for path in (new / "images").iterdir())

    return names


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
    # リポジトリの data への書き込み
    names = write_data(DATA_DIR, NEW_DIR)
    print(f"書き出し: {DATA_DIR} ({SETTINGS['count']} 組)")
    print(f"書き出し: {NEW_DIR} ({len(names)} 組)")


if __name__ == "__main__":
    main()
