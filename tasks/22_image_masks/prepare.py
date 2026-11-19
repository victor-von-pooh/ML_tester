from pathlib import Path
import sys

from ml_eda import export_image_masks, load_items
from ml_eda.synthetic import MASK_CLASSES
from make_data import DATA_DIR, SETTINGS

# このタスクのディレクトリと, 書き出す先
TASK_DIR = Path(__file__).resolve().parent
DATASET_DIR = TASK_DIR / "outputs" / "dataset"


def main() -> None:
    """
    画像と領域の対を dataset.yaml の版 2 に書き出し, 読み直して確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元のデータが無ければ作り方の案内と終了
    if not (DATA_DIR / "images").is_dir():
        print("data/image_masks がありません. 先に make_data.py を実行して"
              "ください")
        sys.exit(1)

    # 正解は見本を作る関数の出力そのもの (作った正確な正解)
    origin = {
        "type": "synthetic", "name": "ml_eda.synthetic.write_image_masks",
        "settings": SETTINGS,
    }
    try:
        info = export_image_masks(
            DATA_DIR / "images", DATA_DIR / "masks", DATASET_DIR,
            split={"method": "random", "valid": 0.2, "test": 0.2, "seed": 0},
            classes=MASK_CLASSES, origin=origin, name="image_masks",
            overwrite=True
        )
    except (ValueError, FileExistsError) as error:
        print(f"書き出せません: {error}")
        sys.exit(1)
    print(f"書き出し: {DATASET_DIR}")
    print(f"版: {info['format_version']}, 受け持ち: {info['role']}, "
          f"正解の種類: {info['target']['kind']}")
    print(f"正解の出どころ: {info['target']['origin']}")
    print(f"分割: {info['splits']}, 注意: {info['notes'] or 'なし'}")
    for name, entry in info["entries"].items():
        print(f"項目 {name}: {entry['kind']} ({entry['role']}), 形 "
              f"{entry['shape']}, 型 {entry['dtype']}")

    # 学習のスクリプトと同じ読み方での 1 件の読み直し
    dataset = load_items(DATASET_DIR)
    item = dataset.split("train")[0]
    print(f"読み直し: id {item['id']}, 画像 {item['image'].shape} "
          f"{item['image'].dtype}, 領域 {item['mask'].shape} "
          f"{item['mask'].dtype}")


if __name__ == "__main__":
    main()
