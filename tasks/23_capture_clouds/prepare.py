from pathlib import Path
import sys

import numpy as np

from ml_eda import export_capture_clouds, load_items
from make_data import DATA_DIR, SETTINGS

# このタスクのディレクトリと, 書き出す先
TASK_DIR = Path(__file__).resolve().parent
DATASET_DIR = TASK_DIR / "outputs" / "dataset"


def main() -> None:
    """
    多視点の撮影と正確な点群の対を dataset.yaml の版 2 に書き出し,
    読み直して正規化を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元のデータが無ければ作り方の案内と終了
    if not (DATA_DIR / "captures").is_dir():
        print("data/capture_clouds がありません. 先に make_data.py を実行"
              "してください")
        sys.exit(1)

    # 正解は形の式から置いた面の上の点 (作った正確な正解)
    origin = {
        "type": "synthetic",
        "name": "ml_eda.synthetic.write_shape_captures",
        "settings": SETTINGS,
    }
    try:
        info = export_capture_clouds(
            DATA_DIR / "captures", DATA_DIR / "clouds", DATASET_DIR,
            split={"method": "random", "valid": 0.2, "test": 0.2, "seed": 0},
            origin=origin, cloud_units="meter", name="capture_clouds",
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
    print(f"正規化: {info['normalization']}")
    for name, entry in info["entries"].items():
        print(f"項目 {name}: {entry['kind']} ({entry['role']}), 形 "
              f"{entry['shape']}, 単位 {entry['units']}, 決まり "
              f"{entry['convention']}")

    # 学習のスクリプトと同じ読み方で読み直し, 正解の点群の中心が原点の
    # 近くに来たか (正規化は姿勢だけから決めるので, ちょうど 0 ではない)
    dataset = load_items(DATASET_DIR)
    item = dataset.split("train")[0]
    capture = item["capture"]
    center = item["cloud"].mean(axis=0)
    print(f"読み直し: id {item['id']}, 画像 {capture['images'].shape}, "
          f"深度 {capture['depth'].shape}, 姿勢 {capture['poses'].shape}, "
          f"点群 {item['cloud'].shape}")
    print(f"正解の点群の中心 (正規化した座標, m): {np.round(center, 4)}")


if __name__ == "__main__":
    main()
