"""
22 の README の「実行方法」の画面の操作を, 画面と同じ API で順に流し,
段階ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを
表にする. 部品は 27 の journey.py と pairs.py (起動と記録は 25 の apps.py)
"""
import importlib.util
import sys
from pathlib import Path

from ml_eda.synthetic import write_image_masks

from make_data import SETTINGS

# 部品は 27 (pairs.py と journey.py) と 25 (apps.py)
TASKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASKS / "27_m2_image_masks"))
from pairs import run_pairs_journey
from apps import main_with

# 推論する新しい画像の数と乱数の種 (学習の組とは別の種)
NEW_COUNT = 4
NEW_SEED = 1

# 学習のエポックの数 (None は骨組みの既定の 30)
EPOCHS = None


def load_parts() -> object:
    """
    27 の journey.py を別の名前の部品として読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: object
        27 の journey.py (adapt_unet, apply_unet_info, check_samples,
        check_outputs, CLASSES を使う)
    """
    # 同じ名前の journey.py が並ぶので, 場所を指して読む
    path = TASKS / "27_m2_image_masks" / "journey.py"
    spec = importlib.util.spec_from_file_location("journey_27", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def build(work: Path) -> dict:
    """
    データを作り, 通しの設定を返す関数

    Parameters
    ----------
    work: Path
        作業用の場所

    Returns
    ----------
    setup: dict
        pairs.run_pairs_journey に渡す設定
    """
    # make_data.py と同じ設定の組と, 推論する新しい画像 (正解は比べるだけ)
    parts = load_parts()
    size = tuple(SETTINGS["size"])
    data = work / "data" / "image_masks"
    write_image_masks(data, count=SETTINGS["count"], size=size,
                      seed=SETTINGS["seed"])
    new = work / "data" / "image_masks_new"
    write_image_masks(new, count=NEW_COUNT, size=size, seed=NEW_SEED)
    names = sorted(path.name for path in (new / "images").iterdir())
    setup = {
        "name": "画像の領域を塗る (そろった大きさ)",
        "answers": [
            ("issue.pain", "写真の中の物の領域を塗るのに時間がかかる"),
            ("issue.kpi.name", "1 枚を塗る時間"),
            ("decomposition", [
                {"cause": "人が 1 枚ずつ塗る",
                 "lever": "領域を予測して人は直すだけにする", "ml": True}
            ]),
            ("action.what", "写真の物の領域"),
            ("ml.reason", "形と大きさがまちまちで, 決まった規則では塗れない"),
            ("ml.parts.0.what", "画素ごとの領域 (背景, 円, 四角)"),
            ("ml.parts.0.item", "写真 1 枚"),
            ("ml.parts.0.unit", "pair"),
            ("ml.parts.0.output", "region"),
            ("ml.parts.0.target.kind", "mask"),
            ("ml.parts.0.target.source", "files"),
            ("ml.parts.0.evaluation.metric", "iou"),
            ("ml.parts.0.evaluation.link", "当たるほど直す時間が減る")
        ],
        "data": data,
        "origin": {"type": "synthetic",
                   "name": "ml_eda.synthetic.write_image_masks",
                   "settings": SETTINGS},
        "options": {"classes": parts.CLASSES},
        "item_steps": None,
        "example": "unet_segmentation",
        "adapt": parts.adapt_unet,
        "apply_info": parts.apply_unet_info,
        "epochs": EPOCHS,
        "check_samples": parts.check_samples,
        "new": new / "images",
        "new_names": names,
        "truth": new / "masks",
        "check_outputs": parts.check_outputs
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "画像と領域 (そろった大きさ)", sys.argv[1:],
                       runner=run_pairs_journey, prefix="t22_")
    sys.exit(1 if failed else 0)
