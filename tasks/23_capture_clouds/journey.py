"""
23 の README の「実行方法」の画面の操作を, 画面と同じ API で順に流し,
段階ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを
表にする. 部品は 28 の journey.py と 27 の pairs.py (起動と記録は 25 の
apps.py)
"""
import importlib.util
import sys
from pathlib import Path

from make_data import SETTINGS, write_data

# 部品は 28 (journey.py), 27 (pairs.py) と 25 (apps.py)
TASKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASKS / "27_m2_image_masks"))
from pairs import run_pairs_journey
from apps import main_with

# 学習のエポックの数 (None は骨組みの既定の 30)
EPOCHS = None


def load_parts() -> object:
    """
    28 の journey.py を別の名前の部品として読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: object
        28 の journey.py (adapt_cloud, check_samples, check_outputs を使う)
    """
    # 同じ名前の journey.py が並ぶので, 場所を指して読む
    path = TASKS / "28_m2_capture_clouds" / "journey.py"
    spec = importlib.util.spec_from_file_location("journey_28", path)
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
    # make_data.py と同じ設定の撮影と点群と, 推論する新しい撮影 (正解は
    # 比べるだけ)
    parts = load_parts()
    data = work / "data" / "capture_clouds"
    new = work / "data" / "capture_clouds_new"
    names = write_data(data, new)
    setup = {
        "name": "撮影から形の点群を作る (原点の近くの形)",
        "answers": [
            ("issue.pain", "撮った物の形を図面にするのに時間がかかる"),
            ("issue.kpi.name", "1 つの物の形を起こす時間"),
            ("decomposition", [
                {"cause": "人が撮影から形を起こす",
                 "lever": "撮影から点群を予測して人は確かめるだけにする",
                 "ml": True}
            ]),
            ("action.what", "物の形の点群"),
            ("ml.reason", "見えない面を補うのに, 決まった規則では足りない"),
            ("ml.parts.0.what", "物の表面の点群"),
            ("ml.parts.0.item", "1 つの物の多視点の撮影"),
            ("ml.parts.0.unit", "bundle"),
            ("ml.parts.0.output", "shape"),
            ("ml.parts.0.target.kind", "point_cloud"),
            ("ml.parts.0.target.source", "files"),
            ("ml.parts.0.evaluation.metric", "chamfer"),
            ("ml.parts.0.evaluation.link", "近いほど確かめの手間が減る")
        ],
        "data": data,
        "origin": {"type": "synthetic",
                   "name": "ml_eda.synthetic.write_shape_captures",
                   "settings": SETTINGS},
        "options": {"cloud_units": "meter"},
        "item_steps": None,
        "example": "point_cloud_predictor",
        "adapt": parts.adapt_cloud,
        "epochs": EPOCHS,
        "check_samples": parts.check_samples,
        "new": new / "captures",
        "new_names": names,
        "truth": new / "clouds",
        "check_outputs": parts.check_outputs
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "多視点の撮影から点群 (原点の近くの形)",
                       sys.argv[1:], runner=run_pairs_journey, prefix="t23_")
    sys.exit(1 if failed else 0)
