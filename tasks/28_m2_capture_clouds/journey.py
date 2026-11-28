"""
多視点の撮影から点群 (合成の形) を, 問題の設定から推論まで 5 つの画面の
アプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で写すか
打つか (手), 誤りかを表にする (点群と領域を画面で通す節目の確かめ)
"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml_eda.synthetic import write_shape_captures

# 部品は 27 の pairs.py と 25 の apps.py
TASKS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASKS / "25_m1_churn"))
sys.path.insert(0, str(TASKS / "27_m2_image_masks"))
from apps import main_with
from pairs import (Client, load_npy, match_rows, run_pairs_journey, short,
                   yaml_of)

# 「データセットから入力と出力を作る」の値を例に当てる所は 27 の journey.py
# (同じ名前の journey.py が並ぶので, 場所を指した読み込み)
_SPEC_27 = importlib.util.spec_from_file_location(
    "journey_27", TASKS / "27_m2_image_masks" / "journey.py"
)
_JOURNEY_27 = importlib.util.module_from_spec(_SPEC_27)
_SPEC_27.loader.exec_module(_JOURNEY_27)
apply_info = _JOURNEY_27.apply_unet_info

# 形の数, 1 つの撮影の枚数, 画像の (高さ, 幅), 正解の点群の点の数
SHAPES = 40
NEW_SHAPES = 2
VIEWS = 8
SIZE = (24, 32)
POINTS = 2000

# 学習のエポックの数 (None は骨組みの既定の 30. 40 件なら数十秒)
EPOCHS = None

# 推論した点群の重心と正解の重心のずれの上限 (m. 形の置き場は ±2 m)
CENTER_TOLERANCE = 0.1


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
    # 中心は原点から離して置く (元の座標に戻したかを重心で見分けるため)
    rng = np.random.default_rng(seed)
    shapes = []
    for index in range(count):
        shape = {
            "name": f"{prefix}_{index:03d}",
            "center": [float(rng.uniform(-2.0, 2.0)),
                       float(rng.uniform(0.0, 0.3)),
                       float(rng.uniform(-2.0, 2.0))],
            "color": [int(value) for value in rng.integers(60, 240, 3)]
        }
        if index % 2 == 0:
            shape.update(type="sphere", radius=float(rng.uniform(0.08, 0.2)))
        else:
            shape.update(type="box", size=[
                float(value) for value in rng.uniform(0.1, 0.35, 3)
            ])
        shapes.append(shape)

    return shapes


def read_ply(path: Path) -> np.ndarray:
    """
    頂点の x, y, z だけを持つ PLY (ascii か binary_little_endian の
    float) を読む関数

    Parameters
    ----------
    path: Path
        PLY のファイル

    Returns
    ----------
    points: np.ndarray
        (点の数, 3) の座標
    """
    # 頭の行から形式と点の数, 続きを数で読む
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    count = next(int(line.split()[2]) for line in header
                 if line.startswith("element vertex"))
    if any("binary_little_endian" in line for line in header):
        points = np.frombuffer(data[end:], dtype="<f4", count=count * 3)
    else:
        points = np.array(data[end:].split(), dtype=float)[:count * 3]
    points = points.reshape(count, 3).astype(float)

    return points


def input_points(dataset: Path) -> int:
    """
    骨組みの config (既定と実験) の data.input_points を読む関数

    Parameters
    ----------
    dataset: Path
        書き出したデータセット (プロジェクトの中)

    Returns
    ----------
    points: int
        学習で入力に渡す点の数

    Raises
    ----------
    ValueError
        config が見つからない場合と, input_points が無い場合
    """
    # データセットから上へたどったプロジェクトの configs. 実験の config
    # の値が既定より先
    project = next(
        (place for place in dataset.resolve().parents
         if (place / "configs" / "default" / "train.yaml").is_file()), None
    )
    if project is None:
        raise ValueError(f"{dataset} の上に骨組みの config が無い")
    found = None
    for name in ("default", "experiment"):
        path = project / "configs" / name / "train.yaml"
        data = yaml_of(path).get("data")
        if isinstance(data, dict) and data.get("input_points") is not None:
            found = data["input_points"]
    if found is None:
        raise ValueError("骨組みの config に data.input_points が無い")
    points = int(found)

    return points


def adapt_cloud(spec: dict, dataset: Path) -> tuple[dict, list[str]]:
    """
    同梱の点群の例の入力の点の数を, 骨組みの config の data.input_points
    に合わせる関数 (画面でノードの欄を書き換えるのと同じ)

    Parameters
    ----------
    spec: dict
        例の定義
    dataset: Path
        書き出したデータセット (骨組みの config を探す起点)

    Returns
    ----------
    result: tuple[dict, list[str]]
        定義と, 直した所の説明
    """
    # 骨組みの input_points と例の入力の形を比べる
    points = input_points(dataset)
    changes = []
    name, entry = next(iter(spec["inputs"].items()))
    if entry["shape"][1] != points:
        changes.append(f"入力 {name} の点の数 {entry['shape'][1]} → "
                       f"{points} (骨組みの data.input_points)")
        entry["shape"][1] = points
    result = (spec, changes)

    return result


def check_samples(
    dash: Client, run_path: str, results: dict[str, Any]
) -> tuple[str, str]:
    """
    ML-Dashboard の結果の見本に, 元の座標に戻せる点群があるかを見る関数

    Parameters
    ----------
    dash: Client
        ML-Dashboard の客
    run_path: str
        実験の API のパス
    results: dict[str, Any]
        実験の結果

    Returns
    ----------
    result: tuple[str, str]
        判定と説明
    """
    # 点群の見本 (kind point_cloud) を元の座標で読む
    samples = results.get("samples") or []
    clouds = [s for s in samples if s.get("kind") == "point_cloud"]
    if not clouds:
        result = ("手", f"見本が無い (見本 {len(samples)} 件). 骨組みの "
                  "train.py が見本を書かないので, 見るには save_sample を"
                  "書き足す")
        return result
    path = clouds[0].get("path")
    view = dash.ok("GET", f"{run_path}/samples/cloud?path={path}")
    result = ("OK" if view.get("original") else "誤り",
              f"点群の見本 {len(clouds)} 件, 元の座標 "
              f"{'あり' if view.get('original') else '無し'}")

    return result


def check_outputs(
    client: Client, result: dict[str, Any], rows: dict[str, Any],
    setup: dict
) -> tuple[str, str]:
    """
    推論した点群が元の座標 (撮影の世界の座標) にあるかを, 新しい形の正解
    の重心と比べる関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    result: dict[str, Any]
        推論の結果
    rows: dict[str, Any]
        結果の行 (GET /api/results/{id}/rows の応答)
    setup: dict
        通しの設定 (truth は新しい形の正解の点群のディレクトリ)

    Returns
    ----------
    outcome: tuple[str, str]
        判定と説明
    """
    # 件の名前 (撮影のディレクトリ) での正解との引き合わせ
    matched, unmatched = match_rows(rows, setup["new_names"])
    gaps, centers, shapes = [], [], set()
    for row, name in matched:
        cloud = load_npy(client, result["id"], row)
        shapes.add(tuple(cloud.shape))
        truth = read_ply(setup["truth"] / f"{name}.ply")
        gaps.append(float(np.linalg.norm(
            cloud.mean(axis=0) - truth.mean(axis=0)
        )))
        centers.append(float(np.linalg.norm(truth.mean(axis=0))))
    near = (bool(matched) and not unmatched
            and all(gap < CENTER_TOLERANCE for gap in gaps))
    note = (f"{len(matched)} 件 (件の名前で正解と引き合わせ). 食い違い "
            f"{'; '.join(unmatched) or '無し'}. 形 {sorted(shapes)}, 正解の"
            f"重心とのずれ {[round(gap, 3) for gap in gaps]} m (正解の重心の"
            f"原点からの距離 {[round(c, 2) for c in centers]} m. 上限 "
            f"{CENTER_TOLERANCE} m). 文: "
            f"{short(result.get('sentences'), 120)}")
    outcome = ("OK" if near else "誤り", note)

    return outcome


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
    # 学習の撮影と点群と, 推論する新しい形 (正解は確かめにだけ使う)
    data = work / "data" / "capture_clouds"
    common = {"views": VIEWS, "size": SIZE, "points": POINTS}
    write_shape_captures(data, shapes=make_shapes(SHAPES, 0, "shape"),
                         seed=0, **common)
    new = work / "data" / "capture_clouds_new"
    new_shapes = make_shapes(NEW_SHAPES, 1, "new")
    write_shape_captures(new, shapes=new_shapes, seed=1, **common)
    setup = {
        "name": "撮影から形の点群を作る",
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
        "origin": {"type": "synthetic", "name": "合成の見本",
                   "settings": {"shapes": SHAPES, "views": VIEWS}},
        "options": {"cloud_units": "meter"},
        "item_steps": None,
        "example": "point_cloud_predictor",
        "adapt": adapt_cloud,
        "apply_info": apply_info,
        "epochs": EPOCHS,
        "check_samples": check_samples,
        "new": new / "captures",
        "new_names": sorted(shape["name"] for shape in new_shapes),
        "truth": new / "clouds",
        "check_outputs": check_outputs
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "多視点の撮影から点群 (合成の形)", sys.argv[1:],
                       runner=run_pairs_journey, prefix="m2_")
    sys.exit(1 if failed else 0)
