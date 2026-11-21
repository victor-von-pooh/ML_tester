"""
画像と領域 (大きさの混ざる合成の画像) を, 問題の設定から推論まで 5 つの
画面のアプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で
写すか打つか (手), 誤りかを表にする (点群と領域を画面で通す節目の確かめ)
"""
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml_eda.synthetic import write_image_masks
from pairs import Client, load_npy, run_pairs_journey, yaml_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import main_with

# 大きさごとの組の数 (高さ, 幅). 学習用と推論用で違う種
SIZES = {(32, 32): 30, (48, 40): 30}
NEW_SIZES = {(32, 32): 2, (48, 40): 2}

# ML-Fathom の書き出しでかける中身の手順 (U-Net の例は 4 の倍数の大きさ)
RESIZE = {"op": "resize", "width": 32, "height": 32}

# 領域の値の名前 (合成の見本の作り方. 利用者が知っている値)
CLASSES = {"0": "background", "1": "circle", "2": "square"}

# 学習のエポックの数 (None は骨組みの既定の 30. 60 組の 32x32 なら数秒)
EPOCHS = None


def make_pairs(root: Path, sizes: dict, seed: int) -> None:
    """
    大きさの違う画像と領域の組を, 1 つの images と masks に並べる関数

    Parameters
    ----------
    root: Path
        書くディレクトリ (images と masks を作る)
    sizes: dict
        (高さ, 幅) から組の数
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 大きさごとに書き, 名前の頭を付けて 1 つのディレクトリへ
    for folder in ("images", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index, (size, count) in enumerate(sizes.items()):
        part = root / f"part{index}"
        write_image_masks(part, count=count, size=size, seed=seed + index)
        for folder in ("images", "masks"):
            for path in sorted((part / folder).iterdir()):
                path.rename(root / folder / f"s{index}_{path.name}")
        shutil.rmtree(part)


def adapt_unet(spec: dict, dataset: Path) -> tuple[dict, list[str]]:
    """
    同梱の U-Net の例を, データセットのチャンネル数, 大きさ, クラスの数に
    合わせる関数 (画面でノードの欄を書き換えるのと同じ)

    Parameters
    ----------
    spec: dict
        例の定義
    dataset: Path
        書き出したデータセット

    Returns
    ----------
    result: tuple[dict, list[str]]
        直した定義と, 直した所の説明
    """
    # データセットの説明の入力の形 (高さ, 幅, チャンネル) と領域の値の数
    info = yaml_of(dataset / "dataset.yaml")
    entries = info["entries"]
    image = next(v for v in entries.values() if v.get("role") == "input")
    mask = next(v for v in entries.values() if v.get("role") == "target")
    height, width, channels = image["shape"]
    classes = len(mask.get("classes") or mask.get("values") or [])
    changes = []
    name, entry = next(iter(spec["inputs"].items()))
    shape = [entry["shape"][0], channels, height, width]
    if entry["shape"] != shape:
        changes.append(f"入力 {name} の形 {entry['shape']} → {shape}")
        entry["shape"] = shape
    first = next(n for n in spec["nodes"].values()
                 if n.get("inputs") == name)
    if first["params"]["in_channels"] != channels:
        changes.append(f"最初の畳み込みの in_channels → {channels}")
        first["params"]["in_channels"] = channels
    head = spec["nodes"][next(iter(spec["outputs"].values()))]
    if head["params"]["out_channels"] != classes:
        changes.append(
            f"最後の畳み込みの out_channels {head['params']['out_channels']}"
            f" → {classes} (クラスの数)"
        )
        head["params"]["out_channels"] = classes
    result = (spec, changes)

    return result


def apply_unet_info(spec: dict, info: dict) -> tuple[dict, list[str]]:
    """
    ML-CreateArchitecture の「データセットから入力と出力を作る」と同じく,
    返った入力の名前, 形, 型と出す値の数を U-Net の例に当てはめる関数

    画面の applyDatasetInputs のうち, U-Net の例 (入力 1 つ, 出力 1 つ)
    に要る所だけを写す. 当てられない所は当てずに飛ばす (画面も同じ)

    Parameters
    ----------
    spec: dict
        例の定義
    info: dict
        POST /api/dataset/inputs の応答 (ML-EDA の model_inputs)

    Returns
    ----------
    result: tuple[dict, list[str]]
        当てはめた定義 (同じ辞書) と, 当てた所の説明
    """
    notes: list[str] = []
    inputs = info.get("inputs") or []
    nodes = spec.get("nodes") or {}
    if len(inputs) != 1 or len(spec.get("inputs") or {}) != 1:
        result = (spec, notes)
        return result
    item = inputs[0]

    # 1 つずつで名前が違えば, 名前を変えてつなぎ先も書き換える
    name = next(iter(spec["inputs"]))
    if item.get("name") and item["name"] != name:
        spec["inputs"] = {item["name"]: spec["inputs"][name]}
        for node in nodes.values():
            if node.get("inputs") == name:
                node["inputs"] = item["name"]
        notes.append(f"入力 {name} の名前を {item['name']} にしました")
        name = item["name"]

    # 形と型. データセットで決まらない次元 (dataset_shape が None) は定義の値
    entry = spec["inputs"][name]
    shape = item.get("shape")
    dims = item.get("dataset_shape")
    current = entry.get("shape")
    if isinstance(shape, list):
        if (isinstance(dims, list) and isinstance(current, list)
                and len(dims) == len(shape) == len(current)):
            shape = [
                size if index == 0 or dims[index] is not None
                else current[index]
                for index, size in enumerate(shape)
            ]
        entry["shape"] = shape
        if item.get("dtype"):
            entry["dtype"] = item["dtype"]

        # 入力に直につながる in_channels を, 3 次元以上の入力のチャンネルに
        first = next(
            (n for n in nodes.values() if n.get("inputs") == name), None
        )
        if first is not None and len(shape) >= 3:
            params = first.setdefault("params", {})
            if "in_channels" in params and params["in_channels"] != shape[1]:
                params["in_channels"] = shape[1]
                notes.append(f"最初の畳み込みの in_channels を {shape[1]}"
                             " にしました")

    # 出す値の数. 点群の outputs は点の数なので当てない. 出力が 1 つだけ
    outputs = info.get("outputs")
    refs = list((spec.get("outputs") or {}).values())
    if (isinstance(outputs, int) and not isinstance(outputs, bool)
            and info.get("target_kind") != "point_cloud" and len(refs) == 1
            and isinstance(refs[0], str)):
        head = nodes.get(refs[0].split(".")[0])
        params = head.setdefault("params", {}) if head is not None else {}
        if "out_channels" in params and params["out_channels"] != outputs:
            params["out_channels"] = outputs
            notes.append(f"最後の畳み込みの out_channels を {outputs}"
                         " にしました")
    result = (spec, notes)

    return result

def check_samples(
    dash: Client, run_path: str, results: dict[str, Any]
) -> tuple[str, str]:
    """
    ML-Dashboard の結果の見本に, 凡例付きの領域があるかを見る関数

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
    # 領域の見本 (kind mask) の番号, 名前, 色の表
    samples = results.get("samples") or []
    masks = [s for s in samples if s.get("kind") == "mask"]
    if not masks:
        result = ("手", f"見本が無い (見本 {len(samples)} 件). 骨組みの "
                  "train.py が見本を書かないので, 見るには save_sample を"
                  "書き足す")
        return result
    legend = masks[0].get("classes")
    result = ("OK" if legend else "誤り",
              f"領域の見本 {len(masks)} 件, 凡例 {legend}")

    return result


def check_outputs(
    client: Client, result: dict[str, Any], rows: Any, setup: dict
) -> tuple[str, str]:
    """
    推論した領域の値と形を, 新しい件の正解と比べる関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    result: dict[str, Any]
        推論の結果
    rows: Any
        結果の行
    setup: dict
        通しの設定 (truth は新しい件の正解の領域のディレクトリ)

    Returns
    ----------
    outcome: tuple[str, str]
        判定と説明
    """
    # 行ごとの配列を読み, 値が領域の値の中か, 形が元の画像と同じかを見る.
    # 正解は出力の大きさに最も近い画素で縮めて比べる
    count = int(result.get("count") or len(setup["new_names"]))
    allowed = {int(k) for k in CLASSES}
    other, bad, agree = [], [], []
    for row in range(count):
        array = load_npy(client, result["id"], row)
        values = set(np.unique(array).tolist())
        if not values <= allowed:
            bad.append(sorted(values - allowed))
        truth = Image.open(setup["truth"] / setup["new_names"][row])
        if array.shape != (truth.height, truth.width):
            other.append(f"{truth.height}x{truth.width} → "
                         f"{array.shape[0]}x{array.shape[1]}")
            truth = truth.resize((array.shape[1], array.shape[0]),
                                 Image.Resampling.NEAREST)
        agree.append(float((np.asarray(truth) == array).mean()))
    note = (f"{count} 件. 値 {sorted(allowed)} の外 {bad or '無し'}, 画素の"
            f"一致 (正解を出力の大きさにして) {np.mean(agree):.3f}. 元の"
            "画像と大きさの違う出力 "
            + (f"{len(other)} 件 ({', '.join(sorted(set(other)))}). 中身の"
               "手順 (resize) の後の大きさで返り, 元の画像に重ねるには手で"
               "戻す" if other else "無し"))
    status = "誤り" if bad else ("手" if other else "OK")
    outcome = (status, note)

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
    # 学習のデータと, 推論する新しい画像 (正解は確かめにだけ使う)
    data = work / "data" / "image_masks"
    make_pairs(data, SIZES, seed=0)
    new = work / "data" / "image_masks_new"
    make_pairs(new, NEW_SIZES, seed=100)
    names = sorted(path.name for path in (new / "images").iterdir())
    setup = {
        "name": "画像の領域を塗る",
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
        "origin": {"type": "synthetic", "name": "合成の見本",
                   "settings": {"sizes": [list(s) for s in SIZES]}},
        "options": {"classes": CLASSES},
        "item_steps": {"image": [RESIZE]},
        "example": "unet_segmentation",
        "adapt": adapt_unet,
        "apply_info": apply_unet_info,
        "epochs": EPOCHS,
        "check_samples": check_samples,
        "new": new / "images",
        "new_names": names,
        "truth": new / "masks",
        "check_outputs": check_outputs
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "画像と領域 (大きさの混ざる画像)", sys.argv[1:],
                       runner=run_pairs_journey, prefix="m2_")
    sys.exit(1 if failed else 0)
