"""
画像の分類 (大きさのそろった合成の図形の画像) を, 問題の設定から推論まで
5 つの画面のアプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で
写すか打つか (手), 誤りかを表にする (画像と文の分類を画面で通す節目の確かめ)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from classes import example_model, no_fix, table_columns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Client, main_with

# 図形のクラスと, クラスごとの枚数 (学習用と, 推論とたどり直し用).
# 見本の意思決定 (選ぶ) は 2 クラスの分類にだけ使えるので 2 つ
SHAPES = ["circle", "square"]
COUNT = 60
PAST_COUNT = 10

# 画像の大きさ (全て同じなので中身の手順の resize は要らない)
SIZE = 28

# 学習の後に ML-Pathfinder で直す損得 (連絡 1 件の費用. 25 と同じ)
COST_UPDATE = {"cost": 3000}


def draw_shape(shape: str, rng: np.random.Generator) -> Image.Image:
    """
    雑音の背景に, 色と位置と大きさの違う図形を 1 つ描く関数

    Parameters
    ----------
    shape: str
        circle か square
    rng: np.random.Generator
        乱数

    Returns
    ----------
    image: Image.Image
        SIZE x SIZE の RGB の画像
    """
    # 暗い雑音の背景と, 明るい色の図形
    noise = rng.integers(0, 60, (SIZE, SIZE, 3), dtype=np.uint8)
    image = Image.fromarray(noise, "RGB")
    draw = ImageDraw.Draw(image)
    color = tuple(int(c) for c in rng.integers(120, 256, 3))
    half = int(rng.integers(6, 11))
    cx, cy = (int(c) for c in rng.integers(half + 1, SIZE - half - 1, 2))
    box = (cx - half, cy - half, cx + half, cy + half)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)

    return image


def make_images(root: Path, count: int, seed: int) -> None:
    """
    <クラス>/<画像>.png のディレクトリに合成の図形の画像を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ
    count: int
        クラスごとの枚数
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # クラスごとのディレクトリ (ML-EDA の load_data がラベルとして読む名前)
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        (root / shape).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            draw_shape(shape, rng).save(root / shape / f"{index:03d}.png")


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
        apps.run_journey に渡す設定
    """
    # 学習のデータと, 推論とたどり直しに使う別の乱数の種の画像
    data = work / "data" / "shapes"
    make_images(data, COUNT, seed=0)
    past = work / "data" / "shapes_past"
    make_images(past, PAST_COUNT, seed=1)
    columns = table_columns(data)
    setup = {
        "example": "解約の引き止め",
        "data": data,
        "past": past,
        # 画像の表 (ML-EDA の load_data の列) に合わせた見本の答え
        "answers": {
            "information.columns": columns,
            "ml.parts.0.target.column": "label",
            "ml.parts.0.positive": "circle",
            "ml.parts.0.item": "画像 1 枚",
            "information.known": [c for c in columns
                                  if c not in ("label", "digest")],
            "information.forbidden": ["digest"]
        },
        "recipe_update": None,
        "recipe_steps": [],
        "item_steps_from": same_size,
        "make_model": example_model("cnn_classifier", no_fix),
        # 画像は 1 件の中身で学び, 骨組みは決まりで今のやり方と比べない
        "compare": False,
        "cost_update": COST_UPDATE
    }

    return setup


def same_size(fathom: Client, sid: str) -> tuple[list[dict], str, str]:
    """
    ML-Fathom の分析の結果から画像の大きさを見て, 中身の手順を決める関数

    Parameters
    ----------
    fathom: Client
        ML-Fathom の客
    sid: str
        データの ID

    Returns
    ----------
    result: tuple[list[dict], str, str]
        中身の手順, 判定, 説明
    """
    # 分析の結果の画像の大きさ (画面の画像の分析に出る表). 1 種類なら手順
    # は無し. 混ざれば画面の気付きどおりに最も多い大きさへ resize
    items = fathom.ok("GET", f"/api/sources/{sid}/profile").get("items")
    sizes = (items or {}).get("sizes") or []
    if len(sizes) == 1:
        result = ([], "OK", f"大きさが 1 種類 ({sizes[0]['width']}x"
                  f"{sizes[0]['height']}, {sizes[0]['count']} 枚) なので, "
                  "中身の手順は無し")
        return result
    if not sizes:
        result = ([], "誤り", "分析の結果に画像の大きさが無い")
        return result
    top = sizes[0]
    result = ([{"op": "resize", "width": top["width"],
                "height": top["height"]}], "OK",
              f"大きさが {len(sizes)} 種類. 気付きの勧める resize")

    return result


if __name__ == "__main__":
    failed = main_with(build, "画像の分類 (図形の画像)", sys.argv[1:],
                       prefix="m3_")
    sys.exit(1 if failed else 0)
