from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 書き出す先 (リポジトリの data, git の管理外)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "shapes"

# クラス (図形) の名前
SHAPES = ("circle", "square", "triangle")

# 画像の大きさの候補 (幅, 高さ). 前処理でそろえる必要がある
SIZES = ((48, 48), (64, 64), (64, 48))


def draw_shape(
    shape: str, size: tuple, rng: np.random.Generator, gray: bool
) -> Image.Image:
    """
    1 つの図形を, 位置と大きさと色をばらつかせて描く関数

    Parameters
    ----------
    shape: str
        circle, square, triangle のどれか
    size: tuple
        (幅, 高さ)
    rng: np.random.Generator
        乱数
    gray: bool
        灰色 (L) の画像にするか

    Returns
    ----------
    image: Image.Image
        描いた画像
    """
    # 雑音のある背景
    width, height = size
    noise = rng.normal(40, 15, (height, width, 3)).clip(0, 255)
    image = Image.fromarray(noise.astype(np.uint8))
    canvas = ImageDraw.Draw(image)

    # 中心と半径と色をばらつかせる
    radius = rng.uniform(0.2, 0.35) * min(width, height)
    cx = rng.uniform(radius, width - radius)
    cy = rng.uniform(radius, height - radius)
    color = tuple(int(v) for v in rng.integers(120, 256, 3))
    if shape == "circle":
        canvas.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
        )
    elif shape == "square":
        canvas.rectangle(
            (cx - radius, cy - radius, cx + radius, cy + radius), fill=color
        )
    else:
        canvas.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius),
             (cx + radius, cy + radius)], fill=color
        )
    image = image.convert("L") if gray else image

    return image


def main(per_class: int = 300, seed: int = 0) -> None:
    """
    図形の画像をクラスのフォルダに保存する関数

    大きさの違い, 灰色とカラーの混在, 読めないファイルを含める

    Parameters
    ----------
    per_class: int = 300
        クラスごとの枚数
    seed: int = 0
        乱数のシード

    Returns
    ----------
    None
    """
    # クラスごとに描いて保存
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        folder = DATA_DIR / shape
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            size = SIZES[rng.integers(len(SIZES))]
            gray = bool(rng.random() < 0.3)
            draw_shape(shape, size, rng, gray).save(
                folder / f"{index:04d}.png"
            )

    # 途中で切れたファイル (転送の失敗などで実際に起きる)
    for shape in SHAPES:
        (DATA_DIR / shape / "broken.png").write_bytes(b"\x89PNG\r\n")
    print(f"保存: {DATA_DIR} ({per_class * len(SHAPES)} 枚と読めない 3 枚)")


if __name__ == "__main__":
    main()
