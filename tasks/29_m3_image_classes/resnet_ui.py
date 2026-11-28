"""
本物の ResNet18 (torchvision の ImageNet で学習した重み) の既製のモデルの
一式を, ML-Provide の画面 (ブラウザ) から作って登録し, 新しい画像を推論して
種類の名前が画像の分類の画面に出ることを確かめる. ブラウザの操作は
ML-Provide の画面の Playwright を使う (resnet_ui.mjs). 画面に出た種類の名前
を, 同じ重みを torchvision で直に読んだ予測と引き合わせる
"""
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Apps, Journey, free_port, main_with, signal_group

# 画面で選ぶ出どころ, モデル, 重み (torchvision の ImageNet の ResNet18)
LIBRARY = "torchvision"
MODEL = "resnet18"
WEIGHTS = "IMAGENET1K_V1"

# 推論する画像の大きさ (前処理の resize と切り出しを通るように大きめ)
SIZE = 256

# 画面のビルドと配信と, ブラウザの操作を待つ秒数の上限
BUILD_TIMEOUT = 300
SERVE_TIMEOUT = 60
BROWSER_TIMEOUT = 900


def draw_images(root: Path) -> list[str]:
    """
    種類の違う模様の画像 (無地, 市松, 縞, 同心円) を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ

    Returns
    ----------
    names: list[str]
        書いた画像のファイルの名前
    """
    # 無地 (赤, 緑, 青), 白黒の市松, 黄と黒の縞, 赤と白の同心円
    root.mkdir(parents=True, exist_ok=True)
    grid = np.indices((SIZE, SIZE))
    images = {}
    for index, name in enumerate(["a_red", "b_green", "c_blue"]):
        array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        array[:, :, index] = 255
        images[name] = array
    checker = ((grid[0] // 32 + grid[1] // 32) % 2 * 255).astype(np.uint8)
    images["d_checker"] = np.stack([checker] * 3, axis=-1)
    stripe = (grid[1] // 24 % 2).astype(bool)
    array = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    array[stripe] = (255, 210, 0)
    images["e_stripes"] = array
    radius = np.hypot(grid[0] - SIZE / 2, grid[1] - SIZE / 2)
    ring = (radius // 20 % 2).astype(bool)
    array = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    array[ring] = (220, 0, 0)
    images["f_rings"] = array
    names = []
    for name, array in images.items():
        Image.fromarray(array, "RGB").save(root / f"{name}.png")
        names.append(f"{name}.png")

    return names


def build(work: Path) -> dict[str, Any]:
    """
    推論する画像を作り, 通しの設定を返す関数

    Parameters
    ----------
    work: Path
        作業用の場所

    Returns
    ----------
    setup: dict[str, Any]
        run_ui に渡す設定
    """
    # 新しい画像と, 画面の結果を書く場所
    images = work / "data" / "patterns"
    setup = {
        "images": images,
        "names": draw_images(images),
        "result": work / "ui.json",
        "dist": work / "dist"
    }

    return setup


def direct_labels(images: Path, names: list[str]) -> dict[str, str]:
    """
    同じ重みを torchvision で直に読み, 画像ごとの最も確かな種類を返す関数

    Parameters
    ----------
    images: Path
        画像のディレクトリ
    names: list[str]
        画像のファイルの名前

    Returns
    ----------
    labels: dict[str, str]
        ファイルの名前から種類の名前
    """
    # 重みの前処理 (resize 256, 真ん中の 224 の切り出し, 正規化) と推論
    weights = ResNet18_Weights[WEIGHTS]
    model = resnet18(weights=weights).eval()
    transform = weights.transforms()
    batch = torch.stack([
        transform(Image.open(images / name).convert("RGB"))
        for name in names
    ])
    with torch.no_grad():
        top = model(batch).argmax(dim=1).tolist()
    categories = weights.meta["categories"]
    labels = {name: categories[i] for name, i in zip(names, top)}

    return labels


def wait_http(url: str, timeout: float) -> None:
    """
    URL が答えるまで待つ関数

    Parameters
    ----------
    url: str
        待つ URL
    timeout: float
        待つ秒数の上限

    Returns
    ----------
    None
    """
    # 起動の途中は接続を断られるので, 答えるまでの繰り返し
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"{url} が {timeout} 秒で答えません")


def run_ui(apps: Apps, journey: Journey, setup: dict[str, Any]) -> None:
    """
    画面を作って配信し, ブラウザで一式を作って推論し, 段階を記録する関数

    Parameters
    ----------
    apps: Apps
        起動した ML-Provide
    journey: Journey
        段階の記録
    setup: dict[str, Any]
        build の設定

    Returns
    ----------
    None
    """
    # ML-Provide の画面を作業用の場所に作り (clone の中の配信の場所は変え
    # ない), API を起動した ML-Provide に転送する vite preview で配信.
    # 転送は Host を変えない (ML-Provide は Origin と Host が違う書き込みを
    # 断る. 画面の vite.config.ts の転送は Host を転送先に変える)
    frontend = apps.apps_dir / "ML-Provide" / "frontend"
    work = apps.work
    subprocess.run(
        ["pnpm", "exec", "vite", "build", "--outDir", str(setup["dist"]),
         "--emptyOutDir"],
        cwd=frontend, check=True, timeout=BUILD_TIMEOUT,
        stdout=subprocess.DEVNULL
    )
    config = work / "preview.config.mjs"
    proxy = {"/api": {"target": apps.clients["provide"].base,
                      "changeOrigin": False}}
    config.write_text(
        "export default " + json.dumps({"preview": {"proxy": proxy}}) + "\n",
        encoding="utf-8"
    )
    port = free_port()
    log = (work / "logs" / "preview.log").open("w")
    preview = subprocess.Popen(
        ["pnpm", "exec", "vite", "preview", "--config", str(config),
         "--outDir", str(setup["dist"]), "--port", str(port), "--strictPort",
         "--host", "127.0.0.1"],
        cwd=frontend,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        base = f"http://127.0.0.1:{port}"
        wait_http(base + "/", SERVE_TIMEOUT)
        journey.add("画面を作って配信", "OK",
                    "ML-Provide の main の画面 (vite build と preview)")
        browse(apps, journey, setup, frontend, base)
    finally:
        # 起動に失敗して終わっていれば止めない
        if preview.poll() is None:
            signal_group(preview, signal.SIGTERM)
            preview.wait(timeout=20)
        log.close()


def browse(
    apps: Apps, journey: Journey, setup: dict[str, Any], frontend: Path,
    base: str
) -> None:
    """
    ブラウザの操作 (resnet_ui.mjs) を動かし, 結果を段階として記録する関数

    Parameters
    ----------
    apps: Apps
        起動した ML-Provide
    journey: Journey
        段階の記録
    setup: dict[str, Any]
        build の設定
    frontend: Path
        ML-Provide の画面のディレクトリ (Playwright を読む)
    base: str
        画面の URL

    Returns
    ----------
    None
    """
    # 画面の操作の引数 (トークン, 選ぶもの, 画像, 結果を書く場所)
    task = {
        "base": base, "token": apps.token, "frontend": str(frontend),
        "library": LIBRARY, "model": MODEL, "weights": WEIGHTS,
        "name": MODEL, "images": str(setup["images"]),
        "count": len(setup["names"]), "result": str(setup["result"]),
        "shots": str(apps.work / "shots")
    }
    script = Path(__file__).resolve().parent / "resnet_ui.mjs"
    done = subprocess.run(
        ["node", str(script), json.dumps(task)], timeout=BROWSER_TIMEOUT,
        capture_output=True, text=True
    )
    (apps.work / "logs" / "browser.log").write_text(
        done.stdout + done.stderr, encoding="utf-8"
    )
    result = json.loads(setup["result"].read_text(encoding="utf-8"))
    for step in result["steps"]:
        journey.add(step["step"], step["status"], step["note"])
    if done.returncode != 0:
        raise RuntimeError(f"ブラウザの操作が止まりました: {result.get('error')}")

    # 画面に出た種類の名前を, ImageNet の種類と直の予測に引き合わせる
    categories = set(ResNet18_Weights[WEIGHTS].meta["categories"])
    direct = direct_labels(setup["images"], setup["names"])
    shown = result["labels"]
    unknown = [n for n in setup["names"] if shown.get(n) not in categories]
    journey.add(
        "画面の種類の名前が ImageNet の種類", "誤り" if unknown else "OK",
        ", ".join(f"{n}: {shown.get(n)}" for n in setup["names"])
    )
    same = [n for n in setup["names"] if shown.get(n) == direct[n]]
    journey.add(
        "torchvision で直に読んだ予測と同じ",
        "OK" if len(same) == len(setup["names"]) else "誤り",
        f"{len(same)} / {len(setup['names'])} 枚が同じ"
        + "".join(f". {n}: 直に {direct[n]}" for n in setup["names"]
                  if n not in same)
    )


if __name__ == "__main__":
    failed = main_with(build, "本物の ResNet18 の画面の通し", sys.argv[1:],
                       runner=run_ui, prefix="m3_resnet_", names=("provide",))
    sys.exit(1 if failed else 0)
