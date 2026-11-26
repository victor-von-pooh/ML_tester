"""
似た写真を探す (種類ごとのディレクトリの合成の写真) を, 問題の設定から推論
まで画面のアプリ (ML-Pathfinder, ML-Provide) の API と骨組みの bundle.py で
通し, 段階ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを表に
する (似た写真を探すを画面で通す節目の確かめ)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import JOB_TIMEOUT, Apps, Client, Journey, main_with, problem_file

# ML-Pathfinder の見本の名前
EXAMPLE = "似た商品の写真を探す (既製の埋め込み)"

# 写真の種類と, 種類ごとの枚数 (索引と検証用の問いに分ける写真と, ML-Provide
# で探す問いの写真)
KINDS = ["circle", "square", "triangle", "stripes"]
COUNT = 20
QUERY_COUNT = 3

# 写真の辺 (骨組みの前処理が既製のモデルの大きさにそろえる)
SIZE = 64

# 一式づくりの環境に入れるもの (骨組みの bundle.py の入れ方と同じ)
PACKAGES = [
    "ml-inference @ git+https://github.com/victor-von-pooh/"
    "ML-Inference.git",
    "ml-modelcontrol[torchvision] @ git+https://github.com/victor-von-pooh/"
    "ML-ModelControl.git"
]

# 起動するアプリ (分析, モデルの作成, 学習の段階は通らない)
NAMES = ("pathfinder", "provide")


def draw_photo(kind: str, rng: np.random.Generator) -> Image.Image:
    """
    雑音の背景に, 色と位置と大きさの違う図形を 1 つ描く関数

    Parameters
    ----------
    kind: str
        circle, square, triangle, stripes のどれか
    rng: np.random.Generator
        乱数

    Returns
    ----------
    image: Image.Image
        SIZE x SIZE の RGB の写真
    """
    # 色の違う雑音の背景と, 色の違う図形 (色では種類が分からない)
    tint = rng.integers(0, 120, 3)
    noise = rng.integers(0, 60, (SIZE, SIZE, 3)) + tint
    image = Image.fromarray(noise.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    color = tuple(int(c) for c in rng.integers(120, 256, 3))
    half = int(rng.integers(14, 24))
    cx, cy = (int(c) for c in rng.integers(half + 2, SIZE - half - 2, 2))
    box = (cx - half, cy - half, cx + half, cy + half)
    if kind == "circle":
        draw.ellipse(box, fill=color)
    elif kind == "square":
        draw.rectangle(box, fill=color)
    elif kind == "triangle":
        draw.polygon([(cx, cy - half), (cx - half, cy + half),
                      (cx + half, cy + half)], fill=color)
    else:
        # 横の縞 (4 px の線を 4 px おき)
        for top in range(cy - half, cy + half, 8):
            draw.rectangle((cx - half, top, cx + half, top + 3), fill=color)

    return image


def make_photos(root: Path, count: int, seed: int) -> None:
    """
    <種類>/<写真>.png のディレクトリに合成の写真を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ
    count: int
        種類ごとの枚数
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 種類ごとのディレクトリ (骨組みが評価の正解に使う名前)
    rng = np.random.default_rng(seed)
    for kind in KINDS:
        (root / kind).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            draw_photo(kind, rng).save(root / kind / f"{index:03d}.png")


def build(work: Path) -> dict[str, Any]:
    """
    写真を作り, 通しの設定を返す関数

    Parameters
    ----------
    work: Path
        作業用の場所

    Returns
    ----------
    setup: dict[str, Any]
        run_search に渡す設定
    """
    # 索引にする写真と, 別の乱数の種の問いの写真
    photos = work / "data" / "photos"
    make_photos(photos, COUNT, seed=0)
    queries = work / "data" / "queries"
    make_photos(queries, QUERY_COUNT, seed=1)
    setup = {"example": EXAMPLE, "data": photos, "past": queries}

    return setup


def bundle_environment(
    project: Path, logs: Path, journey: Journey
) -> Path:
    """
    骨組みの README のとおりに, 一式づくりの環境をプロジェクトに作る関数

    Parameters
    ----------
    project: Path
        書き出したプロジェクト
    logs: Path
        ログのディレクトリ (作業用の場所の logs)
    journey: Journey
        結果の記録

    Returns
    ----------
    python: Path
        環境の Python
    """
    # uv の仮想環境と, ML-Inference と ML-ModelControl の torchvision の
    # 追加の依存 (ML_tester の .venv の外)
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    started = time.time()
    venv = project / ".venv"
    python = venv / "bin" / "python"
    log = logs / "bundle_env.log"
    with log.open("w") as out:
        for command in (
            ["uv", "venv", "--python", "3.13", str(venv)],
            ["uv", "pip", "install", "--python", str(python), *PACKAGES]
        ):
            done = subprocess.run(
                command, env=env, stdout=out, stderr=subprocess.STDOUT,
                timeout=JOB_TIMEOUT
            )
            if done.returncode != 0:
                break
    if done.returncode != 0:
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        raise RuntimeError(f"一式づくりの環境を作れません: {lines[-1:]}")
    journey.add(
        "一式づくりの環境 (骨組みの README の入れ方)", "手",
        f"画面に無いのでコマンドで作る. {time.time() - started:.0f} 秒"
    )

    return python


def run_bundle(
    python: Path, project: Path, logs: Path, journey: Journey
) -> dict[str, Any]:
    """
    骨組みの bundle.py で索引と評価と検索の一式を作る関数

    Parameters
    ----------
    python: Path
        一式づくりの環境の Python
    project: Path
        書き出したプロジェクト
    logs: Path
        ログのディレクトリ (作業用の場所の logs)
    journey: Journey
        結果の記録

    Returns
    ----------
    made: dict[str, Any]
        config (bundle.yaml の中身), row (評価の行), index (索引の説明),
        bundle (一式のディレクトリ)
    """
    # 書き換えない骨組みのまま, プロジェクトのディレクトリで動かす
    started = time.time()
    log = logs / "bundle.log"
    with log.open("w") as out:
        done = subprocess.run(
            [str(python), "bundle.py"], cwd=project, stdout=out,
            stderr=subprocess.STDOUT, timeout=JOB_TIMEOUT
        )
    if done.returncode != 0:
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        raise RuntimeError(f"bundle.py が止まりました: {lines[-1:]}")

    # 成果物の評価と索引, 一式の場所 (config の outputs と bundle)
    config = yaml.safe_load(
        (project / "configs" / "default" / "bundle.yaml").read_text(
            encoding="utf-8"
        )
    )
    artifacts = project / config["outputs"]["search"]
    table = json.loads((artifacts / "evaluations" / "search_valid.json")
                       .read_text(encoding="utf-8"))
    index = json.loads((artifacts / "search_index.json")
                       .read_text(encoding="utf-8"))
    row = table["rows"][0]
    made = {"config": config, "row": row, "index": index,
            "bundle": project / config["bundle"]["path"]}
    journey.add(
        "索引と評価と一式づくり (骨組みの bundle.py)", "手",
        f"コマンドで動かす. {time.time() - started:.0f} 秒. 索引 "
        f"{len(index['items'])} 枚, 検証用の問い {row['queries']} 枚, "
        f"上位 {row['k']} 件の Recall@K {row['recall_at_k']:.4f} (縮めた画素 "
        f"{row['recall_at_k_baseline']:.4f}, 差 "
        f"{row['recall_at_k_diff']:+.4f}), NDCG@K {row['ndcg_at_k']:.4f} "
        f"(縮めた画素 {row['ndcg_at_k_baseline']:.4f}, 差 "
        f"{row['ndcg_at_k_diff']:+.4f})"
    )

    return made


def query_hits(provide: Client, result_id: str) -> dict[str, Any]:
    """
    推論の結果の表から, 問いごとの上位 K 件の種類の当たりを数える関数

    Parameters
    ----------
    provide: Client
        ML-Provide の客
    result_id: str
        結果の id

    Returns
    ----------
    hits: dict[str, Any]
        queries (並べた問いの数), top1 (1 位が問いと同じ種類の問いの数),
        share (上位 K 件のうち問いと同じ種類の割合の平均)
    """
    # 問いの種類は読み込みの表の label (予測の列と同じ名前なので「(入力)」)
    rows = provide.ok(
        "GET", f"/api/results/{result_id}/rows?status=ok&limit=500"
    )["rows"]
    found: dict[str, list] = {}
    for row in rows:
        found.setdefault(row["path"], []).append(
            (row["rank"], row["label (入力)"] == row["label"])
        )
    shares = [np.mean([same for _, same in items])
              for items in found.values()]
    hits = {
        "queries": len(found),
        "top1": sum(1 for items in found.values() if min(items)[1]),
        "share": float(np.mean(shares)) if shares else 0.0
    }

    return hits


def link_problem(
    provide: Client, journey: Journey, bundle: dict[str, Any],
    project: Path
) -> None:
    """
    一式の画面の候補を見て, 元の問題の設定をつなぐ関数

    一式が問題の設定を使わない (画面に欄が無い) 場合は, つながずに欄が
    無いことを記録する

    Parameters
    ----------
    provide: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    bundle: dict[str, Any]
        登録した一式 (id と, 問題の設定を使うかの uses_problem)
    project: Path
        書き出したプロジェクト (problem.yaml の写しの origin が元を指す)

    Returns
    ----------
    None
    """
    # 問題の設定を使わない一式は, 画面に欄が無く, つながない
    bid = str(bundle["id"])
    if bundle.get("uses_problem") is False:
        journey.add(
            "問題の設定の欄 (一式の画面)", "OK",
            "問題の設定を使わない一式なので欄が無い (件数と待ち時間の上限"
            "は一式が写して持つ)"
        )
        return

    # 候補は画面の欄に出るだけで, つなぐのは写しの origin が指す元の問題
    # の設定 (ML-Pathfinder で直した値が届く方)
    origin = yaml.safe_load(
        (project / "problem.yaml").read_text(encoding="utf-8")
    ).get("origin")
    state = provide.ok("GET", f"/api/bundles/{bid}/problem")
    hint = state["hint"]
    is_original = bool(hint and origin) and problem_file(hint) == (
        problem_file(str(origin))
    )
    shown = ("候補は元の問題の設定. " if is_original
             else f"候補 {hint or '無し'}. ")
    if not origin:
        journey.add("問題の設定をつなぐ (候補から)", "誤り",
                    shown + "骨組みの problem.yaml に元の場所 (origin) が無い")
        return
    code, value = provide.call(
        "PUT", f"/api/bundles/{bid}/problem", {"path": str(origin)}
    )
    linked = value.get("linked") if code == 200 else None
    if linked is None:
        detail = value.get("detail") if isinstance(value, dict) else value
        journey.add("問題の設定をつなぐ (候補から)", "誤り",
                    shown + f"元の問題の設定をつなぐと {code}: {detail}")
        return

    # 食い違いは誤り, 候補が元でなく選び直すなら手
    status = "OK" if is_original else "手"
    if linked["issues"]:
        status = "誤り"
    journey.add(
        "問題の設定をつなぐ (候補から)", status,
        shown + f"元の問題の設定をつなぐ. {linked.get('text')} 食い違い "
        f"{linked['issues']}"
    )


def run_search(apps: Apps, journey: Journey, setup: dict[str, Any]) -> None:
    """
    問題の設定から推論までを, 画面と同じ API と骨組みの bundle.py で通す
    関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ (ML-Pathfinder と ML-Provide)
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        example (見本の名前), data (索引にする写真), past (問いの写真),
        project (書き出し先)

    Returns
    ----------
    None
    """
    pf, provide = apps.clients["pathfinder"], apps.clients["provide"]
    project = Path(setup["project"]).resolve()

    # 1. 問題の設定: 見本から作り, 写真のディレクトリを入れる
    problem = pf.ok("POST", "/api/problems", {"example": setup["example"]})
    pid = problem["id"]
    problem = pf.ok("POST", f"/api/problems/{pid}/data",
                    {"path": str(setup["data"])})
    journey.add(
        "問題の設定 (見本, 写真のディレクトリ)",
        "OK" if problem["exportable"] and not problem["open_errors"]
        else "誤り",
        f"骨組みの型 {problem['skeleton_type']}. 直すところ "
        f"{len(problem['open_errors'])}"
        + (f". {problem['export_error']}" if problem["export_error"] else "")
    )

    # 2. 骨組みを書き出す (写真はプロジェクトに写り, config が指す)
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    config = yaml.safe_load(
        (project / "configs" / "default" / "bundle.yaml").read_text(
            encoding="utf-8"
        )
    )
    photos = project / config["data"]["photos"]
    count = sum(1 for _ in photos.rglob("*.png"))
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if "bundle.py" in written["files"] and count
        and Path(written["project"]["path"]).resolve() == project
        else "誤り",
        f"bundle.py と config. 写真 {count} 枚を {config['data']['photos']}"
        f" に写し, config の返す件数 {config['search']['k']}, 待ち時間の"
        f"上限 {config['search']['max_latency_ms']} ms"
    )

    # 3. 一式づくり (画面に無い段階. 骨組みの README のとおり)
    logs = apps.work / "logs"
    python = bundle_environment(project, logs, journey)
    made = run_bundle(python, project, logs, journey)

    # 4. ML-Provide: 場所で登録し, 元の問題の設定をつなぐ
    bundle = provide.ok("POST", "/api/bundles",
                        {"path": str(made["bundle"])})
    bid = str(bundle["id"])
    description = bundle.get("description") or {}
    journey.add(
        "ML-Provide で一式を場所で登録", "OK" if bundle["ok"] else "誤り",
        f"一式の場所は config の bundle.path. 種類 "
        f"{description.get('kind')}, 版 {description.get('format_version')}"
    )
    link_problem(provide, journey, bundle, project)

    # 5. 問いの写真ごとの上位 K 件と待ち時間 (件数は一式の既定)
    result = provide.ok("POST", f"/api/bundles/{bid}/predict",
                        {"source": str(setup["past"])})
    summary = result["summary"] or {}
    latency = summary.get("latency") or {}
    hits = query_hits(provide, result["id"])
    total = len(KINDS) * QUERY_COUNT
    journey.add(
        "推論 (問いの写真ごとに上位 K 件と待ち時間)",
        "OK" if summary.get("predicted") == total and latency else "誤り",
        f"問い {summary.get('predicted')} 枚に上位 {summary.get('k')} 件. "
        f"1 位が同じ種類 {hits['top1']} 枚, 上位の同じ種類の割合の平均 "
        f"{hits['share']:.4f}. 待ち時間は中央値 "
        f"{latency.get('median', float('nan')):.1f} ms, 最大 "
        f"{latency.get('max', float('nan')):.1f} ms, 上限 "
        f"{summary.get('max_latency_ms')} ms を超えた問い "
        f"{summary.get('over_latency')} 枚"
    )
    evaluation = summary.get("evaluation") or {}
    same = all(evaluation.get(key) == value
               for key, value in made["row"].items())
    journey.add(
        "骨組みの評価が推論の結果に出る",
        "OK" if same else "誤り",
        " ".join(result["sentences"][-1:])
    )


if __name__ == "__main__":
    failed = main_with(build, "似た写真を探す (合成の写真)", sys.argv[1:],
                       runner=run_search, prefix="search_", names=NAMES)
    sys.exit(1 if failed else 0)
