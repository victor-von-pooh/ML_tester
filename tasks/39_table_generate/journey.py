"""
表を作る (本物の表 California の住宅の表に似た行を作る) を, 問題の設定から
推論まで 5 つの画面のアプリの API だけで通し, 段階ごとに画面だけでできたか
(OK), 手で写すか打つか (手), 誤りかを表にする
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import (  # noqa: E402
    Apps, Client, Journey, main_with, wait_status
)

# sklearn の取得の置き場 (05 と共有) と, 元の表から使う行の数
DATA_HOME = Path(__file__).resolve().parents[2] / "data" / "sklearn"
ROWS = 5000

# 推論で作る行の数 (ML-Provide の 1 回の上限) と種, 結果の行を読む 1 回の数
COUNT = 1000
SEED = 7
PAGE = 500

# 学んだと見なすテスト用の列ごとの近さの平均の上限 (学ばない雑音のままの
# 表では列ごとの KS 統計量が 1 に近く出るため) と, 推論の表の近さの平均の
# テスト用の平均との差の上限 (一式が学習の評価と同じ表を作るかの目安)
LEARNED = 0.25
AGREE = 0.05

# 受け渡しの雛形と, 推論の一式の種類と版
EXAMPLE = "diffusion_denoiser"
BUNDLE = ("noise", 7)

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「総菜の作る量」の答えを, 表を作る課題に直す答え (受け持ちは作るで,
# 正解の出どころは自分自身. ML-Pathfinder の表の生成のテストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "住宅の価格の分析の係"),
    ("issue.pain", "地区ごとの住宅の表を外に出せず, 外の人と試せない"),
    ("issue.current", "手で作った表で試す"),
    ("issue.kpi.name", "試しの準備の日数"),
    ("issue.kpi.unit", "日"),
    ("issue.kpi.better", "lower"),
    ("decomposition", [{"cause": "本物の表を外に出せない",
                        "lever": "似た表を作る", "ml": True}]),
    ("action.what", "作った表を外の人に渡して試す"),
    ("action.who", "住宅の価格の分析の係"),
    ("action.when", "試しの前"),
    ("ml.reason", "元の表に似た行を作りたい"),
    (f"{PART}.role", "generate"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "table"),
    (f"{PART}.target.source", "self"),
    (f"{PART}.target.kind", "table"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.what", "地区ごとの住宅の表に似た行"),
    (f"{PART}.item", "地区の 1 行"),
    (f"{PART}.baseline", "手で作る"),
    (f"{PART}.evaluation.metric", "column_distance"),
    (f"{PART}.evaluation.link", "近いほど元の表の代わりに使える"),
    ("verification.method", "backtest"),
    ("verification.backtest", "元の表と列ごとに比べる"),
    ("verification.success", "準備の日数が半分"),
    ("operation.review", "月に 1 回"),
    ("operation.approver", "係の長"),
    ("operation.stop", "似なくなったら止める")
]


def build(work: Path) -> dict:
    """
    本物の表 (California の住宅の表) の一部を CSV に書き, 通しの設定を
    返す関数

    Parameters
    ----------
    work: Path
        作業用の場所

    Returns
    ----------
    setup: dict
        run_generate に渡す設定
    """
    # 地区ごとの 9 つの数の列 (目的の列 MedHouseVal も作る列) から, 種を
    # 決めて選んだ行 (学習と検証の時間を抑えるため)
    frame = fetch_california_housing(
        data_home=str(DATA_HOME), as_frame=True
    ).frame
    frame = frame.sample(n=ROWS, random_state=0).reset_index(drop=True)
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "housing.csv"
    frame.to_csv(data, index=False)
    setup = {"data": data, "work": work}

    return setup


def ks_distance(real: np.ndarray, made: np.ndarray) -> float:
    """
    2 つの数の並びの KS 統計量 (経験分布の差の最大) を返す関数

    Parameters
    ----------
    real: np.ndarray
        元の表の値
    made: np.ndarray
        作った表の値

    Returns
    ----------
    distance: float
        0 から 1 の KS 統計量
    """
    # 両方の値の全ての点での経験分布の差
    points = np.sort(np.concatenate([real, made]))
    cdf_real = np.searchsorted(np.sort(real), points, side="right")
    cdf_made = np.searchsorted(np.sort(made), points, side="right")
    distance = float(np.max(np.abs(cdf_real / len(real)
                                   - cdf_made / len(made))))

    return distance


def errors_of(view: dict[str, Any]) -> list[str]:
    """
    問題の設定の確かめの誤りの id を返す関数

    Parameters
    ----------
    view: dict[str, Any]
        ML-Pathfinder の問題の設定の画面の値

    Returns
    ----------
    ids: list[str]
        誤り (level が error) の確かめの id と, 答えの残る誤り
    """
    ids = [item["id"] for item in view.get("findings", [])
           if item.get("level") == "error"]
    ids += [item.get("id") for item in view.get("open_errors") or []]

    return ids


def set_problem(
    pf: Client, journey: Journey, setup: dict[str, Any], project: Path
) -> dict[str, Any]:
    """
    ML-Pathfinder で問題の設定を作り, 表を入れ, 骨組みを書き出す関数

    Parameters
    ----------
    pf: Client
        ML-Pathfinder の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        build の設定
    project: Path
        骨組みの書き出し先

    Returns
    ----------
    view: dict[str, Any]
        問題の設定の画面の値
    """
    # 見本から名前を付けた作成と, 表を作る課題への答え直しと表の入力
    view = pf.ok("POST", "/api/problems", {
        "example": "総菜の作る量", "name": "住宅の表に似た表"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    view = pf.ok("POST", f"/api/problems/{pid}/data",
                 {"path": str(setup["data"])})
    handoff = view.get("handoff") or {}
    model = handoff.get("model") or {}
    training = handoff.get("training") or {}
    errors = errors_of(view)
    journey.add(
        "問題の設定 (作るで表, 正解の出どころ self, 列ごとの分布の近さ)",
        "OK" if view["exportable"] and not errors
        and model.get("example") == EXAMPLE
        and training.get("monitor") == "valid/column_distance" else "誤り",
        f"雛形 {model.get('example')}, 監視 {training.get('monitor')} "
        f"({training.get('mode')}), 分割 "
        f"{(handoff.get('data') or {}).get('split')}, 誤り {errors}, "
        f"書き出せない理由 {view.get('export_error')}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    copied = f"data/{setup['data'].name}" in written["files"]
    origin = "正解の出どころ `self`" in readme
    journey.add(
        "骨組みの書き出し (表の写しと場所の記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and copied and origin and "元の表に似た行を作る" in readme
        else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. 表の写し {copied}. README に"
        f"正解の出どころ self の ML-Found の要件 {origin}"
    )
    view = {**view, "record": record}

    return view


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 目的の列の無いレシピで
    プロジェクトへ書き出す関数

    Parameters
    ----------
    fathom: Client
        ML-Fathom の客
    journey: Journey
        結果の記録
    name: str
        問題の設定の名前
    record: dict[str, Any]
        骨組みの記録

    Returns
    ----------
    entry: dict[str, Any]
        ML-Fathom の一覧の問題の設定
    """
    # 一覧から開き, 雛形のレシピ (task: none) のままの書き戻しと書き出し
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    check = fathom.ok(
        "POST", f"/api/sources/{sid}/problem/check", {"recipe": recipe}
    )
    fathom.ok(
        "POST", f"/api/sources/{sid}/problem/write",
        {"recipe": recipe, "version": check["version"]}
    )
    exported = fathom.ok(
        "POST", f"/api/sources/{sid}/export_project", {"recipe": recipe}
    )
    journey.add(
        "ML-Fathom (目的の列の無いレシピ, プロジェクトへ書き出す)",
        "OK" if recipe.get("task") == "none" and not recipe.get("target")
        and not check["conflicts"]
        and Path(exported["path"]).resolve()
        == Path(record["dataset"]).resolve() else "誤り",
        f"task {recipe.get('task')}, 分割 {recipe['split']}, 手順 "
        f"{[s['op'] for s in recipe['steps']]}, 食い違い "
        f"{check['conflicts']}, 注意 {started.get('notes')}"
    )

    return entry


def make_model(
    ca: Client, journey: Journey, original: Path, project: Path
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 雑音の予測器の雛形を
    符号化した行の幅に書き換えてプロジェクトに保存する関数

    Parameters
    ----------
    ca: Client
        ML-CreateArchitecture の客
    journey: Journey
        結果の記録
    original: Path
        元の問題の設定 (problem.yaml)
    project: Path
        書き出したプロジェクト

    Returns
    ----------
    None
    """
    # 一覧から選び, 選んだ問題の設定の雛形とデータセット
    listed = ca.ok("GET", "/api/problems")["problems"]
    chosen = next(
        (p for p in listed if Path(p["path"]).resolve() == original), None
    )
    if chosen is None or not chosen["usable"]:
        reason = chosen["reason"] if chosen else "一覧にありません"
        raise RuntimeError(f"ML-CreateArchitecture の問題の設定: {reason}")
    target = ca.ok("POST", "/api/project/target", {"path": chosen["path"]})
    journey.add(
        "ML-CreateArchitecture で問題の設定を一覧から選ぶ",
        "OK" if target.get("example") == EXAMPLE and target.get("dataset")
        and Path(chosen["project"]).resolve() == project else "誤り",
        f"雛形 {target.get('example')}, データセットの欄 "
        f"{Path(target.get('dataset') or '-').name}"
    )

    # 符号化した行の幅での雛形の書き換えと保存
    info = ca.ok(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    table = info.get("table") or {}
    template = ca.ok("GET", f"/api/examples/{EXAMPLE}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": EXAMPLE,
        "values": {"features": table.get("width")}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    journey.add(
        "ML-CreateArchitecture (雛形を行の幅に書き換え, 保存)",
        "OK" if table.get("width") and not saved.get("notes")
        and spec["inputs"]["x"]["shape"][-1] == table["width"] else "誤り",
        f"幅 {table.get('width')} (数の列 {table.get('numbers')}, 種類の列 "
        f"{table.get('categories')}), 入力 {spec['inputs']['x']['shape']}, "
        f"出力 {sorted(spec['outputs'])}, 注意 "
        f"{len(saved.get('notes') or [])} 件"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> tuple[str, float]:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 列ごとの分布の近さを読む関数

    Parameters
    ----------
    dash: Client
        ML-Dashboard の客
    journey: Journey
        結果の記録
    original: Path
        元の問題の設定 (problem.yaml)
    project: Path
        書き出したプロジェクト

    Returns
    ----------
    result: tuple[str, float]
        実験の id と, テスト用の列ごとの近さの平均
    """
    # 一覧からの登録と, 画面からの環境づくり
    recorded = dash.ok("GET", "/api/problems/projects")["projects"]
    found = next(
        (p for p in recorded
         if Path(p["problem_path"]).resolve() == original), None
    )
    if found is None:
        raise RuntimeError("ML-Dashboard の一覧に書き出したプロジェクトが"
                           "ありません")
    did = dash.ok("POST", "/api/projects", {"path": found["path"]})["id"]
    ready = dash.ok("GET", f"/api/projects/{did}/readiness")
    started_at = time.time()
    dash.ok("POST", f"/api/projects/{did}/environment",
            {"commands": ready["commands"]})
    job = wait_status(dash, f"/api/projects/{did}/environment", ("running",))
    ready = dash.ok("GET", f"/api/projects/{did}/readiness?refresh=true")
    journey.add(
        "ML-Dashboard で登録し環境を作る",
        "OK" if Path(found["path"]).resolve() == project
        and job["status"] == "succeeded" and ready["ready"] else "誤り",
        f"{job['status']}, {time.time() - started_at:.0f} 秒, 足りない "
        f"{ready['missing']}"
    )

    # 骨組みを書き換えない学習と, 列ごとの分布の近さと見本の読み込み
    started_at = time.time()
    launched = dash.ok(
        "POST", f"/api/projects/{did}/jobs",
        {"script": "train.py", "config_path": "configs/experiment/train.yaml"}
    )
    job = wait_status(
        dash, f"/api/jobs/{launched['id']}", ("running",),
        stop=f"/api/jobs/{launched['id']}/stop"
    )
    run = job.get("run") or {}
    if job.get("return_code") != 0 or not run:
        tail = dash.ok("GET", f"/api/jobs/{launched['id']}/output")["text"]
        lines = tail.strip().splitlines() or [str(job.get("status"))]
        raise RuntimeError(f"学習: {lines[-1]}")
    results = dash.ok("GET", f"/api/projects/{did}/runs/{run['id']}/results")
    tables = {e["name"]: e["rows"] for e in results["evaluations"]
              if e["name"].startswith("column_distance_")}
    test = tables.get("column_distance_test") or []
    mean = next((r for r in test if r.get("type") is None), {})
    worst = max((r for r in test if r.get("type") is not None),
                key=lambda r: r["distance"], default={})
    samples = [s["name"] for s in results.get("samples") or []]
    journey.add(
        "学習 (骨組みを書き換えない) と列ごとの分布の近さ",
        "OK" if mean.get("distance", np.nan) < LEARNED
        and "column_distance_valid" in tables else "誤り",
        f"{time.time() - started_at:.0f} 秒, {run['best']['monitor']} "
        f"{run['best']['value']:.4f}. テスト用の列の平均の近さ "
        f"{mean.get('distance', float('nan')):.4f} (同じ分布の目安 "
        f"{mean.get('train_distance', float('nan')):.4f}), 最も遠い列 "
        f"{worst.get('column')} {worst.get('distance', float('nan')):.4f}. "
        f"見本 {samples}. 上限 {LEARNED}"
    )
    result = (str(run["id"]), float(mean.get("distance", np.nan)))

    return result


def read_rows(client: Client, result_id: str, count: int) -> pd.DataFrame:
    """
    ML-Provide の結果の行を全て読む関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    result_id: str
        結果の id
    count: int
        行の数

    Returns
    ----------
    frame: pd.DataFrame
        結果の行
    """
    # 1 回の上限ずつの読みと, 読んだ行の数の確かめ
    rows: list[dict] = []
    for offset in range(0, count, PAGE):
        page = client.ok(
            "GET",
            f"/api/results/{result_id}/rows?offset={offset}&limit={PAGE}"
        )
        rows += page["rows"]
    if not len(rows) == page["total"] == count:
        raise RuntimeError(f"結果の行の数 {len(rows)} (全体 "
                           f"{page['total']}) が {count} と合いません")
    frame = pd.DataFrame(rows)

    return frame


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    data: Path, test_mean: float
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 行の数と種で表を作り, 元の表と
    列ごとに比べる関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    project: Path
        書き出したプロジェクト
    run_id: str
        ML-Dashboard の実験の id
    data: Path
        元の表
    test_mean: float
        学習の評価のテスト用の列ごとの近さの平均

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と版と出力の形の確かめ
    listing = client.ok("GET", "/api/experiments")
    row = next(
        (r for r in listing["experiments"]
         if Path(r["project"]).resolve() == project
         and r["experiment"] == run_id), None
    )
    if row is None:
        raise RuntimeError("ML-Provide の実験の一覧に学習した実験が"
                           "ありません")
    bundle = client.ok("POST", "/api/bundles", {
        "path": row["path"], "name": f"{row['problem']} ({row['experiment']})"
    })
    bid = str(bundle["id"])
    if not bundle.get("ok", True) or not bundle.get("description"):
        raise RuntimeError(f"一式を読めません: {bundle.get('error')}")
    description = bundle["description"]
    form = (description.get("output") or {}).get("form")
    columns = [c["name"] for c in
               (description.get("output") or {}).get("columns") or []]
    got = (description.get("kind"), description.get("format_version"))
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got == BUNDLE and form == "table" and columns else "誤り",
        f"種類と版 {got}, 出力の形 {form}, 列 {columns}"
    )

    # 行の数と種での推論と, 同じ種での作り直しの一致
    url = f"/api/bundles/{bid}/predict"
    result = client.ok("POST", url, {"count": COUNT, "seed": SEED})
    again = client.ok("POST", url, {"count": COUNT, "seed": SEED})
    made = read_rows(client, result["id"], COUNT)
    remade = read_rows(client, again["id"], COUNT)
    missing = [name for name in columns if name not in made]
    if missing:
        raise RuntimeError(f"結果の表に出力の列 {missing} がありません")
    same = made[columns].equals(remade[columns])

    # 作った表 (作れた行) と元の表の列ごとの KS 統計量 (学習の評価の
    # 数え方の数の列と同じ)
    real = pd.read_csv(data)
    ok = made[made["status"] == "ok"]
    distances = {
        name: round(ks_distance(real[name].to_numpy(float),
                                ok[name].dropna().to_numpy(float)), 4)
        for name in columns
    }
    done = int((made["status"] == "ok").sum())
    mean = float(np.mean(list(distances.values())))
    journey.add(
        "推論 (行の数と種で表を作り, 元の表と比べる)",
        "OK" if result["summary"].get("predicted") == COUNT
        and done == COUNT and same
        and abs(mean - test_mean) < AGREE else "誤り",
        f"{result['summary']} (種 {SEED}), 同じ種で作り直して一致 {same}. "
        f"元の表との列ごとの KS 統計量 "
        f"{json.dumps(distances, ensure_ascii=False)}, 平均 {mean:.4f} "
        f"(テスト用の平均との差 {abs(mean - test_mean):.4f}, 上限 {AGREE})"
    )


def run_generate(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    問題の設定から推論までを画面と同じ API で通す関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        build の設定

    Returns
    ----------
    None
    """
    clients = apps.clients
    project = (setup["work"] / "project").resolve()
    view = set_problem(clients["pathfinder"], journey, setup, project)
    entry = analyze(clients["fathom"], journey, view["name"], view["record"])
    original = Path(entry["path"]).resolve()
    make_model(clients["createarchitecture"], journey, original, project)
    run_id, test_mean = train(clients["dashboard"], journey, original,
                              project)
    provide(clients["provide"], journey, project, run_id, setup["data"],
            test_mean)


if __name__ == "__main__":
    failed = main_with(build, "表を作る (住宅の表)", sys.argv[1:],
                       runner=run_generate, prefix="generate_")
    sys.exit(1 if failed else 0)
