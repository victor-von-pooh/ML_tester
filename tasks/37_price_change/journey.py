"""
値段を変える判断 (試しに値段を変えた売れ行きの記録から, 商品ごとに利益が
最も大きい値段を選ぶ) を, 問題の設定から推論まで 5 つの画面のアプリの API
だけで通し, 段階ごとに画面だけでできたか (OK), 手で写すか打つか (手),
誤りかを表にする
"""
import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import (  # noqa: E402
    Apps, Client, Journey, main_with, wait_status
)

# 商品の数と, 商品ごとの学習の記録の行の数
PRODUCTS = 20
DAYS = 500

# 値段の候補 (今の値段からの割合) と変える幅の上限
CANDIDATES = [-0.1, -0.05, 0, 0.05, 0.1]
MAX_CHANGE = 0.1

# 試しに変えた値段の倍率
SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]

# 種類ごとの値段の弾力性 (食品は値段に敏感で下げると得, 道具は鈍く上げると
# 得)
ELASTICITY = {"food": -3.0, "tool": -1.2}

# 推論で値段を下げる食品の数の下限. 真の弾力性 -3.0 からは食品の 10 商品の
# 全てで 1 割下げるのが最も良い. 学習が足りず食品の値段への敏感さを小さく
# 学ぶと 1 つも下げないので, その向きを学んだしるしとして半数を線にする
FOOD_LOWERED = 5

# 骨組みの雛形と推論の一式の版
EXAMPLE = "tabular_price"
VERSION = 13

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「解約の引き止め」の答えを, 値段の課題に直す答え (受け持ちは効果を
# 見積もるで, 今ある記録で値段を変えた. ML-Pathfinder の値段を変える判断の
# テストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "値付けの担当"),
    ("issue.pain", "値段を全ての商品で据え置いていて, 利益が伸びない"),
    ("issue.current", "全ての商品を今の値段のまま売る"),
    ("issue.kpi.name", "1 日あたりの利益"),
    ("issue.kpi.unit", "円 / 日"),
    ("issue.kpi.better", "higher"),
    ("issue.kpi.current", 50000),
    ("issue.kpi.target", 55000),
    ("issue.guard.name", "今の値段からの変える幅"),
    ("issue.guard.unit", "割合"),
    ("issue.guard.better", "lower"),
    ("issue.guard.line", MAX_CHANGE),
    ("issue.guard.measure", "選んだ値段と今の値段の比を商品ごとに数える"),
    ("action.what", "商品ごとの値段"),
    ("action.who", "値付けの担当"),
    ("action.when", "毎週月曜"),
    ("ml.reason", "商品ごとに値段への売れ行きの敏感さが違い, 一律には"
                  "決められない"),
    (f"{PART}.role", "effect"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "effect"),
    (f"{PART}.target.source", "intervention"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.autonomous", False),
    (f"{PART}.collect", "log"),
    (f"{PART}.varied", "price"),
    (f"{PART}.what", "値段ごとの売れた数"),
    (f"{PART}.item", "商品と日ごとの 1 行"),
    (f"{PART}.baseline", "今の値段のまま"),
    ("verification.trial", "商品を 2 つに分け, 片方だけ選んだ値段にする"),
    ("verification.success", "選んだ値段の側の利益が, 今の値段の側より"
                             "大きい"),
    ("operation.stop", "2 週続けて利益が今の値段の側を下回ったら止める")
]

# 表を入れた後に答える, 値段の列, 評価と値段の判断の問い (画面は候補を
# カンマで区切った文で送る)
PRICE_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.price.price", "price"),
    (f"{PART}.price.units", "units"),
    (f"{PART}.price.cost", "unit_cost"),
    (f"{PART}.price.product", "product_id"),
    (f"{PART}.price.id", "record_id"),
    (f"{PART}.price.columns", ["category", "weekday",
                                "list_price"]),
    (f"{PART}.evaluation.metric", "mae"),
    (f"{PART}.evaluation.link", "売れた数が当たるほど, 選んだ値段の利益が"
                                "見込みに近い"),
    ("action.decision", {
        "type": "price",
        "candidates": ", ".join(f"{c:g}" for c in CANDIDATES),
        "max_change": str(MAX_CHANGE)
    })
]


def products(seed: int) -> pd.DataFrame:
    """
    商品ごとの id, 種類, 今の値段と原価を返す関数

    Parameters
    ----------
    seed: int
        乱数の種

    Returns
    ----------
    frame: pd.DataFrame
        product_id, category, base (今の値段), unit_cost (今の値段の 6 割)
    """
    # 半分が食品, 半分が道具. 今の値段は 100 から 300 の 10 刻み
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "product_id": [f"p{index:02d}" for index in range(PRODUCTS)],
        "category": ["food" if index % 2 == 0 else "tool"
                     for index in range(PRODUCTS)],
        "base": rng.integers(10, 31, PRODUCTS) * 10.0,
    })
    frame["unit_cost"] = (frame["base"] * 0.6).round(0)

    return frame


def true_profit(changes: dict[str, float]) -> float:
    """
    商品ごとの値段の変え方での, 合成のデータの真の売れた数の平均から数えた
    利益の和を返す関数 (週末の倍率は値段に依らないので除く)

    Parameters
    ----------
    changes: dict[str, float]
        商品の id ごとの今の値段からの割合

    Returns
    ----------
    total: float
        利益の和
    """
    # 利益は (値段 - 原価) × 20 × 倍率 ^ 弾力性
    total = 0.0
    for item in products(0).itertuples():
        scale = 1 + changes.get(item.product_id, 0.0)
        total += ((item.base * scale - item.unit_cost) * 20
                  * scale ** ELASTICITY[item.category])

    return total


def make_sales(path: Path, seed: int = 0, days: int = DAYS) -> None:
    """
    商品ごとに日ごとの値段を無作為に変えた合成の売れ行きの記録 (売れた後に
    決まる列 stock_left を持つ) を書く関数

    Parameters
    ----------
    path: Path
        書く CSV
    seed: int = 0
        日ごとの乱数の種 (商品の今の値段と原価は種 0 で固定)
    days: int = DAYS
        商品ごとの日の数 (1 なら全て今の値段)

    Returns
    ----------
    None
    """
    # 売れた数の平均は 20 × 倍率 ^ 弾力性 × 週末に 1.3 倍
    items = products(0)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for item in items.itertuples():
        weekday = rng.integers(0, 7, days)
        scale = (rng.choice(SCALES, days) if days > 1
                 else np.ones(days))
        mean = (20 * scale ** ELASTICITY[item.category]
                * np.where(weekday >= 5, 1.3, 1.0))
        units = rng.poisson(mean)
        for day in range(days):
            rows.append({
                "record_id": f"r{seed}_{item.product_id}_{day:03d}",
                "product_id": item.product_id, "category": item.category,
                "weekday": int(weekday[day]),
                "list_price": item.base,
                "price": float(round(item.base * scale[day], 0)),
                "unit_cost": item.unit_cost, "units": int(units[day]),
                "stock_left": int(max(0, 40 - units[day]))
            })
    pd.DataFrame(rows).to_csv(path, index=False)


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
        run_price に渡す設定
    """
    # 学習の記録と, 推論で値段を選ぶ商品ごとの今の値段の 1 行 (別の乱数の
    # 種で売れた数も持つので誤差を数え直せる)
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "sales.csv"
    make_sales(data)
    trial = work / "data" / "sales_now.csv"
    make_sales(trial, seed=1, days=1)
    setup = {"data": data, "trial": trial, "work": work}

    return setup


def rounded(value: Any) -> Any:
    """
    表に出す値の小数を 4 桁に丸める関数 (辞書は値ごと)

    Parameters
    ----------
    value: Any
        数, 辞書, その他

    Returns
    ----------
    result: Any
        小数を丸めた値 (小数でなければそのまま)
    """
    # 辞書は値ごと, 小数だけの丸め
    if isinstance(value, dict):
        result = {key: rounded(item) for key, item in value.items()}
    elif isinstance(value, float):
        result = round(value, 4)
    else:
        result = value

    return result


def within(change: float, limit: float) -> bool:
    """
    選んだ割合が変える幅の上限の中かを 10 進の表記どおりに比べる関数

    Parameters
    ----------
    change: float
        選んだ割合
    limit: float
        変える幅の上限

    Returns
    ----------
    inside: bool
        割合の絶対値が上限以下なら真
    """
    # 0.1 と -0.1 を上限 0.1 の中とする扱い (2 進の誤差で外にしない)
    inside = abs(Decimal(repr(change))) <= Decimal(repr(limit))

    return inside


def set_problem(
    pf: Client, journey: Journey, setup: dict[str, Any], project: Path
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    ML-Pathfinder で問題の設定を作り, 表を入れて答え, 骨組みを書き出す
    関数

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
    result: tuple[str, dict[str, Any], dict[str, Any]]
        問題の設定の id, 画面の値, 骨組みの記録
    """
    # 見本から名前を付けた作成と, 受け持ちと課題の値段への答え直し
    view = pf.ok("POST", "/api/problems", {
        "example": "解約の引き止め", "name": "値段を変える判断"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in PRICE_ANSWERS:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    price = handoff.get("price") or {}
    model = handoff.get("model") or {}
    decision = (handoff.get("inference") or {}).get("decision") or {}
    errors = errors_of(view) + [
        item.get("id") for item in view.get("open_errors") or []
    ]
    journey.add(
        "問題の設定 (表, 値段の列, 候補と変える幅の上限)",
        "OK" if view["exportable"] and not errors
        and price.get("price") == "price"
        and model.get("example") == EXAMPLE
        and decision.get("candidates") == CANDIDATES
        and decision.get("max_change") == MAX_CHANGE else "誤り",
        f"値段の列 {price}, 雛形 {model.get('example')}, 判断 {decision}, "
        f"監視 {(handoff.get('training') or {}).get('monitor')}, 除く列 "
        f"{(handoff.get('data') or {}).get('drop')}, 誤り {errors}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and "price_estimator.json" in readme else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}"
    )
    result = (pid, view, record)

    return result


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
        誤り (level が error) の確かめの id
    """
    ids = [item["id"] for item in view.get("findings", [])
           if item.get("level") == "error"]

    return ids


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 値段を変えた記録を要約し, 前処理
    してプロジェクトへ書き出す関数

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
        ML-Fathom の一覧の問題の設定 (hints に書き出しの入力の形)
    """
    # 一覧から開き, レシピの列で値段を変えた記録の要約 (値段の幅, 商品, 値段
    # と売れた数の関わり)
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    summary = fathom.ok("POST", f"/api/sources/{sid}/price", {
        "price": recipe.get("price"), "units": recipe.get("target"),
        "cost": recipe.get("cost"), "product": recipe.get("product"),
        "id": recipe.get("id")
    })
    dropped = sum([step["columns"] for step in recipe["steps"]
                   if step["op"] == "drop_columns"], [])
    codes = {item["code"]: item.get("columns")
             for item in summary["findings"]}
    elasticity = (summary.get("relation") or {}).get("elasticity")
    journey.add(
        "ML-Fathom の値段を変えた記録の要約",
        "OK" if summary["rows"] == PRODUCTS * DAYS
        and (summary.get("products") or {}).get("count") == PRODUCTS
        and elasticity is not None and elasticity < 0
        and "stock_left" in dropped else "誤り",
        f"{summary['rows']} 行, 値段の種類 "
        f"{summary['price_stats']['distinct']}, 商品 "
        f"{(summary.get('products') or {}).get('count')}, 値段と売れた数の"
        f"弾力性 {rounded(elasticity)}, 気付き {codes}. 雛形が落とす列 "
        f"{dropped}"
    )

    # 種類の列を数の列にする手順の, 雛形の標準化の前への足しと書き出し
    first = next((i for i, s in enumerate(recipe["steps"])
                  if s["op"] == "standardize"), len(recipe["steps"]))
    recipe["steps"].insert(first, {"op": "one_hot", "columns": ["category"]})
    fathom.ok("PUT", f"/api/sources/{sid}/recipe", {"recipe": recipe})
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
        "ML-Fathom (前処理, プロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"]) else "誤り",
        f"分割 {recipe['split']['method']}, 入力 "
        f"{exported['hints']['input_shape']}, 出す値 "
        f"{exported['hints']['outputs']}"
    )

    entry = {**entry, "hints": exported["hints"]}

    return entry


def make_model(
    ca: Client, journey: Journey, original: Path, project: Path,
    hints: dict[str, Any]
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 売れ行きの雛形を
    データセットの列の数に書き換えてプロジェクトに保存する関数

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
    hints: dict[str, Any]
        ML-Fathom の書き出しの入力の形 (データセットの入力が読めない
        ときに手で写す値)

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

    # 入力の幅 (特徴の列と値段の列の数) での雛形の書き換え
    code, info = ca.call(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    if code == 200:
        columns = int(info["inputs"][0]["shape"][-1])
        names = {key: value for key, value in (info.get("price") or {}).items()
                 if key != "columns"}
        status, note = "OK", f"データセットの入力から (列の名前 {names})"
    else:
        # 読めなければ, ML-Fathom の入力の形の手での写し
        columns = int(hints["input_shape"][-1])
        journey.add(
            "ML-CreateArchitecture のデータセットの入力", "誤り",
            f"{code}: {info}. 列の数を ML-Fathom の書き出しから写して続ける"
        )
        status, note = "手", "列の数を手で写す"
    template = ca.ok("GET", f"/api/examples/{EXAMPLE}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": EXAMPLE,
        "values": {"features": columns}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    outputs = sorted(spec["outputs"])
    journey.add(
        "ML-CreateArchitecture (雛形を列の数に書き換え, 保存)",
        # データセットを読めない注意は, 上の誤りと同じ理由なので手の扱い
        status if not saved.get("notes") or code != 200 else "誤り",
        f"{note}. 列 {columns}, 入力 {spec['inputs']['x']['shape']}, 出力 "
        f"{outputs}, 注意 {len(saved.get('notes') or [])} 件"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> str:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 値段の評価を読む関数

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
    run_id: str
        実験の id
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

    # 骨組みを書き換えない学習と, 売れた数の誤差, 弾力性, 見込みの利益の
    # 比べの読み込み
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
    rows = {e["name"]: e["rows"][0] for e in results["evaluations"]
            if e["name"].startswith("price_") and e["rows"]}
    test = rows.get("price_test") or {}
    keys = ("mae", "elasticity", "profit", "current_profit", "profit_ratio")
    journey.add(
        "学習 (骨組みを書き換えない) と値段の評価",
        # 値段を上げると売れた数が減る向き (弾力性の平均が負) を学んだこと
        "OK" if set(rows) == {"price_valid", "price_test"}
        and all(test.get(key) is not None for key in keys)
        and test["elasticity"] < 0 else "誤り",
        f"{time.time() - started_at:.0f} 秒, {run['best']['monitor']} "
        f"{run['best']['value']:.4f}. テスト用: "
        + ", ".join(f"{key} {rounded(test.get(key))}"
                    for key in (*keys, "changed", "products"))
    )
    run_id = str(run["id"])

    return run_id


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    trial: Path
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 商品ごとの今の値段の表を推論して,
    変える幅の上限の中の候補から利益が最も大きい値段を選ぶ関数

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
    trial: Path
        商品ごとの今の値段の 1 行 (売れた数の列も持つ)

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と版と値段の判断の確かめ
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
    decision = description.get("decision") or {}
    got = (description.get("kind"), description.get("format_version"))
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got == ("price", VERSION)
        and decision.get("candidates") == CANDIDATES
        and decision.get("max_change") == MAX_CHANGE else "誤り",
        f"種類と版 {got}. 一式の判断 {decision}. 列 "
        f"{ {k: v for k, v in (description.get('price') or {}).items()
             if k in ('price', 'cost', 'product', 'id')} }"
    )

    # 値段の一式は問題の設定を使わない (候補と上限は骨組みの config から
    # 一式が写して持つ). 欄が無く, 一式の判断のままの推論
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    uses = bundle.get("uses_problem")
    journey.add(
        "問題の設定の欄 (一式の画面)",
        "OK" if uses is False and state.get("linked") is None else "誤り",
        "問題の設定を使わない一式なので欄が無い (候補と上限は一式が持つ)"
        if uses is False else f"uses_problem {uses}, 候補 {state.get('hint')}"
    )
    result = client.ok("POST", f"/api/bundles/{bid}/predict",
                       {"source": str(trial)})
    summary = result["summary"]
    measured = summary.get("measured") or {}
    totals = (result.get("decision") or {}).get("summary") or {}
    prices = client.ok(
        "GET", f"/api/results/{result['id']}/prices?limit={PRODUCTS}"
    )
    rows = prices["rows"]
    inside = all(within(r["change"], MAX_CHANGE) for r in rows)
    chosen = {}
    for item in rows:
        chosen.setdefault(item["change"], []).append(item["product_id"])
    kinds = dict(zip(*products(0)[["product_id", "category"]].T.values))
    food = [r["change"] for r in rows if kinds[r["product_id"]] == "food"]
    tool = [r["change"] for r in rows if kinds[r["product_id"]] == "tool"]
    lowered = sum(change < 0 for change in food)
    # 選んだ値段と, 候補の中で真に最も良い値段の, 今の値段に対する真の
    # 利益の比
    current = true_profit({})
    ratio = true_profit({r["product_id"]: r["change"] for r in rows}) / current
    best = sum(
        max(true_profit({product: c}) - current for c in CANDIDATES)
        for product in kinds
    ) / current + 1
    journey.add(
        "推論 (商品ごとに上限の中で利益が最も大きい値段を選ぶ)",
        # 道具 (真の弾力性 -1.2) の真の最良は上限まで上げること, 食品
        # (-3.0) は下げること. 候補に 0 があり見込みの利益は今の値段を
        # 下回らないので, 利益の比べは補助
        "OK" if prices["total"] == PRODUCTS and len(rows) == PRODUCTS
        and inside and len(tool) == PRODUCTS // 2
        and all(change == MAX_CHANGE for change in tool)
        and len(food) == PRODUCTS // 2 and lowered >= FOOD_LOWERED
        and measured else "誤り",
        f"{summary['predicted']} 行を推論, 商品 {prices['total']}, 上限の中 "
        f"{inside}, 変える商品 {totals.get('changed')} (上限で抑えた "
        f"{totals.get('limited')}). 選んだ割合ごとの商品 "
        f"{ {k: len(v) for k, v in sorted(chosen.items())} }, 食品の割合 "
        f"{sorted(set(food))} (下げる {lowered} / {len(food)}), 道具の割合 "
        f"{sorted(set(tool))}. 真の利益の比 {ratio:.3f} (候補の中で真に最も"
        f"良い値段なら {best:.3f}). 見込みの利益 "
        f"{rounded(totals.get('total'))} (今の値段 "
        f"{rounded(totals.get('current_total'))}). 弾力性の平均 "
        f"{rounded((summary.get('elasticity') or {}).get('mean'))}. 今の値段"
        f"の記録で数え直した誤差 "
        f"{json.dumps(rounded(measured), ensure_ascii=False)}. "
        + " ".join(result["sentences"][:1])
    )


def run_price(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    値段を変える判断を, 問題の設定から推論までを画面と同じ API で通す関数

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
    try:
        _, view, record = set_problem(
            clients["pathfinder"], journey, setup, project
        )
        entry = analyze(clients["fathom"], journey, view["name"], record)
        original = Path(entry["path"]).resolve()
        make_model(clients["createarchitecture"], journey, original, project,
                   entry["hints"])
        run_id = train(clients["dashboard"], journey, original, project)
        provide(clients["provide"], journey, project, run_id,
                setup["trial"])
    except Exception as error:
        # 止まった段階の記録
        journey.add("止まった段階", "誤り", f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    failed = main_with(build, "値段を変える判断", sys.argv[1:],
                       runner=run_price, prefix="price_")
    sys.exit(1 if failed else 0)
