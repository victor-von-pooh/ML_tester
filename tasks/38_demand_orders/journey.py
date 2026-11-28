"""
時間をまたぐ量の判断 (商品ごとの日ごとの売れた数の記録から, 先の期間の
需要を分位点で予測し, 商品ごとに発注の数を決める) を, 問題の設定から推論
まで 5 つの画面のアプリの API だけで通し, 段階ごとに画面だけでできたか
(OK), 手で写すか打つか (手), 誤りかを表にする
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import (  # noqa: E402
    Apps, Client, Journey, main_with, wait_status
)

# 商品の数, 学習の記録の日の数, 推論で発注の数を決める試しの記録の日の数
PRODUCTS = 12
DAYS = 240
TRIAL_DAYS = 60

# 先の期間の日の数と, 品切れの損, 売れ残りの損, 発注の数の上限 (水準
# τ = 3 / (3 + 1) = 0.75)
HORIZON = 7
SHORTAGE = 3
HOLDING = 1
MAX_ORDER = 150

# 受け渡しの雛形, 学ぶ水準 (既定), 推論の一式の種類と版
EXAMPLE = "tabular_demand"
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
BUNDLE = ("demand", 14)

# テスト用の需要の評価の行の数 (12 商品の 7 行ずつ)
TEST_ROWS = 84

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「解約の引き止め」の答えを, 発注の課題に直す答え (受け持ちは予測
# するで, 正解の種類は先の期間の需要. ML-Pathfinder の時間をまたぐ量の
# 判断のテストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "店の発注の担当"),
    ("issue.pain", "発注の数を勘で決めていて, 品切れと売れ残りが多い"),
    ("issue.current", f"前の {HORIZON} 日と同じ数を発注する"),
    ("issue.kpi.name", "品切れと売れ残りの損"),
    ("issue.kpi.unit", "円 / 週"),
    ("issue.kpi.better", "lower"),
    ("issue.kpi.current", 1000),
    ("issue.kpi.target", 700),
    ("issue.guard.name", "1 商品の発注の数"),
    ("issue.guard.unit", "個 / 回"),
    ("issue.guard.better", "lower"),
    ("issue.guard.line", MAX_ORDER),
    ("issue.guard.measure", "1 回に発注した数を商品ごとに数える"),
    ("action.what", "商品ごとの発注の数"),
    ("action.who", "店の発注の担当"),
    ("action.when", "毎週月曜日"),
    ("ml.reason", "商品ごとに売れ行きの波が違い, 勘では品切れと売れ残りの"
                  "釣り合いが取れない"),
    (f"{PART}.role", "predict"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "distribution"),
    (f"{PART}.target.kind", "demand"),
    (f"{PART}.target.source", "column"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.what", f"先の {HORIZON} 日の売れる数"),
    (f"{PART}.item", "商品と発注を決める日ごとの 1 行"),
    (f"{PART}.baseline", f"前の {HORIZON} 日と同じ数を発注する"),
    ("verification.trial", "店を 2 つに分け, 片方だけ予測から発注する"),
    ("verification.success", "予測から発注した店の品切れと売れ残りの損が, "
                             "前と同じ数を発注した店より小さい"),
    ("operation.stop", "2 週続けて損が前のやり方より大きければ止める")
]

# 表を入れた後に答える, 時間をまたぐ量の判断の列と評価の問い, 発注の判断
DEMAND_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.demand.product", "product_id"),
    (f"{PART}.demand.date", "date"),
    (f"{PART}.demand.units", "units"),
    (f"{PART}.demand.stock", "stock"),
    (f"{PART}.demand.id", "record_id"),
    (f"{PART}.demand.columns", ["category"]),
    (f"{PART}.demand.horizon", str(HORIZON)),
    (f"{PART}.evaluation.metric", "loss"),
    (f"{PART}.evaluation.link", "分位点の損失が小さいほど発注の損の見込み"
                                "が当たる"),
    ("action.decision", {
        "type": "order", "shortage_cost": str(SHORTAGE),
        "holding_cost": str(HOLDING), "max_order": str(MAX_ORDER)
    })
]


def make_daily(
    path: Path, seed: int = 0, days: int = DAYS, blank_last: bool = False
) -> None:
    """
    商品ごとに水準と曜日の波の違う, 合成の日ごとの売れた数の記録 (在庫の
    数と, 売れた後に決まる列 waste を持つ) を書く関数

    Parameters
    ----------
    path: Path
        書く CSV
    seed: int = 0
        乱数の種
    days: int = DAYS
        日の数
    blank_last: bool = False
        商品ごとの最後の日 (発注を決める日) の売れた数を空にするか

    Returns
    ----------
    None
    """
    # 乱数の種に依らない商品ごとの水準 (種類 1 は多く売れる), 週末の多い
    # 売れ行き, 学習の記録の日の数で 1 から 1.3 へゆっくり上がる傾き
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days)
    rows = []
    for product in range(PRODUCTS):
        category = product % 2
        level = 2 + product % 6 + 4 * category
        for number, day in enumerate(dates):
            mean = level * (1.8 if day.weekday() >= 5 else 1.0) * (
                1 + 0.3 * number / DAYS
            )
            units = int(rng.poisson(mean))
            rows.append({
                "record_id": f"r{seed}_{product:02d}_{number:03d}",
                "product_id": f"p{product:02d}",
                "date": day.strftime("%Y-%m-%d"), "category": category,
                "units": units, "stock": int(rng.integers(0, 10)),
                "waste": int(rng.integers(0, 3))
            })
    frame = pd.DataFrame(rows)
    if blank_last:
        last = frame.groupby("product_id").tail(1).index
        frame["units"] = frame["units"].astype("float")
        frame.loc[last, "units"] = None
    frame.to_csv(path, index=False)


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
        run_demand に渡す設定
    """
    # 元の記録と, 別の乱数の種の試しの記録 (商品ごとの最後の日に発注の数
    # を決める. 学習に使った行で決めないため)
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "daily.csv"
    make_daily(data)
    trial = work / "data" / "daily_trial.csv"
    make_daily(trial, seed=1, days=TRIAL_DAYS, blank_last=True)
    setup = {"data": data, "trial": trial, "work": work}

    return setup


def rounded(value: Any) -> Any:
    """
    表に出す値の小数を 4 桁に丸める関数 (辞書と並びは値ごと)

    Parameters
    ----------
    value: Any
        数, 辞書, 並び, その他

    Returns
    ----------
    result: Any
        小数を丸めた値 (小数でなければそのまま)
    """
    # 辞書と並びは値ごと, 小数だけの丸め
    if isinstance(value, dict):
        result = {key: rounded(item) for key, item in value.items()}
    elif isinstance(value, list):
        result = [rounded(item) for item in value]
    elif isinstance(value, float):
        result = round(value, 4)
    else:
        result = value

    return result


def set_problem(
    pf: Client, journey: Journey, setup: dict[str, Any], project: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    result: tuple[dict[str, Any], dict[str, Any]]
        画面の値と, 骨組みの記録
    """
    # 見本から名前を付けた作成と, 受け持ちと課題の発注への答え直し
    view = pf.ok("POST", "/api/problems", {
        "example": "解約の引き止め", "name": "商品ごとの発注の数"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in DEMAND_ANSWERS:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    demand = handoff.get("demand") or {}
    model = handoff.get("model") or {}
    decision = (handoff.get("inference") or {}).get("decision")
    errors = [item["id"] for item in view.get("findings", [])
              if item.get("level") == "error"] + [
        item.get("id") for item in view.get("open_errors") or []
    ]
    journey.add(
        "問題の設定 (表, 時間をまたぐ量の判断の列, 発注の判断)",
        "OK" if view["exportable"] and not errors
        and demand.get("levels") == LEVELS
        and demand.get("horizon") == HORIZON
        and model.get("example") == EXAMPLE
        and (decision or {}).get("type") == "order" else "誤り",
        f"雛形 {model.get('example')} (出す値 {model.get('outputs')}), "
        f"水準 {demand.get('levels')}, 期間 {demand.get('horizon')} 日, "
        f"分割 {(handoff.get('data') or {}).get('split')}, 落とす列 "
        f"{(handoff.get('data') or {}).get('drop')}, 判断 {decision}, 監視 "
        f"{(handoff.get('training') or {}).get('monitor')}, 誤り {errors}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and "demand_estimator.json" in readme else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. README に需要の決め方 "
        f"{'demand_estimator.json' in readme}"
    )
    result = (view, record)

    return result


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 需要の記録を要約し, 雛形の
    レシピのまま書き戻してプロジェクトへ書き出す関数

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
    # 一覧から開き, 受け渡しの列で需要の記録の要約 (商品, 日付の範囲,
    # 抜けの日, 在庫が 0 の日)
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    summary = fathom.ok("POST", f"/api/sources/{sid}/demand", {
        "product": recipe.get("product"), "date": recipe.get("date"),
        "units": recipe.get("target"), "stock": recipe.get("stock"),
        "horizon": recipe.get("horizon")
    })
    ops = [step["op"] for step in recipe["steps"]]
    dropped = sum((step["columns"] for step in recipe["steps"]
                   if step["op"] == "drop_columns"), [])
    warnings = [item.get("code") for item in summary["findings"]
                if item.get("severity") == "warning"]
    journey.add(
        "ML-Fathom の需要の記録の要約",
        "OK" if recipe.get("task") == "demand"
        and summary["products"]["count"] == PRODUCTS
        and summary["rows"] == PRODUCTS * DAYS and "lag" in ops
        and "datetime_parts" in ops and "waste" in dropped else "誤り",
        f"{summary['rows']} 行, 商品 {summary['products']['count']}, 日付 "
        f"{summary['dates']['min']} から {summary['dates']['max']} "
        f"({summary['dates']['days']} 日), 抜けの日 "
        f"{summary['products']['gaps']}, 売れた数の平均 "
        f"{rounded(summary['unit_stats']['mean'])}. 雛形の手順 {ops}, 落とす"
        f"列 {dropped}, 注意 {warnings}"
    )

    # 雛形のレシピのままの書き戻しと書き出し
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
    ML-CreateArchitecture で問題の設定を一覧から選び, 需要の雛形を
    データセットの列の数と学ぶ水準の数に書き換えてプロジェクトに保存する
    関数

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

    # 入力の幅 (列の数) と学ぶ水準の数での雛形の書き換え (画面の「データ
    # セットから入力と出力を作る」と同じ値)
    code, info = ca.call(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    if code == 200 and (info.get("demand") or {}).get("levels"):
        columns = int(info["inputs"][0]["shape"][-1])
        levels = len(info["demand"]["levels"])
        status = "OK"
        note = (f"データセットの入力から (水準 {info['demand']['levels']}, "
                f"{info['demand'].get('levels_from')} から)")
    else:
        # 読めなければ, ML-Fathom の入力の形と受け渡しの水準の手での写し
        columns = int(hints["input_shape"][-1])
        levels = len(LEVELS)
        journey.add(
            "ML-CreateArchitecture のデータセットの入力", "誤り",
            f"{code}: {info}. 列と水準の数を ML-Fathom の書き出しと問題の"
            "設定から写して続ける"
        )
        status, note = "手", "列と水準の数を手で写す"
    template = ca.ok("GET", f"/api/examples/{EXAMPLE}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": EXAMPLE,
        "values": {"features": columns, "levels": levels}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    journey.add(
        "ML-CreateArchitecture (雛形を列と水準の数に書き換え, 保存)",
        # データセットを読めない注意は, 上の誤りと同じ理由なので手の扱い
        status if not saved.get("notes") or code != 200 else "誤り",
        f"{note}. 列 {columns}, 水準の数 {levels}, 入力 "
        f"{spec['inputs']['x']['shape']}, 出力 {sorted(spec['outputs'])}, "
        f"注意 {len(saved.get('notes') or [])} 件"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> str:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 需要の評価を読む関数

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

    # 骨組みを書き換えない学習と, 需要の評価 (mae, 分位点の損失, 発注の
    # 損と前の期間と同じ数の発注の損の比べ) の読み込み
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
            if e["name"].startswith("demand_") and e["rows"]}
    test = rows.get("demand_test") or {}
    keys = ("mae", "pinball_loss", "order_cost", "baseline_cost")
    journey.add(
        "学習 (骨組みを書き換えない) と需要の評価",
        # テスト用は時間の順の最後の 7 日の商品ごとの 1 行. 水準 τ の分位点
        # での発注の損が, 前の期間と同じ数の発注の損より小さいこと
        "OK" if all(test.get(key) is not None for key in keys)
        and "demand_valid" in rows and test.get("rows") == TEST_ROWS
        and test.get("products") == PRODUCTS
        and test["order_cost"] < test["baseline_cost"] else "誤り",
        f"{time.time() - started_at:.0f} 秒, {run['best']['monitor']} "
        f"{run['best']['value']:.4f}. テスト用 ({test.get('rows')} 行, 商品 "
        f"{test.get('products')}): mae {rounded(test.get('mae'))}, 分位点の"
        f"損失 {rounded(test.get('pinball_loss'))}, 水準 "
        f"{test.get('level')} の発注の損 {rounded(test.get('order_cost'))}"
        f", 前の期間と同じ数の発注の損 {rounded(test.get('baseline_cost'))}"
        f" (差 {rounded(test.get('order_cost_saving'))}, 比 "
        f"{rounded(test.get('order_cost_ratio'))}), 上限で切った行 "
        f"{test.get('capped')}. 評価 {sorted(rows)}"
    )

    return str(run["id"])


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    trial: Path
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 試しの記録を推論して商品ごとに
    発注の数の上限の中で発注の数を決める関数

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
        商品ごとの最後の日の売れた数が空の試しの記録

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と版と発注の判断の確かめ
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
    demand = description.get("demand") or {}
    got = (description.get("kind"), description.get("format_version"))
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got == BUNDLE and decision.get("type") == "order"
        and decision.get("max_order") == MAX_ORDER
        and demand.get("levels") == LEVELS else "誤り",
        f"種類と版 {got}. 一式の判断 {decision}, 水準 "
        f"{demand.get('levels')}, 期間 {demand.get('horizon')} 日"
    )

    # 問題の設定を使わない需要の一式 (発注の判断は骨組みの config から
    # 一式が写して持つ) の欄の無さと, 一式の判断のままの試しの記録の推論
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    uses = bundle.get("uses_problem")
    journey.add(
        "問題の設定の欄 (一式の画面)",
        "OK" if uses is False and state.get("linked") is None else "誤り",
        "問題の設定を使わない一式なので欄が無い (発注の判断は一式が持つ)"
        if uses is False else f"uses_problem {uses}, 候補 {state.get('hint')}"
    )
    result = client.ok("POST", f"/api/bundles/{bid}/predict",
                       {"source": str(trial)})
    summary = result["summary"]
    totals = (result.get("decision") or {}).get("summary") or {}
    found = result.get("demand") or {}
    orders = client.ok(
        "GET", f"/api/results/{result['id']}/orders?limit={PRODUCTS}"
    )
    counts = [r["order"] for r in orders["rows"]]
    ordered = all(a >= b for a, b in zip(counts, counts[1:]))
    within = all(0 <= c <= MAX_ORDER and float(c).is_integer()
                 for c in counts)
    products = {str(r["product_id"]) for r in orders["rows"]}
    top = orders["rows"][0] if orders["rows"] else {}
    # 発注の数は分位点から在庫を引いた数の切り下げか切り上げ (損の見込み
    # の小さい方) を 0 から上限の間に収めたもの
    derived = all(
        r["order"] in {
            min(max(np.floor(r["quantile"] - r["on_hand"]), 0), MAX_ORDER),
            min(max(np.ceil(r["quantile"] - r["on_hand"]), 0), MAX_ORDER)
        } for r in orders["rows"]
    )
    # 発注に使う分位点は, 学ぶ水準のうち水準 τ の分位点
    tau = LEVELS.index(SHORTAGE / (SHORTAGE + HOLDING))
    picked = all(
        np.isclose(r["quantile"], r["quantiles"][tau])
        for r in orders["rows"]
    )
    read = len(pd.read_csv(trial))
    journey.add(
        "推論 (商品ごとに上限の中で発注の数を決める)",
        "OK" if summary["predicted"] == PRODUCTS
        and orders["total"] == len(products) == PRODUCTS and within
        and derived and picked and ordered
        and sum(counts) == totals.get("total")
        and totals.get("level") == SHORTAGE / (SHORTAGE + HOLDING)
        else "誤り",
        f"{read} 行の記録のうち発注を決める日の {summary['predicted']} 行 (商品 "
        f"{len(products)}) を推論, 水準 τ = {totals.get('level')} の分位点"
        f"から在庫を引く {derived}, 分位点が水準 τ のもの {picked}, 発注の数の和 {totals.get('total')} (上限 "
        f"{orders['max_order']} で切った商品 {totals.get('capped')}, 大きい"
        f"順 {ordered}, 上限の中の整数 {within}, 最も多い "
        f"{top.get('product_id')} の {top.get('order')} 個. 分位点 "
        f"{rounded(top.get('quantiles'))}, 在庫 {top.get('on_hand')}, 前の"
        f"期間と同じ発注 {top.get('baseline_order')}), 損の"
        f"見込みの和 {rounded(totals.get('expected_cost'))}. 分位点の平均 "
        f"{json.dumps(rounded(found.get('quantiles')))}. "
        + " ".join(result["sentences"][:1])
    )


def run_demand(
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
    project = setup["project"].resolve()
    view, record = set_problem(clients["pathfinder"], journey, setup, project)
    entry = analyze(clients["fathom"], journey, view["name"], record)
    original = Path(entry["path"]).resolve()
    make_model(clients["createarchitecture"], journey, original, project,
               entry["hints"])
    run_id = train(clients["dashboard"], journey, original, project)
    provide(clients["provide"], journey, project, run_id, setup["trial"])


if __name__ == "__main__":
    failed = main_with(build, "時間をまたぐ量の判断 (発注の数)", sys.argv[1:],
                       runner=run_demand, prefix="demand_")
    sys.exit(1 if failed else 0)
