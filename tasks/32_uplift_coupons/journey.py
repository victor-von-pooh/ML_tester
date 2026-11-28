"""
効果を見積もる (無作為に配ったクーポンの記録から, 顧客ごとに買う確率が
どれだけ上がるか) を, 効果の決め方 2 つ (T-learner と S-learner) で,
問題の設定から推論まで 5 つの画面のアプリの API だけで通し, 段階ごとに
画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを表にする
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

# 学習の顧客の数と, 推論で数え直す試しの顧客の数
CUSTOMERS = 3000
TRIAL_CUSTOMERS = 1000

# 1 回に配れる数と, 評価で見る上位の割合 (試しの 1,000 人の 3 割)
BUDGET = 300
TOP_K = 0.3

# 効果の決め方ごとの, 印, 骨組みの雛形, 推論の一式の版
LEARNERS = {
    "t_learner": ("T", "tabular_uplift", 12),
    "s_learner": ("S", "tabular_uplift_s", 18)
}

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「解約の引き止め」の答えを, クーポンの課題に直す答え (受け持ちは
# 効果を見積もるで, 今ある記録から集める. ML-Pathfinder の効果の見積もりの
# テストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "販促の担当"),
    ("issue.pain", "クーポンを全員に配っていて, 費用の割に売上が増えない"),
    ("issue.current", "会員の全員にクーポンを配る"),
    ("issue.kpi.name", "クーポンで増えた購入の件数"),
    ("issue.kpi.unit", "件 / 回"),
    ("issue.kpi.better", "higher"),
    ("issue.kpi.current", 60),
    ("issue.kpi.target", 90),
    ("issue.guard.name", "配ったクーポンの数"),
    ("issue.guard.unit", "枚 / 回"),
    ("issue.guard.better", "lower"),
    ("issue.guard.line", BUDGET),
    ("issue.guard.measure", "1 回に配ったクーポンの数を数える"),
    ("action.what", "クーポンを配る顧客"),
    ("action.who", "販促の担当"),
    ("action.when", "毎月 1 日"),
    ("ml.reason", f"1 回に配れるのは {BUDGET} 枚までで, 誰に配るかを"
                  "選ぶ必要がある"),
    (f"{PART}.role", "effect"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "effect"),
    (f"{PART}.target.source", "intervention"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.autonomous", False),
    (f"{PART}.collect", "log"),
    (f"{PART}.what", "クーポンで買う確率がどれだけ上がるか"),
    (f"{PART}.item", "顧客ごとの 1 行"),
    (f"{PART}.baseline", "全員に配る"),
    ("verification.trial", "顧客を 2 つに分け, 片方だけに効果の上位から"
                           "配る"),
    ("verification.success", "効果の上位から配った側の購入が, 無作為に"
                             "配った側より多い"),
    ("operation.stop", "2 回続けて差が出ないか, 配った数が上限を超えたら"
                       "止める")
]

# 表を入れた後に答える, 効果の見積もりの列と評価の問い
UPLIFT_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.uplift.treatment", "coupon"),
    (f"{PART}.uplift.outcome", "purchased"),
    (f"{PART}.uplift.outcome_kind", "binary"),
    (f"{PART}.uplift.id", "customer_id"),
    (f"{PART}.uplift.columns", ["age", "visits", "region"]),
    (f"{PART}.uplift.k", str(TOP_K)),
    (f"{PART}.evaluation.metric", "qini"),
    (f"{PART}.evaluation.link", "効果の順が当たるほど, 配る数の上限の中で"
                                "増える購入が多い"),
    ("action.decision", {"type": "treat", "budget": str(BUDGET)})
]


def make_coupons(
    path: Path, seed: int = 0, customers: int = CUSTOMERS
) -> None:
    """
    無作為にクーポンを配った合成の記録 (配った後に決まる列 used を持つ) を
    書く関数

    Parameters
    ----------
    path: Path
        書く CSV
    seed: int = 0
        乱数の種
    customers: int = CUSTOMERS
        顧客の数

    Returns
    ----------
    None
    """
    # 来店の多い顧客と北の地域の顧客ほど大きい配ったときの効果 (年齢は
    # 配らないときの買う確率だけ)
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 70, customers).round()
    visits = rng.poisson(4, customers)
    region = rng.choice(["north", "south", "west"], customers,
                        p=[0.3, 0.4, 0.3])
    coupon = rng.integers(0, 2, customers)
    base = 0.08 + 0.002 * (age - 20)
    lift = 0.03 + 0.25 * (visits >= 5) + 0.1 * (region == "north")
    purchased = (rng.random(customers) < base + lift * coupon).astype(int)
    used = coupon * purchased * (rng.random(customers) < 0.8)
    pd.DataFrame({
        "customer_id": [f"C{seed}{i:05d}" for i in range(customers)],
        "age": age, "visits": visits, "region": region, "coupon": coupon,
        "used": used.astype(int), "purchased": purchased
    }).to_csv(path, index=False)


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
        run_uplift に渡す設定
    """
    # 元の記録と, 推論で効果を数え直す別の乱数の種の試しの記録 (学習に
    # 使った行で測らないため)
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "coupons.csv"
    make_coupons(data)
    trial = work / "data" / "coupons_trial.csv"
    make_coupons(trial, seed=1, customers=TRIAL_CUSTOMERS)
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


def set_problem(
    pf: Client, journey: Journey, setup: dict[str, Any], learner: str,
    project: Path
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
    learner: str
        効果の決め方 (t_learner か s_learner)
    project: Path
        骨組みの書き出し先

    Returns
    ----------
    result: tuple[str, dict[str, Any], dict[str, Any]]
        問題の設定の id, 画面の値, 骨組みの記録
    """
    # 見本から名前を付けた作成と, 受け持ちと課題のクーポンへの答え直し
    mark, example, _ = LEARNERS[learner]
    view = pf.ok("POST", "/api/problems", {
        "example": "解約の引き止め", "name": f"クーポンの効果 ({mark})"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    answers = [*UPLIFT_ANSWERS, (f"{PART}.uplift.learner", learner)]
    for key, value in answers:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    uplift = handoff.get("uplift") or {}
    model = handoff.get("model") or {}
    errors = errors_of(view) + [
        item.get("id") for item in view.get("open_errors") or []
    ]
    journey.add(
        f"[{mark}] 問題の設定 (表, 効果の見積もりの列, 配る数の上限)",
        "OK" if view["exportable"] and not errors
        and uplift.get("learner") == learner
        and model.get("example") == example else "誤り",
        f"決め方 {uplift.get('learner')}, 雛形 {model.get('example')}, "
        f"判断 {(handoff.get('inference') or {}).get('decision')}, 監視 "
        f"{(handoff.get('training') or {}).get('monitor')}, 誤り {errors}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    _, _, version = LEARNERS[learner]
    journey.add(
        f"[{mark}] 骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and f"の効果の一式 (版 {version}." in readme else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. README の一式の版 {version}"
    )
    result = (pid, view, record)

    return result


def analyze(
    fathom: Client, journey: Journey, mark: str, name: str,
    record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 介入の記録を要約し, 前処理して
    プロジェクトへ書き出す関数

    Parameters
    ----------
    fathom: Client
        ML-Fathom の客
    journey: Journey
        結果の記録
    mark: str
        効果の決め方の印
    name: str
        問題の設定の名前
    record: dict[str, Any]
        骨組みの記録

    Returns
    ----------
    entry: dict[str, Any]
        ML-Fathom の一覧の問題の設定 (hints に書き出しの入力の形)
    """
    # 一覧から開き, 受け渡しの列で介入の記録の要約 (群の数, 結果の差,
    # 配った後に決まる列)
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    summary = fathom.ok("POST", f"/api/sources/{sid}/uplift", {
        "treatment": recipe.get("treatment"),
        "outcome": recipe.get("target"),
        "outcome_kind": recipe.get("outcome"), "id": recipe.get("id")
    })
    groups = summary["groups"]
    posts = [item["columns"] for item in summary["findings"]
             if item["code"] == "post_treatment"]
    dropped = [step["columns"] for step in recipe["steps"]
               if step["op"] == "drop_columns"]
    journey.add(
        f"[{mark}] ML-Fathom の介入の記録の要約",
        "OK" if posts and all(c in sum(dropped, []) for p in posts for c in p)
        else "誤り",
        f"処置の群 {groups['treated']['rows']} 行 (平均 "
        f"{groups['treated']['mean']:.3f}), 対照の群 "
        f"{groups['control']['rows']} 行 (平均 "
        f"{groups['control']['mean']:.3f}), 差 {summary['effect']:.3f}. "
        f"配った後に決まる列 {posts} を雛形が落とす {dropped}"
    )

    # 種類の列を数の列にする手順の, 雛形の標準化の前への足しと書き出し
    first = next((i for i, s in enumerate(recipe["steps"])
                  if s["op"] == "standardize"), len(recipe["steps"]))
    recipe["steps"].insert(first, {"op": "one_hot", "columns": ["region"]})
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
        f"[{mark}] ML-Fathom (前処理, プロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"]) else "誤り",
        f"分割 {recipe['split']['method']}, 入力 "
        f"{exported['hints']['input_shape']}, 出す値 "
        f"{exported['hints']['outputs']}"
    )

    entry = {**entry, "hints": exported["hints"]}

    return entry


def make_model(
    ca: Client, journey: Journey, mark: str, example: str, original: Path,
    project: Path, hints: dict[str, Any]
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 効果の雛形を
    データセットの列の数に書き換えてプロジェクトに保存する関数

    Parameters
    ----------
    ca: Client
        ML-CreateArchitecture の客
    journey: Journey
        結果の記録
    mark: str
        効果の決め方の印
    example: str
        受け渡しの雛形の名前
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
        f"[{mark}] ML-CreateArchitecture で問題の設定を一覧から選ぶ",
        "OK" if target.get("example") == example and target.get("dataset")
        and Path(chosen["project"]).resolve() == project else "誤り",
        f"雛形 {target.get('example')}, データセットの欄 "
        f"{Path(target.get('dataset') or '-').name}"
    )

    # 入力の幅 (列の数と, S-learner の処置の列の足す数) での雛形の書き換え
    code, info = ca.call(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    if code == 200:
        columns = int(info["inputs"][0]["shape"][-1])
        offset = int((info.get("offsets") or {}).get("x", 0))
        status, note = "OK", "データセットの入力から"
    else:
        # 読めなければ, ML-Fathom の入力の形と決め方の足す数の手での写し
        columns = int(hints["input_shape"][-1])
        offset = 1 if example.endswith("_s") else 0
        journey.add(
            f"[{mark}] ML-CreateArchitecture のデータセットの入力", "誤り",
            f"{code}: {info}. 列の数を ML-Fathom の書き出しから写して続ける"
        )
        status, note = "手", "列の数を手で写す"
    template = ca.ok("GET", f"/api/examples/{example}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": example,
        "values": {"features": columns + offset}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    outputs = sorted(spec["outputs"])
    journey.add(
        f"[{mark}] ML-CreateArchitecture (雛形を列の数に書き換え, 保存)",
        # データセットを読めない注意は, 上の誤りと同じ理由なので手の扱い
        status if not saved.get("notes") or code != 200 else "誤り",
        f"{note}. 列 {columns} (足す数 {offset}), 入力 "
        f"{spec['inputs']['x']['shape']}, 出力 {outputs}, 注意 "
        f"{len(saved.get('notes') or [])} 件"
    )


def train(
    dash: Client, journey: Journey, mark: str, original: Path, project: Path
) -> tuple[str, dict[str, Any]]:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 効果の評価を読む関数

    Parameters
    ----------
    dash: Client
        ML-Dashboard の客
    journey: Journey
        結果の記録
    mark: str
        効果の決め方の印
    original: Path
        元の問題の設定 (problem.yaml)
    project: Path
        書き出したプロジェクト

    Returns
    ----------
    result: tuple[str, dict[str, Any]]
        実験の id と, 分割ごとの効果の評価の行
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
        f"[{mark}] ML-Dashboard で登録し環境を作る",
        "OK" if Path(found["path"]).resolve() == project
        and job["status"] == "succeeded" and ready["ready"] else "誤り",
        f"{job['status']}, {time.time() - started_at:.0f} 秒, 足りない "
        f"{ready['missing']}"
    )

    # 骨組みを書き換えない学習と, 効果の評価と Qini の曲線の読み込み
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
            if e["name"].startswith("uplift_") and e["rows"]}
    curves = [e["name"] for e in results["evaluations"]
              if e["name"].startswith("qini_curve_")]
    test = rows.get("uplift_test") or {}
    journey.add(
        f"[{mark}] 学習 (骨組みを書き換えない) と効果の評価",
        "OK" if test and len(curves) == 2 else "誤り",
        f"{time.time() - started_at:.0f} 秒, {run['best']['monitor']} "
        f"{run['best']['value']:.4f}. テスト用: Qini の係数 "
        f"{rounded(test.get('qini'))}, 上位 {TOP_K:g} の効果の差 "
        f"{rounded(test.get('uplift_at_k'))}, 全体の差 "
        f"{rounded(test.get('effect'))}. 曲線 {curves}"
    )
    result = (str(run["id"]), rows)

    return result


def provide(
    client: Client, journey: Journey, mark: str, version: int,
    learner: str, project: Path, run_id: str, trial: Path
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 試しの記録を推論して効果の上位から
    配る数の上限まで配る関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    mark: str
        効果の決め方の印
    version: int
        期待する推論の一式の版
    learner: str
        効果の決め方
    project: Path
        書き出したプロジェクト
    run_id: str
        ML-Dashboard の実験の id
    trial: Path
        処置と結果の列を持つ試しの記録

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と版と決め方の確かめ
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
    uplift = description.get("uplift") or {}
    got = (description.get("kind"), description.get("format_version"),
           uplift.get("learner", "t_learner"))
    journey.add(
        f"[{mark}] ML-Provide で実験を一覧から登録",
        "OK" if got == ("uplift", version, learner) else "誤り",
        f"種類, 版, 決め方 {got}. 一式の判断 {description.get('decision')}"
    )

    # 効果の一式は問題の設定を使わない (配る数の上限は骨組みの config から
    # 一式が写して持つ). 欄が無く, 一式の判断のままの試しの記録の推論
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    uses = bundle.get("uses_problem")
    journey.add(
        f"[{mark}] 問題の設定の欄 (一式の画面)",
        "OK" if uses is False and state.get("linked") is None else "誤り",
        "問題の設定を使わない一式なので欄が無い (配る数の上限は一式が持つ)"
        if uses is False else f"uses_problem {uses}, 候補 {state.get('hint')}"
    )
    result = client.ok("POST", f"/api/bundles/{bid}/predict",
                       {"source": str(trial)})
    summary = result["summary"]
    measured = summary.get("measured") or {}
    decided = (result.get("decision") or {}).get("summary") or {}
    treated = client.ok(
        "GET", f"/api/results/{result['id']}/treated?limit={BUDGET}"
    )
    effects = [r["effect"] for r in treated["rows"]]
    ordered = all(a >= b for a, b in zip(effects, effects[1:]))
    journey.add(
        f"[{mark}] 推論 (効果の上位から配る数の上限まで配る)",
        # 配るのは上位の上限の行のうち効果が正の行 (acted). 読んだ行の数
        # (total) と合い, 上限を超えないことの確かめ
        "OK" if treated["budget"] == BUDGET
        and treated["total"] == decided.get("acted") <= BUDGET
        and len(effects) == treated["total"] and ordered and measured
        else "誤り",
        f"{summary['predicted']} 行を推論, 配る {decided.get('acted')} 行 "
        f"(上限 {treated['budget']}, 効果の高い順 {ordered}, 最上位 "
        f"{treated['rows'][0]['id'] if treated['rows'] else None} の効果 "
        f"{rounded(effects[0] if effects else None)}). 試しの記録で数え"
        f"直し: {json.dumps(rounded(measured), ensure_ascii=False)}. "
        + " ".join(result["sentences"][:1])
    )


def run_uplift(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    効果の決め方ごとに, 問題の設定から推論までを画面と同じ API で通す
    関数 (片方が止まっても, もう片方は通す)

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
    for learner, (mark, example, version) in LEARNERS.items():
        project = (setup["work"] / f"project_{mark.lower()}").resolve()
        try:
            pid, view, record = set_problem(
                clients["pathfinder"], journey, setup, learner, project
            )
            entry = analyze(clients["fathom"], journey, mark, view["name"],
                            record)
            original = Path(entry["path"]).resolve()
            make_model(clients["createarchitecture"], journey, mark, example,
                       original, project, entry["hints"])
            run_id, _ = train(clients["dashboard"], journey, mark, original,
                              project)
            provide(clients["provide"], journey, mark, version, learner,
                    project, run_id, setup["trial"])
        except Exception as error:
            # 止まった段階を残し, 次の決め方へ
            journey.add(f"[{mark}] 止まった段階", "誤り",
                        f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    failed = main_with(build, "効果を見積もる (クーポン)", sys.argv[1:],
                       runner=run_uplift, prefix="uplift_")
    sys.exit(1 if failed else 0)
