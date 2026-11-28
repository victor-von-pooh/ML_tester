"""
異常の検知 (正常の行が多い機械の計測の表から, 異常の度合いの高い計測を
人の確認に回す) を, 自己符号化器の度合いで, 問題の設定から推論まで 5 つの
画面のアプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で
写すか打つか (手), 誤りかを表にする
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

# 学習の計測の数と, 推論で度合いを数える試しの計測の数
READINGS = 3000
TRIAL_READINGS = 1000

# 異常の割合と, 学習の記録で正解を付けた行の割合 (正解は少し)
ANOMALY_RATE = 0.03
LABELED_RATE = 0.3

# 人が確かめる件数の上限 (評価の上位 K 件も同じ)
CAPACITY = 20

# 度合いの決め方, 骨組みの雛形
SCORE = "autoencoder"
EXAMPLE = "tabular_autoencoder"

# 度合いに使う列
COLUMNS = ["temp", "vibration", "pressure", "current"]

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「解約の引き止め」の答えを, 機械の計測の課題に直す答え (受け持ちは
# 予測するで正解の種類が異常. ML-Pathfinder の異常の検知のテストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "工場の保全の担当"),
    ("issue.pain", "機械の故障の前の兆しを見逃し, 止まってから直している"),
    ("issue.current", "温度が決まった値を超えたら点検する"),
    ("issue.kpi.name", "点検で見つけた異常の件数"),
    ("issue.kpi.unit", "件 / 日"),
    ("issue.kpi.better", "higher"),
    ("issue.kpi.current", 2),
    ("issue.kpi.target", 5),
    ("issue.guard.name", "点検に回す計測の数"),
    ("issue.guard.unit", "件 / 日"),
    ("issue.guard.better", "lower"),
    ("issue.guard.line", CAPACITY),
    ("issue.guard.measure", "1 日に点検に回した計測の数を数える"),
    ("action.what", "点検に回す計測"),
    ("action.who", "保全の担当"),
    ("action.when", "毎日の始業の前"),
    ("ml.reason", f"1 日に確かめられるのは {CAPACITY} 件までで, どの計測を"
                  "確かめるかを選ぶ必要がある"),
    (f"{PART}.role", "predict"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "value"),
    (f"{PART}.target.source", "few"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.target.kind", "anomaly"),
    (f"{PART}.what", "機械の異常の度合い"),
    (f"{PART}.item", "1 回の計測"),
    (f"{PART}.baseline", "温度が決まった値を超えたら"),
    ("verification.trial", "1 週間, 度合いの上位から点検し, 異常だった"
                           "割合を数える"),
    ("verification.success", "度合いの上位から点検した異常の割合が, 温度の"
                             "決まりで点検した割合より高い"),
    ("operation.stop", "2 週続けて上位に異常が見つからなければ止める")
]

# 表を入れた後に答える, 異常の検知の列と評価の問い
ANOMALY_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.anomaly.label", "is_anomaly"),
    (f"{PART}.anomaly.id", "reading_id"),
    (f"{PART}.anomaly.columns", COLUMNS),
    (f"{PART}.anomaly.score", SCORE),
    (f"{PART}.anomaly.capacity", str(CAPACITY)),
    (f"{PART}.evaluation.metric", "average_precision"),
    (f"{PART}.evaluation.link", "上位に異常が多いほど点検での見逃しが減る")
]


def make_readings(
    path: Path, seed: int = 0, readings: int = READINGS,
    labeled: float = LABELED_RATE
) -> None:
    """
    正常の行が多い機械の計測の合成の表 (正解は一部の行だけ) を書く関数

    Parameters
    ----------
    path: Path
        書く CSV
    seed: int = 0
        乱数の種
    readings: int = READINGS
        計測の数
    labeled: float = LABELED_RATE
        正解を付ける行の割合 (1 なら全ての行)

    Returns
    ----------
    None
    """
    # 正常は温度と電流が負荷に沿って動き, 振動と圧力が落ち着く. 異常は
    # 負荷との関係が崩れる (温度か振動が高く, 圧力が下がる)
    rng = np.random.default_rng(seed)
    load = rng.uniform(0.2, 1.0, readings)
    temp = 40 + 30 * load + rng.normal(0, 1.5, readings)
    vibration = 0.5 + 0.8 * load + rng.normal(0, 0.08, readings)
    pressure = 5 - 1.5 * load + rng.normal(0, 0.15, readings)
    current = 10 + 20 * load + rng.normal(0, 1.0, readings)
    anomaly = rng.random(readings) < ANOMALY_RATE
    count = int(anomaly.sum())
    temp[anomaly] += rng.uniform(2, 8, count)
    vibration[anomaly] += rng.uniform(0.1, 0.4, count)
    pressure[anomaly] -= rng.uniform(0.1, 0.5, count)
    label = pd.Series(anomaly.astype(int), dtype="Int64")
    label[rng.random(readings) >= labeled] = pd.NA
    pd.DataFrame({
        "reading_id": [f"R{seed}{i:05d}" for i in range(readings)],
        "temp": temp.round(2), "vibration": vibration.round(3),
        "pressure": pressure.round(3), "current": current.round(2),
        "memo": "定期", "is_anomaly": label
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
        run_anomaly に渡す設定
    """
    # 元の記録と, 推論で度合いを数える別の乱数の種の試しの記録 (全ての
    # 行に正解があり, 人の確認の判断を正解から決める)
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "sensors.csv"
    make_readings(data)
    trial = work / "data" / "sensors_trial.csv"
    make_readings(trial, seed=1, readings=TRIAL_READINGS, labeled=1.0)
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
    # 見本から名前を付けた作成と, 受け持ちと課題の計測への答え直し
    view = pf.ok("POST", "/api/problems", {
        "example": "解約の引き止め", "name": "機械の異常の検知"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in ANOMALY_ANSWERS:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    anomaly = handoff.get("anomaly") or {}
    model = handoff.get("model") or {}
    decision = (handoff.get("inference") or {}).get("decision")
    errors = [item["id"] for item in view.get("findings", [])
              if item.get("level") == "error"] + [
        item.get("id") for item in view.get("open_errors") or []
    ]
    journey.add(
        "問題の設定 (表, 異常の検知の列, 人が確かめる件数の上限)",
        "OK" if view["exportable"] and not errors
        and anomaly.get("score") == SCORE
        and anomaly.get("capacity") == CAPACITY
        and model.get("example") == EXAMPLE
        and decision == {"type": "triage", "capacity": CAPACITY}
        else "誤り",
        f"度合い {anomaly.get('score')}, 正解 {anomaly.get('label')}, "
        f"上限 {anomaly.get('capacity')}, 雛形 {model.get('example')}, "
        f"判断 {decision}, 監視 "
        f"{(handoff.get('training') or {}).get('monitor')}, 除く列 "
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
        and "anomaly_scores.csv" in readme else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. 書いたファイル "
        f"{len(written.get('files') or [])}"
    )
    result = (pid, view, record)

    return result


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 異常の正解の数を数え, 前処理を
    書き戻してプロジェクトへ書き出す関数

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
    # 一覧から開き, 正解の列を異常の正解として数える分析
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    profile = fathom.ok("GET", f"/api/sources/{sid}/profile?task=anomaly")
    counts = profile.get("anomaly") or {}
    dropped = sum((step["columns"] for step in recipe["steps"]
                   if step["op"] == "drop_columns"), [])
    journey.add(
        "ML-Fathom の異常の正解の数",
        "OK" if recipe.get("task") == "anomaly" and counts.get("labeled")
        and counts.get("anomaly") else "誤り",
        f"タスク {recipe.get('task')}, 正解の付いた行 "
        f"{counts.get('labeled_rows')} (異常 {counts.get('anomaly')}, 正常 "
        f"{counts.get('normal')}, 割合 {rounded(counts.get('ratio'))}), "
        f"正解の欠けた行 {counts.get('missing')}. 雛形が落とす列 {dropped}"
    )

    # 書き戻しとプロジェクトへの書き出し (正解の有る無しの分割の行の数)
    check = fathom.ok(
        "POST", f"/api/sources/{sid}/problem/check", {"recipe": recipe}
    )
    fathom.ok(
        "POST", f"/api/sources/{sid}/problem/write",
        {"recipe": recipe, "version": check["version"]}
    )
    preview = fathom.ok("POST", f"/api/sources/{sid}/preview",
                        {"recipe": recipe, "upto": len(recipe["steps"])})
    exported = fathom.ok(
        "POST", f"/api/sources/{sid}/export_project", {"recipe": recipe}
    )
    rows = (preview.get("anomaly") or {}).get("rows")
    journey.add(
        "ML-Fathom (前処理, プロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"]) else "誤り",
        f"分割 {recipe['split']['method']}, 入力 "
        f"{exported['hints']['input_shape']}, 分割ごとの行 "
        f"{json.dumps(rows, ensure_ascii=False)}"
    )
    entry = {**entry, "hints": exported["hints"]}

    return entry


def make_model(
    ca: Client, journey: Journey, original: Path, project: Path,
    hints: dict[str, Any]
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 自己符号化器の雛形を
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

    # 入力の幅 (度合いに使う列の数) での雛形の書き換え
    code, info = ca.call(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    if code == 200:
        columns = int(info["inputs"][0]["shape"][-1])
        status, note = "OK", "データセットの入力から"
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
    journey.add(
        "ML-CreateArchitecture (雛形を列の数に書き換え, 保存)",
        status if not saved.get("notes") or code != 200 else "誤り",
        f"{note}. 列 {columns}, 入力 {spec['inputs']['x']['shape']}, "
        f"出力 {sorted(spec['outputs'])}, 注意 "
        f"{saved.get('notes') or []}"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> str:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 度合いの評価を読む関数

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

    # 骨組みを書き換えない学習と, 度合いの評価の読み込み
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
    names = [e["name"] for e in results["evaluations"]]
    rows = {e["name"]: e["rows"][0] for e in results["evaluations"]
            if e["rows"] and e["name"] in ("anomaly_valid", "anomaly_test")}
    test = rows.get("anomaly_test") or {}
    best = run.get("best") or {}
    journey.add(
        "学習 (骨組みを書き換えない) と度合いの評価",
        # PR-AUC が異常の割合 (でたらめな度合いの PR-AUC) を超えること
        "OK" if test.get("precision_at_k") is not None
        and test.get("rows") and test.get("average_precision", 0)
        > test.get("anomalies", 0) / test["rows"] else "誤り",
        f"{time.time() - started_at:.0f} 秒, {best.get('monitor')} "
        f"{rounded(best.get('value'))}. テスト用 ({test.get('rows')} 行, "
        f"異常 {test.get('anomalies')} 行): 上位 {CAPACITY} 件の"
        f"適合率 {rounded(test.get('precision_at_k'))}, PR-AUC "
        f"{rounded(test.get('average_precision'))}. 評価 {names}"
    )
    run_id = str(run["id"])

    return run_id


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    trial: Path, work: Path
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 試しの記録を推論して度合いの上位から
    件数の上限まで人の確認に回し, 確認の結果を 1 件ごとに記録する関数

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
        正解の列を持つ試しの記録
    work: Path
        作業用の場所 (確認の記録のファイルを探す)

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と判断の確かめ
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
    got = (description.get("kind"), description.get("format_version"))
    decision = description.get("decision")
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got[0] == "anomaly"
        and (decision or {}).get("capacity") == CAPACITY else "誤り",
        f"種類, 版 {got}. 一式の判断 {decision}. 問題の設定を使うか "
        f"{bundle.get('uses_problem')}"
    )

    # 一式の判断のままの試しの記録の推論と, 人の確認に回した行
    result = client.ok("POST", f"/api/bundles/{bid}/predict",
                       {"source": str(trial)})
    rid = result["id"]
    queue = client.ok("GET", f"/api/results/{rid}/queue")
    scores = [r["score"] for r in queue["rows"]]
    ordered = all(a >= b for a, b in zip(scores, scores[1:]))
    truth = pd.read_csv(trial).set_index("reading_id")["is_anomaly"]
    hits = sum(int(truth.get(r["id"], 0)) for r in queue["rows"])
    journey.add(
        "推論 (度合いの上位から件数の上限まで人の確認に回す)",
        "OK" if queue["problem"] is None and queue["capacity"] == CAPACITY
        and len(queue["rows"]) == CAPACITY and ordered
        and all(r["key"] for r in queue["rows"])
        # 回した行の異常の割合が, 試しの記録の全体の異常の割合を超えること
        and hits / CAPACITY > float(truth.mean()) else "誤り",
        f"{result['summary'].get('predicted')} 行を推論, 回した "
        f"{len(queue['rows'])} 行 (上限 {queue['capacity']}, 度合いの高い"
        f"順 {ordered}, 行の id の列 {queue['id']}). 回した行のうち正解が"
        f"異常 {hits} 行 (試しの記録の異常 {int(truth.sum())} 行). "
        + " ".join(result.get("sentences", [])[:1])
    )

    # 1 件ごとの確認の記録 (判断は試しの記録の正解から) と読み直し
    for item in queue["rows"]:
        judgment = "anomaly" if int(truth.get(item["id"], 0)) else "normal"
        client.ok("POST", f"/api/results/{rid}/reviews", {
            "row": item["key"], "judgment": judgment, "reviewer": "保全の担当"
        })
    records = client.ok("GET", f"/api/results/{rid}/reviews")["records"]
    again = client.ok("GET", f"/api/results/{rid}/queue")["rows"]
    judged = sum(1 for r in again if r["judgment"])
    files = [p for p in work.rglob("reviews.csv") if p.parent.name == rid]
    journey.add(
        "人の確認の結果を 1 件ごとに記録",
        "OK" if len(records) == len(queue["rows"]) == judged
        and all(r["reviewed_at"] for r in records) and files else "誤り",
        f"記録 {len(records)} 件 (異常 "
        f"{sum(r['judgment'] == 'anomaly' for r in records)}, 正常 "
        f"{sum(r['judgment'] == 'normal' for r in records)}), 判断の付いた"
        f"回した行 {judged}. 結果のディレクトリの記録 "
        f"{[str(p.relative_to(work)) for p in files]}"
    )


def run_anomaly(
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
                setup["trial"], setup["work"])
    except Exception as error:
        # 止まった段階を残す
        journey.add("止まった段階", "誤り",
                    f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    failed = main_with(build, "異常の検知 (機械の計測)", sys.argv[1:],
                       runner=run_anomaly, prefix="anomaly_")
    sys.exit(1 if failed else 0)
