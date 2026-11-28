"""
推薦 (会員と品物の買った記録から, 会員ごとに勧める品物を上位 K 件並べる)
を, 問題の設定から推論まで 5 つの画面のアプリの API だけで通し, 段階
ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを表にする
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

# 会員, 品物, 好みの群の数と, 会員ごとの反応の数
USERS = 200
ITEMS = 60
GROUPS = 4
REACTIONS = 30

# 学習の記録の日の数と, 推論で数え直す後の期間の会員ごとの反応の数
DAYS = 90
LATER = 5

# 並べる件数, 骨組みの雛形, 推論の一式の版
TOP_K = 10
EXAMPLE = "two_tower_retrieval"
VERSION = 8

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「総菜の作る量」の答えを, 会員に勧める品の課題に直す答え (受け持ちは
# 並べるで, 反応の記録から学ぶ. ML-Pathfinder の並べるのテストと同じ順)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "売り場の係"),
    ("issue.pain", "会員に勧める品が合わず, 勧めた品が買われない"),
    ("issue.current", "全ての会員に人気順に並べる"),
    ("action.what", "会員ごとに勧める品"),
    ("action.who", "売り場の係"),
    ("action.when", "毎朝"),
    (f"{PART}.role", "rank"),
    (f"{PART}.unit", "query"),
    (f"{PART}.output", "ranking"),
    (f"{PART}.target.source", "feedback"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.what", "会員ごとに勧める品"),
    (f"{PART}.item", "会員ごとの品の並び"),
    (f"{PART}.baseline", "人気順"),
    (f"{PART}.evaluation.metric", "ndcg"),
    (f"{PART}.evaluation.link", "上位に買う品が来るほど売上が増える")
]

# 表を入れた後に答える, 反応の表の列と並べる件数の問い (件数は画面と
# 同じく文で送る)
RANK_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.rank.user", "member"),
    (f"{PART}.rank.item", "item_id"),
    (f"{PART}.rank.time", "day"),
    (f"{PART}.rank.reaction", "qty"),
    (f"{PART}.rank.k", str(TOP_K))
]


def make_reactions(
    path: Path, later: Path, users_path: Path, seed: int = 0
) -> None:
    """
    好みの群のある会員が品物を買った合成の記録と, その後の期間の記録と,
    推論に渡す会員の表を書く関数

    Parameters
    ----------
    path: Path
        学習の記録の CSV (列 member, item_id, day, qty)
    later: Path
        学習の記録より後の期間の記録の CSV (列は同じ)
    users_path: Path
        推論に渡す会員の表の CSV (列 member)
    seed: int = 0
        乱数の種

    Returns
    ----------
    None
    """
    # 会員は 4 つの好みの群のどれかで, 8 割は自分の群の品物 (群ごとに 15
    # 品. 群の中では番号の若い品ほど売れる) を, 2 割は全ての品物から買う.
    # 買った数 qty が 0 の行 (見ただけ) を 1 割混ぜた記録
    rng = np.random.default_rng(seed)
    taste = rng.integers(0, GROUPS, USERS)
    per_group = ITEMS // GROUPS
    weights = 1.0 / np.arange(1, per_group + 1)
    weights = weights / weights.sum()

    rows, after = [], []
    for user in range(USERS):
        member = f"M{user:04d}"
        own = rng.random(REACTIONS + LATER) < 0.8
        inside = taste[user] * per_group + rng.choice(
            per_group, REACTIONS + LATER, p=weights
        )
        anywhere = rng.integers(0, ITEMS, REACTIONS + LATER)
        items = np.where(own, inside, anywhere)
        for item in items[:REACTIONS]:
            qty = (rng.integers(0, 4) if rng.random() < 0.1
                   else rng.integers(1, 4))
            rows.append((member, f"I{item:03d}", int(rng.integers(0, DAYS)),
                         int(qty)))
        for item in items[REACTIONS:]:
            after.append((member, f"I{item:03d}",
                          int(DAYS + rng.integers(0, 14)), 1))
    columns = ["member", "item_id", "day", "qty"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    pd.DataFrame(after, columns=columns).to_csv(later, index=False)
    pd.DataFrame({"member": [f"M{u:04d}" for u in range(USERS)]}).to_csv(
        users_path, index=False
    )


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
        run_rank に渡す設定
    """
    # 学習の記録と, 推論で Recall@K と NDCG を数え直す後の期間の記録 (学習
    # に使った反応で測らないため) と, 並べる会員の表
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "reactions.csv"
    later = work / "data" / "reactions_later.csv"
    members = work / "data" / "members.csv"
    make_reactions(data, later, members)
    setup = {"data": data, "later": later, "members": members, "work": work}

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
    # 見本から名前を付けた作成と, 課題と受け持ちの答え直し
    view = pf.ok("POST", "/api/problems", {
        "example": "総菜の作る量", "name": "会員に勧める品"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in RANK_ANSWERS:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    rank = handoff.get("rank") or {}
    model = handoff.get("model") or {}
    training = handoff.get("training") or {}
    errors = [item["id"] for item in view.get("findings", [])
              if item.get("level") == "error"]
    journey.add(
        "問題の設定 (反応の表, 列, 並べる件数)",
        "OK" if view["exportable"] and not errors
        and rank.get("k") == TOP_K and model.get("example") == EXAMPLE
        and training.get("monitor") == "valid/ndcg" else "誤り",
        f"並べ方 {rank}, 分割 {(handoff.get('data') or {}).get('split')}, "
        f"雛形 {model.get('example')}, 監視 {training.get('monitor')}, "
        f"誤り {errors}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and f"版 {VERSION}" in readme else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. README の一式の版 {VERSION}"
    )
    result = (pid, view, record)

    return result


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 反応の表を要約して, プロジェクト
    の骨組みが読む場所へ書き出す関数

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
    # 一覧から開き, 問題の設定の列で反応の表の要約 (会員と品物の数, 反応で
    # ない行, 人気の偏り)
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    summary = fathom.ok("GET", f"/api/sources/{sid}/reactions")["summary"]
    dropped = summary["dropped"]
    journey.add(
        "ML-Fathom の反応の表の要約",
        "OK" if summary["users"] == USERS and summary["items"] == ITEMS
        and dropped["not_positive"] > 0 else "誤り",
        f"{summary['rows']} 行のうち反応 {summary['reactions']} 行 (除く "
        f"{dropped['total']} 行. 反応でない {dropped['not_positive']} 行), "
        f"会員 {summary['users']}, 品物 {summary['items']}, 組 "
        f"{summary['pairs']}, 密度 {rounded(summary['density'])}. 気付き "
        f"{summary['notes']}"
    )

    # 前処理をかけない反応の表の, 骨組みの読む場所への書き出し (ML-
    # Pathfinder の骨組みの書き出しで写した後なので置き換え)
    exported = fathom.ok("POST", f"/api/sources/{sid}/export_reactions",
                         {"overwrite": True})
    journey.add(
        "ML-Fathom (反応の表をプロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"])
        and exported["reactions"] == summary["reactions"] else "誤り",
        f"{Path(exported['path']).name} に写す (写したか "
        f"{exported['copied']}), 反応 {exported['reactions']} 行, 除く "
        f"{exported['dropped']} 行"
    )

    return entry


def make_model(
    ca: Client, journey: Journey, original: Path, project: Path
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 2 塔の雛形を学習用の
    会員と品物の数に書き換えてプロジェクトに保存する関数

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

    # 学習用の反応にいる会員と品物の数での雛形の書き換え
    info = ca.ok(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    shapes = {item["name"]: item["shape"] for item in info["inputs"]}
    sizes = info.get("rank") or {}
    template = ca.ok("GET", f"/api/examples/{EXAMPLE}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": EXAMPLE,
        "values": {"users": int(shapes["user"][-1]),
                   "items": int(shapes["item"][-1])}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    journey.add(
        "ML-CreateArchitecture (雛形を会員と品物の数に書き換え, 保存)",
        "OK" if not saved.get("notes") else "誤り",
        f"データセットの入力から. 会員 {sizes.get('users')}, 品物 "
        f"{sizes.get('items')}, 入力 "
        f"{ {k: v['shape'] for k, v in spec['inputs'].items()} }, 出力 "
        f"{sorted(spec['outputs'])}, 注意 {saved.get('notes') or []}"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> str:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, Recall@K と NDCG を読む関数

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

    # 骨組みを書き換えない学習と, 並べ方の評価の読み込み
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
            if e["name"].startswith("ranking_") and e["rows"]}
    test = rows.get("ranking_test") or {}
    journey.add(
        "学習 (骨組みを書き換えない) と並べ方の評価",
        # 人気順 (学習用の反応の多い順) より良いことの確かめ
        "OK" if test.get("recall_at_k_diff", 0) > 0
        and test.get("ndcg_diff", 0) > 0 else "誤り",
        f"{time.time() - started_at:.0f} 秒, {run['best']['monitor']} "
        f"{run['best']['value']:.4f}. テスト用: "
        f"{json.dumps(rounded(test), ensure_ascii=False)}. 評価の表 "
        f"{sorted(rows)}"
    )
    run_id = str(run["id"])

    return run_id


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    setup: dict[str, Any]
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 会員ごとに上位 K の品物を並べて,
    後の期間の記録で Recall@K と NDCG を数え直す関数

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
    setup: dict[str, Any]
        build の設定

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と版と件数の確かめ
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
    rank = description.get("rank") or {}
    got = (description.get("kind"), description.get("format_version"),
           rank.get("k"))
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got == ("rank", VERSION, TOP_K) else "誤り",
        f"種類, 版, 件数 {got}. 会員 {rank.get('users')}, 品物 "
        f"{rank.get('items')}, 列 {rank.get('columns')}"
    )

    # 問題の設定を使わない一式 (件数と列は一式が持つ) の, 欄の無さの確かめ
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    uses = bundle.get("uses_problem")
    journey.add(
        "問題の設定の欄 (一式の画面)",
        "OK" if uses is False and state.get("linked") is None else "誤り",
        "問題の設定を使わない一式なので欄が無い (件数と列は一式が持つ)"
        if uses is False else f"uses_problem {uses}, 候補 {state.get('hint')}"
    )

    # 学習の記録で前に買った品物を除いた会員の表の推論と, 後の期間の記録
    # での数え直し
    result = client.ok("POST", f"/api/bundles/{bid}/predict", {
        "source": str(setup["members"]), "k": TOP_K,
        "seen": str(setup["data"]), "truth": str(setup["later"])
    })
    summary = result["summary"]
    ranking = result.get("ranking") or {}
    metrics = ranking.get("metrics") or {}
    table = client.ok(
        "GET", f"/api/results/{result['id']}/rows?limit={TOP_K}"
    )
    first = [r for r in table["rows"] if r.get("rank") is not None]
    user = first[0]["user"] if first else None
    items = [r["item"] for r in first if r["user"] == user]
    seen = pd.read_csv(setup["data"], dtype=str)
    bought = set(seen.loc[(seen["member"] == user)
                          & (seen["qty"].astype(int) > 0), "item_id"])
    journey.add(
        "推論 (会員ごとに上位 K の品物を並べる)",
        "OK" if summary.get("predicted") == USERS
        and summary.get("k") == TOP_K and ranking.get("excluded")
        and len(items) == TOP_K and not set(items) & bought
        and "recall_at_k" in metrics and "ndcg" in metrics else "誤り",
        f"会員 {summary.get('rows')} 人のうち {summary.get('predicted')} 人"
        f"に上位 {summary.get('k')} 件 (前に買った品物を除く "
        f"{ranking.get('excluded')}). {user} の並び {items}. 後の期間の"
        f"記録で数え直し: {json.dumps(rounded(metrics), ensure_ascii=False)}"
        ". " + " ".join(result["sentences"][:1])
    )


def run_rank(apps: Apps, journey: Journey, setup: dict[str, Any]) -> None:
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
    _, view, record = set_problem(clients["pathfinder"], journey, setup,
                                  project)
    entry = analyze(clients["fathom"], journey, view["name"], record)
    original = Path(entry["path"]).resolve()
    make_model(clients["createarchitecture"], journey, original, project)
    run_id = train(clients["dashboard"], journey, original, project)
    provide(clients["provide"], journey, project, run_id, setup)


if __name__ == "__main__":
    failed = main_with(build, "推薦 (会員に勧める品)", sys.argv[1:],
                       runner=run_rank, prefix="rank_")
    sys.exit(1 if failed else 0)
