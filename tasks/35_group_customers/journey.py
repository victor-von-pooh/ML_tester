"""
群分け (顧客の記録から, 似た顧客の群を作り, 群ごとに案内を変える) を,
問題の設定から推論まで 5 つの画面のアプリの API だけで通し, 段階ごとに
画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを表にする
"""
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

# 学習の顧客の数と, 推論で群に割り当てる新しい顧客の数
CUSTOMERS = 1200
TRIAL_CUSTOMERS = 400

# 群の数と, 群の安定性の数え方 (標本を変えて作り直す)
GROUPS = 3
STABILITY = "sample"

# 群を作る列と, 行を見分ける列
COLUMNS = ["spend", "visits", "region"]
ID_COLUMN = "customer_id"

# 骨組みの雛形と, 推論の一式の種類
EXAMPLE = "contrastive_encoder"
KIND = "group"

# 問題の設定の受け持ちの場所 (ML-Quest の PART)
PART = "ml.parts.0"

# 見本「解約の引き止め」の答えを, 顧客の群分けの課題に直す答え (受け持ちは
# まとめるで, 正解の列を使わない)
PROBLEM_ANSWERS: list[tuple[str, Any]] = [
    ("issue.owner", "販促の担当"),
    ("issue.pain", "全ての顧客に同じ案内を送っていて, 反応が薄い"),
    ("issue.current", "使う額で 3 段に分けて案内を変える"),
    ("issue.kpi.name", "案内からの来店の件数"),
    ("issue.kpi.unit", "件 / 月"),
    ("issue.kpi.better", "higher"),
    ("issue.kpi.current", 200),
    ("issue.kpi.target", 260),
    ("action.what", "群ごとの案内の文"),
    ("action.who", "販促の担当"),
    ("action.when", "毎月 1 日"),
    ("ml.reason", "顧客の使い方の違いは使う額だけでは分からず, 列を合わせて"
                  "似た顧客をまとめる必要がある"),
    (f"{PART}.role", "group"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "cluster"),
    (f"{PART}.target.source", "none"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.what", "顧客の群"),
    (f"{PART}.item", "顧客ごとの 1 行"),
    (f"{PART}.baseline", "使う額で 3 段に分ける"),
    (f"{PART}.evaluation.metric", "silhouette"),
    (f"{PART}.evaluation.link", "群ごとに案内を変えると, 案内からの来店が"
                                "増える"),
    ("verification.trial", "群ごとの案内と, 使う額の 3 段の案内を 1 か月"
                           "ずつ送って比べる"),
    ("verification.success", "群ごとの案内の来店の件数が多い"),
    ("operation.stop", "群の安定性が 2 回続けて下がったら止める")
]

# 表を入れた後に答える, 群の問い
GROUP_ANSWERS: list[tuple[str, Any]] = [
    (f"{PART}.group.columns", COLUMNS),
    (f"{PART}.group.id", ID_COLUMN),
    (f"{PART}.group.k", str(GROUPS)),
    (f"{PART}.group.stability", STABILITY)
]


def make_customers(
    path: Path, seed: int = 0, customers: int = CUSTOMERS
) -> None:
    """
    使い方の違う 3 つの群のある, 合成の顧客の記録を書く関数

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
    # 群ごとに違う, 使う額, 来店の回数, 多い地域 (群の境は少し重なる)
    rng = np.random.default_rng(seed)
    segment = rng.choice(GROUPS, customers, p=[0.5, 0.3, 0.2])
    spend = rng.normal(np.array([3000, 8000, 15000])[segment], 1200)
    visits = rng.poisson(np.array([2, 6, 4])[segment])
    regions = np.array(["north", "south", "west"])
    weights = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    region = [regions[rng.choice(3, p=weights[s])] for s in segment]
    pd.DataFrame({
        ID_COLUMN: [f"K{seed}{i:05d}" for i in range(customers)],
        "spend": spend.round(), "visits": visits, "region": region
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
        run_group に渡す設定
    """
    # 元の記録と, 推論で群に割り当てる別の乱数の種の新しい顧客
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "customers.csv"
    make_customers(data)
    trial = work / "data" / "customers_new.csv"
    make_customers(trial, seed=1, customers=TRIAL_CUSTOMERS)
    setup = {"data": data, "trial": trial, "work": work}

    return setup


def rounded(value: Any) -> Any:
    """
    表に出す値の小数を 4 桁に丸める関数 (辞書とリストは値ごと)

    Parameters
    ----------
    value: Any
        数, 辞書, リスト, その他

    Returns
    ----------
    result: Any
        小数を丸めた値 (小数でなければそのまま)
    """
    # 辞書とリストは値ごと, 小数だけの丸め
    if isinstance(value, dict):
        result = {key: rounded(item) for key, item in value.items()}
    elif isinstance(value, list):
        result = [rounded(item) for item in value]
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


def shares(groups: list[dict[str, Any]]) -> str:
    """
    群の要約の行を, 群ごとの行の数と安定性の短い文にする関数

    Parameters
    ----------
    groups: list[dict[str, Any]]
        群の要約の行 (最後の行は全ての行)

    Returns
    ----------
    text: str
        群ごとの「<群>: <行> 行 (安定性 <値>)」をつないだ文
    """
    text = ", ".join(
        f"{row['group']}: {row['rows']} 行 (安定性 "
        f"{rounded(row.get('stability'))})" for row in groups
    )

    return text


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
    # 見本から名前を付けた作成と, 受け持ちと課題の群分けへの答え直し
    view = pf.ok("POST", "/api/problems", {
        "example": "解約の引き止め", "name": "顧客の群分け"
    })
    pid = view["id"]
    for key, value in PROBLEM_ANSWERS:
        pf.ok("PUT", f"/api/problems/{pid}/answers",
              {"id": key, "value": value})
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in GROUP_ANSWERS:
        view = pf.ok("PUT", f"/api/problems/{pid}/answers",
                     {"id": key, "value": value})
    handoff = view.get("handoff") or {}
    group = handoff.get("group") or {}
    model = handoff.get("model") or {}
    training = handoff.get("training") or {}
    errors = errors_of(view) + [
        item.get("id") for item in view.get("open_errors") or []
    ]
    journey.add(
        "問題の設定 (表, 群を作る列, 群の数, 安定性の数え方)",
        "OK" if view["exportable"] and not errors
        and group.get("k") == GROUPS and model.get("example") == EXAMPLE
        and training.get("monitor") == "valid/silhouette" else "誤り",
        f"群 {group}, 雛形 {model.get('example')}, 監視 "
        f"{training.get('monitor')}, 分割 "
        f"{(handoff.get('data') or {}).get('split')}, 判断 "
        f"{(handoff.get('inference') or {}).get('decision')}, 誤り {errors}"
    )

    # 骨組みの書き出しと, 場所の問題の設定への記録
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(project)})
    record = written["project"]
    readme = (project / "README.md").read_text(encoding="utf-8")
    named = all(f"{name}.json" in readme
                for name in ("group_stability", "group_summary"))
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec")
        and named else "誤り",
        f"記録: {Path(record['dataset']).name}, "
        f"{Path(record['model_spec']).name}. README に群の安定性と群の要約"
        f"の評価 {named}"
    )
    result = (pid, view, record)

    return result


def analyze(
    fathom: Client, journey: Journey, name: str, record: dict[str, Any]
) -> dict[str, Any]:
    """
    ML-Fathom で問題の設定を一覧から開き, 答えの列で群の要約を作り, 表を
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
        ML-Fathom の一覧の問題の設定 (summary に群の要約)
    """
    # 一覧から群分けのタブで開き, 問題の設定の答えの読み込み
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    answers = fathom.ok("GET", f"/api/sources/{sid}/group_problem")
    group = answers["group"] or {}

    # 答えの列と群の数での, 学習に通さない群の要約 (群の安定性とまとまり)
    summary = fathom.ok("POST", f"/api/sources/{sid}/groups", {
        "columns": group.get("columns"), "k": group.get("k"),
        "stability": group.get("stability")
    })
    stability = summary.get("stability") or {}
    journey.add(
        "ML-Fathom で問題の設定を開き, 群の要約を作る",
        "OK" if started.get("tab") == "groups"
        and group.get("columns") == COLUMNS and summary["k"] == GROUPS
        and stability.get("ari") is not None else "誤り",
        f"タブ {started.get('tab')}, 答え {group.get('columns')} (群 "
        f"{group.get('k')}, {group.get('stability_label')}). 幅 "
        f"{summary['width']}, 安定性 ari {rounded(stability.get('ari'))} "
        f"(最低 {rounded(stability.get('ari_min'))}), シルエット係数 "
        f"{rounded(summary.get('silhouette'))}. {shares(summary['groups'])}"
    )

    # 骨組みが読む場所への表の書き出し (置き換えは確かめてから)
    place = fathom.ok("GET", f"/api/sources/{sid}/project")
    exported = fathom.ok(
        "POST", f"/api/sources/{sid}/export_group_project",
        {"overwrite": place.get("state") == "file"}
    )
    journey.add(
        "ML-Fathom (群を作る表をプロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"])
        and exported["columns"] == COLUMNS
        and exported["id"] == ID_COLUMN else "誤り",
        f"置き換え前の状態 {place.get('state')}, {exported['rows']} 行, 列 "
        f"{exported['columns']}, 行の id {exported['id']}"
    )
    entry = {**entry, "summary": summary}

    return entry


def make_model(
    ca: Client, journey: Journey, original: Path, project: Path,
    summary: dict[str, Any]
) -> None:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 対照学習の雛形を
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
    summary: dict[str, Any]
        ML-Fathom の群の要約 (データセットの入力が読めないときに手で写す
        符号化した行の幅)

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

    # 符号化した行の幅での雛形の書き換え (データセットの入力から読めなければ
    # ML-Fathom の群の要約の幅を手で写す)
    code, info = ca.call(
        "POST", "/api/dataset/inputs",
        {"path": target["dataset"], "problem": chosen["path"]}
    )
    if code == 200:
        width = int(info["inputs"][0]["shape"][-1])
        status, note = "OK", "データセットの入力から"
    else:
        width = int(summary["width"])
        journey.add(
            "ML-CreateArchitecture のデータセットの入力", "誤り",
            f"{code}: {info}. 幅を ML-Fathom の群の要約から写して続ける"
        )
        status, note = "手", "幅を ML-Fathom の群の要約から手で写す"
    template = ca.ok("GET", f"/api/examples/{EXAMPLE}")
    adapted = ca.ok("POST", "/api/templates/adapt", {
        "spec": template.get("spec", template), "name": EXAMPLE,
        "values": {"features": width}
    })
    spec = adapted["spec"]
    saved = ca.ok("PUT", "/api/project/model", {
        "path": chosen["path"], "spec": spec, "overwrite": target["exists"]
    })
    notes = saved.get("notes") or []
    journey.add(
        "ML-CreateArchitecture (雛形を行の幅に書き換え, 保存)",
        status if not notes or code != 200 else "誤り",
        f"{note}. 幅 {width} (ML-Fathom の群の要約は {summary['width']}), "
        f"入力 {spec['inputs']['x']['shape']}, 出力 "
        f"{sorted(spec['outputs'])}, 注意 {notes}"
    )


def train(
    dash: Client, journey: Journey, original: Path, project: Path
) -> str:
    """
    ML-Dashboard で書き出したプロジェクトを一覧から登録し, 環境を作って
    学習し, 群の安定性と群の要約を読む関数

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

    # 骨組みを書き換えない学習と, 群の安定性と群の要約の読み込み
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
    tables = {e["name"]: e["rows"] for e in results["evaluations"]}
    stability = (tables.get("group_stability") or [{}])[0]
    groups = tables.get("group_summary") or []
    best = run.get("best") or {}
    journey.add(
        "学習 (骨組みを書き換えない) と群の安定性, 群の要約",
        "OK" if stability and len(groups) == GROUPS + 1 else "誤り",
        f"{time.time() - started_at:.0f} 秒, {best.get('monitor')} "
        f"{rounded(best.get('value'))} (エポック {best.get('epoch')}). "
        f"群の安定性 {rounded(stability)}. 群の要約 {shares(groups)}"
    )
    run_id = str(run["id"])

    return run_id


def provide(
    client: Client, journey: Journey, project: Path, run_id: str,
    trial: Path
) -> None:
    """
    ML-Provide で実験を一覧から登録し, 新しい顧客を群に割り当てて群の
    要約を読む関数

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
        新しい顧客の表

    Returns
    ----------
    None
    """
    # 実験の一覧からの登録と, 説明の種類と群を作る列の確かめ
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
    group = description.get("group") or {}
    got = (description.get("kind"), description.get("format_version"),
           (description.get("output") or {}).get("form"))
    journey.add(
        "ML-Provide で実験を一覧から登録",
        "OK" if got[0] == KIND and got[2] == "cluster"
        and group.get("columns") == COLUMNS else "誤り",
        f"種類, 版, 出力の形 {got}. 群を作る列 {group.get('columns')}"
    )

    # 群分けの一式は問題の設定を使わない (欄が無い)
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    uses = bundle.get("uses_problem")
    journey.add(
        "問題の設定の欄 (一式の画面)",
        "OK" if uses is False and state.get("linked") is None else "誤り",
        "問題の設定を使わない一式なので欄が無い" if uses is False
        else f"uses_problem {uses}, 候補 {state.get('hint')}"
    )

    # 新しい顧客の推論と, 群の要約, 代表の行, 群で絞った行
    result = client.ok("POST", f"/api/bundles/{bid}/predict",
                       {"source": str(trial)})
    summary = result["summary"]
    groups = summary.get("groups") or []
    first = client.ok(
        "GET", f"/api/results/{result['id']}/rows?group=0&limit=5"
    )
    representatives = summary.get("representatives") or []
    journey.add(
        "推論 (新しい顧客を群に割り当て, 群の要約を出す)",
        "OK" if summary.get("predicted") == TRIAL_CUSTOMERS
        and summary.get("k") == GROUPS and len(groups) == GROUPS + 1
        and {r["group"] for r in first["rows"]} <= {0}
        and len(representatives) == GROUPS else "誤り",
        f"{summary.get('rows')} 行のうち {summary.get('predicted')} 行を "
        f"{summary.get('k')} 個の群に割り当て. 群の要約 {shares(groups)}. "
        f"群 0 の行 {first['total']} 件, 代表の行 "
        f"{[len(r['rows']) for r in representatives]} 件. "
        + " ".join(result["sentences"])
    )
    # 群の要約の行の表示 (README の表に写す)
    for item in groups:
        print(rounded(item), flush=True)


def run_group(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    群分けの問題の設定から推論までを画面と同じ API で通す関数

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
    _, view, record = set_problem(clients["pathfinder"], journey, setup,
                                  project)
    entry = analyze(clients["fathom"], journey, view["name"], record)
    original = Path(entry["path"]).resolve()
    make_model(clients["createarchitecture"], journey, original, project,
               entry["summary"])
    run_id = train(clients["dashboard"], journey, original, project)
    provide(clients["provide"], journey, project, run_id, setup["trial"])


if __name__ == "__main__":
    failed = main_with(build, "群分け (顧客)", sys.argv[1:],
                       runner=run_group, prefix="group_")
    sys.exit(1 if failed else 0)
