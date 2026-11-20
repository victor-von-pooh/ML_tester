"""
ファイルの組の受け持ち (画像の領域と点群) を, 問題の設定から推論まで 5 つの
画面のアプリの API だけで通す部品 (27 と 28 で使う. アプリの起動と記録は
25 の apps.py)
"""
import io
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

# 部品は 25 のディレクトリの apps.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Apps, Client, Journey, wait_status

# 型に依らない問題の設定の答え (画面で答える問いの順)
COMMON_ANSWERS = [
    ("issue.owner", "検査の担当"),
    ("issue.current", "人が 1 件ずつ目で見て書き込む"),
    ("issue.kpi.better", "lower"),
    ("action.who", "検査の担当"),
    ("action.when", "撮った後すぐ"),
    ("ml.use", True),
    ("ml.parts.0.role", "predict"),
    ("ml.parts.0.method", "scratch"),
    ("ml.parts.0.baseline", "人が書き込んだもの"),
    ("verification.method", "trial"),
    ("verification.trial", "新しい 10 件で人の書き込みと比べる"),
    ("verification.success", "書き込む時間が半分になる"),
    ("operation.review", "月に 1 回")
]


def fetch_bytes(client: Client, path: str) -> bytes:
    """
    API からファイルの中身をそのまま読む関数 (画面のダウンロードと同じ)

    Parameters
    ----------
    client: Client
        アプリの客
    path: str
        /api からのパス

    Returns
    ----------
    data: bytes
        読んだ中身
    """
    # 2xx でなければ urllib の HTTPError で停止
    with client.opener.open(client.base + path, timeout=300) as response:
        data = response.read()

    return data


def make_problem(
    pf: Client, journey: Journey, setup: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """
    ML-Pathfinder で空の問題の設定を作り, 問いに答えてデータを入れる関数

    Parameters
    ----------
    pf: Client
        ML-Pathfinder の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定

    Returns
    ----------
    result: tuple[str, dict[str, Any]]
        問題の設定の id と, 最後の画面の値
    """
    # 見本が無いので空から作り, 画面の問いの順に答える
    problem = pf.ok("POST", "/api/problems", {"name": setup["name"]})
    pid = problem["id"]
    for key, value in [*COMMON_ANSWERS, *setup["answers"]]:
        problem = pf.ok(
            "PUT", f"/api/problems/{pid}/answers", {"id": key, "value": value}
        )
    problem = pf.ok(
        "POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])}
    )
    insight = pf.ok("GET", f"/api/problems/{pid}/insight")
    journey.add(
        "問題の設定 (受け持ち, 正解の種類, 元のデータ)",
        "OK" if problem["exportable"] else "誤り",
        f"元のディレクトリを入れる. 確かめ: {insight_note(insight)}"
    )
    result = (pid, problem)

    return result


def insight_note(insight: dict[str, Any]) -> str:
    """
    ML-Pathfinder のデータの確かめを 1 行にする関数

    Parameters
    ----------
    insight: dict[str, Any]
        GET /api/problems/{id}/insight の応答

    Returns
    ----------
    note: str
        確かめの文を並べたもの
    """
    # 確かめ (findings) の文を並べる. 無ければ「無し」
    texts = []
    for finding in insight.get("findings") or []:
        text = (finding.get("message") or finding.get("text")
                if isinstance(finding, dict) else finding)
        texts.append(str(text or finding))
    note = short(" / ".join(texts) or "無し", 240)

    return note


def export_skeleton(
    pf: Client, journey: Journey, setup: dict[str, Any], pid: str
) -> dict[str, Any]:
    """
    ML-Pathfinder で骨組みを書き出し, プロジェクトの記録を返す関数

    Parameters
    ----------
    pf: Client
        ML-Pathfinder の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定
    pid: str
        問題の設定の id

    Returns
    ----------
    record: dict[str, Any]
        問題の設定に残ったプロジェクトの記録
    """
    # 書き出し先は利用者が選ぶ (最初に用意するものの場所なので OK)
    written = pf.ok(
        "POST", f"/api/problems/{pid}/project",
        {"path": str(setup["project"])}
    )
    record = written["project"]
    spec = record.get("model_spec")
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") else "誤り",
        f"記録: データセット {rel(record.get('dataset'), setup)}, "
        f"モデルの定義 {rel(spec, setup) if spec else '無し (config の '
                        'model.spec が null で, 仮のモデル)'}"
    )

    return record


def rel(path: Optional[str], setup: dict[str, Any]) -> str:
    """
    プロジェクトの中の場所を, プロジェクトからの相対の文字にする関数

    Parameters
    ----------
    path: Optional[str]
        場所
    setup: dict[str, Any]
        通しの設定 (project が書き出し先)

    Returns
    ----------
    text: str
        相対の場所 (外なら名前だけ, 無ければ「無し」)
    """
    # 作業用の場所のパスを表に残さないため
    if not path:
        text = "無し"
        return text
    target = Path(path).resolve()
    root = Path(setup["project"]).resolve()
    text = (str(target.relative_to(root)) if target.is_relative_to(root)
            else target.name)

    return text


def fathom_export(
    fathom: Client, journey: Journey, setup: dict[str, Any],
    problem_name: str, record: dict[str, Any]
) -> Path:
    """
    ML-Fathom で問題の設定を一覧から始め, ファイルの組をプロジェクトへ
    書き出す関数

    Parameters
    ----------
    fathom: Client
        ML-Fathom の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定 (origin, options, item_steps)
    problem_name: str
        問題の設定の名前
    record: dict[str, Any]
        プロジェクトの記録

    Returns
    ----------
    original: Path
        ML-Pathfinder の置き場の問題の設定 (problem.yaml)
    """
    # 一覧から選び, ファイルの組の書き出しの画面へ
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == problem_name), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    draft = fathom.ok(
        "GET", f"/api/problems/pairs?path={entry['path']}"
    )
    export = draft["export"]
    filled = bool(export.get("inputs") and export.get("targets"))
    journey.add(
        "ML-Fathom で問題の設定を一覧から始める",
        "OK" if started.get("form") and filled else "誤り",
        f"書き出しの画面 ({started.get('form')}) に移り, 入力と正解の"
        f"ディレクトリ ({Path(export['inputs']).name}, "
        f"{Path(export['targets'] or '-').name}) と分割 "
        f"{export['split']} が下書きで入る"
    )

    # 正解の出どころと型ごとの欄は画面で選ぶか答える (利用者が知る値)
    export["origin"] = setup["origin"]
    # 中身の手順も同じ欄 (書き出しの設定の options.item_steps)
    export["options"] = {**export.get("options", {}), **setup["options"]}
    if setup.get("item_steps"):
        export["options"]["item_steps"] = setup["item_steps"]
    body = {"path": draft["path"], "version": draft["version"],
            "export": export, "overwrite": False}
    done = fathom.ok("POST", "/api/problems/pairs/export", body)
    spec = yaml_of(Path(record["dataset"]) / "dataset.yaml")
    same = Path(done["path"]).resolve() == Path(record["dataset"]).resolve()
    steps = {name: entry.get("steps") for name, entry
             in (spec.get("entries") or {}).items() if entry.get("steps")}
    journey.add(
        "ML-Fathom でプロジェクトへ書き出す (書き出しの設定を書き戻す)",
        "OK" if same else "誤り",
        f"版 {spec.get('format_version')}, 分割 "
        f"{ {k: v.get('items') for k, v in spec.get('splits', {}).items()} }"
        f", 中身の手順 {steps or '無し'}, 注意 "
        f"{short(done.get('notes') or '無し', 200)}"
    )
    reread = fathom.ok("GET", f"/api/problems/pairs?path={entry['path']}")
    if setup.get("item_steps"):
        kept = (reread["export"].get("options") or {}).get("item_steps")
        journey.add(
            "開き直した書き出しの画面に中身の手順が残る",
            "OK" if kept == setup["item_steps"] else "手",
            f"問題の設定に書き戻した中身の手順 {kept}" if kept
            else "中身の手順 (resize) は問題の設定に残らない. 書き出し"
            "直すたびに選び直す"
        )
    original = Path(entry["path"]).resolve()
    if original.is_dir():
        original = original / "problem.yaml"

    return original


def yaml_of(path: Path) -> dict[str, Any]:
    """
    YAML のファイルを辞書で読む関数

    Parameters
    ----------
    path: Path
        YAML のファイル

    Returns
    ----------
    value: dict[str, Any]
        読んだ辞書 (無ければ空)
    """
    # 無いファイルは空
    value = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
             if path.is_file() else {})

    return value


def build_model(
    ca: Client, journey: Journey, setup: dict[str, Any], original: Path,
    record: dict[str, Any]
) -> str:
    """
    ML-CreateArchitecture で問題の設定を一覧から選び, 同梱の例から定義を
    作り, プロジェクトに保存する関数

    Parameters
    ----------
    ca: Client
        ML-CreateArchitecture の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定 (example, adapt)
    original: Path
        元の問題の設定
    record: dict[str, Any]
        プロジェクトの記録

    Returns
    ----------
    saved: str
        保存したモデルの定義の場所
    """
    # 一覧から選べるか (選べなければプロジェクトのディレクトリを打つ)
    listed = ca.ok("GET", "/api/problems")["problems"]
    chosen = next(
        (p for p in listed
         if Path(p["path"]).resolve() in (original, original.parent)), None
    )
    project_dir = Path(setup["project"]).resolve()
    target_from = str(project_dir)
    if chosen is not None and chosen.get("usable"):
        code, target = ca.call(
            "POST", "/api/project/target", {"path": chosen["path"]}
        )
        if code == 200:
            target_from = chosen["path"]
            journey.add("ML-CreateArchitecture で問題の設定を一覧から選ぶ",
                        "OK", "保存先が問題の設定の記録から決まる")
        else:
            journey.add(
                "ML-CreateArchitecture で問題の設定を一覧から選ぶ", "手",
                f"選ぶと {code}: {target}. プロジェクトのディレクトリを"
                "打って保存先 (既定の models/model.yaml) を決める"
            )
    else:
        reason = (chosen or {}).get("reason") or "一覧にありません"
        journey.add(
            "ML-CreateArchitecture で問題の設定を一覧から選ぶ", "手",
            f"選べない ({reason}). プロジェクトのディレクトリを打つ"
        )
    target = ca.ok("POST", "/api/project/target", {"path": target_from})

    # 同梱の例から開き, データセットから入力と出力を作れるか
    spec = ca.ok("GET", f"/api/examples/{setup['example']}")
    spec = spec.get("spec", spec)
    dataset = target.get("dataset") or record["dataset"]
    code, info = ca.call("POST", "/api/dataset/inputs", {"path": dataset})
    # 作れたら, 画面と同じく返った値を例に当ててから, 残る手直しを数える
    if code == 200 and isinstance(info, dict) and setup.get("apply_info"):
        spec = setup["apply_info"](spec, info)
    spec, changes = setup["adapt"](spec, Path(record["dataset"]))
    if code == 200:
        # 作れても, 例を手で直した所があれば手 (隠れた手を数えないため)
        status = "手" if changes else "OK"
        note = (f"データセットから入力と出力: {info}"
                + (f". 例の定義を手で直す: {'; '.join(changes)}"
                   if changes else ""))
    else:
        status = "手" if changes else "OK"
        note = (f"データセットから作れない ({code}: {short(info)}). "
                + (f"例の定義を手で直す: {'; '.join(changes)}" if changes
                   else "例の形が骨組みの既定と同じで, 直す所なし"))
    journey.add(
        f"ML-CreateArchitecture (同梱の例 {setup['example']}, 入力と出力)",
        status, note
    )
    ca.ok("PUT", "/api/project/model",
          {"path": target_from, "spec": spec, "overwrite": target["exists"]})
    saved = target["path"]
    journey.add(
        "ML-CreateArchitecture でプロジェクトに保存",
        "OK", f"保存先 {rel(saved, setup)}"
    )

    return saved


def short(value: Any, size: int = 160) -> str:
    """
    応答を表に収まる長さの文字にする関数

    Parameters
    ----------
    value: Any
        応答
    size: int
        最大の文字の数

    Returns
    ----------
    text: str
        切り詰めた文字
    """
    # FastAPI の誤りは detail の中
    if isinstance(value, dict) and "detail" in value:
        value = value["detail"]
    text = str(value).replace("\n", " ")
    text = text if len(text) <= size else text[:size] + "…"

    return text


def train(
    dash: Client, journey: Journey, setup: dict[str, Any], original: Path,
    saved: str
) -> str:
    """
    ML-Dashboard でプロジェクトを一覧から登録し, 環境を作り, 数エポック
    学習して結果と見本を見る関数

    Parameters
    ----------
    dash: Client
        ML-Dashboard の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定 (epochs, check_samples)
    original: Path
        元の問題の設定
    saved: str
        ML-CreateArchitecture が保存したモデルの定義

    Returns
    ----------
    run_id: str
        実験の id
    """
    # 問題の設定が記録したプロジェクトを一覧から登録
    project_dir = Path(setup["project"]).resolve()
    recorded = dash.ok("GET", "/api/problems/projects")["projects"]
    found = next(
        (p for p in recorded
         if Path(p["problem_path"]).resolve() in (original, original.parent)),
        None
    )
    if found is None:
        raise RuntimeError("ML-Dashboard の一覧に書き出したプロジェクトが"
                           "ありません")
    project = dash.ok("POST", "/api/projects", {"path": found["path"]})
    did = project["id"]
    journey.add(
        "ML-Dashboard で書き出したプロジェクトを一覧から登録",
        "OK" if Path(found["path"]).resolve() == project_dir else "誤り",
        f"問題の設定「{found['problem']}」が記録したプロジェクト"
    )
    ready = dash.ok("GET", f"/api/projects/{did}/readiness")
    journey.add(
        "学習の準備 (データセット, モデルの定義, 環境)",
        "OK" if ready["dataset"]["exists"] else "誤り",
        f"データセット {ready['dataset']['exists']}, モデルの定義 "
        f"{ready['model_spec']}, 足りない機能 {ready['missing']}"
    )
    started_at = time.time()
    dash.ok("POST", f"/api/projects/{did}/environment",
            {"commands": ready["commands"]})
    job = wait_status(
        dash, f"/api/projects/{did}/environment", ("running",)
    )
    ready = dash.ok("GET", f"/api/projects/{did}/readiness?refresh=true")
    journey.add(
        "環境を作る (画面から)",
        "OK" if job["status"] == "succeeded" and ready["ready"] else "誤り",
        f"{job['status']}, {time.time() - started_at:.0f} 秒, 足りない "
        f"{ready['missing']}"
    )

    # 起動の画面の config の欄: エポックの数を減らす (画面の欄の書き換え).
    # 骨組みの config がモデルの定義を指さなければ, 保存した場所を打つ
    config = yaml_of(project_dir / "configs" / "default" / "train.yaml")
    experiment = yaml_of(project_dir / "configs" / "experiment"
                         / "train.yaml")
    typed = []
    epochs = (config.get("trainer") or {}).get("epochs")
    if setup.get("epochs") and setup["epochs"] != epochs:
        experiment.setdefault("trainer", {})["epochs"] = setup["epochs"]
        epochs = setup["epochs"]
    if not (config.get("model") or {}).get("spec"):
        spec = Path(saved).resolve().relative_to(project_dir)
        experiment.setdefault("model", {})["spec"] = str(spec)
        requires = list(config.get("requires") or [])
        if "modelcontrol" not in requires:
            experiment["requires"] = [*requires, "modelcontrol"]
        typed.append(f"model.spec: {spec} と requires に modelcontrol")
    launched = dash.ok(
        "POST", f"/api/projects/{did}/jobs",
        {"script": "train.py",
         "config_text": yaml.safe_dump(experiment, allow_unicode=True)}
    )
    journey.add(
        "学習の起動の config (エポックの数, モデルの定義)",
        "手" if typed else "OK",
        f"エポックの数 {epochs}"
        + (" (欄で直す)" if "trainer" in experiment else " (骨組みのまま)")
        + (f". 骨組みの config の model.spec が null なので, 保存した"
           f"定義の場所を打つ ({'; '.join(typed)})" if typed else "")
    )
    started_at = time.time()
    job = wait_status(
        dash, f"/api/jobs/{launched['id']}", ("running",),
        stop=f"/api/jobs/{launched['id']}/stop"
    )
    run = job.get("run") or {}
    if job.get("return_code") != 0 or not run:
        tail = dash.ok("GET", f"/api/jobs/{launched['id']}/output")["text"]
        lines = tail.strip().splitlines() or [str(job.get("status"))]
        raise RuntimeError(f"学習が止まった: {lines[-1]}")
    journey.add(
        "学習", "OK",
        f"{run['best']['monitor']} {run['best']['value']:.4f} "
        f"({epochs} エポックまで, {time.time() - started_at:.0f} 秒)"
    )
    run_path = f"/api/projects/{did}/runs/{run['id']}"
    results = dash.ok("GET", f"{run_path}/results")
    evaluations = {e["name"]: e["rows"] for e in results["evaluations"]}
    journey.add(
        "テスト用の評価 (ML-Dashboard の結果)",
        "OK" if evaluations else "誤り",
        short(evaluations, 240)
    )
    log = dash.ok("GET", f"{run_path}/log")
    lines = [str(entry.get("message", entry)) if isinstance(entry, dict)
             else str(entry) for entry in (log if isinstance(log, list)
                                           else log.get("lines", []))]
    bundle_lines = [line for line in lines if line.startswith("推論の一式")]
    journey.add(
        "学習の後の推論の一式 (ML-Dashboard のログ)",
        "OK" if any("合う: True" in line for line in bundle_lines)
        else "誤り",
        short(" / ".join(bundle_lines) or "一式の行が無い", 240)
    )
    status, note = setup["check_samples"](dash, run_path, results)
    journey.add("見本を見る (ML-Dashboard の結果)", status, note)
    run_id = str(run["id"])

    return run_id


def provide(
    client: Client, journey: Journey, setup: dict[str, Any], run_id: str
) -> None:
    """
    ML-Provide で実験の一式を一覧から登録し, 新しい件を推論して出力を
    確かめる関数

    Parameters
    ----------
    client: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        通しの設定 (new, check_outputs)
    run_id: str
        ML-Dashboard の実験の id

    Returns
    ----------
    None
    """
    # 実験の一覧から学習した実験の行を選んで登録
    project_dir = Path(setup["project"]).resolve()
    listing = client.ok("GET", "/api/experiments")
    row = next(
        (r for r in listing["experiments"]
         if Path(r["project"]).resolve() == project_dir
         and r["experiment"] == run_id), None
    )
    if row is None:
        raise RuntimeError("ML-Provide の実験の一覧に学習した実験が"
                           "ありません")
    bundle = client.ok(
        "POST", "/api/bundles",
        {"path": row["path"],
         "name": f"{row['problem']} ({row['experiment']})"}
    )
    bid = str(bundle["id"])
    detail = client.ok("GET", f"/api/bundles/{bid}")
    checked = bundle_check(detail)
    matches = isinstance(checked, dict) and checked.get("matches")
    journey.add(
        "ML-Provide で実験の一式を一覧から登録 (一式の確かめ)",
        "OK" if matches else "誤り",
        f"問題の設定「{row['problem']}」の実験 {row['experiment']}. "
        f"一式の確かめ: {short(checked, 200)}"
    )
    state = client.ok("GET", f"/api/bundles/{bid}/problem")
    journey.add(
        "問題の設定をつなぐ (候補)",
        "OK",
        "候補 " + ("あり" if state.get("hint") else "無し")
        + " (ファイルの組の受け持ちは意思決定が無く, 候補に出さない決まり)"
    )
    result = client.ok(
        "POST", f"/api/bundles/{bid}/predict", {"source": str(setup["new"])}
    )
    rows = client.ok("GET", f"/api/results/{result['id']}/rows")
    status, note = setup["check_outputs"](client, result, rows, setup)
    journey.add("新しい件の推論と出力", status, note)


def bundle_check(detail: dict[str, Any]) -> Any:
    """
    一式の詳細から, 一式での評価と実験の記録との比べを取り出す関数

    Parameters
    ----------
    detail: dict[str, Any]
        GET /api/bundles/{id} の応答

    Returns
    ----------
    check: Any
        比べの値 (無ければ「無し」)
    """
    # 一式の説明 (ML-Inference の describe_bundle) の check
    description = detail.get("description") or {}
    found = description.get("check")
    check = ({key: found.get(key) for key in
              ("split", "metric", "value", "recorded", "matches")}
             if isinstance(found, dict) else "無し")

    return check


def run_pairs_journey(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    ファイルの組の受け持ちを, 問題の設定から推論まで画面と同じ API で
    通す関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        name, answers, data, origin, options, item_steps, example, adapt,
        epochs, check_samples, new, check_outputs, project

    Returns
    ----------
    None
    """
    # 1, 2. ML-Pathfinder: 問題の設定と骨組み
    pf, fathom = apps.clients["pathfinder"], apps.clients["fathom"]
    pid, problem = make_problem(pf, journey, setup)
    record = export_skeleton(pf, journey, setup, pid)
    # 3. ML-Fathom
    original = fathom_export(fathom, journey, setup, problem["name"], record)
    # 4. ML-CreateArchitecture
    saved = build_model(apps.clients["createarchitecture"], journey, setup,
                        original, record)
    # 5. ML-Dashboard
    run_id = train(apps.clients["dashboard"], journey, setup, original,
                   saved)
    # 6. ML-Provide
    provide(apps.clients["provide"], journey, setup, run_id)


def load_npy(client: Client, result_id: str, row: int) -> np.ndarray:
    """
    推論の結果の行の配列 (.npy) を読む関数 (画面のダウンロードと同じ)

    Parameters
    ----------
    client: Client
        ML-Provide の客
    result_id: str
        結果の id
    row: int
        行の位置

    Returns
    ----------
    array: np.ndarray
        行の配列
    """
    # ダウンロードの中身を NumPy で読む
    data = fetch_bytes(
        client, f"/api/results/{result_id}/artifacts/{row}/file"
    )
    array = np.load(io.BytesIO(data), allow_pickle=False)

    return array
