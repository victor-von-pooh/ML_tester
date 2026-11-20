"""
5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture,
ML-Dashboard, ML-Provide) を作業用の置き場で起動し, 画面と同じ API で
問題の設定から推論までを通す部品 (25 と 26 で使う)
"""
import argparse
import http.cookiejar
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

# アプリの名前, 起動のコマンド, 置き場の環境変数, トークンの環境変数
APPS = {
    "pathfinder": ("ML-Pathfinder", "ml-pathfinder", "ML_PATHFINDER_HOME",
                   "ML_PATHFINDER_TOKEN"),
    "fathom": ("ML-Fathom", "ml-fathom", "ML_FATHOM_HOME",
               "ML_FATHOM_TOKEN"),
    "createarchitecture": ("ML-CreateArchitecture", "ml-createarchitecture",
                           "ML_CREATEARCHITECTURE_HOME",
                           "ML_CREATEARCHITECTURE_TOKEN"),
    "dashboard": ("ML-Dashboard", "ml-dashboard", "ML_DASHBOARD_HOME",
                  "ML_DASHBOARD_TOKEN"),
    "provide": ("ML-Provide", "ml-provide", "ML_PROVIDE_HOME",
                "ML_PROVIDE_TOKEN")
}

# 起動を待つ秒数と, 環境づくりと学習を待つ秒数の上限
START_TIMEOUT = 120
JOB_TIMEOUT = 1800


def free_port() -> int:
    """
    空いているポートを返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    port: int
        空いているポート
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    return port


# 起動ごとのトークンを Cookie とヘッダーで渡す API の客 (base はアプリの URL)
class Client:
    def __init__(self, base: str, token: str):
        # URL, トークン, Cookie を持つ opener
        self.base = base
        self.token = token
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )

    def login(self) -> None:
        """
        トークンを Cookie に保存する関数 (画面を開くときと同じ)

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # /api/auth は画面に移る応答. 画面が無くても Cookie は保存
        try:
            self.opener.open(f"{self.base}/api/auth?token={self.token}")
        except urllib.error.HTTPError:
            pass

    def call(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> tuple[int, Any]:
        """
        API を呼ぶ関数

        Parameters
        ----------
        method: str
            GET, POST, PUT など
        path: str
            /api からのパス
        body: Optional[dict]
            JSON の本文

        Returns
        ----------
        result: tuple[int, Any]
            状態の番号と, 読んだ JSON (読めなければ文字)
        """
        # ML-Pathfinder にはヘッダーのトークンも
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data, method=method
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("x-pathfinder-token", self.token)
        try:
            with self.opener.open(request) as response:
                code, text = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, text = error.code, error.read()
        try:
            value = json.loads(text)
        except ValueError:
            value = text.decode(errors="replace")
        result = (code, value)

        return result

    def ok(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> Any:
        """
        API を呼び, 2xx でなければ止める関数

        Parameters
        ----------
        method: str
            GET, POST, PUT など
        path: str
            /api からのパス
        body: Optional[dict]
            JSON の本文

        Returns
        ----------
        value: Any
            読んだ JSON
        """
        code, value = self.call(method, path, body)
        if not 200 <= code < 300:
            raise RuntimeError(f"{method} {path} が {code}: {value}")

        return value


# 5 つのアプリを作業用の置き場で起動し, 止める仕組み (apps_dir は clone の親)
class Apps:
    def __init__(self, apps_dir: Path, work: Path):
        # 起動したプロセスと客
        self.apps_dir = apps_dir
        self.work = work
        self.token = secrets.token_urlsafe(16)
        self.processes: list[subprocess.Popen] = []
        self.logs: list[Any] = []
        self.clients: dict[str, Client] = {}

    def start(self) -> None:
        """
        全てのアプリを起動し, API が答えるまで待つ関数

        利用者の置き場 (~/.ml-*) を使わないように, 置き場の環境変数を
        全て作業用の置き場にする. ML-Fathom は ML-Pathfinder の置き場の
        問題の設定の一覧を読むので, 同じ置き場を渡す

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 置き場の環境変数 (全てのアプリに同じものを渡す)
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        for name, (_, _, home, token) in APPS.items():
            env[home] = str(self.work / "homes" / name)
            env[token] = self.token
        (self.work / "logs").mkdir(parents=True, exist_ok=True)
        for name, (repo, command, _, _) in APPS.items():
            backend = self.apps_dir / repo / "backend"
            if not backend.is_dir():
                raise SystemExit(
                    f"{backend} がありません. 5 つのアプリを clone して "
                    "make setup したディレクトリを --apps で渡してください"
                )
            port = free_port()
            # ML-CreateArchitecture の入力と出力はその追加の依存 project
            extra = (
                ["--extra", "project"] if name == "createarchitecture"
                else []
            )
            log = (self.work / "logs" / f"{name}.log").open("w")
            self.logs.append(log)
            process = subprocess.Popen(
                ["uv", "run", *extra, command, "--port", str(port)],
                cwd=backend, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            self.processes.append(process)
            self.clients[name] = Client(f"http://127.0.0.1:{port}",
                                        self.token)
        for name, client in self.clients.items():
            self.wait_ready(name, client)
            client.login()

    def wait_ready(self, name: str, client: Client) -> None:
        """
        アプリが答えるまで待つ関数

        Parameters
        ----------
        name: str
            アプリの名前
        client: Client
            アプリの客

        Returns
        ----------
        None
        """
        # 起動の途中は接続を断られるので, 答えるまでの繰り返し
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            try:
                client.opener.open(client.base + "/api/auth?token=x")
                return
            except urllib.error.HTTPError:
                return
            except OSError:
                time.sleep(1)
        raise SystemExit(
            f"{name} が {START_TIMEOUT} 秒で起動しません "
            f"({self.work / 'logs' / (name + '.log')} を見る)"
        )

    def stop(self) -> None:
        """
        全てのアプリを止める関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # uv run の子のサーバーも止まるように, 終わらせてからの待ち
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in self.logs:
            log.close()


# 段階ごとの結果 (OK, 手, 誤り) の記録
class Journey:
    def __init__(self, title: str):
        # 段階, 判定, 説明の並び
        self.title = title
        self.steps: list[dict[str, str]] = []

    def add(self, step: str, status: str, note: str = "") -> None:
        """
        段階の結果を記録して表示する関数

        Parameters
        ----------
        step: str
            段階の名前
        status: str
            OK (画面だけ), 手 (手で写す, 打つ), 誤り
        note: str
            説明

        Returns
        ----------
        None
        """
        self.steps.append({"step": step, "status": status, "note": note})
        print(f"[{status}] {step}: {note}", flush=True)

    def report(self, path: Path) -> int:
        """
        結果を JSON に書き, 表にして返す関数

        Parameters
        ----------
        path: Path
            書く JSON のファイル

        Returns
        ----------
        failed: int
            誤りの段階の数 (手の段階は数えない)
        """
        path.write_text(json.dumps(
            {"title": self.title, "steps": self.steps},
            ensure_ascii=False, indent=1
        ), encoding="utf-8")
        print(f"\n## {self.title}\n")
        print("| 段階 | 判定 | 説明 |\n| --- | --- | --- |")
        for item in self.steps:
            print(f"| {item['step']} | {item['status']} | {item['note']} |")
        manual = sum(1 for item in self.steps if item["status"] == "手")
        failed = sum(1 for item in self.steps if item["status"] == "誤り")
        print(f"\n手の段階 {manual}, 誤りの段階 {failed}")

        return failed


def wait_status(
    client: Client, path: str, running: tuple[str, ...]
) -> dict[str, Any]:
    """
    ジョブが終わるまで待つ関数

    Parameters
    ----------
    client: Client
        アプリの客
    path: str
        ジョブの状態の API
    running: tuple[str, ...]
        まだ終わっていない状態

    Returns
    ----------
    state: dict[str, Any]
        終わったときの状態
    """
    # 画面と同じく, 数秒ごとの状態の読み直し
    deadline = time.time() + JOB_TIMEOUT
    state = client.ok("GET", path)
    while state.get("status") in running:
        if time.time() > deadline:
            raise SystemExit(f"{path} が {JOB_TIMEOUT} 秒で終わりません")
        time.sleep(3)
        state = client.ok("GET", path)

    return state


def model_from_inputs(info: dict[str, Any]) -> dict[str, Any]:
    """
    データセットの入力と出す値の数から, ML-CreateArchitecture の画面の
    「入力と出力を作る」と同じ定義 (入力 x, 全結合の head, 出力 out) を
    作る関数

    Parameters
    ----------
    info: dict[str, Any]
        POST /api/dataset/inputs の結果

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl のモデルの定義
    """
    # 形のバッチの次元 (null) は例の大きさ 4
    first = info["inputs"][0]
    shape = [4 if size is None else int(size) for size in first["shape"]]
    spec = {
        "format_version": 1,
        "name": "model",
        "inputs": {"x": {"shape": shape, "dtype": first["dtype"]}},
        "nodes": {
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": shape[-1],
                           "out_features": int(info["outputs"])},
                "inputs": "x"
            }
        },
        "outputs": {"out": "head"}
    }

    return spec


def run_journey(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    問題の設定から推論までを, 画面と同じ API で通す関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        example (ML-Pathfinder の見本の名前), data (元のデータ), answers
        (足す答えの辞書), recipe_steps (ML-Fathom で足す手順),
        recipe_update (ML-Fathom の雛形の手順を直す関数. 無ければ None),
        past (推論とたどり直しのデータ. 目的の列を持つ), project
        (書き出し先)

    Returns
    ----------
    None
    """
    pf, fathom = apps.clients["pathfinder"], apps.clients["fathom"]
    ca, dash = apps.clients["createarchitecture"], apps.clients["dashboard"]
    provide = apps.clients["provide"]

    # 1. 問題の設定: 見本から作り, データを入れ, 今のやり方などに答える
    problem = pf.ok("POST", "/api/problems", {"example": setup["example"]})
    pid = problem["id"]
    pf.ok("POST", f"/api/problems/{pid}/data", {"path": str(setup["data"])})
    for key, value in setup["answers"].items():
        problem = pf.ok(
            "PUT", f"/api/problems/{pid}/answers", {"id": key, "value": value}
        )
    insight = pf.ok("GET", f"/api/problems/{pid}/insight")
    baseline = insight.get("baseline") or {}
    journey.add(
        "問題の設定 (データ, 今のやり方)",
        "OK" if problem["exportable"] else "誤り",
        baseline.get("sentence") or baseline.get("reason") or ""
    )

    # 2. 骨組みを書き出し, 場所を問題の設定に記録する
    written = pf.ok(
        "POST", f"/api/problems/{pid}/project",
        {"path": str(setup["project"])}
    )
    record = written["project"]
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if record.get("dataset") and record.get("model_spec") else "誤り",
        "書き出し先は利用者が選ぶ. 記録: "
        f"{Path(record['dataset']).name}, {Path(record['model_spec']).name}"
    )

    # 3. ML-Fathom: 一覧から問題の設定を開き, 前処理してプロジェクトへ
    listed = fathom.ok("GET", "/api/problems")["problems"]
    entry = next((p for p in listed if p["name"] == problem["name"]), None)
    if entry is None:
        raise RuntimeError("ML-Fathom の一覧に問題の設定がありません")
    started = fathom.ok("POST", "/api/problems/start", {"path": entry["path"]})
    sid = started["source"]["id"]
    recipe = fathom.ok("GET", f"/api/sources/{sid}/recipe")
    if setup.get("recipe_update"):
        setup["recipe_update"](recipe)
    recipe["steps"] += setup["recipe_steps"]
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
        "ML-Fathom (一覧から開く, 前処理, プロジェクトへ書き出す)",
        "OK" if Path(exported["path"]) == Path(record["dataset"]) else "誤り",
        f"入力 {exported['hints']['input_shape']}, 出す値 "
        f"{exported['hints']['outputs']}"
    )

    # 4. ML-CreateArchitecture: 保存先を確かめ, データセットから入出力
    journey.add(
        "ML-CreateArchitecture でプロジェクトを選ぶ", "手",
        "ML-Pathfinder が記録したプロジェクトの場所を打つ"
    )
    target = ca.ok(
        "POST", "/api/project/target", {"path": str(setup["project"])}
    )
    dataset = target.get("dataset")
    status, note = "OK", "保存先の確かめでデータセットの欄が入る"
    if not dataset:
        status, note = "手", "データセットの場所を手で打つ"
        dataset = record["dataset"]
    info = ca.ok("POST", "/api/dataset/inputs", {"path": dataset})
    spec = model_from_inputs(info)
    ca.ok(
        "PUT", "/api/project/model",
        {"path": str(setup["project"]), "spec": spec,
         "overwrite": target["exists"]}
    )
    journey.add(
        "ML-CreateArchitecture (入力と出力, プロジェクトに保存)", status,
        f"{note}. 入力 {spec['inputs']['x']['shape']}, 出す値 "
        f"{info['outputs']}"
    )

    # 5. ML-Dashboard: 登録, 学習の準備, 環境づくり, 学習
    project = dash.ok("POST", "/api/projects", {"path": str(setup["project"])})
    did = project["id"]
    journey.add(
        "ML-Dashboard にプロジェクトを登録", "手",
        "ML-Pathfinder が記録したプロジェクトの場所を打つ"
    )
    ready = dash.ok("GET", f"/api/projects/{did}/readiness")
    journey.add(
        "学習の準備 (データセット, モデルの定義, 環境)",
        "OK" if ready["dataset"]["exists"] and ready["model_spec"]["exists"]
        else "誤り",
        f"足りない機能 {ready['missing']}"
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
    launched = dash.ok(
        "POST", f"/api/projects/{did}/jobs",
        {"script": "train.py", "config_path": "configs/experiment/train.yaml"}
    )
    job = wait_status(
        dash, f"/api/jobs/{launched['id']}", ("running",)
    )
    run = job.get("run") or {}
    if job.get("return_code") != 0 or not run:
        tail = dash.ok("GET", f"/api/jobs/{launched['id']}/output")["text"]
        lines = tail.strip().splitlines() or [str(job.get("status"))]
        journey.add("学習", "誤り", lines[-1])
        return
    journey.add(
        "学習", "OK",
        f"{run['best']['monitor']} {run['best']['value']:.4f}"
    )
    run_path = f"/api/projects/{did}/runs/{run['id']}"
    results = dash.ok("GET", f"{run_path}/results")
    rows = {e["name"]: e["rows"][0] for e in results["evaluations"]
            if e["name"].startswith("baseline_") and e["rows"]}
    journey.add(
        "比べる相手の評価 (ML-Dashboard の結果)",
        "OK" if "baseline_test" in rows else "誤り",
        json.dumps(rows.get("baseline_test"), ensure_ascii=False)
    )

    # 6. ML-Provide: 実験の一式, 問題の設定, 推論, 意思決定, たどり直し
    bundle = provide.ok(
        "POST", "/api/bundles",
        {"path": str(Path(setup["project"]) / "outputs" / run["id"]
                     / "inference")}
    )
    bid = bundle["id"]
    journey.add(
        "ML-Provide に実験の一式を登録", "手",
        "実験のディレクトリの中の inference を場所で選ぶ"
    )
    hint = provide.ok("GET", f"/api/bundles/{bid}/problem")["hint"]
    linked = provide.ok(
        "PUT", f"/api/bundles/{bid}/problem", {"path": hint}
    )["linked"]
    if not linked:
        raise RuntimeError("一式に問題の設定をつなげません")
    journey.add(
        "問題の設定をつなぐ (候補から)",
        "OK" if not linked["issues"] else "誤り", linked["text"]
    )
    result = provide.ok(
        "POST", f"/api/bundles/{bid}/predict",
        {"source": str(setup["past"]), "decision": linked["decision"],
         "rules": linked["conclusion"]}
    )
    journey.add(
        "推論と意思決定", "OK" if result.get("decision") else "誤り",
        " ".join(result["sentences"][:2])
    )
    tested = provide.ok("POST", f"/api/results/{result['id']}/backtest", {})
    policies = (tested.get("backtest") or {}).get("policies") or {}
    journey.add(
        "今のやり方とのたどり直し (問題の設定から作る)",
        "OK" if "今のやり方" in policies else "誤り",
        json.dumps(policies, ensure_ascii=False)
    )


def main_with(
    build: Callable[[Path], dict[str, Any]], title: str, argv: list[str]
) -> int:
    """
    引数を読み, アプリを起動して通しを行い, 止める関数

    Parameters
    ----------
    build: Callable[[Path], dict[str, Any]]
        作業用の場所にデータを作り, run_journey に渡す設定を返す関数
    title: str
        通しの名前
    argv: list[str]
        コマンドの引数

    Returns
    ----------
    failed: int
        誤りの段階の数
    """
    # アプリの親のディレクトリ (既定は ML_tester と同じ並び) と作業用の場所
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
        "--apps", type=Path,
        default=Path(__file__).resolve().parents[3],
        help="5 つのアプリを clone して make setup したディレクトリの親"
    )
    parser.add_argument(
        "--work", type=Path, default=None,
        help="置き場, プロジェクト, ログを置くディレクトリ (既定は一時)"
    )
    args = parser.parse_args(argv)
    work = (args.work or Path(tempfile.mkdtemp(prefix="m1_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    print(f"作業用の場所: {work}", flush=True)
    setup = {**build(work), "project": work / "project"}
    apps = Apps(args.apps.resolve(), work)
    journey = Journey(title)
    try:
        apps.start()
        run_journey(apps, journey, setup)
    except (RuntimeError, LookupError, TypeError, ValueError) as error:
        # 途中で止まっても, それまでの段階と止まった理由を表に残す
        journey.add("止まった段階", "誤り", f"{type(error).__name__}: {error}")
    finally:
        apps.stop()
    failed = journey.report(work / "journey.json")

    return failed


if __name__ == "__main__":
    sys.exit("25 と 26 の journey.py から使う")
