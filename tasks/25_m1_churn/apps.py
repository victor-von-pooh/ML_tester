"""
5 つの画面のアプリ (ML-Pathfinder, ML-Fathom, ML-CreateArchitecture,
ML-Dashboard, ML-Provide) を作業用の置き場で起動し, 画面と同じ API で
問題の設定から推論までを通す部品 (25 から 30 で使う)
"""
import argparse
import http.client
import http.cookiejar
import json
import os
import secrets
import signal
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
# API の 1 回の呼び出しの待ちの上限 (秒. 答えないアプリで止まらないため)
CALL_TIMEOUT = 300

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

# 止めるときに, プロセスのまとまりが終わるのを待つ秒数
STOP_TIMEOUT = 20


# 起動か待ちが時間の上限を超えたときの例外 (表に誤りとして残す)
class JourneyTimeout(RuntimeError):
    pass


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
            with self.opener.open(request, timeout=CALL_TIMEOUT) as response:
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
                raise RuntimeError(
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
            # 止めるときに uv run の子のサーバーまで届くように, 別の
            # プロセスのまとまりで起動
            process = subprocess.Popen(
                ["uv", "run", *extra, command, "--port", str(port)],
                cwd=backend, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True
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
        raise JourneyTimeout(
            f"{name} が {START_TIMEOUT} 秒で起動しません "
            f"({self.work / 'logs' / (name + '.log')} を見る)"
        )

    def stop_jobs(self) -> None:
        """
        ML-Dashboard の動いている学習のジョブを止める関数

        学習は ML-Dashboard とは別のプロセスのまとまりで動くので, アプリを
        止めても残る. 止まった通しが学習を残さないように, 先に止める

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 起動していないか応答しない場合は何もしない
        dash = self.clients.get("dashboard")
        if dash is None:
            return
        try:
            code, jobs = dash.call("GET", "/api/jobs")
            for job_id in running_ids(jobs) if code == 200 else []:
                dash.call("POST", f"/api/jobs/{job_id}/stop",
                          {"force": True})
        except (OSError, http.client.HTTPException) as error:
            print(f"学習のジョブを止められませんでした: {error}", flush=True)

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
        # ML-Dashboard の学習は別のプロセスのまとまりで動くので, 先に API
        # で止める. その後 uv run の子のサーバーも止まるように, プロセスの
        # まとまりごとの終わらせと待ち. 待っても残れば強制の終わり
        self.stop_jobs()
        for process in self.processes:
            signal_group(process, signal.SIGTERM)
        for process in self.processes:
            try:
                process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                signal_group(process, signal.SIGKILL)
                process.wait()
        for log in self.logs:
            log.close()


def running_ids(jobs: Any) -> list[str]:
    """
    ジョブの一覧から, 動いているジョブの ID を取り出す関数

    Parameters
    ----------
    jobs: Any
        ML-Dashboard の GET /api/jobs の応答

    Returns
    ----------
    ids: list[str]
        動いているジョブの ID
    """
    # 一覧でない応答は無し
    ids = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict) and job.get("status") == "running":
                ids.append(str(job.get("id")))

    return ids


def signal_group(process: subprocess.Popen, number: int) -> None:
    """
    起動したプロセスのまとまりの全てにシグナルを送る関数

    Parameters
    ----------
    process: subprocess.Popen
        start_new_session で起動したプロセス
    number: int
        送るシグナル

    Returns
    ----------
    None
    """
    # 既に終わったまとまりは飛ばす
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass


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
    client: Client, path: str, running: tuple[str, ...],
    stop: Optional[str] = None
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
    stop: Optional[str]
        時間の上限を超えたときに呼ぶ, ジョブを止める API (無ければ None)

    Returns
    ----------
    state: dict[str, Any]
        終わったときの状態

    Raises
    ----------
    JourneyTimeout
        時間の上限を超えた場合 (止める API を先に呼ぶ)
    """
    # 画面と同じく, 数秒ごとの状態の読み直し
    deadline = time.time() + JOB_TIMEOUT
    state = client.ok("GET", path)
    while state.get("status") in running:
        if time.time() > deadline:
            if stop:
                # 本文が要る (StopRequest). 止められなくても時間切れは出す
                code, _ = client.call("POST", stop, {"force": True})
                if code != 200:
                    print(f"止める依頼が {code} で失敗しました: {stop}",
                          flush=True)
            raise JourneyTimeout(
                f"{path} が {JOB_TIMEOUT} 秒で終わりません"
            )
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
        任意で item_steps (ML-Fathom の客とデータの ID から, 中身の手順,
        判定, 説明を返す関数) と make_model (ML-CreateArchitecture の客,
        入力と出力, データセットから, 定義, 手で直す所, 画面が当てる所の
        説明を返す関数. 無ければ全結合の head だけの定義), past (推論と
        たどり直しのデータ. 目的の列を持つ), cost_update (学習の後に
        ML-Pathfinder で直す損得の値), project (書き出し先)

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
    if setup.get("item_steps"):
        # 画像と文の中身の手順 (画面で選ぶ値. 表と時系列には無い)
        steps, status, note = setup["item_steps"](fathom, sid)
        recipe["item_steps"] = steps
        journey.add("ML-Fathom の中身の手順 (画像と文)", status, note)
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

    # 4. ML-CreateArchitecture: 問題の設定を一覧から選び, データセットから
    # 入力と出力を作ってプロジェクトに保存
    original = Path(entry["path"]).resolve()
    listed = ca.ok("GET", "/api/problems")["problems"]
    chosen = next(
        (p for p in listed if Path(p["path"]).resolve() == original), None
    )
    if chosen is None or not chosen["usable"]:
        reason = chosen["reason"] if chosen else "一覧にありません"
        raise RuntimeError(f"ML-CreateArchitecture の問題の設定: {reason}")
    project_dir = Path(setup["project"]).resolve()
    journey.add(
        "ML-CreateArchitecture で問題の設定を一覧から選ぶ",
        "OK" if Path(chosen["project"]).resolve() == project_dir
        else "誤り",
        "一覧の問題の設定が, 記録したプロジェクトとデータセットを持つ"
    )
    target = ca.ok("POST", "/api/project/target", {"path": chosen["path"]})
    dataset = target.get("dataset")
    status, note = "OK", "選んだ問題の設定でデータセットの欄が入る"
    if not dataset:
        status, note = "手", "データセットの場所を手で打つ"
        dataset = record["dataset"]
    info = ca.ok("POST", "/api/dataset/inputs", {"path": dataset})
    if setup.get("make_model"):
        # 同梱の例に画面と同じく入力と出力を当て, 残る手直しを数える
        spec, manual, applied = setup["make_model"](ca, info, Path(dataset))
        note += f". 画面が当てる: {applied}"
        if manual:
            status, note = "手", f"{note}. 手で直す: {manual}"
    else:
        spec = model_from_inputs(info)
    ca.ok(
        "PUT", "/api/project/model",
        {"path": chosen["path"], "spec": spec,
         "overwrite": target["exists"]}
    )
    journey.add(
        "ML-CreateArchitecture (入力と出力, プロジェクトに保存)", status,
        f"{note}. 入力 {next(iter(spec['inputs'].values()))['shape']}, "
        f"出す値 {info['outputs']}"
    )

    # 5. ML-Dashboard: 一覧から登録, 学習の準備, 環境づくり, 学習
    recorded = dash.ok("GET", "/api/problems/projects")["projects"]
    found = next(
        (p for p in recorded
         if Path(p["problem_path"]).resolve() == original), None
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
        dash, f"/api/jobs/{launched['id']}", ("running",),
        stop=f"/api/jobs/{launched['id']}/stop"
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

    # 6. ML-Provide と, ML-Pathfinder で損得を直した後の推論の既定
    bid = provide_journey(provide, journey, setup, original, run["id"])
    check_cost_update(pf, provide, journey, setup, pid, bid)


def problem_file(path: str) -> Path:
    """
    問題の設定の場所 (problem.yaml かそのディレクトリ) をファイルにする
    関数

    Parameters
    ----------
    path: str
        problem.yaml かそのディレクトリ

    Returns
    ----------
    file: Path
        problem.yaml の場所 (リンクをたどった形)
    """
    # ディレクトリなら中の problem.yaml
    file = Path(path).resolve()
    if file.is_dir():
        file = file / "problem.yaml"

    return file


def provide_journey(
    provide: Client, journey: Journey, setup: dict[str, Any],
    original: Path, run_id: str
) -> str:
    """
    ML-Provide で, 実験を一覧から登録し, 問題の設定をつなぎ, 推論と意思
    決定とたどり直しをする関数

    Parameters
    ----------
    provide: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        run_journey に渡した設定
    original: Path
        ML-Pathfinder の置き場の元の問題の設定 (problem.yaml)
    run_id: str
        ML-Dashboard の実験の id (出力先からの実験の場所)

    Returns
    ----------
    bid: str
        登録した一式の id
    """
    # 実験の一覧から, 学習した実験の行を選んで登録 (名前は画面と同じ)
    project_dir = Path(setup["project"]).resolve()
    listing = provide.ok("GET", "/api/experiments")
    row = next(
        (r for r in listing["experiments"]
         if Path(r["project"]).resolve() == project_dir
         and r["experiment"] == run_id), None
    )
    if row is None:
        raise RuntimeError("ML-Provide の実験の一覧に学習した実験が"
                           "ありません")
    bundle = provide.ok(
        "POST", "/api/bundles",
        {"path": row["path"],
         "name": f"{row['problem']} ({row['experiment']})"}
    )
    bid = str(bundle["id"])
    journey.add(
        "ML-Provide で実験を一覧から登録", "OK",
        f"問題の設定「{row['problem']}」の実験 {row['experiment']}"
    )

    # 候補の問題の設定は, 学習の時点の写しでなく元のもの
    state = provide.ok("GET", f"/api/bundles/{bid}/problem")
    hint = state["hint"]
    if not hint:
        raise RuntimeError("一式に問題の設定の候補がありません")
    linked = provide.ok(
        "PUT", f"/api/bundles/{bid}/problem", {"path": hint}
    )["linked"]
    if not linked:
        raise RuntimeError("一式に問題の設定をつなげません")
    is_original = problem_file(hint) == original
    journey.add(
        "問題の設定をつなぐ (候補から)",
        "OK" if is_original and not linked["issues"] else "誤り",
        ("元の問題の設定. " if is_original
         else f"写し ({state.get('hint_note')}). ") + linked["text"]
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

    return bid


def check_cost_update(
    pf: Client, provide: Client, journey: Journey, setup: dict[str, Any],
    pid: str, bid: str
) -> None:
    """
    学習の後に ML-Pathfinder で元の問題の設定の損得を直し, ML-Provide の
    次の推論の既定に入るかを確かめる関数

    Parameters
    ----------
    pf: Client
        ML-Pathfinder の客
    provide: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        run_journey に渡した設定 (cost_update は直す損得の値)
    pid: str
        ML-Pathfinder の問題の設定の id
    bid: str
        ML-Provide の一式の id

    Returns
    ----------
    None
    """
    # 今の損得の答えに直す値を重ね, 画面の答えと同じ API で保存
    view = pf.ok("GET", f"/api/problems/{pid}")
    current = next(
        q["value"] for q in view["questions"] if q["id"] == "action.decision"
    )
    changed = {**current, **setup["cost_update"]}
    pf.ok("PUT", f"/api/problems/{pid}/answers",
          {"id": "action.decision", "value": changed})

    # 一式の画面を開き直したときの既定と, その既定での推論
    linked = provide.ok("GET", f"/api/bundles/{bid}/problem")["linked"]
    decision = (linked or {}).get("decision") or {}
    got = {key: decision.get(key) for key in setup["cost_update"]}
    result = provide.ok(
        "POST", f"/api/bundles/{bid}/predict",
        {"source": str(setup["past"]), "decision": decision,
         "rules": linked["conclusion"] if linked else None}
    )
    # 結果の decision の中の decision が, 推論で使った意思決定の値
    used = (result.get("decision") or {}).get("decision") or {}
    applied = {key: used.get(key) for key in setup["cost_update"]}
    journey.add(
        "ML-Pathfinder で損得を直すと次の推論の既定に入る",
        "OK" if got == setup["cost_update"] and applied == got else "誤り",
        f"直した値 {setup['cost_update']}, 推論の既定 {got}, 推論で使った "
        f"値 {applied}"
    )


def main_with(
    build: Callable[[Path], dict[str, Any]], title: str, argv: list[str],
    runner: Optional[Callable[[Apps, Journey, dict[str, Any]], None]] = None,
    prefix: str = "m1_"
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
    runner: Optional[Callable[[Apps, Journey, dict[str, Any]], None]]
        通しの関数 (省略で表の通しの run_journey)
    prefix: str
        既定の作業用の場所 (一時ディレクトリ) の名前の頭

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
    work = (args.work or Path(tempfile.mkdtemp(prefix=prefix))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    print(f"作業用の場所: {work}", flush=True)
    setup = {**build(work), "project": work / "project"}
    apps = Apps(args.apps.resolve(), work)
    journey = Journey(title)
    try:
        apps.start()
        (runner or run_journey)(apps, journey, setup)
    except Exception as error:
        # 途中で止まっても, それまでの段階と止まった理由を表に残す
        journey.add("止まった段階", "誤り", f"{type(error).__name__}: {error}")
    finally:
        apps.stop()
    failed = journey.report(work / "journey.json")

    return failed


if __name__ == "__main__":
    sys.exit("25 から 30 の journey.py から使う")
