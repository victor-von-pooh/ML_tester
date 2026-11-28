"""
方策の問題の設定 (policy/problem.yaml) を ML-Pathfinder に取り込んで
骨組みを書き出し, ML-Dashboard で骨組みを書き換えずに学習し, ML-Provide の
方策の実験の一覧から一式を登録して, 観測から行動と守りの結果を出すまでを
3 つの画面のアプリの API だけで通し, 段階ごとに画面だけでできたか (OK),
手で写すか打つか (手), 誤りかを表にする
"""
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import (  # noqa: E402
    Apps, Client, Journey, main_with, wait_status
)

# このタスクのディレクトリと, 取り込む方策の問題の設定
TASK_DIR = Path(__file__).resolve().parent
PROBLEM = TASK_DIR / "policy" / "problem.yaml"

# 学習の config に足す守り (README の実験 2 と同じ数. 問題の設定の守りの
# 文「左右に押すだけ」「台車が中心から 2.4 を出たら」を数にしたもの)
GUARDS = {
    "allowed_actions": [0, 1],
    "observation_low": [-2.4, None, None, None],
    "observation_high": [2.4, None, None, None]
}

# 守りの範囲の中と外の観測 (台車の位置, 速さ, 棒の角度, 角速度). 外は
# 位置が範囲の外と, 数でない値
OBSERVATIONS = {
    "範囲の中": [0.0, 0.0, 0.02, 0.0],
    "位置が範囲の外": [3.0, 0.0, 0.0, 0.0],
    "数でない値": [float("nan"), 0.0, 0.0, 0.0]
}

# 範囲の中と外で期待する守りの結果 (外れたときの動きは問題の設定の答えの
# action で, 行動 0 で続ける)
EXPECTED = {"範囲の中": "ok", "位置が範囲の外": "fallback",
            "数でない値": "fallback"}

# ML-Provide の act で CartPole を動かすエピソードの乱数の種 (学習の検証と
# 守り付きの評価, infer.py の種と重ならない値)
EPISODE_SEED = 40000

# 環境の名前 (問題の設定と骨組みの config と同じ)
ENV_ID = "CartPole-v1"


def build(work: Path) -> dict[str, Any]:
    """
    作業用の場所に置くものは無いので, 問題の設定の場所だけを返す関数

    Parameters
    ----------
    work: Path
        作業用の場所

    Returns
    ----------
    setup: dict[str, Any]
        problem (取り込む problem.yaml)
    """
    setup = {"problem": PROBLEM}

    return setup


def set_problem(
    pf: Client, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    ML-Pathfinder に問題の設定を取り込み, 骨組みを書き出す関数

    Parameters
    ----------
    pf: Client
        ML-Pathfinder の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        problem と project

    Returns
    ----------
    None
    """
    # 画面の「取り込む」と同じ (元のファイルは変えずに置き場へ写す)
    view = pf.ok("POST", "/api/problems/import",
                 {"path": str(setup["problem"])})
    pid = view["id"]
    journey.add(
        "問題の設定を取り込む", "OK" if view["exportable"] else "誤り",
        f"「{view['name']}」, 直すところ {len(view['open_errors'])} 件"
    )
    written = pf.ok("POST", f"/api/problems/{pid}/project",
                    {"path": str(setup["project"])})
    files = written["files"]
    journey.add(
        "骨組みの書き出し (場所を問題の設定に記録)",
        "OK" if written["project"].get("path") else "誤り",
        f"書いたファイル {len(files)}"
    )


def train(
    dash: Client, journey: Journey, setup: dict[str, Any]
) -> str:
    """
    ML-Dashboard で, 問題の設定が記録したプロジェクトを一覧から登録し,
    守りを数にした experiment の config で骨組みを書き換えずに学習する関数

    Parameters
    ----------
    dash: Client
        ML-Dashboard の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        project

    Returns
    ----------
    run_id: str
        学習した実験の id (出力先からの実験の場所)
    """
    # 問題の設定が記録したプロジェクトの一覧から登録
    project_dir = Path(setup["project"]).resolve()
    recorded = dash.ok("GET", "/api/problems/projects")["projects"]
    found = next((p for p in recorded
                  if Path(p["path"]).resolve() == project_dir), None)
    if found is None:
        raise RuntimeError("ML-Dashboard の一覧に書き出したプロジェクトが"
                           "ありません")
    did = dash.ok("POST", "/api/projects", {"path": found["path"]})["id"]
    journey.add("ML-Dashboard で書き出したプロジェクトを一覧から登録", "OK",
                f"問題の設定「{found['problem']}」が記録したプロジェクト")

    # 環境づくり (Gymnasium は骨組みの requires の rl)
    ready = dash.ok("GET", f"/api/projects/{did}/readiness")
    dash.ok("POST", f"/api/projects/{did}/environment",
            {"commands": ready["commands"]})
    job = wait_status(dash, f"/api/projects/{did}/environment", ("running",))
    ready = dash.ok("GET", f"/api/projects/{did}/readiness?refresh=true")
    # 環境の項目だけで判定 (全てがそろったかの ready は, データセットと
    # モデルの定義の有無も含むので説明に添える)
    environment = (ready["environment_exists"] and ready["ml_found"]
                   and not ready["missing"] and not ready["outdated"])
    spec = ready.get("model_spec") or {}
    journey.add(
        "環境を作る (画面から)",
        "OK" if job["status"] == "succeeded" and environment else "誤り",
        f"{job['status']}, 足りない {ready['missing']}. 全てがそろったか "
        f"{ready['ready']} (データセット {ready['dataset'].get('exists')}, "
        f"モデルの定義 {spec.get('exists')}, Python の付け替え "
        f"{ready['switch_python']})"
    )

    # 守りの数は experiment の config の編集で渡す (画面の config の編集と
    # 同じ. 骨組みの train.py と default の config は書き換えない)
    text = yaml.safe_dump({"guards": GUARDS}, allow_unicode=True,
                          sort_keys=False)
    launched = dash.ok("POST", f"/api/projects/{did}/jobs",
                       {"script": "train.py", "config_text": text})
    job = wait_status(dash, f"/api/jobs/{launched['id']}", ("running",),
                      stop=f"/api/jobs/{launched['id']}/stop")
    run = job.get("run") or {}
    if job.get("return_code") != 0 or not run:
        tail = dash.ok("GET", f"/api/jobs/{launched['id']}/output")["text"]
        lines = tail.strip().splitlines() or [str(job.get("status"))]
        raise RuntimeError(f"学習が止まりました: {lines[-1]}")
    journey.add(
        "学習 (守りを experiment の config で数に)", "OK",
        f"{run['best']['monitor']} {run['best']['value']:.1f}"
    )
    run_id = str(run["id"])

    return run_id


def register(
    provide: Client, journey: Journey, setup: dict[str, Any], run_id: str
) -> str:
    """
    ML-Provide の方策の実験の一覧から, 学習した実験の一式を登録する関数

    Parameters
    ----------
    provide: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        project
    run_id: str
        学習した実験の id

    Returns
    ----------
    policy_id: str
        登録した方策の一式の id
    """
    # 一覧の行を選んで登録 (名前は推論の一式の登録と同じ形)
    project_dir = Path(setup["project"]).resolve()
    rows = provide.ok("GET", "/api/experiments/policies")["experiments"]
    row = next((r for r in rows
                if Path(r["project"]).resolve() == project_dir
                and r["experiment"] == run_id), None)
    if row is None:
        raise RuntimeError("ML-Provide の方策の実験の一覧に学習した実験が"
                           "ありません")
    view = provide.ok("POST", "/api/policies",
                      {"path": row["path"],
                       "name": f"{row['problem']} ({row['experiment']})"})
    policy_id = str(view["id"])

    # 登録の後の一覧の行は登録済みの id を持つ
    rows = provide.ok("GET", "/api/experiments/policies")["experiments"]
    marked = next((r for r in rows if r["path"] == row["path"]), {})
    guards = (view.get("spec") or {}).get("guards") or {}
    journey.add(
        "ML-Provide で方策の実験を一覧から登録",
        "OK" if marked.get("policy_id") == policy_id else "誤り",
        f"問題の設定「{row['problem']}」の実験 {row['experiment']}. 守り "
        f"allowed_actions {guards.get('allowed_actions')}, "
        f"observation_high {guards.get('observation_high')}"
    )

    return policy_id


def act(
    provide: Client, journey: Journey, policy_id: str
) -> None:
    """
    ML-Provide の act で, 守りの範囲の中と外の観測から行動と守りの結果を
    出し, CartPole の 1 エピソードを動かす関数

    Parameters
    ----------
    provide: Client
        ML-Provide の客
    journey: Journey
        結果の記録
    policy_id: str
        方策の一式の id

    Returns
    ----------
    None
    """
    # 範囲の中と外の観測 (nan は JSON の NaN で渡す)
    path = f"/api/policies/{policy_id}/act"
    for label, observation in OBSERVATIONS.items():
        result = provide.ok("POST", path, {"observation": observation})
        journey.add(
            f"観測から行動と守りの結果 ({label})",
            "OK" if result["status"] == EXPECTED[label]
            and result["action"] is not None else "誤り",
            f"action {result['action']}, status {result['status']}, "
            f"reasons {result['reasons']}"
        )

    # 1 歩ごとに観測を渡して行動を受け取り, 環境を進める (環境は手元)
    env = gym.make(ENV_ID)
    observation, _ = env.reset(seed=EPISODE_SEED)
    total, statuses, done = 0.0, {}, False
    while not done:
        result = provide.ok("POST", path,
                            {"observation": observation.tolist()})
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        if result["action"] is None:
            break
        observation, reward, terminated, truncated, _ = env.step(
            int(result["action"])
        )
        total += float(reward)
        done = terminated or truncated
    env.close()
    journey.add(
        "act で CartPole の 1 エピソードを動かす",
        "OK" if statuses.get("ok", 0) > 0 else "誤り",
        f"種 {EPISODE_SEED}, 報酬の和 {total:.0f}, 守りの結果 {statuses}"
    )


def run_policy(
    apps: Apps, journey: Journey, setup: dict[str, Any]
) -> None:
    """
    問題の設定から方策の一式の登録と行動までを通す関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        problem と project

    Returns
    ----------
    None
    """
    clients = apps.clients
    set_problem(clients["pathfinder"], journey, setup)
    run_id = train(clients["dashboard"], journey, setup)
    policy_id = register(clients["provide"], journey, setup, run_id)
    act(clients["provide"], journey, policy_id)


if __name__ == "__main__":
    failed = main_with(build, "方策の一式を一覧から登録 (CartPole)",
                       sys.argv[1:], runner=run_policy, prefix="policy_",
                       names=("pathfinder", "dashboard", "provide"))
    sys.exit(1 if failed else 0)
