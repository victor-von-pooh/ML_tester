"""
学習した方策を ε-greedy にして, 行動を選んだ確率 (action_prob) 付きの軌跡を
書き出し, ML-Fathom の画面の API で分析し, 別の ε の方策の良さを環境で
動かさずに見積もって, 環境で動かした値と比べる
"""
import argparse
import math
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
from ml_found import save_evaluation
from ml_found.rl import (
    Trajectory, estimate_policy_value, evaluate_policy, load_policy,
    read_policy_spec, save_trajectory
)

from model import PolicyNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Apps, Journey  # noqa: E402

# このタスクのディレクトリと, 実験の出力先
TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"

# 記録する方策の ε と, 見積もる方策の ε (記録と同じ ε は重みが全て 1 に
# なる確かめ. 0 は最もスコアの高い行動だけを選ぶ方策で, 記録した方策が
# ε で替えた歩を含むエピソードの重みが 0 になり, 見積もりが決まらない)
BEHAVIOR_EPSILON = 0.2
TARGET_EPSILONS = (0.2, 0.19, 0.15, 0.1, 0.0)

# 記録するエピソード数と最初の乱数の種, 見積もる方策を環境で動かす
# エピソード数と最初の乱数の種 (学習の検証と守り付きの評価と重ならない値)
EPISODES = 100
BEHAVIOR_SEED = 40000
TARGET_SEED = 50000

# ε で行動を選ぶ乱数の種
CHOICE_SEED = 0

# 見積もりの割引率 (エピソードの報酬の和)
GAMMA = 1.0

# 軌跡の名前と, 見積もりの表の名前
TRAJECTORY_NAME = "epsilon_greedy"
EVALUATION_NAME = "off_policy"


class EpsilonGreedy:
    def __init__(self, policy: Any, epsilon: float, seed: int):
        # 守り付きの方策, ε, 選んでよい行動, 行動を選ぶ乱数
        self.policy = policy
        self.epsilon = epsilon
        guards = policy.spec["guards"]
        allowed = guards.get("allowed_actions")
        if allowed is None:
            allowed = list(range(int(policy.spec["action_space"]["n"])))
        self.allowed = [int(action) for action in allowed]
        self.rng = np.random.default_rng(seed)

    def prob(self, greedy: int, action: int) -> float:
        """
        最もスコアの高い行動が greedy のときに action を選ぶ確率を返す関数

        Parameters
        ----------
        greedy: int
            守りを当てた方策の行動
        action: int
            選んだ行動

        Returns
        ----------
        prob: float
            選ぶ確率 (選んでよい行動から ε で一様に選ぶ分を含む)
        """
        prob = self.epsilon / len(self.allowed)
        if action == greedy:
            prob += 1.0 - self.epsilon

        return prob

    def act(self, observation: Any) -> dict:
        """
        守りを当てた方策の行動を, ε の確率で選んでよい行動からの一様な
        選択に替え, 選んだ確率を添えて返す関数

        Parameters
        ----------
        observation: Any
            観測

        Returns
        ----------
        result: dict
            act の結果 (action, status, reasons) と action_prob. 守りが
            働いた判断 (ok でない) はそのまま (行動があれば確率 1)
        """
        # 守りが働いた判断は替えない
        result = self.policy.act(observation)
        if result["action"] is None:
            return result
        if result["status"] != "ok":
            result = {**result, "action_prob": 1.0}
            return result
        greedy = int(result["action"])
        action = greedy
        if self.rng.random() < self.epsilon:
            action = int(self.rng.choice(self.allowed))
        result = {**result, "action": action,
                  "action_prob": self.prob(greedy, action)}

        return result


def latest_run(run: Optional[Path]) -> Path:
    """
    方策の一式を持つ実験の場所を返す関数

    Parameters
    ----------
    run: Optional[Path]
        渡された実験の場所 (None なら outputs/ の中で最も新しい実験)

    Returns
    ----------
    place: Path
        実験のディレクトリ
    """
    # 名前の新しい順 (日付と時刻のディレクトリ)
    if run is not None:
        place = run.resolve()
    else:
        runs = sorted(OUTPUT_DIR.glob("*/*/*/policy/policy.json"),
                      key=lambda path: path.parts[-5:-2])
        if not runs:
            sys.exit("方策の一式がありません. 先に train.py を動かしてください")
        place = runs[-1].parents[1]

    return place


def target_prob_of(policy: Any, epsilon: float) -> Any:
    """
    見積もる ε-greedy の方策の, 観測で行動を選ぶ確率の関数を返す関数

    Parameters
    ----------
    policy: Any
        守り付きの方策 (最もスコアの高い行動を返す)
    epsilon: float
        見積もる方策の ε

    Returns
    ----------
    target_prob: Callable
        (観測, 行動) から確率
    """
    chooser = EpsilonGreedy(policy, epsilon, CHOICE_SEED)

    def target_prob(observation: Any, action: Any) -> float:
        # 守りが働く観測は, 守りが返す行動を確率 1 で選ぶ
        result = policy.act(observation)
        if result["action"] is None:
            prob = 0.0
        elif result["status"] != "ok":
            prob = float(int(result["action"]) == int(action))
        else:
            prob = chooser.prob(int(result["action"]), int(action))

        return prob

    return target_prob


def record_and_estimate(place: Path) -> dict[str, Any]:
    """
    ε-greedy の方策で軌跡を記録して書き出し, 見積もりと環境で動かした値を
    実験の evaluations/ に書く関数

    Parameters
    ----------
    place: Path
        実験のディレクトリ

    Returns
    ----------
    setup: dict[str, Any]
        trajectory (書いた軌跡の表), rows (見積もりの行), behavior
        (記録したエピソードの行)
    """
    # 一式の metadata のモデルの引数で読み直し
    spec = read_policy_spec(place / "policy")
    meta = spec["metadata"]
    model_args = meta["model"]["args"]
    env_id = meta["env_id"]
    policy = load_policy(place / "policy", PolicyNet(**model_args))

    # 記録する方策で動かし, 1 歩ごとの行動の確率を軌跡に
    behavior = EpsilonGreedy(policy, BEHAVIOR_EPSILON, CHOICE_SEED)
    trajectory = Trajectory()
    episodes = evaluate_policy(behavior, env_id, EPISODES,
                               seed=BEHAVIOR_SEED, trajectory=trajectory)
    path = save_trajectory(
        place, TRAJECTORY_NAME, trajectory, epoch=meta.get("best_epoch"),
        description=f"学習した方策を ε {BEHAVIOR_EPSILON} の ε-greedy に"
        "して, 守り付きで動かした軌跡 (action_prob は選んだ行動の確率)"
    )
    behavior_mean = float(np.mean([row["reward"] for row in episodes]))
    print(f"軌跡: {path} ({EPISODES} エピソード, 報酬の和の平均 "
          f"{behavior_mean:.1f})")

    # 見積もる方策ごとに, 書いた表からの見積もりと環境で動かした値
    rows = []
    for epsilon in TARGET_EPSILONS:
        estimate = estimate_policy_value(
            path, target_prob_of(policy, epsilon), gamma=GAMMA
        )
        actual = evaluate_policy(
            EpsilonGreedy(policy, epsilon, CHOICE_SEED + 1), env_id,
            EPISODES, seed=TARGET_SEED
        )
        row = {
            "target_epsilon": epsilon,
            "value": estimate["value"],
            "ordinary": estimate["ordinary"],
            "per_decision": estimate["per_decision"],
            "behavior_value": estimate["behavior_value"],
            "effective_sample_size": estimate["effective_sample_size"],
            "max_weight_share": estimate["max_weight_share"],
            "low_ess": estimate["low_ess"],
            "zero_weight_episodes": estimate["zero_weight_episodes"],
            "actual": float(np.mean([r["reward"] for r in actual]))
        }
        rows.append(row)
        print(" ".join(f"{key} {value}" for key, value in row.items()))
        for note in estimate["notes"]:
            print(f"  {note}")
    save_evaluation(
        place, EVALUATION_NAME, rows, "simulator",
        description=f"ε {BEHAVIOR_EPSILON} で記録した軌跡 "
        f"({TRAJECTORY_NAME}) からの ε ごとの方策の価値の見積もり (重点"
        "サンプリング) と, 同じ方策を環境で動かした報酬の和の平均 (actual)",
        epoch=meta.get("best_epoch")
    )
    setup = {"trajectory": path, "rows": rows, "behavior": episodes}

    return setup


def analyze(apps: Apps, journey: Journey, setup: dict[str, Any]) -> None:
    """
    書いた軌跡を ML-Fathom に登録して要約を読み, 記録した値と比べる関数

    Parameters
    ----------
    apps: Apps
        起動したアプリ
    journey: Journey
        結果の記録
    setup: dict[str, Any]
        record_and_estimate の結果

    Returns
    ----------
    None
    """
    fathom = apps.clients["fathom"]
    path = str(setup["trajectory"])
    episodes = setup["behavior"]

    # 画面の「データを登録」と同じ (候補の先頭が記録した軌跡)
    candidates = fathom.ok("POST", "/api/detect", {"path": path})
    first = candidates[0]["loader"] if candidates else None
    source = fathom.ok("POST", "/api/sources",
                       {"path": path, "loader": first})
    journey.add("ML-Fathom で軌跡を登録",
                "OK" if first == "trajectory" else "誤り",
                f"読み込みの候補の先頭 {first}")
    summary = fathom.ok("GET", f"/api/sources/{source['id']}/trajectory")
    steps = sum(row["length"] for row in episodes)
    stops = sum(row["stop"] for row in episodes)
    same = (summary["episodes"] == len(episodes)
            and summary["steps"] == steps
            and math.isclose(summary["return"]["mean"],
                             np.mean([row["reward"] for row in episodes])))
    journey.add(
        "ML-Fathom の要約 (エピソード, 歩数, 報酬の和)",
        "OK" if same else "誤り",
        f"エピソード {summary['episodes']}, 歩数 {summary['steps']}, 報酬の和"
        f"の平均 {summary['return']['mean']:.1f}, 終わり方 {summary['ends']}"
        f" (記録: 歩数 {steps}, 止めた判断 {stops})"
    )
    prob = summary["action_prob"] or {}
    journey.add(
        "ML-Fathom の行動を選んだ確率の要約",
        "OK" if summary["with_action_prob"] and prob.get("out_of_range") == 0
        else "誤り",
        f"欠け {prob.get('missing')}, 範囲の外 {prob.get('out_of_range')}, "
        f"最小 {prob.get('min')}, 最大 {prob.get('max')}, 平均 "
        f"{prob.get('mean'):.4f}. 守りの結果 "
        f"{(summary['guard'] or {}).get('results')}, 働いた理由の種類 "
        f"{len((summary['guard'] or {}).get('reasons') or [])}"
    )

    # 見積もり (記録と同じ ε は記録した方策の平均と同じになり, ε 0 は
    # 全ての重みが 0 で決まらない)
    for row in setup["rows"]:
        if row["target_epsilon"] == BEHAVIOR_EPSILON:
            good = row["value"] is not None and math.isclose(
                row["value"], row["behavior_value"]
            )
        elif row["target_epsilon"] == 0.0:
            good = row["value"] is None and row["zero_weight_episodes"] == len(
                episodes
            )
        else:
            good = row["value"] is not None
        status = "OK" if good else "誤り"
        journey.add(
            f"見積もり (ε {row['target_epsilon']})", status,
            f"見積もり {rounded(row['value'])}, 歩ごとの見積もり "
            f"{rounded(row['per_decision'])}, 環境で動かした値 "
            f"{row['actual']:.1f}, 有効なサンプルの数 "
            f"{rounded(row['effective_sample_size'])}, "
            f"当てにならない {row['low_ess']}"
        )


def rounded(value: Optional[float]) -> Optional[float]:
    """
    表に出す数を小数 1 桁に丸める関数

    Parameters
    ----------
    value: Optional[float]
        数 (決まらなければ None)

    Returns
    ----------
    result: Optional[float]
        丸めた数 (None はそのまま)
    """
    result = None if value is None else round(float(value), 1)

    return result


def main() -> int:
    """
    軌跡の記録と見積もり, ML-Fathom での分析を通す関数

    Parameters
    ----------
    None

    Returns
    ----------
    failed: int
        誤りの段階の数
    """
    # 実験の場所, アプリの親のディレクトリ, 作業用の場所
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None,
                        help="実験の場所 (既定は outputs/ の最も新しい実験)")
    parser.add_argument(
        "--apps", type=Path, default=TASK_DIR.parents[2],
        help="ML-Fathom を clone して make setup したディレクトリの親"
    )
    parser.add_argument("--work", type=Path, default=None,
                        help="ML-Fathom の置き場 (既定は実験の中の fathom)")
    args = parser.parse_args()
    place = latest_run(args.run)
    setup = record_and_estimate(place)

    # ML-Fathom を起動して分析 (止まっても表に残す)
    work = (args.work or place / "fathom").resolve()
    apps = Apps(args.apps.resolve(), work, ("fathom",))
    journey = Journey("記録した経験からの方策の見積もり (CartPole)")
    try:
        apps.start()
        analyze(apps, journey, setup)
    except Exception as error:
        journey.add("止まった段階", "誤り", f"{type(error).__name__}: {error}")
    finally:
        apps.stop()
    failed = journey.report(place / "evaluations" / "off_policy_journey.json")

    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
