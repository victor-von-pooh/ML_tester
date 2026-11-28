from pathlib import Path
import sys

import ml_quest

# このタスクのディレクトリ
TASK_DIR = Path(__file__).resolve().parent

# 受け持ちの場所
PART = "ml.parts.0"

# 画像の生成 (11 と同じ手書きの数字) の答え. 問いの順
GENERATE = [
    ("issue.owner", "字を読む仕組みの試験の担当"),
    ("issue.pain", "試験に使う手書きの数字の画像が足りず, 集めるのに手間が"
                   "かかる"),
    ("issue.current", "人に頼んで手で書いてもらう"),
    ("issue.kpi.name", "試験の画像を集める手間"),
    ("issue.kpi.unit", "時間 / 月"),
    ("issue.kpi.better", "lower"),
    ("issue.kpi.current", 40),
    ("issue.kpi.target", 10),
    ("decomposition", [
        {"cause": "手で書く人が少ない",
         "lever": "本物に似た数字の画像を作って足す", "ml": True},
        {"cause": "頼むたびに書き方の説明がいる",
         "lever": "書き方の用紙を決めておく", "ml": False},
    ]),
    ("action.what", "試験に足す数字の画像"),
    ("action.who", "試験の担当"),
    ("action.when", "毎月の初め"),
    ("ml.use", True),
    ("ml.reason", "手書きの字の揺れは規則で書けない"),
    (f"{PART}.role", "generate"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "image"),
    (f"{PART}.target.source", "self"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.target.kind", "image"),
    (f"{PART}.what", "手書きの数字の画像"),
    (f"{PART}.item", "1 枚の画像"),
    (f"{PART}.baseline", "人が手で書く"),
    (f"{PART}.evaluation.metric", "loss"),
    (f"{PART}.evaluation.link", "作った画像を人が見て, 数字に見える割合で"
                                "確かめる"),
    ("verification.method", "trial"),
    ("verification.trial", "1 か月の試験で作った画像を半分混ぜる"),
    ("verification.success", "集める手間が月 10 時間以下"),
    ("operation.review", "月に 1 回"),
    ("operation.approver", "試験の責任者"),
    ("operation.stop", "数字に見えない画像が 1 割を超えたら止める"),
    ("operation.remeasure", "月に 1 回, 作った画像を人が見て確かめる"),
]

# 方策 (24 と同じ CartPole) の答え. 問いの順
POLICY = [
    ("issue.owner", "台車の設備の担当"),
    ("issue.pain", "台車の上の棒を立て続けるのに, 人が付きっきりで押す"),
    ("issue.current", "人が棒の傾きを見て左右に押す"),
    ("issue.kpi.name", "人が押す時間"),
    ("issue.kpi.unit", "時間 / 日"),
    ("issue.kpi.better", "lower"),
    ("issue.kpi.current", 8),
    ("issue.kpi.target", 1),
    ("decomposition", [
        {"cause": "押す向きを人が決める",
         "lever": "押す向きを方策が決める", "ml": True},
        {"cause": "押す人が 1 人で交代できない",
         "lever": "交代の順を決めておく", "ml": False},
    ]),
    ("action.what", "台車を押す向き"),
    ("action.who", "方策"),
    ("action.when", "各歩の観測を見た時点"),
    ("ml.use", True),
    ("ml.reason", "棒の動きに合わせて押す向きを続けて選ぶ"),
    (f"{PART}.role", "effect"),
    (f"{PART}.unit", "row"),
    (f"{PART}.output", "effect"),
    (f"{PART}.target.source", "feedback"),
    (f"{PART}.method", "scratch"),
    (f"{PART}.what", "押す向きを選ぶ方策"),
    (f"{PART}.item", "1 歩"),
    (f"{PART}.autonomous", True),
    (f"{PART}.collect", "simulator"),
    (f"{PART}.simulator", "other"),
    (f"{PART}.env_id", "CartPole-v1"),
    (f"{PART}.baseline", "人が押す"),
    (f"{PART}.evaluation.metric", "reward"),
    (f"{PART}.evaluation.link", "報酬 (棒が立っていた歩数) が大きいほど人が"
                                "押す時間が減る"),
    (f"{PART}.policy.action_limits", "左右に押すだけ"),
    (f"{PART}.policy.stop", "台車が中心から 2.4 を出たら"),
    (f"{PART}.policy.fallback", "action"),
    (f"{PART}.policy.fallback_action", "0"),
    (f"{PART}.policy.approver", "設備の責任者"),
    ("verification.method", "trial"),
    ("verification.trial", "シミュレータで守りを当てて確かめる"),
    ("verification.success", "守り付きの評価で報酬の平均が 195 以上"),
    ("operation.review", "週に 1 回"),
    ("operation.approver", "設備の責任者"),
    ("operation.stop", "止める条件に当たった歩が 1 日に 1 回でも出たら止める"),
    ("operation.remeasure", "週に 1 回, シミュレータで報酬を数え直す"),
]


def same_answers(first: dict, second: dict) -> bool:
    """
    2 つの問題の設定の答えが, 作った時刻と更新の時刻を除いて同じかを返す
    関数

    Parameters
    ----------
    first: dict
        問題の設定
    second: dict
        問題の設定

    Returns
    ----------
    same: bool
        時刻以外が同じなら True
    """
    # 時刻の項目を除いた比べ
    stamps = ("created_at", "updated_at")
    same = {k: v for k, v in first.items() if k not in stamps} == {
        k: v for k, v in second.items() if k not in stamps
    }

    return same


def build(name: str, answers: list) -> dict:
    """
    空の問題の設定に, 問いの順に答えを入れる関数

    Parameters
    ----------
    name: str
        問題の設定の名前
    answers: list
        問いの id と答えの組の並び

    Returns
    ----------
    problem: dict
        答えを入れた問題の設定
    """
    # 問いの順の 1 つずつの答え (要らなくなった問いは ML-Quest が外す)
    problem = ml_quest.new_problem(name)
    for key, value in answers:
        problem = ml_quest.answer(problem, key, value)

    return problem


def write(problem: dict, folder: Path) -> None:
    """
    問題の設定を確かめ, problem.yaml と ML-Found のプロジェクトの骨組みを
    書き出す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    folder: Path
        problem.yaml と project/ を置くディレクトリ

    Returns
    ----------
    None
    """
    # 確かめ (直すところが無いこと) と筋道の文
    print(f"== {problem['name']}")
    for item in ml_quest.check(problem):
        print(f"[{item['level']}] {item['id']}: {item['message']}")
    if not ml_quest.is_ready(problem):
        print("問題の設定に直すところがあります")
        sys.exit(1)
    for sentence in ml_quest.describe(problem):
        print(sentence)

    # 問題の設定と骨組み (前のものは置き換え). 答えが同じなら前の
    # problem.yaml (時刻を含む) をそのまま使い, 実行のたびの差分を防止
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "problem.yaml"
    if path.is_file() and same_answers(ml_quest.load_problem(path), problem):
        problem = ml_quest.load_problem(path)
    else:
        problem = ml_quest.save_problem(problem, path)
    written = ml_quest.make_project(problem, folder / "project",
                                    overwrite=True)
    print(f"骨組み: {', '.join(written)}")


def main() -> None:
    """
    画像の生成と方策の 2 つの問題の設定を確かめ, それぞれの problem.yaml と
    骨組みを書き出す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像の生成 (generate/) と方策 (policy/)
    write(build("手書きの数字の画像を作る", GENERATE), TASK_DIR / "generate")
    write(build("台車の棒を立て続ける", POLICY), TASK_DIR / "policy")


if __name__ == "__main__":
    main()
