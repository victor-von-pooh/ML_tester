from pathlib import Path
import sys

import ml_quest

# このタスクのディレクトリと, 書き出す ML-Found のプロジェクト
TASK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TASK_DIR / "project"


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


def main() -> None:
    """
    見本の問題の設定 (総菜の作る量) を確かめ, problem.yaml と ML-Found の
    プロジェクトの骨組みを書き出す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 問いの順に答えた見本と, 確かめ (直すところが無いこと)
    problem = ml_quest.example_problem("総菜の作る量")
    findings = ml_quest.check(problem)
    for item in findings:
        print(f"[{item['level']}] {item['id']}: {item['message']}")
    if not ml_quest.is_ready(problem):
        print("問題の設定に直すところがあります")
        sys.exit(1)
    for sentence in ml_quest.describe(problem):
        print(sentence)

    # 問題の設定と骨組み (前のものは置き換え). 答えが同じなら前の
    # problem.yaml (時刻を含む) をそのまま使い, 実行のたびの差分を防止
    path = TASK_DIR / "problem.yaml"
    if path.is_file() and same_answers(ml_quest.load_problem(path), problem):
        problem = ml_quest.load_problem(path)
    else:
        problem = ml_quest.save_problem(problem, path)
    written = ml_quest.make_project(problem, PROJECT_DIR, overwrite=True)
    print(f"骨組み: {', '.join(written)}")


if __name__ == "__main__":
    main()
