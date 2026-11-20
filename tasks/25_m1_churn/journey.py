"""
表の分類 (解約) を, 問題の設定から推論まで 5 つの画面のアプリの API だけで
通し, 段階ごとに画面だけでできたか (OK), 手で写すか打つか (手), 誤りかを
表にする (表の予測を画面だけで通す節目の確かめ)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from apps import main_with

# 会員の数と, 今のやり方で連絡した人数
MEMBERS = 3000
CONTACTED = 600


def make_churn(path: Path, seed: int = 0) -> None:
    """
    合成した会員の表 (今のやり方で連絡したかの列 contacted を持つ) を
    書く関数

    Parameters
    ----------
    path: Path
        書く CSV
    seed: int
        乱数の種

    Returns
    ----------
    None
    """
    # 問い合わせが多く, 入って間もなく, 安い料金の会員ほど高い解約の確率
    rng = np.random.default_rng(seed)
    plan = rng.choice(
        ["basic", "standard", "premium"], MEMBERS, p=[0.5, 0.3, 0.2]
    )
    months = rng.integers(1, 60, MEMBERS)
    inquiries = rng.poisson(1.5, MEMBERS)
    fee = np.select([plan == "basic", plan == "standard"], [1500, 2500],
                    4000)
    logit = -2.0 + 0.6 * inquiries - 0.04 * months + (plan == "basic") * 0.8
    churned = np.where(
        rng.random(MEMBERS) < 1 / (1 + np.exp(-logit)), "yes", "no"
    )
    # 今のやり方: 問い合わせの多い順に連絡
    order = np.argsort(-inquiries + rng.random(MEMBERS) * 0.1)
    contacted = np.full(MEMBERS, "no", dtype=object)
    contacted[order[:CONTACTED]] = "yes"
    pd.DataFrame({
        "member_id": [f"M{i:05d}" for i in range(MEMBERS)], "plan": plan,
        "months": months, "inquiries": inquiries, "monthly_fee": fee,
        "contacted": contacted, "churned": churned
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
        apps.run_journey に渡す設定
    """
    # 元のデータと, 推論とたどり直しに使う最後の 800 人
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "churn.csv"
    make_churn(data)
    past = work / "data" / "churn_past.csv"
    pd.read_csv(data).tail(800).to_csv(past, index=False)
    setup = {
        "example": "解約の引き止め",
        "data": data,
        "past": past,
        # 連絡したかの列は予測に使わず, 今のやり方の列
        "answers": {
            "information.forbidden": ["member_id", "contacted"],
            "action.current": {"type": "column", "column": "contacted"}
        },
        "recipe_update": None,
        "recipe_steps": [
            {"op": "one_hot", "columns": ["plan"]},
            {"op": "standardize",
             "columns": ["months", "inquiries", "monthly_fee"]}
        ]
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "表の分類 (解約の引き止め)", sys.argv[1:])
    sys.exit(1 if failed else 0)
