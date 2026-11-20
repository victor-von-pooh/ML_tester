"""
時系列の予測 (総菜の売れる数) を, 問題の設定から推論まで 5 つの画面の
アプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で写すか
打つか (手), 誤りかを表にする (表の予測を画面だけで通す節目の確かめ).
今のやり方は 7 期前と同じ量 (k 期前の決まり)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 部品は 25 のディレクトリの apps.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import main_with

# 店と品と日の数
STORES = ["s1", "s2"]
ITEMS = ["karaage", "salad", "korokke"]
DAYS = 365


def make_sozai(path: Path, seed: int = 0) -> None:
    """
    合成した店と品ごとの日々の売れた数の表を書く関数

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
    # 品の水準, 週末, 特売, 気温 (サラダは暑いほど, 他は寒いほど売れる)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=DAYS, freq="D")
    season = 15 + 10 * np.sin((dates.day_of_year - 100) / 365 * 2 * np.pi)
    frames = []
    for store in STORES:
        for item in ITEMS:
            base = rng.uniform(30, 80)
            temp = season + rng.normal(0, 2, DAYS)
            promo = (rng.random(DAYS) < 0.1).astype(int)
            weekend = np.where(dates.day_of_week >= 5, 1.3, 1.0)
            sign = 1 if item == "salad" else -1
            mean = (base * weekend * (1 + 0.4 * promo)
                    * (1 + 0.01 * (temp - 15) * sign))
            sales = np.maximum(0, rng.normal(mean, mean * 0.1)).astype(int)
            frames.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"), "store": store,
                "item": item, "sales": sales,
                "temperature_forecast": temp.round(1), "promo": promo,
                "customers": (sales * 3 + rng.normal(0, 10, DAYS)).astype(
                    int
                )
            }))
    pd.concat(frames).to_csv(path, index=False)


def add_lags(recipe: dict) -> None:
    """
    ML-Fathom の雛形の過去の値の手順に, 7 日前と 7 日の移動平均を足す関数

    決める時点で分かるのは 2 日前までなので, 移動平均は 2 つ前から始める

    Parameters
    ----------
    recipe: dict
        ML-Fathom のレシピ (書き換える)

    Returns
    ----------
    None
    """
    # 雛形に過去の値の手順が無ければ, 問題の設定の誤りとして停止
    lags = [step for step in recipe["steps"] if step["op"] == "lag"]
    if not lags:
        raise RuntimeError("ML-Fathom の雛形に過去の値の手順がありません")
    for step in lags:
        step.update(lags=[2, 7], windows=[7], gap=2)


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
    # 元のデータと, 推論とたどり直しに使う最後の 90 日
    (work / "data").mkdir(exist_ok=True)
    data = work / "data" / "sozai.csv"
    make_sozai(data)
    frame = pd.read_csv(data)
    past = work / "data" / "sozai_past.csv"
    frame[frame["date"] >= "2025-10-03"].to_csv(past, index=False)
    setup = {
        "example": "総菜の作る量",
        "data": data,
        "past": past,
        # 今のやり方: 前の週の同じ曜日と同じ数 (7 期前)
        "answers": {"action.current": {"type": "lag", "periods": 7}},
        "recipe_update": add_lags,
        "recipe_steps": [
            {"op": "drop_missing"},
            {"op": "datetime_parts", "columns": ["date"],
             "parts": ["weekday", "month"]},
            {"op": "one_hot", "columns": ["store", "item"]},
            {"op": "standardize", "columns": "auto"}
        ]
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "時系列の予測 (総菜の作る量)", sys.argv[1:])
    sys.exit(1 if failed else 0)
