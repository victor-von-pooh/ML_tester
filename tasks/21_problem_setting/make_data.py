from pathlib import Path

import numpy as np
import pandas as pd

# 書き出す先 (リポジトリの data, git の管理外)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sozai.csv"

# 店と品 (品ごとに売れる数の水準と, 暑さへの反応が違う)
STORES = ["east", "west", "north"]
ITEMS = {"karaage": (120, 0.6), "salad": (60, 1.8), "croquette": (90, 0.2),
         "nimono": (45, -1.2)}

# 曜日 (月曜から) ごとの売れ方の倍率
WEEKDAY = [1.0, 0.9, 0.9, 1.0, 1.15, 1.35, 1.3]


def main(days: int = 540, seed: int = 0) -> None:
    """
    店と品ごとの総菜の日々の売れた数の表を書く関数

    売れた数は品の水準, 店の大きさ, 曜日, 天気予報の気温, 特売, 前の日からの
    続き (自己相関) で決まる. 当日の客数は売れた数と一緒に決まる (予測の
    時点では分からない列)

    Parameters
    ----------
    days: int = 540
        日数
    seed: int = 0
        乱数のシード

    Returns
    ----------
    None
    """
    # 共通の気温 (予報) と曜日
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    season = np.sin(2 * np.pi * (dates.day_of_year - 110) / 365)
    forecast = 17 + 9 * season + rng.normal(0, 2.5, days)
    weekday = np.array(WEEKDAY)[dates.day_of_week]

    # 店と品ごとの系列 (前の日の外れの 4 割が残る)
    rows = []
    for store, size in zip(STORES, (1.2, 1.0, 0.8)):
        for item, (level, heat) in ITEMS.items():
            carry = 0.0
            for day in range(days):
                promo = rng.random() < 0.07
                mean = level * size * weekday[day] * (
                    1 + heat * (forecast[day] - 17) / 100
                ) * (1.35 if promo else 1.0)
                carry = 0.4 * carry + rng.normal(0, 0.12)
                sold = max(0, int(round(mean * np.exp(carry))))
                rows.append({
                    "date": dates[day].strftime("%Y-%m-%d"), "store": store,
                    "item": item, "sales": sold,
                    "temperature_forecast": round(float(forecast[day]), 1),
                    "promo": int(promo),
                    "customers": int(sold * 3.1 + rng.normal(0, 15)),
                })

    # 書き出し
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA_PATH, index=False)
    print(f"{DATA_PATH.name}: {len(rows)} 行")


if __name__ == "__main__":
    main()
