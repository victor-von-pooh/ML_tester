from pathlib import Path

import numpy as np
import pandas as pd

# 書き出す先 (リポジトリの data, git の管理外)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "store_sales.csv"

# 店の名前 (表記のゆれを入れる店を含む)
STORES = [
    "渋谷店", "新宿店", "池袋店", "上野店", "品川店", "横浜店", "大宮店",
    "千葉店", "立川店", "町田店", "川崎店", "船橋店",
]


def main(days: int = 730, seed: int = 0) -> None:
    """
    店ごとの日々の売上の表を書く関数

    売上は店ごとの水準, 曜日, 季節, 気温, 販促, 前の日からの続き (自己相関)
    で決まる. 学習の前に直すことを含める: 円とカンマの付いた売上, 3 つの
    書き方の混ざった日付, 店の名前の表記のゆれ, 休みの日 (行が無い),
    同じ行の重複

    Parameters
    ----------
    days: int = 730
        日数 (2 年)
    seed: int = 0
        乱数のシード

    Returns
    ----------
    None
    """
    # 店ごとの水準と, 共通の気温
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    season = np.sin(2 * np.pi * (dates.day_of_year - 30) / 365)
    temperature = 16 + 10 * season + rng.normal(0, 3, days)
    rows = []
    for number, store in enumerate(STORES):
        level = rng.uniform(30, 120) * 10000
        weekend = rng.uniform(1.1, 1.6)
        heat = rng.uniform(-0.02, 0.02)
        noise = 0.0
        for day in range(days):
            # 前の日の揺れの続き (店ごと)
            noise = 0.7 * noise + rng.normal(0, 0.08)
            promo = rng.random() < 0.08
            dow = dates[day].day_of_week
            value = level * (weekend if dow >= 5 else 1.0) * (
                1 + 0.15 * season[day] + heat * (temperature[day] - 16)
                + 0.3 * promo + noise
            )
            if rng.random() < 0.03:
                continue
            rows.append({
                "date": dates[day], "store": store,
                "temperature": round(float(temperature[day]), 1),
                "promo": int(promo), "sales": max(0, int(value)),
            })
    frame = pd.DataFrame(rows)

    # 日付の書き方の混在 (2024/01/03, 2024-01-03, 2024年1月3日)
    style = rng.integers(0, 3, len(frame))
    frame["date"] = [
        d.strftime("%Y/%m/%d") if s == 0 else d.strftime("%Y-%m-%d")
        if s == 1 else f"{d.year}年{d.month}月{d.day}日"
        for d, s in zip(frame["date"], style)
    ]

    # 売上は円とカンマ付きの文字, 店の名前の表記のゆれ, 重複
    frame["sales"] = [f"{v:,}円" for v in frame["sales"]]
    shibuya = frame["store"] == "渋谷店"
    frame.loc[shibuya & (rng.random(len(frame)) < 0.3), "store"] = "渋谷店 "
    frame = pd.concat([frame, frame.sample(150, random_state=seed)])
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False, encoding="utf-8")
    print(f"保存: {DATA_PATH} ({len(frame)} 行, {len(STORES)} 店)")


if __name__ == "__main__":
    main()
