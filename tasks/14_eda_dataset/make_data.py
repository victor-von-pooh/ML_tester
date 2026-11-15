from pathlib import Path

import numpy as np
import pandas as pd

# 書き出す先 (リポジトリの data, git の管理外)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def make_churn(rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    """
    解約の予測に使う, 前処理が必要な顧客の表を作る関数

    欠損, 文字のカテゴリ, カンマ付きの数, 偏った分布, ID, 答えから作った
    列 (リーク) を含める

    Parameters
    ----------
    rows: int = 3000
        行の数
    seed: int = 0
        乱数のシード

    Returns
    ----------
    frame: pd.DataFrame
        顧客の表 (目的の列は churn)
    """
    # 顧客の属性と利用の様子
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, rows)
    monthly = rng.normal(70, 25, rows).clip(15, 150).round(2)
    plan = rng.choice(["basic", "standard", "premium"], rows,
                      p=[0.5, 0.35, 0.15])
    contract = rng.choice(["monthly", "yearly", "two_year"], rows,
                          p=[0.55, 0.3, 0.15])
    support_calls = rng.poisson(1.5, rows)
    data_gb = np.exp(rng.normal(2.5, 1.0, rows)).round(1)

    # 解約の確率 (短い契約, 多い問い合わせ, 短い利用期間で上がる)
    logit = (
        -1.0 + 1.2 * (contract == "monthly") - 0.8 * (contract == "two_year")
        + 0.35 * support_calls - 0.03 * tenure + 0.01 * (monthly - 70)
    )
    churn = rng.random(rows) < 1 / (1 + np.exp(-logit))

    # 表にし, 欠損とカンマ付きの数とリークを入れる
    frame = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(rows)],
        "tenure_months": tenure,
        "monthly_fee": monthly,
        "total_paid": [f"{v:,.0f}" for v in tenure * monthly],
        "plan": plan,
        "contract": contract,
        "support_calls": support_calls,
        "data_gb": data_gb,
        "churn": np.where(churn, "yes", "no"),
    })
    frame.loc[rng.random(rows) < 0.15, "monthly_fee"] = np.nan
    frame.loc[rng.random(rows) < 0.05, "plan"] = None
    frame["cancel_reason_code"] = np.where(
        churn, rng.integers(1, 5, rows), 0
    )

    return frame


def main() -> None:
    """
    顧客の表を data/churn.csv に保存する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 保存先を作って書き出す
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "churn.csv"
    make_churn().to_csv(path, index=False)
    print(f"保存先: {path}")


if __name__ == "__main__":
    main()
