from pathlib import Path
import sys

import ml_quest
import pandas as pd
from ml_eda import load_dataset
from ml_inference import backtest

# このタスクのディレクトリ, 元のデータ, ML-Found のプロジェクト
TASK_DIR = Path(__file__).resolve().parent
SOURCE = TASK_DIR.parents[1] / "data" / "sozai.csv"
PROJECT_DIR = TASK_DIR / "project"

# 過去の値の列のために, テスト用の期間の前に渡す日数
HISTORY_DAYS = 30

# 骨組みの実験の名前 (目的の列とタスクから)
EXPERIMENT = "sales_forecasting"


def latest_bundle() -> Path:
    """
    最も新しい実験の中の推論の一式を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    bundle: Path
        <実験>/inference
    """
    # 骨組みの実験の名前の, 学習を終えた (最終の重みのある) 最も新しい実験
    found = sorted(
        run / "inference" for run in (
            PROJECT_DIR / "outputs" / EXPERIMENT
        ).glob("*/*") if (run / "final_model.pt").is_file()
        and (run / "inference" / "inference.yaml").is_file()
    )
    if not found:
        print("推論の一式がありません. 先に project/train.py を実行して"
              "ください")
        sys.exit(1)
    bundle = found[-1]

    return bundle


def with_last_week(frame: pd.DataFrame) -> pd.DataFrame:
    """
    今のやり方 (前の週の同じ曜日と同じ数を作る) の列を足す関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ

    Returns
    ----------
    result: pd.DataFrame
        last_week (店と品ごとの 7 日前の売れた数. 無ければ空) を足した表
    """
    # 7 日後の日付に付け替えて, 同じ店と品と日付でつなぐ
    week = frame[["date", "store", "item", "sales"]].copy()
    week["date"] = (pd.to_datetime(week["date"]) + pd.Timedelta(days=7)
                    ).dt.strftime("%Y-%m-%d")
    week = week.rename(columns={"sales": "last_week"})
    result = frame.merge(week, on=["date", "store", "item"], how="left")

    return result


def main() -> None:
    """
    問題の設定の損得で, 機械学習の方針と今のやり方の実際の損を, テスト用の
    期間でたどり直して比べる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 前の手順の確かめ (データ, 問題の設定, データセット, 推論の一式)
    dataset_dir = PROJECT_DIR / "outputs" / "dataset"
    if not SOURCE.is_file() or not (TASK_DIR / "problem.yaml").is_file() or (
        not (dataset_dir / "dataset.yaml").is_file()
    ):
        print("データ, 問題の設定かデータセットがありません. README の実行"
              "方法の順に実行してください")
        sys.exit(1)
    bundle = latest_bundle()

    # 問題の設定の意思決定と成功の基準
    problem = ml_quest.load_problem(TASK_DIR / "problem.yaml")
    stages = ml_quest.handoff(problem)
    decision = dict(stages["inference"]["decision"], current="last_week")
    print(f"成功の基準: {ml_quest.get_value(problem, 'verification.success')}")

    # テスト用の期間 (学習のときの時刻の順の分割の最後) と, その前の日
    dataset = load_dataset(dataset_dir)
    start = pd.to_datetime(dataset.frames["test"]["date"], utc=True).min()
    frame = with_last_week(pd.read_csv(SOURCE))
    dates = pd.to_datetime(frame["date"], utc=True)
    window = frame[dates >= start - pd.Timedelta(days=HISTORY_DAYS)]
    window = window.reset_index(drop=True)
    in_test = pd.to_datetime(window["date"], utc=True) >= start
    count = (in_test & window["last_week"].notna()).to_numpy()
    days = window.loc[count, "date"].nunique()
    print(f"テスト用の期間: {window.loc[count, 'date'].min()} から "
          f"{window.loc[count, 'date'].max()} ({days} 日, {count.sum()} 行)")

    # 実際の損 (円) と, 30 日あたり
    result = backtest(bundle, window, decision, count=count)
    for name, value in result["policies"].items():
        loss = -value
        print(f"  {name}: 損 {loss:,.0f} 円 (30 日あたり "
              f"{loss / days * 30:,.0f} 円)")
    for sentence in result["sentences"]:
        print(f"  {sentence}")
    policy = -result["policies"]["この方針"]
    current = -result["policies"]["今のやり方"]
    print(f"今のやり方より {1 - policy / current:.1%} 少ない")


if __name__ == "__main__":
    main()
