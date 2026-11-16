import argparse
from pathlib import Path
import sys
import unicodedata

import pandas as pd
import yaml

from ml_eda import load_data, load_dataset
from ml_inference import (
    explain, forecast, infer, make_bundle, predict, write_result
)

# このタスクのディレクトリと, 学習の結果を使うタスクのディレクトリ
TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
DATA_DIR = TASKS.parent / "data"

# 文字は全く同じで, どの点が良い (悪い) かだけが違うレビューの組 (17 と同じ)
PROBES = [
    ("本体は良い、配送は悪い。", "良い"),
    ("配送は良い、本体は悪い。", "悪い"),
    ("本体は悪くない、梱包は満足できない。", "良い"),
    ("梱包は悪くない、本体は満足できない。", "悪い"),
    ("説明書は問題があった、本体は期待通りだ。", "良い"),
    ("説明書は期待通りだ、本体は問題があった。", "悪い"),
]


def latest_run(experiment: Path) -> Path:
    """
    実験の名前のディレクトリから, 最も新しい実験のディレクトリを返す関数

    Parameters
    ----------
    experiment: Path
        outputs/<experiment.name>

    Returns
    ----------
    run: Path
        <日付>/<時刻> の最も新しいもの

    Raises
    ----------
    SystemExit
        学習を終えた実験が無い場合
    """
    # final_model.pt のある実験だけ (日付と時刻の名前の順)
    runs = sorted(p.parent for p in experiment.glob("*/*/final_model.pt"))
    if not runs:
        print(f"{experiment} に学習を終えた実験がありません. 先に README の"
              "手順で学習してください")
        sys.exit(1)
    run = runs[-1]

    return run


def run_settings(task: Path, run: Path) -> tuple[Path, Path]:
    """
    実験の config から, 学習に使ったデータセットとモデルの定義を返す関数

    Parameters
    ----------
    task: Path
        タスクのディレクトリ
    run: Path
        実験のディレクトリ

    Returns
    ----------
    paths: tuple[Path, Path]
        データセットのディレクトリと, モデルの定義のファイル
    """
    # 上書き (data.dataset=... など) も含めた, 学習のときの config
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    paths = (task / config["data"]["dataset"], task / config["model"]["spec"])

    return paths


def print_check(info: dict) -> None:
    """
    一式を作るときに検証用の分割を推論した評価と, 実験の記録を表示する関数

    Parameters
    ----------
    info: dict
        make_bundle が書いた inference.yaml の中身

    Returns
    ----------
    None
    """
    # 合わない場合は注意も
    check = info["check"]
    print(f"  一式での {check['split']} の {check['metric']}: "
          f"{check['value']:.6g} (実験の記録 {check['recorded']:.6g}, "
          f"合う: {check['matches']})")
    for note in info["notes"]:
        print(f"  注意: {note}")


def reviews(out: Path) -> None:
    """
    17 の部分語の BERT の一式を作り, テスト用の分割, 見分けの組, ラベルの
    無いレビューを推論する関数

    Parameters
    ----------
    out: Path
        一式と結果を書く場所

    Returns
    ----------
    None
    """
    # 学習のスクリプトが先頭に足した [CLS] (語彙の数 + 1) の一式への記録
    task = TASKS / "17_ja_reviews_bert"
    run = latest_run(task / "outputs" / "natural_subword_scratch_48k")
    dataset_dir, spec = run_settings(task, run)
    dataset = load_dataset(dataset_dir)
    vocabulary = next(
        s["state"]["vocabulary"] for s in dataset.recipe["item_steps"]
        if s["op"] == "vocabulary"
    )
    bundle = out / "bundle_reviews"
    info = make_bundle(
        bundle, spec, dataset_dir, run,
        inputs={"ids": {"source": "items", "prepend": [len(vocabulary) + 1]}},
        output="logits", conclusion={"min_confidence": 0.9},
        name="ja_reviews_subword", overwrite=True
    )
    print(f"17 の一式: {bundle.name} (実験 {run.parent.name}/{run.name}, "
          f"{info['run']['monitor']} {info['run']['best_value']:.4f})")
    print_check(info)

    # テスト用の分割の文 (書き出した表のラベルはクラスの番号)
    test = dataset.frames["test"]
    answers = [dataset.info["classes"][i] for i in test["label"]]
    result = predict(bundle, test[["review"]])
    accuracy = (result["prediction"] == pd.Series(answers)).mean()
    print(f"  テスト用の正解率 (一式で推論): {accuracy:.4f} "
          f"({len(test):,} 件, 予測しなかった行 "
          f"{(result['status'] != 'ok').sum()})")

    # 見分けの組
    probes = pd.DataFrame(PROBES, columns=["review", "answer"])
    result = predict(bundle, probes)
    hits = int((result["prediction"] == result["answer"]).sum())
    print(f"  見分けの組の正解: {hits}/{len(PROBES)}")

    # ラベルの無いレビューの結論
    unlabeled = DATA_DIR / "reviews_ja_natural_unlabeled_48000.csv"
    result = infer(bundle, unlabeled)
    write_result(result, out / "result_reviews")
    for sentence in result["sentences"][:2]:
        print(f"  {sentence}")


def sales(out: Path) -> None:
    """
    18 の売上のモデルの一式を作り, 元のデータのテスト用の期間を推論する
    関数

    Parameters
    ----------
    out: Path
        一式と結果を書く場所

    Returns
    ----------
    None
    """
    # 学習用の平均と標準偏差での目的の値の変換 (train.py と同じ求め方)
    task = TASKS / "18_store_sales"
    run = latest_run(task / "outputs" / "store_sales")
    dataset_dir, spec = run_settings(task, run)
    dataset = load_dataset(dataset_dir)
    _, y_train = dataset.tensors("train")
    scale = {"center": y_train.mean().item(), "scale": y_train.std().item()}
    bundle = out / "bundle_sales"
    bands = [{"label": "少ない", "upper": 500000},
             {"label": "ふつう", "upper": 1000000}, {"label": "多い"}]
    info = make_bundle(
        bundle, spec, dataset_dir, run,
        target_scale=scale, conclusion={"bands": bands, "group_by": "store"},
        name="store_sales", overwrite=True
    )
    print(f"18 の一式: {bundle.name} (実験 {run.parent.name}/{run.name}, "
          f"{info['run']['monitor']} {info['run']['best_value']:,.0f})")
    print_check(info)

    # 汚れた元のデータをそのまま推論 (過去の値は渡した行から作る)
    result = infer(bundle, DATA_DIR / "store_sales.csv")
    write_result(result, out / "result_sales")
    for sentence in result["sentences"][:3]:
        print(f"  {sentence}")

    # テスト用の期間の誤り. 重複は学習のときと同じく, 店の表記をそろえて
    # から元の全ての列で比べて除く
    frame = result["frame"]
    start = dataset.frames["test"]["date"].min()
    period = frame[(frame["date"] >= start) & (frame["status"] == "ok")]
    original = period[["date", "store", "temperature", "promo", "sales"]]
    original = original.assign(store=original["store"].map(
        lambda name: unicodedata.normalize("NFKC", str(name)).strip()
    ))
    period = period.loc[~original.duplicated()]
    error = (period["prediction"] - period["sales"]).abs()
    print(f"  テスト用の期間の平均絶対誤差 (一式で推論): {error.mean():,.0f} 円 "
          f"({len(period):,} 行), 平均の割合の誤差: "
          f"{(error / period['sales'].clip(lower=1)).mean() * 100:.3f}%")
    skipped = frame[frame["status"] != "ok"]
    print(f"  予測しなかった行: {len(skipped)} (理由の例: "
          f"{skipped['reason'].iloc[0] if len(skipped) else 'なし'})")


def sales_ahead(out: Path, days: int = 7) -> None:
    """
    18 の一式で, テスト用の期間の始めから先の数日を何段も先まで予測し,
    同じ行の 1 つ先の予測と実際の売上と比べ, 列の効きも求める関数

    Parameters
    ----------
    out: Path
        一式のある場所
    days: int = 7
        予測する先の日の数

    Returns
    ----------
    None
    """
    # テスト用の期間より前の過去の行と, 始めの数日の先の行. 先の行の
    # 店の表記はそろえる (系列を渡した値で分けるため. 重複もここで除く)
    bundle = out / "bundle_sales"
    frame = load_data(DATA_DIR / "store_sales.csv").frame
    start = load_dataset(
        TASKS / "18_store_sales" / "outputs" / "dataset"
    ).frames["test"]["date"].min()
    history = frame[frame["date"] < start]
    ahead = frame[(frame["date"] >= start)
                  & (frame["date"] < start + pd.Timedelta(days=days))]
    store = ahead["store"].map(
        lambda name: unicodedata.normalize("NFKC", str(name)).strip()
    )
    ahead = ahead.assign(store=store).drop_duplicates()
    actual = ahead["sales"].to_numpy()

    # 何段も先の予測 (先の行の売上は空) と, 同じ行の 1 つ先の予測 (先の
    # 行の前の日の売上は実際の値)
    given = pd.concat([history, ahead.assign(sales=float("nan"))],
                      ignore_index=True)
    result = forecast(bundle, given).iloc[len(history):]
    known = pd.concat([history, ahead], ignore_index=True)
    single = predict(bundle, known).iloc[len(history):]
    result = result.assign(
        actual=actual, single=single["prediction"].to_numpy()
    )
    done = result[result["status"] == "ok"]
    print(f"18 の先の {days} 日 ({len(done)}/{len(result)} 行を予測):")
    for step, part in [*done.groupby("step"), ("全て", done)]:
        scale = part["actual"].clip(lower=1)
        many = (part["prediction"] - part["actual"]).abs()
        one = (part["single"] - part["actual"]).abs()
        label = f"{int(step)} 段目" if step != "全て" else "全て"
        print(f"  {label}: {len(part)} 行, 何段も先 {many.mean():,.0f} 円 "
              f"({(many / scale).mean() * 100:.1f}%), 1 つ先 "
              f"{one.mean():,.0f} 円 ({(one / scale).mean() * 100:.1f}%)")

    # テスト用の期間と前の 30 日の行の列の効き
    window = frame[frame["date"] >= start - pd.Timedelta(days=30)]
    effects = explain(bundle, window)
    print("  列の効き: " + ", ".join(
        f"{item['column']} {item['effect']:,.0f}"
        for item in effects["columns"] if item["effect"] is not None
    ))


def main() -> None:
    """
    17 と 18 の学習の結果から推論の一式を作り, 推論と結論を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式と結果の書き出し先
    parser = argparse.ArgumentParser(description="推論の一式の確かめ")
    parser.add_argument(
        "--out", default="outputs", help="書き出し先 (タスクのディレクトリから)"
    )
    args = parser.parse_args()
    out = TASK_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)

    # 文の分類と, 売上の予測
    reviews(out)
    sales(out)
    sales_ahead(out)


if __name__ == "__main__":
    main()
