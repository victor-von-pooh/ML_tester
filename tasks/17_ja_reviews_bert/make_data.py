from pathlib import Path

import numpy as np
import pandas as pd

# 書き出す先 (リポジトリの data, git の管理外). ラベル付きのレビューと,
# 事前学習に使うラベルの無いレビュー
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_PATH = DATA_DIR / "reviews_ja.csv"
UNLABELED_PATH = DATA_DIR / "reviews_ja_unlabeled.csv"

# 評価の対象 (本体の評価がラベル. 他の点は本体と関係の無い良し悪し)
TARGET = "本体"
ASPECTS = ["配送", "梱包", "説明書", "サポートの対応", "付属品", "箱"]

# 良い評価と悪い評価の述語 (否定すると意味が逆になる組と, 悪いことの否定)
PREDICATES = [
    ("良い", "良くない"), ("満足できる", "満足できない"),
    ("しっかりしている", "しっかりしていない"), ("期待通りだ", "期待通りではない"),
    ("悪くない", "悪い"), ("不満はない", "不満だ"), ("問題なかった", "問題があった"),
]

# どちらにも出る言葉
FILLERS = ["正直に言うと", "届いてすぐ見たが", "一週間使って", "", "", ""]

# 文の終わり
ENDINGS = ["。", "!", "…", "。星の数で判断して。"]


def clause(subject: str, positive: bool, rng: np.random.Generator) -> str:
    """
    1 つの点についての評価の句を作る関数

    Parameters
    ----------
    subject: str
        評価の対象
    positive: bool
        良い評価か
    rng: np.random.Generator
        乱数

    Returns
    ----------
    text: str
        句
    """
    # 否定で意味が逆になる述語と, 悪いことの否定の混在
    good, bad = PREDICATES[rng.integers(len(PREDICATES))]
    filler = FILLERS[rng.integers(len(FILLERS))]
    text = f"{filler}{subject}は{good if positive else bad}"

    return text


def review(positive: bool, rng: np.random.Generator) -> str:
    """
    1 件のレビューを作る関数

    本体の評価 (ラベル) と, 1 から 3 つの他の点の評価 (良し悪しは無作為)
    を, 無作為な順に並べる. 文字を数えるだけでは, どの評価が本体のものか
    分からない

    Parameters
    ----------
    positive: bool
        本体が良い評価か
    rng: np.random.Generator
        乱数

    Returns
    ----------
    text: str
        レビュー
    """
    # 本体の句と他の点の句の, 無作為な順の並び
    count = int(rng.integers(1, 4))
    aspects = rng.choice(ASPECTS, count, replace=False)
    parts = [clause(TARGET, positive, rng)] + [
        clause(str(a), bool(rng.random() < 0.5), rng) for a in aspects
    ]
    order = rng.permutation(len(parts))
    text = "、".join(parts[k] for k in order) + ENDINGS[
        rng.integers(len(ENDINGS))
    ]

    return text


def main(rows: int = 12000, unlabeled: int = 100000, seed: int = 0) -> None:
    """
    日本語のレビューの表を Shift_JIS の csv で書く関数

    学習の前に直すことを含める: ID, 答えから作った列 (星の数), 同じ文の
    重複, 空の文, 極端に長い文, 全角と半角の混在, クラスの偏り

    Parameters
    ----------
    rows: int = 12000
        ラベル付きの行の数 (重複を足す前)
    unlabeled: int = 100000
        ラベルの無いレビューの数 (事前学習用)
    seed: int = 0
        乱数のシード

    Returns
    ----------
    None
    """
    # 本体が良い評価が 7 割. 星の数は評価から決まる (リーク)
    rng = np.random.default_rng(seed)
    positive = rng.random(rows) < 0.7
    texts = [review(bool(p), rng) for p in positive]
    stars = np.where(
        positive, rng.integers(4, 6, rows), rng.integers(1, 3, rows)
    )
    frame = pd.DataFrame({
        "review_id": [f"R{100000 + i}" for i in range(rows)],
        "posted": pd.date_range("2025-01-01", periods=rows, freq="37min")
        .strftime("%Y/%m/%d %H:%M"),
        "product": rng.choice(["イヤホン", "ケトル", "ﾘｭｯｸ", "リュック"], rows),
        "stars": stars,
        "review": texts,
        "label": np.where(positive, "良い", "悪い"),
    })

    # 空の文, 極端に長い文 (同じ文の繰り返し), 同じ文の重複 (2 回の投稿)
    frame.loc[rng.choice(rows, 40, replace=False), "review"] = ""
    long_rows = rng.choice(rows, 5, replace=False)
    frame.loc[long_rows, "review"] = frame.loc[long_rows, "review"] * 30
    copies = frame.sample(200, random_state=seed).copy()
    copies["review_id"] = [f"R{900000 + i}" for i in range(len(copies))]
    frame = pd.concat([frame, copies]).sample(frac=1, random_state=seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False, encoding="cp932")
    print(f"保存: {DATA_PATH} ({len(frame)} 行, Shift_JIS)")

    # ラベルの無いレビュー (同じ作り方. 本体の良し悪しは半々)
    texts = [review(bool(rng.random() < 0.5), rng) for _ in range(unlabeled)]
    pd.DataFrame({"review": texts}).to_csv(
        UNLABELED_PATH, index=False, encoding="utf-8"
    )
    print(f"保存: {UNLABELED_PATH} ({unlabeled} 行, UTF-8)")


if __name__ == "__main__":
    main()
