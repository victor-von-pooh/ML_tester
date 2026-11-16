import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 書き出す先 (リポジトリの data, git の管理外). ラベル付きのレビューと,
# 事前学習に使うラベルの無いレビュー
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_PATH = DATA_DIR / "reviews_ja.csv"
UNLABELED_PATH = DATA_DIR / "reviews_ja_unlabeled.csv"
# 言い方を増やした版 (--natural. 元の版とは別のファイル)
NATURAL_PATH = DATA_DIR / "reviews_ja_natural.csv"
NATURAL_UNLABELED_PATH = DATA_DIR / "reviews_ja_natural_unlabeled.csv"

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

# 言い方を増やした版 (--natural). 主語, 助詞, 間に挟む言葉, 述語, つなぎ
# の言い方を増やし, 句がまるごと 1 つの部分語にならないようにする.
# {product} は商品の名前
NATURAL_TARGETS = [
    "本体", "商品自体", "製品そのもの", "物自体", "{product}本体",
    "{product}自体", "肝心の本体", "これ自体", "本体の作り",
]
NATURAL_ASPECTS = {
    "配送": ["配送", "届くまでの日数", "発送の早さ"],
    "梱包": ["梱包", "包装", "梱包の仕方"],
    "説明書": ["説明書", "取扱説明書", "マニュアル"],
    "サポートの対応": ["サポートの対応", "問い合わせへの返事", "窓口の対応"],
    "付属品": ["付属品", "付属のケーブル", "おまけ"],
    "箱": ["箱", "外箱", "パッケージ"],
}
NATURAL_PARTICLES = ["は", "は", "も", "については", "に関しては"]
NATURAL_MODIFIERS = [
    "", "", "", "正直", "思ったより", "かなり", "少し", "値段の割に",
    "今のところ", "使ってみると", "個人的には", "最初は気になったが結局",
]
NATURAL_PREDICATES = [
    ("良い", "良くない"), ("良かった", "良くなかった"),
    ("満足できる", "満足できない"), ("気に入った", "気に入らなかった"),
    ("しっかりしている", "しっかりしていない"), ("丈夫だ", "すぐ壊れた"),
    ("期待通りだ", "期待外れだ"), ("期待以上だった", "期待以下だった"),
    ("悪くない", "悪い"), ("悪くはなかった", "ひどかった"),
    ("不満はない", "不満だ"), ("文句なしだ", "文句しかない"),
    ("問題なかった", "問題があった"), ("使いやすい", "使いにくい"),
    ("おすすめできる", "おすすめできない"), ("買ってよかった", "買って後悔した"),
]
NATURAL_JOINS = ["、", "、", "けど、", "が、", "し、", "。ただ、", "。一方で", "。それに"]
NATURAL_FILLERS = [
    "", "", "", "正直に言うと", "届いてすぐ見たが", "一週間使って",
    "家族にも聞いたら", "前の物と比べると",
]


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


def natural_clause(
    subject: str, positive: bool, rng: np.random.Generator
) -> str:
    """
    言い方を増やした版の, 1 つの点についての評価の句を作る関数

    Parameters
    ----------
    subject: str
        評価の対象の言い方
    positive: bool
        良い評価か
    rng: np.random.Generator
        乱数

    Returns
    ----------
    text: str
        句 (前置き, 主語, 助詞, 間に挟む言葉, 述語)
    """
    # 間に挟む言葉は良し悪しに関係しない
    good, bad = NATURAL_PREDICATES[rng.integers(len(NATURAL_PREDICATES))]
    filler = NATURAL_FILLERS[rng.integers(len(NATURAL_FILLERS))]
    particle = NATURAL_PARTICLES[rng.integers(len(NATURAL_PARTICLES))]
    modifier = NATURAL_MODIFIERS[rng.integers(len(NATURAL_MODIFIERS))]
    text = (
        f"{filler}{subject}{particle}{modifier}{good if positive else bad}"
    )

    return text


def natural_review(positive: bool, rng: np.random.Generator) -> str:
    """
    言い方を増やした版の 1 件のレビューを作る関数

    作り方は review と同じ (本体の評価がラベル, 他の点は無作為) で, 主語,
    助詞, 間に挟む言葉, 述語, 句のつなぎの言い方を増やす

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
    # 本体の言い方には商品の名前も入る
    product = str(rng.choice(["イヤホン", "ケトル", "リュック"]))
    target = NATURAL_TARGETS[rng.integers(len(NATURAL_TARGETS))].format(
        product=product
    )
    count = int(rng.integers(1, 4))
    aspects = rng.choice(list(NATURAL_ASPECTS), count, replace=False)
    parts = [natural_clause(target, positive, rng)] + [
        natural_clause(
            str(rng.choice(NATURAL_ASPECTS[str(a)])),
            bool(rng.random() < 0.5), rng
        )
        for a in aspects
    ]
    order = rng.permutation(len(parts))
    text = parts[order[0]]
    for k in order[1:]:
        text += NATURAL_JOINS[rng.integers(len(NATURAL_JOINS))] + parts[k]
    text += ENDINGS[rng.integers(len(ENDINGS))]

    return text


def main(
    rows: int = 12000, unlabeled: int = 100000, seed: int = 0,
    natural: bool = False
) -> None:
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
    natural: bool = False
        言い方を増やした版を, 元の版とは別のファイルに書くか

    Returns
    ----------
    None
    """
    # 本体が良い評価が 7 割. 星の数は評価から決まる (リーク)
    rng = np.random.default_rng(seed)
    make = natural_review if natural else review
    data_path = NATURAL_PATH if natural else DATA_PATH
    unlabeled_path = NATURAL_UNLABELED_PATH if natural else UNLABELED_PATH
    positive = rng.random(rows) < 0.7
    texts = [make(bool(p), rng) for p in positive]
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
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False, encoding="cp932")
    print(f"保存: {data_path} ({len(frame)} 行, Shift_JIS)")

    # ラベルの無いレビュー (同じ作り方. 本体の良し悪しは半々)
    texts = [make(bool(rng.random() < 0.5), rng) for _ in range(unlabeled)]
    pd.DataFrame({"review": texts}).to_csv(
        unlabeled_path, index=False, encoding="utf-8"
    )
    print(f"保存: {unlabeled_path} ({unlabeled} 行, UTF-8)")


if __name__ == "__main__":
    # 元の版 (既定) と, 言い方を増やした版 (--natural)
    parser = argparse.ArgumentParser(description="レビューのデータを作る")
    parser.add_argument(
        "--natural", action="store_true",
        help="言い方を増やした版を reviews_ja_natural*.csv に書く"
    )
    main(natural=parser.parse_args().natural)
