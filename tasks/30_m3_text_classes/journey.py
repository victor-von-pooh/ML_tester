"""
文の分類 (合成した問い合わせの文) を, 問題の設定から推論まで 5 つの画面の
アプリの API だけで通し, 段階ごとに画面だけでできたか (OK), 手で写すか
打つか (手), 誤りかを表にする (画像と文の分類を画面で通す節目の確かめ)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "29_m3_image_classes"))
from classes import example_model, recipe_of, table_columns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "25_m1_churn"))
from apps import Client, main_with

# クラスごとの文の部品 (主語, 中身, 結び). 中身がクラスを決める. 見本の
# 意思決定 (選ぶ) は 2 クラスの分類にだけ使えるので 2 つ
SUBJECTS = ["先月から", "昨日", "契約してから", "今朝", "乗り換えてから",
            "家族の分も", "引っ越してから", "毎月", "先週", "最近"]
BODIES = {
    "complaint": ["料金が高すぎて納得できません", "請求が二重になっています",
                  "通信が何度も切れて困っています", "窓口の対応がひどかったです",
                  "説明と違う金額を引き落とされました",
                  "解約の手続きが分かりにくくて腹が立ちます"],
    "question": ["支払い方法を変えられますか", "料金の明細はどこで見られますか",
                 "家族割は使えますか", "契約の期間を教えてください",
                 "住所の変更はどうすればいいですか",
                 "途中でプランを変えると費用はかかりますか"]
}
ENDINGS = ["", "。", "!", "。よろしくお願いします", "。至急お願いします",
           "。以上です"]

# クラスごとの文の数 (学習用と, 推論とたどり直し用)
COUNT = 80
PAST_COUNT = 10

# 学習の後に ML-Pathfinder で直す損得 (連絡 1 件の費用. 25 と同じ)
COST_UPDATE = {"cost": 3000}


def make_texts(
    root: Path, count: int, seed: int, skip: set[str]
) -> set[str]:
    """
    <クラス>/<文>.txt のディレクトリに合成の問い合わせの文を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ
    count: int
        クラスごとの文の数
    seed: int
        乱数の種
    skip: set[str]
        書かない文 (学習のデータと同じ文を推論に混ぜないため)

    Returns
    ----------
    written: set[str]
        書いた文
    """
    # 部品の組み合わせからの, 同じ文の無い選び出し
    rng = np.random.default_rng(seed)
    written: set[str] = set()
    for label, bodies in BODIES.items():
        (root / label).mkdir(parents=True, exist_ok=True)
        index = 0
        while index < count:
            text = (SUBJECTS[rng.integers(len(SUBJECTS))]
                    + bodies[rng.integers(len(bodies))]
                    + ENDINGS[rng.integers(len(ENDINGS))])
            if text in written or text in skip:
                continue
            written.add(text)
            (root / label / f"{index:03d}.txt").write_text(
                text, encoding="utf-8"
            )
            index += 1

    return written


def recommended_steps(
    fathom: Client, sid: str
) -> tuple[list[dict], str, str]:
    """
    ML-Fathom の分析の結果が勧める文の区切り方を, 中身の手順にする関数
    (画面の「勧めを使う」と同じ値)

    Parameters
    ----------
    fathom: Client
        ML-Fathom の客
    sid: str
        データの ID

    Returns
    ----------
    result: tuple[list[dict], str, str]
        中身の手順, 判定, 説明
    """
    # 分析の結果の tokenizers の recommended (ML-EDA の compare_tokenizers)
    items = fathom.ok("GET", f"/api/sources/{sid}/profile").get("items")
    tokenizers = (items or {}).get("tokenizers") or {}
    chosen = tokenizers.get("recommended")
    if not chosen:
        result = ([], "誤り", f"勧めが無い ({tokenizers.get('reason')})")
        return result
    result = ([chosen], "OK",
              f"分析の勧め {chosen} ({tokenizers.get('reason')})")

    return result


def fit_vocabulary(spec: dict, dataset: Path) -> list[str]:
    """
    埋め込みの語彙の数を, 書き出したデータセットの語彙の大きさにする関数
    (画面が当てていなければ, 利用者が ML-Fathom の書き出しの結果から写す
    値. 当てていれば直す所は無し)

    Parameters
    ----------
    spec: dict
        定義
    dataset: Path
        書き出したデータセット

    Returns
    ----------
    changes: list[str]
        直した所の説明
    """
    # レシピの vocabulary の当てはめた語彙 (ML-Fathom の書き出しの hints の
    # vocabulary と同じ値)
    steps = recipe_of(dataset).get("item_steps") or []
    size = next((len(s["state"]["vocabulary"]) for s in steps
                 if s.get("op") == "vocabulary"), None)
    changes = []
    for name, node in spec["nodes"].items():
        params = node.get("params") or {}
        if (node.get("unit") == "nn.Embedding" and size is not None
                and params.get("num_embeddings") != size):
            changes.append(f"{name} の num_embeddings "
                           f"{params.get('num_embeddings')} → {size} (語彙の"
                           "大きさを ML-Fathom の書き出しの結果から写す)")
            params["num_embeddings"] = size

    return changes


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
    # 学習のデータと, 推論とたどり直しに使う別の乱数の種の文 (学習の文と
    # 重ならないもの)
    data = work / "data" / "inquiries"
    seen = make_texts(data, COUNT, seed=0, skip=set())
    past = work / "data" / "inquiries_past"
    make_texts(past, PAST_COUNT, seed=1, skip=seen)
    columns = table_columns(data)
    setup = {
        "example": "解約の引き止め",
        "data": data,
        "past": past,
        # 文の表 (ML-EDA の load_data の列) に合わせた見本の答え
        "answers": {
            "information.columns": columns,
            "ml.parts.0.target.column": "label",
            "ml.parts.0.positive": "complaint",
            "ml.parts.0.item": "問い合わせの文 1 件",
            "information.known": [c for c in columns
                                  if c not in ("label", "file")],
            "information.forbidden": ["file"]
        },
        "recipe_update": None,
        "recipe_steps": [],
        "item_steps_from": recommended_steps,
        "make_model": example_model("lstm_classifier", fit_vocabulary),
        "cost_update": COST_UPDATE
    }

    return setup


if __name__ == "__main__":
    failed = main_with(build, "文の分類 (問い合わせの文)", sys.argv[1:],
                       prefix="m3_")
    sys.exit(1 if failed else 0)
