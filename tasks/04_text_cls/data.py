from collections import Counter
import csv
from pathlib import Path
import re
import sys

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset

# このタスクのディレクトリ (データの保存先の相対パスの基準)
TASK_DIR = Path(__file__).resolve().parent

# ラベルの各ビットが表す分野 (1: 量子物理, 2: 機械学習, 4: 最適化)
CATEGORIES = ["quant-ph", "ML", "math.OC"]

# 単語として切り出す文字の並び (英数字と, 単語の中のハイフン)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# 語彙の特別な番号 (0 はパディング, 1 は語彙に無い単語)
PAD_ID = 0
UNK_ID = 1


def get_path(cfg: DictConfig) -> Path:
    """
    データの CSV ファイルのパスを取得する関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    path: Path
        CSV ファイルのパス, 相対パスの場合はタスクのディレクトリから辿る
    """
    # 相対パスはタスクのディレクトリを基準に解決
    path = Path(cfg.data.path).expanduser()
    if not path.is_absolute():
        path = (TASK_DIR / path).resolve()

    return path


def decode_label(label: int) -> list[float]:
    """
    整数のラベルを分野ごとの 0 と 1 に分解する関数

    Parameters
    ----------
    label: int
        1 から 7 の整数, 各ビットが分野を表す

    Returns
    ----------
    targets: list[float]
        CATEGORIES の順に, その分野に属する場合は 1
    """
    # ビットごとに分野に属するかを判定
    targets = [float((label >> i) & 1) for i in range(len(CATEGORIES))]

    return targets


def get_yymm(url: str) -> str:
    """
    arXiv の URL から投稿年月を取り出す関数

    Parameters
    ----------
    url: str
        http://arxiv.org/abs/2201.00260v2 のような URL

    Returns
    ----------
    yymm: str
        2201 のような年 (下 2 桁) と月の文字列
    """
    # 新しい形式の arXiv ID の先頭 4 桁が投稿年月
    match = re.search(r"abs/(\d{4})\.\d+", url)
    if match is None:
        raise ValueError(f"arXiv ID を読み取れません: {url}")
    yymm = match.group(1)

    return yymm


def load_rows(cfg: DictConfig) -> list[dict]:
    """
    CSV ファイルを読み込む関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    rows: list[dict]
        論文ごとの本文 (タイトルと要旨), 分野のラベル, 投稿年月
    """
    # 要旨が長いので CSV の 1 項目の長さの上限を外して読み込み
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(get_path(cfg), newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            rows.append({
                "text": f"{record['Title']} {record['Abstract']}",
                "targets": decode_label(int(record["Label"])),
                "yymm": get_yymm(record["URL"])
            })

    return rows


def tokenize(text: str) -> list[str]:
    """
    文章を小文字の単語に分割する関数

    Parameters
    ----------
    text: str
        タイトルと要旨をつないだ文章

    Returns
    ----------
    tokens: list[str]
        小文字にした単語の列
    """
    # 小文字にして英数字の並びを切り出し
    tokens = TOKEN_PATTERN.findall(text.lower())

    return tokens


def build_vocab(texts: list[list[str]], size: int) -> dict[str, int]:
    """
    学習用データの単語から語彙を作る関数

    Parameters
    ----------
    texts: list[list[str]]
        学習用データの単語の列
    size: int
        語彙の大きさ (パディングと未知語を含む)

    Returns
    ----------
    vocab: dict[str, int]
        単語から番号への対応
    """
    # 出現回数の多い順に番号を割り当て, 0 と 1 は特別な番号として予約
    counter = Counter(token for tokens in texts for token in tokens)
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for token, _ in counter.most_common(size - len(vocab)):
        vocab[token] = len(vocab)

    return vocab


def encode(
    tokens: list[str], vocab: dict[str, int], max_len: int
) -> list[int]:
    """
    単語の列を語彙の番号の列に変換する関数

    Parameters
    ----------
    tokens: list[str]
        単語の列
    vocab: dict[str, int]
        単語から番号への対応
    max_len: int
        残す単語の数の上限, 先頭から数える

    Returns
    ----------
    ids: list[int]
        番号の列, 語彙に無い単語は UNK_ID
    """
    # 先頭から max_len 語までを番号に変換
    ids = [vocab.get(t, UNK_ID) for t in tokens[:max_len]]

    return ids


class ArxivDataset(Dataset):
    def __init__(
        self, token_ids: list[list[int]], targets: list[list[float]]
    ):
        # 親クラスのコンストラクタを呼び出す
        super(ArxivDataset, self).__init__()

        # 単語の番号の列と分野のラベルを保存
        self.token_ids = token_ids
        self.targets = targets

    def __len__(self) -> int:
        """
        サンプル数を取得する関数

        Parameters
        ----------
        None

        Returns
        ----------
        n: int
            論文の数
        """
        # 論文の数を返却
        n = len(self.token_ids)

        return n

    def __getitem__(self, index: int) -> tuple[list[int], list[float]]:
        """
        1 つの論文を取り出す関数

        Parameters
        ----------
        index: int
            論文の番号

        Returns
        ----------
        token_ids: list[int]
            単語の番号の列
        targets: list[float]
            分野ごとの 0 と 1
        """
        # 番号の列とラベルを取り出し
        token_ids = self.token_ids[index]
        targets = self.targets[index]

        return token_ids, targets


def collate(batch: list[tuple[list[int], list[float]]]) -> dict:
    """
    長さの違う単語の列をパディングして辞書のバッチにする関数

    Parameters
    ----------
    batch: list[tuple[list[int], list[float]]]
        Dataset から取り出した論文のリスト

    Returns
    ----------
    collated: dict
        input_ids, attention_mask, labels のテンソル
    """
    # バッチの中で最も長い列に合わせてパディング
    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # パディングでない位置の印とラベルを集約
    collated = {
        "input_ids": input_ids,
        "attention_mask": (input_ids != PAD_ID).float(),
        "labels": torch.tensor([t for _, t in batch], dtype=torch.float32)
    }

    return collated


def make_datasets(
    cfg: DictConfig
) -> tuple[ArxivDataset, ArxivDataset, dict[str, int]]:
    """
    投稿年月で学習用と検証用に分け, 単語の番号の列にする関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: ArxivDataset
        学習用のデータセット
    valid_dataset: ArxivDataset
        検証用のデータセット
    vocab: dict[str, int]
        学習用データから作った語彙
    """
    # 未来の論文が学習に漏れないように, 投稿年月で分割
    rows = load_rows(cfg)
    train_rows = [r for r in rows if r["yymm"] < cfg.data.valid_from]
    valid_rows = [r for r in rows if r["yymm"] >= cfg.data.valid_from]

    # 学習用データだけで語彙を作成
    train_tokens = [tokenize(r["text"]) for r in train_rows]
    valid_tokens = [tokenize(r["text"]) for r in valid_rows]
    vocab = build_vocab(train_tokens, cfg.data.vocab_size)

    # 先頭から max_len 語までを番号の列に変換
    max_len = cfg.data.max_len
    train_dataset = ArxivDataset(
        [encode(t, vocab, max_len) for t in train_tokens],
        [r["targets"] for r in train_rows]
    )
    valid_dataset = ArxivDataset(
        [encode(t, vocab, max_len) for t in valid_tokens],
        [r["targets"] for r in valid_rows]
    )

    return train_dataset, valid_dataset, vocab


def make_loader(
    dataset: Dataset, cfg: DictConfig, shuffle: bool
) -> DataLoader:
    """
    データセットから DataLoader を作る関数

    Parameters
    ----------
    dataset: Dataset
        バッチにするデータセット
    cfg: DictConfig
        学習に使う config
    shuffle: bool
        エポックごとに並べ替えるかどうか

    Returns
    ----------
    loader: DataLoader
        長さをそろえて辞書のバッチにする DataLoader
    """
    # パディングして辞書にする関数を渡して作成
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=shuffle,
        collate_fn=collate
    )

    return loader
