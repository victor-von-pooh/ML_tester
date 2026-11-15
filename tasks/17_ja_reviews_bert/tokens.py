from pathlib import Path
from typing import Optional

import torch
from torch import nn

from ml_eda import Dataset
from ml_modelcontrol import build_model


def special_ids(dataset: Dataset) -> tuple[int, int, int]:
    """
    データセットの語彙の大きさと, 足す 2 つの特別な番号を返す関数

    ML-EDA の語彙は 0 が詰め物 (<pad>), 1 が語彙に無い文字 (<unk>). その
    後ろに, 隠した文字の印 ([MASK]) と文の先頭の印 ([CLS]) を足す

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット (中身の手順に vocabulary を持つ)

    Returns
    ----------
    ids: tuple[int, int, int]
        モデルの語の数 (語彙 + 2), [MASK] の番号, [CLS] の番号
    """
    # 当てはめた語彙の長さ
    vocabulary = next(
        step["state"]["vocabulary"] for step in dataset.recipe["item_steps"]
        if step["op"] == "vocabulary"
    )
    size = len(vocabulary)
    ids = (size + 2, size, size + 1)

    return ids


def add_cls(ids: torch.Tensor, cls_id: int) -> torch.Tensor:
    """
    番号の並びの先頭に [CLS] を足す関数 (モデルは先頭の表現で分類する)

    Parameters
    ----------
    ids: torch.Tensor
        (件数, 長さ) の番号の並び
    cls_id: int
        [CLS] の番号

    Returns
    ----------
    added: torch.Tensor
        (件数, 長さ + 1) の番号の並び
    """
    # 先頭の列を足す
    head = torch.full((len(ids), 1), cls_id, dtype=ids.dtype)
    added = torch.cat([head, ids], dim=1)

    return added


def load_model(
    task_dir: Path, spec: str, init: Optional[str], seed: int, vocab: int
) -> nn.Module:
    """
    ML-CreateArchitecture で組んだ定義からモデルを作り, 重みを読む関数

    Parameters
    ----------
    task_dir: Path
        タスクのディレクトリ
    spec: str
        定義のパス (タスクのディレクトリから)
    init: Optional[str]
        読み込む重みのパス (None なら読まない)
    seed: int
        重みの初期値の乱数のシード
    vocab: int
        データセットから求めた語の数 (定義の埋め込みの大きさと比べる)

    Returns
    ----------
    model: nn.Module
        組み立てたモデル

    Raises
    ----------
    ValueError
        定義の語の数がデータセットと合わない場合, 重みが無い場合
    """
    # 定義の埋め込みの大きさを確かめてから組み立てる
    model = build_model(task_dir / spec, seed=seed)
    sizes = {
        module.num_embeddings for name, module in model.named_modules()
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag))
        and not name.endswith("pos")
    }
    if vocab not in sizes:
        raise ValueError(
            f"定義の語の埋め込みの大きさ {sorted(sizes)} がデータセットの "
            f"{vocab} (語彙 + [MASK] + [CLS]) と合いません. "
            "ML-CreateArchitecture で num_embeddings を直してください"
        )
    if init:
        path = task_dir / init
        if not path.is_file():
            raise ValueError(f"{path} がありません. 先に pretrain.py を実行"
                             "してください (使わない場合は model.init=null)")
        model.load_state_dict(torch.load(path, map_location="cpu"))

    return model
