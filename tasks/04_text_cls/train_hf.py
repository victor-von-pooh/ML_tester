from pathlib import Path
from typing import Any

from omegaconf import DictConfig
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding,
    PreTrainedTokenizerBase, get_linear_schedule_with_warmup
)

from data import CATEGORIES, load_rows
from ml_found import load_project_config, start_experiment
from ml_found.training import build_optimizer
from train import SHORT_NAMES, MultiLabelTrainer

# このタスクのディレクトリ (ML-Found のプロジェクトのディレクトリ)
TASK_DIR = Path(__file__).resolve().parent


class TextDataset(Dataset):
    def __init__(self, texts: list[str], targets: list[list[float]]):
        # 親クラスのコンストラクタを呼び出す
        super(TextDataset, self).__init__()

        # 文章と分野のラベルを保存
        self.texts = texts
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
        n = len(self.texts)

        return n

    def __getitem__(self, index: int) -> tuple[str, list[float]]:
        """
        1 つの論文を取り出す関数

        Parameters
        ----------
        index: int
            論文の番号

        Returns
        ----------
        text: str
            タイトルと要旨をつないだ文章
        targets: list[float]
            分野ごとの 0 と 1
        """
        # 文章とラベルを取り出し
        text = self.texts[index]
        targets = self.targets[index]

        return text, targets


class TokenizeCollate:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        # トークナイザと最大の長さを保存
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(
        self, batch: list[tuple[str, list[float]]]
    ) -> BatchEncoding:
        """
        文章をトークンに分けてパディングし, ラベルを加える関数

        Parameters
        ----------
        batch: list[tuple[str, list[float]]]
            Dataset から取り出した論文のリスト

        Returns
        ----------
        encoded: BatchEncoding
            input_ids, attention_mask, labels を持つトークナイザの出力
        """
        # トークナイザの出力 (BatchEncoding) をそのままバッチに使用
        texts = [text for text, _ in batch]
        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_len, padding=True,
            return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(
            [t for _, t in batch], dtype=torch.float32
        )

        return encoded


class HFTrainer(MultiLabelTrainer):
    def step(self, batch: Any, phase: str) -> dict:
        """
        Hugging Face のモデルで損失と完全一致率を求め, 予測を貯める関数

        Parameters
        ----------
        batch: Any
            input_ids, attention_mask, labels を持つ BatchEncoding
        phase: str
            train または valid

        Returns
        ----------
        outputs: dict
            モデルが求めた二値交差エントロピー損失, 完全一致率, サンプル数
        """
        # labels を渡すとモデルが損失も計算するので, その損失を使用
        result = self.model(**batch)
        loss = result.loss

        # 0.5 を境に分野ごとの予測を決め, エポック全体の指標のために保存
        pred = (result.logits.detach() > 0).float()
        self.preds.append(pred.cpu())
        self.labels.append(batch["labels"].cpu())

        # 全ての分野の予測が当たった割合の計算
        exact = (pred == batch["labels"]).all(dim=1).float().mean()

        # ML-Found は BatchEncoding からサンプル数を推定できないので明示
        outputs = {
            "loss": loss, "exact": exact, "batch_size": len(batch["labels"])
        }

        return outputs


def make_text_datasets(
    cfg: DictConfig
) -> tuple[TextDataset, TextDataset]:
    """
    投稿年月で分け, 学習用を決まった数だけ取り出す関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config

    Returns
    ----------
    train_dataset: TextDataset
        学習用のデータセット, data.n_train 件
    valid_dataset: TextDataset
        検証用のデータセット, data.n_valid 件
    """
    # 段階 1 と同じく投稿年月で分割
    rows = load_rows(cfg)
    train_rows = [r for r in rows if r["yymm"] < cfg.data.valid_from]
    valid_rows = [r for r in rows if r["yymm"] >= cfg.data.valid_from]

    # 事前学習済みモデルは重いので, 専用の乱数で決まった数だけ取り出し
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    train_idx = torch.randperm(len(train_rows), generator=generator)
    valid_idx = torch.randperm(len(valid_rows), generator=generator)
    train_rows = [train_rows[i] for i in train_idx[:cfg.data.n_train]]
    valid_rows = [valid_rows[i] for i in valid_idx[:cfg.data.n_valid]]

    # 文章とラベルのデータセットに集約
    train_dataset = TextDataset(
        [r["text"] for r in train_rows], [r["targets"] for r in train_rows]
    )
    valid_dataset = TextDataset(
        [r["text"] for r in valid_rows], [r["targets"] for r in valid_rows]
    )

    return train_dataset, valid_dataset


def build_scheduler(
    cfg: DictConfig, optimizer: optim.Optimizer, n_batches: int
) -> optim.lr_scheduler.LRScheduler:
    """
    warmup の後に線形に学習率を下げる Scheduler を作る関数

    Parameters
    ----------
    cfg: DictConfig
        学習に使う config
    optimizer: optim.Optimizer
        学習率を調整する Optimizer
    n_batches: int
        1 エポックあたりのバッチ数

    Returns
    ----------
    scheduler: optim.lr_scheduler.LRScheduler
        transformers の warmup 付きの線形減衰 (中身は LambdaLR)
    """
    # 全体のステップ数はバッチ数で決まるので,
    # config だけでは決められずここで計算
    total_steps = cfg.trainer.epochs * n_batches
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    return scheduler


def main() -> None:
    """
    config を読み込んで事前学習済みモデルで論文の分野の分類を学習する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config を読み込んで実験を開始
    cfg, diff = load_project_config(
        str(TASK_DIR), default="configs/default/hf.yaml",
        experiment="configs/experiment/hf.yaml"
    )
    experiment = start_experiment(cfg, diff, str(TASK_DIR))

    # データとトークナイザを用意
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    collate = TokenizeCollate(tokenizer, cfg.data.max_len)
    train_dataset, valid_dataset = make_text_datasets(cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        collate_fn=collate
    )
    experiment.logger.info(
        f"学習用データ: {len(train_dataset)} 件, "
        f"検証用データ: {len(valid_dataset)} 件"
    )

    # 事前学習済みモデルに分野の数だけの出力層を付けて読み込み
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=len(CATEGORIES),
        problem_type="multi_label_classification"
    )

    # config に Scheduler か warmup を書いた場合は ML-Found が作る
    # (warmup の回数と auto はバッチ数から補う). 書かない場合は transformers の
    # warmup 付きの線形減衰を, バッチ数を使って作って Trainer に受け渡し
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    uses_config = (
        cfg.scheduler.name is not None
        or cfg.scheduler.warmup_ratio is not None
        or cfg.scheduler.warmup_steps is not None
    )
    scheduler = (
        None if uses_config
        else build_scheduler(cfg, optimizer, len(train_loader))
    )
    trainer = HFTrainer(model, experiment, optimizer, scheduler)
    history = trainer.fit(train_loader, valid_loader)

    # ベストモデルのエポックの結果を表示
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {experiment.run_dir}")
    print(f"学習したエポック数: {len(history.records)}")
    print(f"ベストモデルのエポック: {best_epoch}")
    keys = ["valid/macro_f1", "valid/micro_f1", "valid/exact"]
    keys += [f"valid/f1_{name}" for name in SHORT_NAMES]
    for key in keys:
        value = history.get(key)[best_epoch - 1]
        print(f"ベストモデルの {key}: {value:.4f}")


if __name__ == "__main__":
    main()
