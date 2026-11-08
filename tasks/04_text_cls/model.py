import torch
from torch import nn


class FastTextClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int, num_labels: int,
        dropout: float
    ):
        # 親クラスのコンストラクタを呼び出す
        super(FastTextClassifier, self).__init__()

        # 語彙の大きさ, 埋め込みの次元数, ラベルの数, ドロップアウトの確率を保存
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.dropout = dropout

        # 単語の埋め込み層 (番号 0 はパディング) と出力層の定義
        self.embedding = nn.Embedding(
            self.vocab_size, self.embed_dim, padding_idx=0
        )
        self.fc = nn.Linear(self.embed_dim, self.num_labels)

        # ドロップアウト層の定義
        self.drop = nn.Dropout(self.dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        input_ids: torch.Tensor
            単語の番号の列, 形状は (バッチ, 系列長)
        attention_mask: torch.Tensor
            パディングでない位置が 1 の印, 形状は (バッチ, 系列長)

        Returns
        ----------
        logits: torch.Tensor
            分野ごとのロジット, 形状は (バッチ, num_labels)
        """
        # 単語を埋め込み, パディングを除いて平均
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        mean = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # 分野ごとのロジットを出力
        logits = self.fc(self.drop(mean))

        return logits
