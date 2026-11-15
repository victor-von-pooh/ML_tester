import torch
from torch import nn


class ChurnClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(ChurnClassifier, self).__init__()

        # 引数の保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 2 層の多層パーセプトロン
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            前処理した特徴量

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 多層パーセプトロンを適用
        logits = self.layers(x)

        return logits
