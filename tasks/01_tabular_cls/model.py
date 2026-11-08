import torch
from torch import nn


class IrisClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(IrisClassifier, self).__init__()

        # 入力次元数, 隠れ層の次元数, クラス数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 隠れ層と出力層の定義
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            標準化した特徴量

        Returns
        ----------
        logits: torch.Tensor
            クラスごとのスコア
        """
        # 隠れ層で特徴量を変換
        h = self.relu(self.fc1(x))

        # 出力層でクラスごとのスコアを計算
        logits = self.fc2(h)

        return logits
