import torch
from torch import nn


class ImbalancedClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        # 親クラスのコンストラクタを呼び出す
        super(ImbalancedClassifier, self).__init__()

        # 入力次元数, 隠れ層の次元数, ドロップアウトの確率を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 隠れ層と出力層の定義
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)

        # 活性化関数とドロップアウト層の定義
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

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
            正例である度合いのロジット, 形状は (バッチ,)
        """
        # 隠れ層を 2 回通して特徴量を変換
        h = self.drop(self.relu(self.fc1(x)))
        h = self.drop(self.relu(self.fc2(h)))

        # 出力層で 1 つのロジットに集約
        logits = self.fc_out(h).squeeze(-1)

        return logits
