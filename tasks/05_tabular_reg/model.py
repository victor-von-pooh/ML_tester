import torch
from torch import nn


class HousingRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(HousingRegressor, self).__init__()

        # 入力次元数と隠れ層の次元数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 隠れ層と出力層の定義
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)

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
        pred: torch.Tensor
            標準化した住宅価格の予測, 形状は (バッチ,)
        """
        # 隠れ層を 2 回通して特徴量を変換
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        # 出力層で 1 つの値に集約
        pred = self.fc_out(h).squeeze(-1)

        return pred
