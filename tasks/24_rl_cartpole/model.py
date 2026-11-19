import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        # 親クラスのコンストラクタを呼び出す
        super(PolicyNet, self).__init__()

        # 観測の次元数, 隠れ層の次元数, 行動の数を保存
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # 線形層と活性化関数
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        logits: torch.Tensor
            (バッチ, 行動の数) の行動ごとのスコア
        """
        # 隠れ層 1 つの全結合
        hidden = self.tanh(self.fc1(x))
        logits = self.fc2(hidden)

        return logits
