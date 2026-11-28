"""
方策のモデル (ML-Quest が問題の設定から書き出した骨組み)

方策の頭 (観測から行動のスコアか平均) は ML-ModelControl の定義 (config の
model.spec) から train.py が組み立てて渡す. このモデルは学習だけに使う価値の
関数と, 連続の行動の対数の標準偏差を足す. 方策の一式には方策の頭の重みと
定義だけを写す. 印 (TODO) の所を書き足す
"""
import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(
        self, actor: nn.Module, obs_dim: int, hidden_dim: int,
        action_dim: int, continuous: bool
    ):
        # 親クラスのコンストラクタを呼び出す
        super(PolicyNet, self).__init__()

        # 方策の頭 (定義から組み立てたモデル), 観測の次元数, 隠れ層の次元数,
        # 行動の数 (連続の行動は要素の数), 連続の行動かを保存
        self.actor = actor
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # TODO: 課題に合う大きさと形に直す. 今は価値の関数を隠れ層 2 つの
        # 全結合 (方策の頭とは重みを分ける)
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()

        # 連続の行動の, 観測によらない対数の標準偏差 (ML-Found の決まり)
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数 (書き出した方策もこの出力だけを使う)

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        output: torch.Tensor
            (バッチ, 行動の数) の行動ごとのスコアか, 連続の行動の平均
        """
        # 方策の頭
        output = self.actor(x)

        return output

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        状態の価値を求める関数 (ActorCriticStep が使う)

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        values: torch.Tensor
            (バッチ, 1) の状態の価値
        """
        # 全結合と活性化を 2 段と, 価値の頭
        hidden = self.tanh(self.fc2(self.tanh(self.fc1(x))))
        values = self.critic(hidden)

        return values
