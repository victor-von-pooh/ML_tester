import torch
from torch import nn


class FunctionMLP(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int):
        # 親クラスのコンストラクタを呼び出す
        super(FunctionMLP, self).__init__()

        # 隠れ層の次元数と層の数を保存
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 1 次元の入力から隠れ層を n_layers 回重ねる線形層を定義
        self.layers = nn.ModuleList()
        in_dim = 1
        for _ in range(self.n_layers):
            self.layers.append(nn.Linear(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim

        # 1 次元の出力層の定義
        self.fc_out = nn.Linear(self.hidden_dim, 1)

        # 滑らかな関数を近似するので活性化関数は tanh を定義
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力, 形状は (サンプル数, 1)

        Returns
        ----------
        y: torch.Tensor
            関数の値の予測, 形状は (サンプル数, 1)
        """
        # 線形層と活性化関数を順に適用
        h = x
        for layer in self.layers:
            h = self.tanh(layer(h))

        # 出力層で 1 つの値に変換
        y = self.fc_out(h)

        return y
