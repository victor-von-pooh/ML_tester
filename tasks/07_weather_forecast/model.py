import math
from typing import Optional

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(LSTMCell, self).__init__()

        # 入力層サイズと隠れ層サイズを保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 各種重みとバイアスの定義
        # OddNN の版は重みを torch.randn (標準偏差 1) で初期化していて大きいので,
        # 重みだけ nn.LSTM と同じく ±1/√hidden_dim の一様分布で初期化
        bound = 1.0 / math.sqrt(self.hidden_dim)
        self.weight_ih = nn.Parameter(
            torch.empty(4 * self.hidden_dim, self.input_dim).uniform_(
                -bound, bound
            )
        )
        self.weight_hh = nn.Parameter(
            torch.empty(4 * self.hidden_dim, self.hidden_dim).uniform_(
                -bound, bound
            )
        )
        self.bias_ih = nn.Parameter(torch.zeros(4 * self.hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(4 * self.hidden_dim))

        # 活性化関数の定義
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x_t: torch.Tensor
            入力データ
        h_prev: torch.Tensor
            1 ステップ前の隠れ状態
        c_prev: torch.Tensor
            1 ステップ前のセル

        Returns
        ----------
        h_t: torch.Tensor
            隠れ状態
        c_t: torch.Tensor
            セル
        """
        # 総合的なゲート計算
        gates = (
            torch.matmul(x_t, self.weight_ih.t())
            + torch.matmul(h_prev, self.weight_hh.t())
            + self.bias_ih + self.bias_hh
        )

        # ゲート分割
        i, f, g, o = gates.chunk(4, dim=1)

        # 活性化関数適用
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        # セル・隠れ状態の更新
        c_t = f * c_prev + i * g
        h_t = o * self.tanh(c_t)

        return h_t, c_t


class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(LSTMLayer, self).__init__()

        # 入力層サイズと隠れ層サイズを保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM セルのインスタンスの作成
        # OddNN の版の双方向と残差接続はこのタスクでは使わないので省略
        self.cell = LSTMCell(self.input_dim, self.hidden_dim)

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ, 形状は (系列長, バッチ, 特徴量)
        h_0: torch.Tensor
            初期化された隠れ状態
        c_0: torch.Tensor
            初期化されたセル

        Returns
        ----------
        outputs: torch.Tensor
            各時刻の隠れ状態, 形状は (系列長, バッチ, hidden_dim)
        h_t: torch.Tensor
            最後の隠れ状態
        c_t: torch.Tensor
            最後のセル
        """
        # 時刻を 1 つずつ進めながら隠れ状態を更新
        h_t, c_t = h_0, c_0
        outputs = []
        for t in range(x.size(0)):
            h_t, c_t = self.cell(x[t], h_t, c_t)
            outputs.append(h_t.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        return outputs, h_t, c_t


class OddLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        # 親クラスのコンストラクタを呼び出す
        super(OddLSTM, self).__init__()

        # 入力層サイズ, 隠れ層サイズ, 層数を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 全体の構造を作成
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            in_dim = self.input_dim if layer == 0 else self.hidden_dim
            self.layers.append(LSTMLayer(in_dim, self.hidden_dim))

    def forward(
        self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ, 形状は (系列長, バッチ, 特徴量)
        h_0: Optional[torch.Tensor] = None
            全ての層で共通の初期の隠れ状態, 省略した場合は 0

        Returns
        ----------
        outputs: torch.Tensor
            最終層の各時刻の隠れ状態, 形状は (系列長, バッチ, hidden_dim)
        """
        # 初期状態の隠れ状態とセルを定義
        batch_size = x.size(1)
        if h_0 is None:
            h_0 = torch.zeros(
                batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
            )
        c_0 = torch.zeros_like(h_0)

        # 層ごとに処理
        layer_input = x
        for layer in self.layers:
            layer_input, _, _ = layer(layer_input, h_0, c_0)
        outputs = layer_input

        return outputs


class WeatherForecaster(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int,
        output_dim: int, lstm_type: str
    ):
        # 親クラスのコンストラクタを呼び出す
        super(WeatherForecaster, self).__init__()

        # 入力次元数, 隠れ層の次元数, 層数, 出力次元数, LSTM の種類を保存
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm_type = lstm_type

        # torch は nn.LSTM, oddnn は OddNN の自作 LSTM を定義
        if self.lstm_type == "torch":
            self.lstm = nn.LSTM(
                self.input_dim, self.hidden_dim, num_layers=self.num_layers,
                batch_first=True
            )
        elif self.lstm_type == "oddnn":
            self.lstm = OddLSTM(
                self.input_dim, self.hidden_dim, self.num_layers
            )
        else:
            raise ValueError(
                f"model.lstm_type は torch か oddnn にしてください: {lstm_type}"
            )

        # 最後の時刻の隠れ状態から予測する出力層の定義
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            過去の気象要素, 形状は (バッチ, 日数, 特徴量)

        Returns
        ----------
        pred: torch.Tensor
            翌日の最高気温と最低気温 (標準化した値), 形状は (バッチ, 2)
        """
        # LSTM で系列を処理し, 最後の日の隠れ状態を取り出し
        if self.lstm_type == "torch":
            outputs, _ = self.lstm(x)
            last = outputs[:, -1]
        else:
            outputs = self.lstm(x.transpose(0, 1))
            last = outputs[-1]

        # 出力層で翌日の値を予測
        pred = self.fc(last)

        return pred
