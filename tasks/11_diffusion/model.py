import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(TimeEmbedding, self).__init__()

        # 埋め込みの次元数を保存
        self.dim = dim

        # 正弦波の埋め込みを変換する MLP の定義
        self.fc1 = nn.Linear(self.dim, 4 * self.dim)
        self.fc2 = nn.Linear(4 * self.dim, 4 * self.dim)

        # 活性化関数の定義
        self.silu = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        t: torch.Tensor
            時刻, 形状は (バッチ,)

        Returns
        ----------
        emb: torch.Tensor
            時刻の埋め込み, 形状は (バッチ, 4 * dim)
        """
        # 周波数の違う正弦波で時刻を表現
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        # MLP で変換
        emb = self.fc2(self.silu(self.fc1(emb)))

        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_dim: int):
        # 親クラスのコンストラクタを呼び出す
        super(ResBlock, self).__init__()

        # 入力と出力のチャネル数, 時刻の埋め込みの次元数を保存
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_dim = t_dim

        # 正規化層と畳み込み層を 2 組と, 時刻の埋め込みを足す線形層の定義
        self.norm1 = nn.GroupNorm(8, self.in_channels)
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.time_proj = nn.Linear(self.t_dim, self.out_channels)
        self.norm2 = nn.GroupNorm(8, self.out_channels)
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )

        # チャネル数が変わる場合は残差接続の側も 1×1 の畳み込みで合わせる
        if self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=1
            )
        else:
            self.skip = nn.Identity()

        # 活性化関数の定義
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力の特徴マップ
        t_emb: torch.Tensor
            時刻の埋め込み

        Returns
        ----------
        out: torch.Tensor
            残差接続を足した特徴マップ
        """
        # 正規化, 活性化, 畳み込みの後に時刻の埋め込みを各チャネルに加算
        h = self.conv1(self.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]

        # もう 1 組の正規化, 活性化, 畳み込みの後に残差接続
        h = self.conv2(self.silu(self.norm2(h)))
        out = h + self.skip(x)

        return out


class UNet(nn.Module):
    def __init__(self, base_channels: int, channel_mults: list[int]):
        # 親クラスのコンストラクタを呼び出す
        super(UNet, self).__init__()

        # チャネル数の基準と, 解像度ごとの倍率を保存
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        channels = [self.base_channels * m for m in self.channel_mults]
        t_dim = 4 * self.base_channels

        # 時刻の埋め込みと入力の畳み込み層の定義
        self.time_embedding = TimeEmbedding(self.base_channels)
        self.conv_in = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)

        # 解像度を半分にしながらチャネル数を増やす部分の定義
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = channels[0]
        for i, ch in enumerate(channels):
            self.down_blocks.append(ResBlock(prev, ch, t_dim))
            prev = ch
            if i < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.downsamples.append(nn.Identity())

        # 最も低い解像度の部分の定義
        self.mid_block = ResBlock(prev, prev, t_dim)

        # 解像度を 2 倍にしながら, 同じ解像度の特徴をつないで戻す部分の定義
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(channels))):
            ch = channels[i]
            self.up_blocks.append(ResBlock(prev + ch, ch, t_dim))
            prev = ch
            if i > 0:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        ch, ch, kernel_size=4, stride=2, padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())

        # 出力の正規化層と畳み込み層の定義
        self.norm_out = nn.GroupNorm(8, channels[0])
        self.conv_out = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

        # 活性化関数の定義
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            ノイズを加えた画像, 形状は (バッチ, 1, 32, 32)
        t: torch.Tensor
            時刻, 形状は (バッチ,)

        Returns
        ----------
        eps: torch.Tensor
            加えられたノイズの予測, 形状は (バッチ, 1, 32, 32)
        """
        # 時刻の埋め込みと入力の畳み込み
        t_emb = self.time_embedding(t)
        h = self.conv_in(x)

        # 解像度を下げながら, 戻すときにつなぐ特徴を保存
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # 最も低い解像度での変換
        h = self.mid_block(h, t_emb)

        # 同じ解像度の特徴をつなぎながら解像度を戻す
        for block, upsample in zip(self.up_blocks, self.upsamples):
            h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = upsample(h)

        # ノイズの予測を出力
        eps = self.conv_out(self.silu(self.norm_out(h)))

        return eps


class Diffusion:
    def __init__(
        self, timesteps: int, beta_start: float, beta_end: float,
        device: torch.device
    ):
        # 時刻の数とデバイスを保存
        self.timesteps = timesteps
        self.device = device

        # 線形に増えるノイズの分散と, その累積の積を計算
        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        元の画像に時刻 t までのノイズを一度に加える関数

        Parameters
        ----------
        x0: torch.Tensor
            元の画像
        t: torch.Tensor
            時刻, 形状は (バッチ,)
        noise: torch.Tensor
            標準正規分布のノイズ

        Returns
        ----------
        x_t: torch.Tensor
            ノイズを加えた画像
        """
        # 累積の積から元の画像とノイズの混ぜ方を決定
        a = self.alphas_cumprod[t][:, None, None, None]
        x_t = a.sqrt() * x0 + (1.0 - a).sqrt() * noise

        return x_t

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, size: int) -> torch.Tensor:
        """
        ノイズから祖先サンプリングで画像を生成する関数

        Parameters
        ----------
        model: nn.Module
            ノイズを予測するモデル
        n: int
            生成する枚数
        size: int
            画像の縦横の大きさ

        Returns
        ----------
        x: torch.Tensor
            生成した画像, 画素値はおよそ -1 から 1
        """
        # 標準正規分布のノイズから開始
        x = torch.randn(n, 1, size, size, device=self.device)

        # 時刻を 1 つずつ戻しながらノイズを除去
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            mean = (
                x - (1.0 - alpha) / (1.0 - alpha_cumprod).sqrt() * eps
            ) / alpha.sqrt()
            if i > 0:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x
