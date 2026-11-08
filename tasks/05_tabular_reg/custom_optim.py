from typing import Callable, Iterable, Optional

import torch
from torch import optim


class Lion(optim.Optimizer):
    def __init__(
        self, params: Iterable, lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0
    ):
        # 引数の範囲の確認
        if lr <= 0.0:
            raise ValueError(f"lr は正の値にしてください: {lr}")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(
                f"betas は 0 以上 1 未満の 2 つの値にしてください: {betas}"
            )
        if weight_decay < 0.0:
            raise ValueError(
                f"weight_decay は 0 以上にしてください: {weight_decay}"
            )

        # 親クラスのコンストラクタを呼び出す
        defaults = {
            "lr": lr, "betas": tuple(betas), "weight_decay": weight_decay
        }
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable] = None
    ) -> Optional[torch.Tensor]:
        """
        パラメータを 1 回更新する関数

        Parameters
        ----------
        closure: Optional[Callable] = None
            損失を計算し直す関数, 使う場合だけ渡す

        Returns
        ----------
        loss: Optional[torch.Tensor]
            closure を渡した場合はその損失, 渡さない場合は None
        """
        # closure が渡された場合は損失を計算し直す
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # パラメータグループごとに更新
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 勾配の移動平均の初期化
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # 重みの減衰
                p.mul_(1.0 - lr * group["weight_decay"])

                # 移動平均と勾配を混ぜた値の符号だけで更新
                update = exp_avg * beta1 + p.grad * (1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # 移動平均の更新
                exp_avg.mul_(beta2).add_(p.grad, alpha=1.0 - beta2)

        return loss
