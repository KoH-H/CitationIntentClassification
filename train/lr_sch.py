# -*- coding: utf-8 -*-
import torch
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
# https://zhuanlan.zhihu.com/p/99568607


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,  # 表示在该区间内学习率保持恒定
        gamma=0.1,  # 学习率下降的幅度，值越小下降幅度越厉害
        warmup_factor=0.1 / 3,  # 指的是模型运行的一开始的学习率，然后逐渐上升到optimize中初始定义的学习率
        warmup_epochs=5,   # 学习率上升曲线的陡峭程度
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):  # 区间要递增
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):  # constant 标量
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        # self.base_lrs = torch.from_numpy(np.repeat(0.0001, 84))
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs  # -0.2
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha # 1.2 / 3 -
        # print(self.base_lrs)  # self.base_lrs 【0.001，.... 0.001】 len = 84
        # lr = [
        #     base_lr  # base_lr是曲线的顶点 由定义optimizer时的lr决定
        #     * warmup_factor
        #     * self.gamma ** bisect_right(self.milestones, self.last_epoch)
        #     for base_lr in self.base_lrs
        # ]
        # plt.plot(50, lr)
        return [
            base_lr  # base_lr是曲线的顶点
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]