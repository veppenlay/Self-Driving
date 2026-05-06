#!/usr/bin/env python
# -*- encoding: utf-8 -*-

class AverageMeter:
    """
    用于跟踪某个指标（如损失或精度）的平均值、当前值和总和。
    """
    def __init__(self, name='Metric', fmt=':6.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """重置统计数据"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """更新平均值

        Args:
            val (float): 当前 batch 的指标值
            n (int): 当前 batch 样本数
        """
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """便于打印日志"""
        fmtstr = '{name} {val' + self.fmt + '} (avg:{avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
