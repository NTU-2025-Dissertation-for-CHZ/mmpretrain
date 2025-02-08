# mmpretrain/models/necks/compressive_neck.py

from ..builder import NECKS
from mmpretrain.registry import MODELS
from ..CompressiveSensingDownsampler.CompressiveDownsampler import CompressiveDownsampler
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CompressiveNeck(nn.Module ):
    def __init__(self,
                 in_channels,
                 sample,
                 kernel_size=7,
                 threshold=0.05,
                 warmup_steps=5000):
        super().__init__()  # 调用nn.Module的初始化
        
        self.pooler = CompressiveDownsampler(  # 作为成员变量而不是父类
            in_channels=in_channels,
            sample= sample,
            kernel_size=kernel_size,
            threshold=threshold,
            warmup_steps=warmup_steps
        )

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]  # 取最后一个特征图
        
        x = self.pooler(inputs)
        
        if self.training:
            self.pooler.update_matrices()
                
        # 明确使用 squeeze 时要小心
        x = x.view(x.size(0), -1)  # 关键是保留 x.size(0)
                
        return (x,)  # 将输出包装成 tuple

    def train(self, mode=True):
        super(CompressiveNeck, self).train(mode)
