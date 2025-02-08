# 在自定义配置文件中添加以下内容
_base_ = [
    '../_base_/models/resnet18_cs.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]


# 使用CosineAnnealingLR
param_scheduler = [
    {
        'type': 'CosineAnnealingLR',
        'by_epoch': True,
        'begin': 0,
        'end': 100,  # 假设总共训练100个epoch
        'eta_min': 0,
        'convert_to_iter_based': True,
    }
]
