# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    # neck=dict(type='GlobalAveragePooling'),
    neck=dict(
        type='CompressiveNeck',
        in_channels=512,  # 输入通道数
        out_channels=512*6,  # 输出通道数 
        kernel_size=4,  # kernel size设为7
        # 其他参数使用默认值
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512*6,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
