# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CS_ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(
        type='CompressiveNeck',
        in_channels=512,  # 输入通道数
        sample=12,  # 输出通道数 
        kernel_size=7,  # kernel size设为7
        threshold=0.05,
        warmup_steps=5000,
    ),
    #neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
