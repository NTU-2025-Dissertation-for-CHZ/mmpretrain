_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

visualizer = dict(
    type='Visualizer',  # 改为默认的 Visualizer
    vis_backends=[dict(type='LocalVisBackend')]
)