_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/whubds1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

data_root = 'D:/Yang/Py/Dataset/WHUBDS1'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root))

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg, num_classes=2),
    auxiliary_head=dict(norm_cfg=norm_cfg, num_classes=2))
