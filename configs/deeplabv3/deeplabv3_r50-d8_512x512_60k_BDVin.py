_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/whubds1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

data_root = 'D:/Yang/Py/Dataset/BDVin'

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
