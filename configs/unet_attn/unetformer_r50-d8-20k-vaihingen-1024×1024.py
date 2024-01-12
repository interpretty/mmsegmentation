_base_ = [
    '../_base_/models/unetformer_r50-d8.py',
    '../_base_/datasets/vaihingen_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
norm_cfg = dict(type='BN')
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
