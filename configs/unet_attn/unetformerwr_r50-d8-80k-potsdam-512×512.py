_base_ = [
    '../_base_/models/unetformerwr_r50-d8.py',
    '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
train_dataloader = dict(batch_size=4)
norm_cfg = dict(type='BN')
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
train_cfg = dict(val_interval=4000)
default_hooks = dict(checkpoint=dict(interval=4000))
