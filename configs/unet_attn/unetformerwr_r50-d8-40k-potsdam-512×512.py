_base_ = [
    '../_base_/models/unetformerwr_r50-d8.py',
    '../_base_/datasets/potsdam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
train_dataloader = dict(batch_size=4)
norm_cfg = dict(type='BN')
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
