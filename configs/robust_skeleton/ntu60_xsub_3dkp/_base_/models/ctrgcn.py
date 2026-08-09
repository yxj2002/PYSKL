model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
