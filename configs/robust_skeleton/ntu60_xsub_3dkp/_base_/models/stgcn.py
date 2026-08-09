model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
