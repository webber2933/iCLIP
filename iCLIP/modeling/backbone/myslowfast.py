import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


import os
import sys
from collections import OrderedDict


from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6


def get_slow_model_cfg(cfg):
    backbone_strs = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[1:]
    error_msg = 'Model backbone {} is not supported.'.format(cfg.MODEL.BACKBONE.CONV_BODY)

    use_temp_convs_1 = [0]
    temp_strides_1 = [1]
    max_pool_stride_1 = 1

    use_temp_convs_2 = [0, 0, 0]
    temp_strides_2 = [1, 1, 1]

    use_temp_convs_3 = [0, 0, 0, 0]
    temp_strides_3 = [1, 1, 1, 1]

    use_temp_convs_5 = [1, 1, 1]
    temp_strides_5 = [1, 1, 1]

    slow_stride = cfg.INPUT.TAU
    avg_pool_stride = int(cfg.INPUT.FRAME_NUM / slow_stride)
    if backbone_strs[0] == 'Resnet50':
        block_config = (3, 4, 6, 3)

        use_temp_convs_4 = [1, 1, 1, 1, 1, 1]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
    elif backbone_strs[0] == 'Resnet101':
        block_config = (3, 4, 23, 3)

        use_temp_convs_4 = [1, ] * 23
        temp_strides_4 = [1, ] * 23
    else:
        raise KeyError(error_msg)

    if len(backbone_strs) > 1:
        raise KeyError(error_msg)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]
    pool_strides_set = [max_pool_stride_1, avg_pool_stride]
    return block_config, use_temp_convs_set, temp_strides_set, pool_strides_set


def get_fast_model_cfg(cfg):
    backbone_strs = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[1:]
    error_msg = 'Model backbone {} is not supported.'.format(cfg.MODEL.BACKBONE.CONV_BODY)

    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    max_pool_stride_1 = 1

    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]

    use_temp_convs_3 = [1, 1, 1, 1]
    temp_strides_3 = [1, 1, 1, 1]

    use_temp_convs_5 = [1, 1, 1]
    temp_strides_5 = [1, 1, 1]

    fast_stride = cfg.INPUT.TAU // cfg.INPUT.ALPHA
    avg_pool_stride = int(cfg.INPUT.FRAME_NUM / fast_stride)

    if backbone_strs[0] == 'Resnet50':
        block_config = (3, 4, 6, 3)

        use_temp_convs_4 = [1, 1, 1, 1, 1, 1]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
    elif backbone_strs[0] == 'Resnet101':
        block_config = (3, 4, 23, 3)

        use_temp_convs_4 = [1, ] * 23
        temp_strides_4 = [1, ] * 23
    else:
        raise KeyError(error_msg)

    if len(backbone_strs) > 1:
        raise KeyError(error_msg)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]
    pool_strides_set = [max_pool_stride_1, avg_pool_stride]
    return block_config, use_temp_convs_set, temp_strides_set, pool_strides_set

class LateralBlock(nn.Module):
    def __init__(self, conv_dim, alpha):
        super(LateralBlock, self).__init__()
        self.conv = nn.Conv3d(conv_dim, conv_dim * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                              padding=(2, 0, 0), bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        return out
