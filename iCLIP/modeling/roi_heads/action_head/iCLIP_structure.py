from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import torch
import torch.nn as nn
from iCLIP.modeling import registry
from iCLIP.utils.IA_helper import has_memory, has_person, has_object, has_hand
import time

class InteractionUnit(nn.Module):
    def __init__(self, structure_config,
                 max_others=20,
                 temp_pos_len=-1,  # configs for memory feature
                 dropout=0.):
        super(InteractionUnit, self).__init__()
        self.dim_person = 512
        self.dim_other = 512
        self.dim_out = 512
        self.dim_inner = 512
        self.max_others = max_others
        self.scale_value = self.dim_inner ** (-0.5)
        # config for temporal position, only used for temporal interaction,
        self.temp_pos_len = temp_pos_len

        bias = not structure_config.NO_BIAS
        init_std = structure_config.CONV_INIT_STD

        self.query = nn.Conv3d(self.dim_person, self.dim_inner, 1, bias)
        init_layer(self.query, init_std, bias)

        self.key = nn.Conv3d(self.dim_other, self.dim_inner, 1, bias)
        init_layer(self.key, init_std, bias)

        self.value = nn.Conv3d(self.dim_other, self.dim_inner, 1, bias)
        init_layer(self.value, init_std, bias)

        self.out = nn.Conv3d(self.dim_inner, self.dim_out, 1, bias)
        if structure_config.USE_ZERO_INIT_CONV:
            out_init = 0
        else:
            out_init = init_std
        init_layer(self.out, out_init, bias)

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.use_ln = structure_config.LAYER_NORM

        if self.dim_person != self.dim_out:
            self.shortcut = nn.Conv3d(self.dim_person, self.dim_out, 1, bias)
            init_layer(self.shortcut, init_std, bias)
        else:
            self.shortcut = None

        if self.temp_pos_len > 0:
            self.temporal_position_k = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))
            self.temporal_position_v = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))

    def forward(self, person, others):
        """
        :param person: [n, channels, t, h, w]
        :param others: [n, num_other, channels, t, h, w]
        """
        device = person.device
        n, dim_person, t, h, w = person.size()
        _, max_others, dim_others, t_others, h_others, w_others = others.size()

        query_batch = person
        key = fuse_batch_num(others)  # [n*num_other, channels, t, h, w]

        query_batch = self.query(query_batch)
        key_batch = self.key(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others, w_others)
        value_batch = self.value(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others,
                                                        w_others)

        if self.temp_pos_len > 0:
            max_person_per_sec = max_others // self.temp_pos_len
            key_batch = key_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner, t_others,
                                                    h_others,
                                                    w_others)
            key_batch = key_batch + self.temporal_position_k
            key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                                    h_others, w_others)

            value_batch = value_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner,
                                                        t_others,
                                                        h_others, w_others)
            value_batch = value_batch + self.temporal_position_v
            value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                                        h_others, w_others)

        query_batch = query_batch.contiguous().view(n, self.dim_inner, -1).transpose(1, 2)  # [n, thw, dim_inner]
        key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        key_batch = key_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1).transpose(
            1, 2)

        qk = torch.bmm(query_batch, key_batch)  # n, thw, max_other * thw

        qk_sc = qk * self.scale_value

        weight = self.softmax(qk_sc)


        value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        value_batch = value_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1)
        out = torch.bmm(weight, value_batch)  # n, thw, dim_inner

        out = out.contiguous().view(n, t * h * w, -1)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n, self.dim_inner, t, h, w)

        if self.use_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = nn.LayerNorm([self.dim_inner, t, h, w], elementwise_affine=False).to(
                    device)
            out = self.layer_norm(out)

        out = self.relu(out)

        out = self.out(out)
        out = self.dropout(out)

        if self.shortcut:
            person = self.shortcut(person)

        out = out + person
        return out


class iCLIPStructure(nn.Module):
    def __init__(self, structure_cfg):
        super(iCLIPStructure, self).__init__()

        self.max_person = structure_cfg.MAX_PERSON
        self.max_object = structure_cfg.MAX_OBJECT
        self.max_keypoints = structure_cfg.MAX_KEYPOINTS
        self.mem_len = structure_cfg.LENGTH[0] + structure_cfg.LENGTH[1] # 60 # webber : memory take image feature
        self.mem_feature_len = self.mem_len * structure_cfg.MAX_PER_SEC # 60 * 5 = 300

        self.I_block_list = structure_cfg.I_BLOCK_LIST

        bias = not structure_cfg.NO_BIAS
        conv_init_std = structure_cfg.CONV_INIT_STD

        self.has_P = has_person(structure_cfg)
        self.has_O = has_object(structure_cfg)
        self.has_M = has_memory(structure_cfg)
        self.has_H = has_hand(structure_cfg)
            
                        
    def forward(self, person, person_boxes, obj_feature, object_boxes, keypoints_feature, keypoints_boxes, mem_feature, person_pooled, phase):
        # RGB stream
        if phase == "rgb":
            query, person_key, object_key, keypoints_key, mem_key = self._reduce_dim(person, person_boxes, obj_feature, object_boxes, keypoints_feature, keypoints_boxes,
                                                                    mem_feature, phase)

            return self._aggregate(person_boxes, query, person_key, object_key, keypoints_key, mem_key)            

    def _reduce_dim(self, person, person_boxes, obj_feature, object_boxes, keypoints_feature, keypoints_boxes, mem_feature, phase):
        query = person
        n = query.size(0)

        if self.has_P:
            person_key = person
        else:
            person_key = None

        if self.has_O and obj_feature != None:
            object_key = separate_roi_per_person(person_boxes, obj_feature, object_boxes,
                                                 self.max_object)
            object_key = fuse_batch_num(object_key)
            object_key = unfuse_batch_num(object_key, n, self.max_object)
        else:
            object_key = None



        keypoint_key = separate_roi_per_person(person_boxes, keypoints_feature, keypoints_boxes,
                                                self.max_keypoints, True)
        keypoint_key = fuse_batch_num(keypoint_key)
        keypoint_key = unfuse_batch_num(keypoint_key, n, self.max_keypoints)

        if self.has_M and mem_feature != None:
            mem_key = separate_batch_per_person(person_boxes, mem_feature)
            mem_key = fuse_batch_num(mem_key)
            mem_key = unfuse_batch_num(mem_key, n, self.mem_feature_len)
        else:
            mem_key = None

        return query, person_key, object_key, keypoint_key, mem_key

    def _aggregate(self, proposals, query, person_key, object_key, keypoints_key, mem_key):
        raise NotImplementedError

    def _make_interaction_block(self, block_type, block_name, structure_cfg):
        dropout = structure_cfg.DROPOUT
        temp_pos_len = -1
        if block_type == "P":
            max_others = self.max_person
        elif block_type == "O":
            max_others = self.max_object
        elif block_type == "H":
            max_others = self.max_keypoints
        elif block_type == "M":
            max_others = self.mem_feature_len
            if structure_cfg.TEMPORAL_POSITION:
                temp_pos_len = self.mem_len
        else:
            raise KeyError("Unrecognized interaction block type '{}'!".format(block_type))

        I_block = InteractionUnit(structure_cfg, max_others, temp_pos_len=temp_pos_len, dropout=dropout)

        self.add_module(block_name, I_block)


@registry.INTERACTION_AGGREGATION_STRUCTURES.register("serial")
class SerialiCLIPStructure(iCLIPStructure):
    # webber : we use this
    def __init__(self, structure_cfg):
        super(SerialiCLIPStructure, self).__init__(structure_cfg)

        block_count = dict()
        for idx, block_type in enumerate(self.I_block_list):
            block_count[block_type] = block_count.get(block_type, 0) + 1
            name = block_type + "_block_{}".format(block_count[block_type])
            self._make_interaction_block(block_type, name, structure_cfg)

    def _aggregate(self, person_boxes, query, person_key, object_key, keypoints_key, mem_key):
        block_count = dict()
        for idx, block_type in enumerate(self.I_block_list):
            block_count[block_type] = block_count.get(block_type, 0) + 1
            name = block_type + "_block_{}".format(block_count[block_type])
            I_block = getattr(self, name)
            if block_type == "P":
                person_key = separate_roi_per_person(person_boxes, person_key, person_boxes, self.max_person, )
                query = I_block(query, person_key)
            if block_type == "O":
                if object_key is not None:
                    query = I_block(query, object_key)
            
            elif block_type == "H":
                query = I_block(query, keypoints_key)
            
            elif block_type == "M":
                query = I_block(query, mem_key)
            person_key = query

        return query

def separate_roi_per_person(proposals, things, other_proposals, max_things, flag = False):
    """
    :param things: [n2, c, t, h, w]
    :param proposals:
    :param max_things:
    :return [n, max_other, c, t, h, w]
    """
    res = []
    _, c, t, h, w = things.size()
    device = things.device
    index = 0
    for i, (person_box, other_box) in enumerate(zip(proposals, other_proposals)):
        person_num = len(person_box)

        if flag == True:
            other_num = 1
        else:
            other_num = len(other_box)

        if other_box.has_field('keep_num'):
            other_num = other_box.get_field('keep_num')

        tmp = torch.zeros((person_num, max_things, c, t, h, w), device=device)
        if other_num > max_things:
            idx = torch.randperm(other_num)[:max_things]
            tmp[:, :max_things] = things[index:index + other_num][idx]
        else:
            tmp[:, :other_num] = things[index:index + other_num]

        res.append(tmp)
        index += other_num
    features = torch.cat(res, dim=0)
    return features


def separate_batch_per_person(proposals, things):
    """

    :param things: [b, max_others, c, t, h, w]
    :return [n, max_others, c, t, h, w]
    """
    res = []
    b, max_others, c, t, h, w = things.size()
    device = things.device
    for i, person_box in enumerate(proposals):
        person_num = len(person_box)
        tmp = torch.zeros((person_num, max_others, c, t, h, w), device=device)
        tmp[:] = things[i]
        res.append(tmp)
    return torch.cat(res, dim=0)


def fuse_batch_num(things):
    n, number, c, t, h, w = things.size()
    return things.contiguous().view(-1, c, t, h, w)


def unfuse_batch_num(things, batch_size, num):
    assert things.size(0) == batch_size * num, "dimension should matches"
    _, c, t, h, w = things.size()
    return things.contiguous().view(batch_size, num, c, t, h, w)


def init_layer(layer, init_std, bias):
    if init_std == 0:
        nn.init.constant_(layer.weight, 0)
    else:
        nn.init.normal_(layer.weight, std=init_std)
    if bias:
        nn.init.constant_(layer.bias, 0)


def make_iCLIP_structure(cfg):
    func = registry.INTERACTION_AGGREGATION_STRUCTURES[
        cfg.MODEL.ICLIP_STRUCTURE.STRUCTURE
    ]
    return func(cfg.MODEL.ICLIP_STRUCTURE)