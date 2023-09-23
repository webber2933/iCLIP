import torch
from torch import nn
from torch.nn import functional as F

from iCLIP.modeling import registry
from iCLIP.modeling.roi_heads.action_head.iCLIP_structure import make_iCLIP_structure
from iCLIP.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from iCLIP.utils.IA_helper import has_object, has_hand
from iCLIP.structures.bounding_box import BoxList

from iCLIP.modeling.roi_heads.action_head.pose_transformer import PoseTransformer

from iCLIP.modeling.roi_heads.action_head.text_feature_gen import CLIPencoder
import time

@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config

        if config.MODEL.ICLIP_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.MODEL.ICLIP_STRUCTURE.MAX_PER_SEC
            self.iCLIP_structure = make_iCLIP_structure(config)

        self.pose_transformer = PoseTransformer()
        self.clip = CLIPencoder(self.config,None,None,None,torch.device("cuda")) # webber : memory take image feature
        self.image_feature_pool = {} # webber : memory take image feature

    def forward(self, image_feature, person_feature, object_feature, proposals, objects=None, keypoints=None, extras={}, part_forward=-1):
        ia_active = hasattr(self, "iCLIP_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            if objects is None:
                object_pooled = None
            else:
                object_pooled = cat([box.get_field("pooled_feature") for box in objects])

            hands_pooled = image_feature

        else:
            person_pooled = person_feature
            object_pooled = object_feature
            hands_pooled = image_feature

        
        if object_pooled.shape[0] == 0:
            object_pooled = None

        x_after = person_pooled
        if ia_active:
            tsfmr = self.iCLIP_structure
            mem_len = self.config.MODEL.ICLIP_STRUCTURE.LENGTH
            mem_rate = self.config.MODEL.ICLIP_STRUCTURE.MEMORY_RATE
            use_penalty = self.config.MODEL.ICLIP_STRUCTURE.PENALTY
            # webber : memory take image feature
            memory_person = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                       self.max_feature_len_per_sec,
                                                                       person_pooled, proposals, use_penalty)
            # RGB stream
            ia_feature = self.iCLIP_structure(person_pooled, proposals, object_pooled, objects, hands_pooled, keypoints, memory_person, None, phase="rgb")
            x_after = ia_feature
        x_after = x_after.view(x_after.size(0), -1)

        return x_after, person_pooled, object_pooled, hands_pooled, None

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, movie_id, mem_ind, max_boxes, cur_loss, use_penalty)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, movie_id, mem_ind, max_boxes, cur_loss, use_penalty)
                                  for mem_ind in after_inds]
            #mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            mem_box_list = mem_box_list_before + mem_box_list_after # webber : memory take image feature
            mem_feature_list += [box_list.get_field("image_feature") # webber : memory take image feature
                                 if box_list is not None
                                 else torch.zeros(0, 512, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) # webber : memory take image feature
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, 512, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        #mem_pos = pad_sequence(mem_pos_list, max_boxes) # webber : memory take image feature
        #mem_pos = mem_pos.view(-1, person_per_seq, 4) # webber : memory take image feature
        #mem_pos = mem_pos.to(device) # webber : memory take image feature

        return mem_feature # webber : memory take image feature

    def check_fetch_mem_feature(self, movie_cache, movie_id, mem_ind, max_num, cur_loss, use_penalty):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)

        # webber : memory take image feature
        image_path = []
        image_root = "data/jhmdb/videos/"
        str_timestamp = str(mem_ind).zfill(5)
        # check whether this image feature exists
        key = movie_id + '/' + str_timestamp
        if key not in self.image_feature_pool:
            image_root = image_root + movie_id + '/' + str_timestamp + '.png'
            image_path.append(image_root)
            image_feature = self.clip.generate_memory_image_feature(image_path)
            self.image_feature_pool[key] = image_feature
        
        box_list.add_field("image_feature", self.image_feature_pool[key])
        ######################################

        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        return box_list

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")


def make_roi_action_feature_extractor(cfg):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
