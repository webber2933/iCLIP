import torch

from .action_head.action_head import build_roi_action_head


class Combined3dROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(Combined3dROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):
        result, loss_action, loss_weight, accuracy_action = self.action(boxes, objects, keypoints, extras, part_forward)

        return result, loss_action, loss_weight, accuracy_action

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child,"c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map


def build_3d_roi_heads(cfg, actionlist, actiondict, actiontoken, device):
    roi_heads = []
    roi_heads.append(("action", build_roi_action_head(cfg, actionlist, actiondict, actiontoken, device)))

    if roi_heads:
        roi_heads = Combined3dROIHeads(cfg, roi_heads)

    return roi_heads
