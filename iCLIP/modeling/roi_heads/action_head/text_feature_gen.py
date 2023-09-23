from torch import nn
from iCLIP.modeling import registry
import torch
import numpy as np
import os

import sys
sys.path.append('clip')
from clip import clip

from PIL import Image
import time
from .cct import CrossFrameCommunicationTransformer
from .MIT import do_mit

image_resolution = 50176
vision_patch_size = 14
vision_width = 64
vision_heads = vision_width // 64
vision_layers = 1
visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=512
        ).to('cuda')

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)
        return super().forward(x)


@registry.TEXT_FEATURE_GENERATOR.register("CLIPencoder")
class CLIPencoder(nn.Module):
    def __init__(self, cfg, actionlist, actiondict, actiontoken, device):
        super(CLIPencoder, self).__init__()

        # for text encoder
        self.actionlist = actionlist
        self.actiondict = actiondict
        self.actiontoken = actiontoken

        self.device = device
        self.clipmodel, self.preprocess = clip.load('ViT-B/16', device=self.device, jit=False, return_intermediate_text_feature=0)

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        self.hidden_size = 512
        self.prefix = cfg.MODEL.ROI_ACTION_HEAD.PREFIX_LEN
        self.postfix = cfg.MODEL.ROI_ACTION_HEAD.POSTFIX_LEN

        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.01)

        # sin
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, 512))


    # if we wanna change prompt, there is the main part to be modified
    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist), 77)  

        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])
            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)
        #print(self.text_embedding[0][1])

    def generate_person_feature(self,image_paths,proposals,write_txt = False):
        images = []
        if proposals is None:
            return None

        for i in range(len(image_paths)):
            path = image_paths[i]
            image = Image.open(path).convert("RGB")
            

            box_num = len(proposals[i])
            for j in range(box_num):
                bbox = proposals[i].bbox
                
                x1 = round(bbox[j][0].item())
                y1 = round(bbox[j][1].item())
                x2 = round(bbox[j][2].item())
                y2 = round(bbox[j][3].item())

                if write_txt:
                    print(path,x1,y1,x2,y2, file=f)

                # crop person or object
                wanted = image.crop((x1, y1, x2, y2))
                images.append(self.preprocess(wanted))


        image_input = torch.tensor(np.stack(images)).to(self.device)
        wanted_features = self.clipmodel.encode_image(image_input).float()
        wanted_features = wanted_features.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        #f.close()
        return wanted_features
    
    def generate_object_feature(self,image_paths,proposals,objects,cal_iou = False,write_txt = False):
        images = []
        if objects is None:
            return None

        for i in range(len(image_paths)):
            path = image_paths[i]
            image = Image.open(path).convert("RGB")
            

            box_num = len(objects[i])
            keep_num = 0
            for j in range(box_num):
                bbox = objects[i].bbox
                
                x1 = round(bbox[j][0].item())
                y1 = round(bbox[j][1].item())
                x2 = round(bbox[j][2].item())
                y2 = round(bbox[j][3].item())

                if cal_iou:
                    keep = False
                    person_box = proposals[i].bbox
                    for p in range(len(proposals[i])):
                        x1p = round(person_box[p][0].item())
                        y1p = round(person_box[p][1].item())
                        x2p = round(person_box[p][2].item())
                        y2p = round(person_box[p][3].item())

                        x_left = max(x1, x1p)
                        y_top = max(y1, y1p)
                        x_right = min(x2, x2p)
                        y_bottom = min(y2, y2p)

                        # object box and person box iou > 0
                        if not (x_right < x_left or y_bottom < y_top):
                            keep = True
                            keep_num += 1
                            break
                    
                    if not keep:
                        continue


                if write_txt:
                    print(path,x1,y1,x2,y2, file=f)

                # crop person or object
                wanted = image.crop((x1, y1, x2, y2))
                images.append(self.preprocess(wanted))

            if cal_iou:
                objects[i].add_field('keep_num',keep_num)
            else:
                objects[i].add_field('keep_num',len(objects[i]))

         # to-do : object may be 0
        if len(images) == 0:
            return torch.zeros(0, 512, 1, 1, 1, dtype=torch.float32, device="cuda")

        image_input = torch.tensor(np.stack(images)).to(self.device)
        wanted_features = self.clipmodel.encode_image(image_input).float()
        wanted_features = wanted_features.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        #f.close()
        return wanted_features


    def generate_image_feature(self,image_paths):
        images = []
        
        for i in range(len(image_paths)):
            path = image_paths[i]
        
            image = Image.open(path).convert("RGB")

            images.append(self.preprocess(image))
        
        image_input = torch.tensor(np.stack(images)).to(self.device)
        image_features = self.clipmodel.encode_image(image_input).float()
        image_features = image_features.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        return image_features
    
    def generate_prompt_iFeature(self,image_paths,proposals):
        images = []

        for i in range(len(image_paths)):
            path = image_paths[i]
            image = Image.open(path).convert("RGB")

            box_num = len(proposals[i])
            for _ in range(box_num):
                images.append(self.preprocess(image))
            
        image_input = torch.tensor(np.stack(images)).to(self.device)
        prompt_iFeature = self.clipmodel.encode_image(image_input).float()

        return prompt_iFeature

    def generate_memory_image_feature(self,image_paths):
        images = []
        
        for i in range(len(image_paths)):
            path = image_paths[i]
            image = Image.open(path).convert("RGB")

            images.append(self.preprocess(image))

        image_input = torch.tensor(np.stack(images)).to(self.device)
        image_features = self.clipmodel.encode_image(image_input).float()
        image_features = image_features.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        return image_features

    # with input image_paths
    def forward(self, image_paths, proposals, objects):

        # replace_text_embedding at every iter
        # self.replace_text_embedding(self.actionlist)

        # encode text
        # tFeature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken)

        # webber original tfeature
        actiontoken = clip.tokenize([a for a in self.actionlist]).cuda()
        tFeature = self.clipmodel.encode_text_original(actiontoken).float()
        # encode image
        iFeature = self.generate_image_feature(image_paths)

        #take person feature
        person_feature = self.generate_person_feature(image_paths,proposals)

        #take object feature
        object_feature = self.generate_object_feature(image_paths,proposals,objects,True)

        prompt_iFeature = self.generate_prompt_iFeature(image_paths,proposals)
        
        return tFeature, iFeature, person_feature, object_feature, prompt_iFeature

def make_text_feature_generator(cfg, actionlist, actiondict, actiontoken, device):
    func = registry.TEXT_FEATURE_GENERATOR[cfg.MODEL.ROI_ACTION_HEAD.TEXT_FEATURE_GENERATOR]
    return func(cfg, actionlist, actiondict, actiontoken, device)
