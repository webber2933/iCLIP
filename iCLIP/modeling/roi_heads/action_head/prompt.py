from iCLIP.modeling import registry
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import sys
sys.path.append("../")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        # print('dim', dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.dim = dim

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        # text
        B_q, N, C = q.shape
        # visual
        B_k, C = k.shape
        # print(q.shape)
        # print(self.dim)
        # print('Q_PROJ', self.q_proj(q).shape)
        q = self.q_proj(q).reshape(B_q, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B_k, 1, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B_k, 1, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        #print(q.shape, k.shape, v.shape)
        #print(k.transpose(-2,-1).shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #print('attn.shape', attn.shape)

        x = (attn @ v).transpose(1, 2).reshape(B_k, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
    ):
        super().__init__()
        # print('d_model', d_model)
        # print('n_head', nhead)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, text, visual):
        # print('visual.shape', visual.shape)
        # print('text.shape', text.shape)
        q = k = v = self.norm1(text)
        # print('q.shape',q.shape)
        
        text = text + self.cross_attn(q, visual, visual)
        text = text + self.dropout(self.mlp(self.norm3(text)))
        return text

@registry.TEXT_FEATURE_GENERATOR.register("VideoSpecificPrompt")
class VideoSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1,):
        super().__init__()
        # print('layers', layers)
        self.norm = nn.LayerNorm(512)
        #print(PromptGeneratorLayer(512, 512//64))
        # print('-----------------')
        self.decoder = nn.ModuleList([PromptGeneratorLayer(512, 512//64) for _ in range(2)])
        self.alpha = nn.Parameter(torch.ones(512).to('cuda') * alpha)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        print('vs', visual.shape)
        B, C, = visual.shape
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        # print('text',text)
        return self.alpha * text

def make_video_prompt(cfg):
    func = registry.TEXT_FEATURE_GENERATOR[cfg.MODEL.ROI_ACTION_HEAD.VIDEO_PROMPT]
    return func(cfg)