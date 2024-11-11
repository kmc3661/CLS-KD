# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from Fackbook, Deit
# {haozhiwei1, jianyuan.guo}@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

from types import MethodType

import torch
from torch import nn as nn
from torch.nn import functional as F

def register_forward(model, model_name, args, no_proj):
    
    # Teacher models
    if model_name.split('_')[1] == 'base':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    elif model_name.split('_')[0] == 'tnt':
        model.forward_features = MethodType(tnt_forward_features, model)
        model.forward = MethodType(tnt_forward, model)
        
    # Student models
    elif model_name.split('_')[1] == 'tiny' or 'small':
        if no_proj:
            if args.dim_match == 'projector' or 'both':
                model.forward_features = MethodType(no_proj_forward_features, model)
                model.forward = MethodType(proj_vit_forward, model)
            elif args.dim_match == 'relation':
                model.forward_features = MethodType(vit_forward_features, model)
                model.forward = MethodType(vit_forward, model)
            else:
                raise RuntimeError(f'Not defined customized method forward for model {args.dim_match}')
        else:
            if args.dim_match == 'projector' or 'both':
                model.forward_features = MethodType(proj_forward_features, model)
                model.forward = MethodType(proj_vit_forward, model)
            elif args.dim_match == 'relation':
                model.forward_features = MethodType(vit_forward_features, model)
                model.forward = MethodType(vit_forward, model)
            else:
                raise RuntimeError(f'Not defined customized method forward for model {args.dim_match}')
    
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')

# deit & vit
def vit_forward_features(self, x, require_feat: bool = False):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    # x = self.blocks(x)
    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)

    x = self.norm(x)
    if require_feat:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), block_outs
        else:
            return (x[:, 0], x[:, 1]), block_outs
    else:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

def proj_forward_features(self, x, require_feat: bool = False):
    
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    # x = self.blocks(x)
    
    block_outs = []
    cls_s = []
    projector = self.projector
    
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)
        cls_i = block_outs[i][:,:1,:]
        cls_i = F.normalize(cls_i, dim=-1)
        cls_s.append(cls_i)
    
    cls_s = torch.stack(cls_s, dim=0)  
    proj_cls_s = projector(cls_s)
    
    x = self.norm(x)
    if require_feat:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), block_outs, proj_cls_s
        else:
            return (x[:, 0], x[:, 1]), block_outs
    else:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        
def no_proj_forward_features(self, x, require_feat: bool = False):
    
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    # x = self.blocks(x)
    
    block_outs = []
    cls_s = []
    # projector = self.projector
    
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)
        cls_i = block_outs[i][:,:1,:]
        cls_i = F.normalize(cls_i, dim=-1)
        cls_s.append(cls_i)

    cls_s = torch.stack(cls_s, dim=0)  
    # proj_cls_s = projector(cls_s)
    

    x = self.norm(x)
    if require_feat:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), block_outs, cls_s
        else:
            return (x[:, 0], x[:, 1]), block_outs
    else:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

def vit_forward(self, x, require_feat: bool = True):
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x = outs[0]
        block_outs = outs[-1]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return (x, x_dist), block_outs
            else:
                return (x + x_dist) / 2, block_outs
        else:
            x = self.head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    
def proj_vit_forward(self, x, require_feat: bool = True):
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x = outs[0]
       
        
        block_outs = outs[1]
        proj_cls_s = outs[2]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return (x, x_dist), block_outs
            else:
                return (x + x_dist) / 2, block_outs
        else:
            x = self.head(x)
        return x, block_outs, proj_cls_s
    else:
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
#tnt

def tnt_forward_features(self, x, require_feat: bool = False):
    
    B = x.shape[0]
    pixel_embed = self.pixel_embed(x, self.pixel_pos)
        
    patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    patch_embed = patch_embed + self.patch_pos
    patch_embed = self.pos_drop(patch_embed)
    
    block_outs = []
    
    for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)
            block_outs.append(patch_embed)
    

    patch_embed = self.norm(patch_embed)
    
    if require_feat:
        return patch_embed[:, 0], block_outs
    else:
        return patch_embed[:, 0]


def tnt_forward(self, x, require_feat: bool = True):
    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x = outs[0]
        block_outs = outs[-1]   
        x = self.head(x)
        
        return x, block_outs
    else:
        x = self.forward_features(x)   
        x = self.head(x)
        return x


# cait
def cait_forward_features(self, x, require_feat: bool = False):
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)

    x = x + self.pos_embed
    x = self.pos_drop(x)

    block_outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        block_outs.append(x)

    for i, blk in enumerate(self.blocks_token_only):
        cls_tokens = blk(x, cls_tokens)

    x = torch.cat((cls_tokens, x), dim=1)

    x = self.norm(x)
    if require_feat:
        return x[:, 0], block_outs
    else:
        return x[:, 0]


def cait_forward(self, x, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_features(x, require_feat=True)
        x = self.head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.head(x)
        return x
