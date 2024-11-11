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
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import linalg as LA


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.beta = args.distillation_beta
        self.w_sample = args.w_sample
        self.w_patch = args.w_patch
        self.w_rand = args.w_rand
        self.w_cls = args.w_cls
        self.w_atn = args.w_atn
        self.w_norm = args.w_norm
        self.w_proj = args.w_proj
        self.w_relation = args.w_relation
        self.K = args.K
        self.dim_match = args.dim_match
        self.manifold = args.manifold
        self.norm_distill = args.norm_distill
        self.last_w = args.last_w
        

    def forward(self, inputs, outputs, labels, teacher_outputs, block_outs_t):
        
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # only consider the case of [outputs, block_outs_s] or [(outputs, outputs_kd), block_outs_s]
        # i.e. 'require_feat' is always True when we compute loss
        block_outs_s = outputs[1]
        
        proj_cls_s = outputs[2]       
        
        if isinstance(outputs[0], torch.Tensor):
            outputs = outputs_kd = outputs[0]
        else:
            outputs, outputs_kd = outputs[0]
        

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss

        # don't backprop throught the teacher
        # with torch.no_grad():
        #     teacher_outputs, block_outs_t = self.teacher_model(inputs)
        
        

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss_base = (1 - self.alpha) * base_loss
        loss_dist = self.alpha * distillation_loss
        
        if self.manifold == True:         
            loss_mf_sample, loss_mf_patch, loss_mf_rand = mf_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                    self.layer_ids_t, self.K, self.w_sample, self.w_patch, self.w_rand,self.w_norm,self.last_w)
            loss_mf_sample = self.beta * loss_mf_sample
            loss_mf_patch = self.beta * loss_mf_patch
            loss_mf_rand = self.beta * loss_mf_rand
            
        if self.norm_distill == True:         
            loss_norm = norm_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                    self.layer_ids_t, self.w_norm,self.last_w)
        
        if self.dim_match == 'both':
            loss_cls_proj, loss_cls_relation = cls_token_loss(block_outs_s, block_outs_t, proj_cls_s, self.layer_ids_s,
                                  self.layer_ids_t, self.K, self.dim_match)
            loss_atn = atn_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                    self.layer_ids_t, self.K)
           
            if self.manifold == True: 
                return loss_base, loss_dist, self.w_proj*loss_cls_proj, self.w_relation*loss_cls_relation, self.w_atn*loss_atn, loss_mf_sample, loss_mf_patch, loss_mf_rand
        
            else:
                return loss_base, loss_dist, self.w_proj*loss_cls_proj, self.w_relation*loss_cls_relation, self.w_atn*loss_atn
            
        else:
            loss_cls,w_adap = cls_token_loss(block_outs_s, block_outs_t, proj_cls_s, self.layer_ids_s,
                                    self.layer_ids_t, self.K, self.dim_match,self.last_w)
        
            loss_atn = atn_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                    self.layer_ids_t, self.K,w_adap,self.last_w)
            
            if self.manifold == True: 
                return loss_base, loss_dist, self.w_cls*loss_cls, self.w_atn*loss_atn, loss_mf_sample, loss_mf_patch, loss_mf_rand
            elif self.norm_distill == True:
                return loss_base, loss_dist, self.w_cls*loss_cls, self.w_atn*loss_atn, loss_norm
            else:
                return loss_base, loss_dist, self.w_cls*loss_cls, self.w_atn*loss_atn
    

def mf_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K,w_sample, w_patch, w_rand,w_norm,last_w,max_patch_num=0):
    losses = [[], [], []]  # loss_mf_sample, loss_mf_patch, loss_mf_rand
    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]
        F_s = block_outs_s[id_s][:, extra_tk_num:, :]  # remove additional tokens
        F_t = block_outs_t[id_t]
        
        if max_patch_num > 0:
            F_s = merge(F_s, max_patch_num)
            F_t = merge(F_t, max_patch_num)

        loss_mf_patch, loss_mf_sample, loss_mf_rand = layer_mf_loss(
            F_s, F_t, K)
        losses[0].append(w_sample * loss_mf_sample)
        losses[1].append(w_patch * loss_mf_patch)
        losses[2].append(w_rand * loss_mf_rand)
        
    loss_mf_sample = sum(losses[0]) / len(losses[0])
    loss_mf_patch = sum(losses[1]) / len(losses[1])
    loss_mf_rand = sum(losses[2]) / len(losses[2])
    

    return loss_mf_sample, loss_mf_patch, loss_mf_rand

def norm_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, w_norm,last_w):
    
    losses = []
    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]       
        patch_s = block_outs_s[id_s][:, extra_tk_num:, :]
        patch_t = block_outs_t[id_t][:, extra_tk_num:, :]
        
        
        norm_s = LA.norm(patch_s,dim=-1)
        norm_t = LA.norm(patch_t,dim=-1)
        
        norm_s = F.normalize(norm_s, dim=-1)
        norm_t = F.normalize(norm_t, dim=-1)
        ####################################
        norm_diff = norm_t - norm_s
        loss_norm = (norm_diff * norm_diff).mean()
        
        losses.append(loss_norm)
    
    losses[len(losses)-1]*=last_w
    loss_norm = w_norm*sum(losses) / len(losses)
    
    return loss_norm

def atn_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K,w_adap,last_w):
    losses_atn = []  # loss_attention map
    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        
        extra_tk_num_s = 1 # remove cls token
        extra_tk_num_t = 1 # remove cls token
        
        patch_s = block_outs_s[id_s][:, extra_tk_num_s:, :]
        patch_t = block_outs_t[id_t][:, extra_tk_num_t:, :]
        
        ####################################
        cls_s = block_outs_s[id_s][:,:1,:] # 64, 1, 192, [# layer][B][# token][dim]
        cls_t = block_outs_t[id_t][:,:1,:] # 64, 1, 192
        loss_atn = layer_atn_loss(
            patch_s,patch_t,cls_s, cls_t)
        losses_atn.append(loss_atn)
    losses_atn_adap = []
    w_adap = []
    
    for i in range(len(losses_atn)):
        w_adap.append(losses_atn[i] / sum(losses_atn)*len(losses_atn))
        
        losses_atn_adap.append(losses_atn[i]*(w_adap[i]))
        
        if i == (len(losses_atn)-1):
            losses_atn_adap[i]*=last_w
    
    loss_atn = sum(losses_atn_adap) / len(losses_atn)
    
    return loss_atn

def layer_atn_loss(patch_s,patch_t,cls_s, cls_t):
    # normalize at feature dim
    cls_s = F.normalize(cls_s, dim=-1) # (64, 1, 192)
    cls_t = F.normalize(cls_t, dim=-1)
    patch_s = F.normalize(patch_s, dim=-1)# (64, 196, 192)
    patch_t = F.normalize(patch_t, dim=-1)

    # manifold loss among different patches (intra-sample)
    
    atn_s = cls_s.bmm(patch_s.transpose(-1,-2))
    atn_t = cls_t.bmm(patch_t.transpose(-1,-2))

    atn_diff = atn_t - atn_s
    loss_atn = (atn_diff * atn_diff).mean()

    return loss_atn

def cls_token_loss(block_outs_s, block_outs_t, proj_cls_s, layer_ids_s, layer_ids_t, K, dim_match,last_w):
    losses = []  # loss_cls_token
    proj_cls = []
    realtion_cls = []
    
    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]
        cls_s = block_outs_s[id_s][:,0,:] # 64, 192, [# layer][B][# token][dim]
        cls_t = block_outs_t[id_t][:,0,:] # 64, 768
        cls_s_proj = proj_cls_s[id_s]
        if dim_match == 'both':
            loss_proj_cls, loss_relation_cls = layer_cls_loss(
            cls_s, cls_t,cls_s_proj, K, dim_match)
            proj_cls.append(loss_proj_cls)
            realtion_cls.append(loss_relation_cls)
        else:
            loss_cls_token = layer_cls_loss(
                cls_s, cls_t,cls_s_proj, K, dim_match)
            losses.append(loss_cls_token)
    
    if dim_match == 'both':
        loss_cls_proj = sum(proj_cls) / len(proj_cls)
        loss_cls_relation = sum(realtion_cls) / len(realtion_cls)
        return loss_cls_proj, loss_cls_relation
    else:
        w_adap = []
        losses_adap=[]
        for i in range(len(losses)):
            w_adap.append(losses[i] / sum(losses)*len(losses))
            # w_adap[i]=w_adap[i]**2
            losses_adap.append(losses[i]*w_adap[i])
            if i == (len(losses)-1):
                losses_adap[i]*=last_w
        # loss_cls = sum(losses) / len(losses)
        loss_cls = sum(losses_adap) / len(losses)
        
        return loss_cls,w_adap

def layer_cls_loss(cls_s, cls_t, cls_s_proj, K, dim_match):
    # normalize at feature dim
    
    cls_s = F.normalize(cls_s, dim=-1)
    cls_t = F.normalize(cls_t, dim=-1)
    if dim_match == 'relation':
        # manifold loss among different images (inter-sample)
        
        M_s = cls_s.mm(cls_s.transpose(0,1))
        M_t = cls_t.mm(cls_t.transpose(0,1))

        M_diff = M_t - M_s
        loss_cls_token = (M_diff * M_diff).mean()       
        return loss_cls_token
    
    elif dim_match == 'projector':
        
        M_diff = cls_t - cls_s_proj.squeeze()
        loss_cls_token = (M_diff * M_diff).mean()       
        return loss_cls_token
    
    
    elif dim_match == 'both':
        M_diff = cls_t - cls_s_proj.squeeze()
        loss_proj_cls = (M_diff * M_diff).mean()
        
        M_s = cls_s.mm(cls_s.transpose(0,1))
        M_t = cls_t.mm(cls_t.transpose(0,1))

        M_diff = M_t - M_s
        loss_relation_cls = (M_diff * M_diff).mean()
        
        return loss_proj_cls, loss_relation_cls
    
    else: 
        print("Error! Please check the argument for dim-match")

def layer_mf_loss(F_s, F_t, K):
    # normalize at feature dim
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)

    # manifold loss among different patches (intra-sample)
    M_s = F_s.bmm(F_s.transpose(-1, -2))
    M_t = F_t.bmm(F_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_mf_patch = (M_diff * M_diff).mean()

    # manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_mf_sample = (M_diff * M_diff).mean()

    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    loss_mf_rand = (M_diff * M_diff).mean()

    return loss_mf_patch, loss_mf_sample, loss_mf_rand

def adap_layer_mf_loss(F_s, F_t, K,w_norm):
    # normalize at feature dim
    norm_t = LA.norm(F_t,dim=-1)#128,197
    norm_t = norm_t.detach().cpu().numpy()
    
    idx = [np.where(norm_t[i]>2.0*np.mean(norm_t[i])) for i in range(len(norm_t))]
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    

    # manifold loss among different patches (intra-sample)
    M_s = F_s.bmm(F_s.transpose(-1, -2))
    M_t = F_t.bmm(F_t.transpose(-1, -2))
    M_diff = M_t - M_s
    

    for i, current_idx in enumerate(idx):
        if current_idx[0].size > 0:
            M_diff[i][current_idx[0], :] *= w_norm
            M_diff[i][:, current_idx[0]] *= w_norm             
    
    loss_mf_patch = (M_diff * M_diff).mean()

    # manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_mf_sample = (M_diff * M_diff).mean()

    # manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    loss_mf_rand = (M_diff * M_diff).mean()

    return loss_mf_patch, loss_mf_sample, loss_mf_rand


def merge(x, max_patch_num=196):
    B, P, C = x.shape
    if P <= max_patch_num:
        return x
    n = int(P ** (1/2))  # original patch num at each dim
    m = int(max_patch_num ** (1/2))  # target patch num at each dim
    merge_num = n // m  # merge every (merge_num x merge_num) adjacent patches
    x = x.view(B, m, merge_num, m, merge_num, C)
    merged = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, m * m, -1)
    return merged
