# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import os
import logging



def train_one_epoch(model: torch.nn.Module,teacher_model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args= None, writer= None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.no_grad():
            teacher_outputs, block_outs_t = teacher_model(samples)

        with torch.cuda.amp.autocast(enabled=False):
        # with torch.cuda.amp.autocast():
            if args.dim_match == 'relation' or 'projector' or 'both':
                
                outputs = model(samples)
            else:
                print('dim_match is not defined')
            
            if args.dim_match == 'both':
                
                if args.manifold == True:
                    base_loss, distillation_loss, loss_cls_proj, loss_cls_relation, loss_atn, loss_mf_sample, loss_mf_patch, loss_mf_rand = criterion(samples, outputs, targets,teacher_outputs, block_outs_t)
                    loss = base_loss + distillation_loss + loss_cls_proj + loss_cls_relation + loss_atn + loss_mf_sample + loss_mf_patch + loss_mf_rand
                
                else:
                    base_loss, distillation_loss, loss_cls_proj, loss_cls_relation, loss_atn = criterion(samples, outputs, targets,teacher_outputs, block_outs_t)
                    loss = base_loss + distillation_loss + loss_cls_proj + loss_cls_relation + loss_atn
                    
            
            else:
                
                if args.manifold == True:
                    base_loss, distillation_loss, loss_cls, loss_atn, loss_mf_sample, loss_mf_patch, loss_mf_rand = criterion(samples, outputs, targets,teacher_outputs, block_outs_t)
                    loss = base_loss + distillation_loss + loss_cls + loss_atn + loss_mf_sample + loss_mf_patch + loss_mf_rand
                elif args.norm_distill == True:
                    base_loss, distillation_loss, loss_cls, loss_atn, loss_norm = criterion(samples, outputs, targets,teacher_outputs, block_outs_t)
                    loss = base_loss + distillation_loss + loss_cls + loss_atn + loss_norm
                else:
                    base_loss, distillation_loss, loss_cls, loss_atn = criterion(samples, outputs, targets,teacher_outputs, block_outs_t)
                    loss = base_loss + distillation_loss + loss_cls + loss_atn
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if args.dim_match == 'both':
            metric_logger.update(loss=loss_value)
            metric_logger.update(base_loss=base_loss.item())
            metric_logger.update(loss_dist=distillation_loss.item())
            metric_logger.update(loss_cls_proj=loss_cls_proj.item())
            metric_logger.update(loss_cls_relation=loss_cls_relation.item())
            metric_logger.update(loss_atn=loss_atn.item())     
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if args.manifold == True:
                metric_logger.update(loss_mf_sample=loss_mf_sample.item())
                metric_logger.update(loss_mf_patch=loss_mf_patch.item())
                metric_logger.update(loss_mf_rand=loss_mf_rand.item())
        else:
            metric_logger.update(loss=loss_value)
            metric_logger.update(base_loss=base_loss.item())
            metric_logger.update(loss_dist=distillation_loss.item())
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(loss_atn=loss_atn.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if args.manifold == True:
                metric_logger.update(loss_mf_sample=loss_mf_sample.item())
                metric_logger.update(loss_mf_patch=loss_mf_patch.item())
                metric_logger.update(loss_mf_rand=loss_mf_rand.item())
            elif args.norm_distill == True:
                metric_logger.update(loss_norm=loss_norm.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    writer.add_scalar("Loss/train", loss_value, epoch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, require_feat=False)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
