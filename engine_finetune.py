# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import wandb
import pdb


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # targets = targets.unsqueeze(1)
        # pdb.set_trace()
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        original_min = 33
        original_max = 300

        # Define the new min and max range
        new_min = 0.669
        new_max = 1.669
        scaled_targets = (targets - original_min) / (original_max - original_min) * (new_max - new_min) + new_min
        scaled_targets = scaled_targets.float()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            x = outputs.squeeze()
            x = x.float()
            #pdb.set_trace()
            loss = criterion(x, scaled_targets)
            #print("loss is ;", loss)
        #pdb.set_trace()
        x_val = float(x.mean().item())
        loss_value = loss.item()
        mae_loss = torch.nn.L1Loss()
        inverse_scaled_targets = (x - new_min) / (new_max - new_min) * (original_max - original_min) + original_min
        mae = mae_loss(inverse_scaled_targets, targets)
        
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        # import pdb
        # pdb.set_trace()
        

        wandb.log({"lr": max_lr, "train_loss": loss_value, "train mae": mae, 'x':x_val})

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            x = output.squeeze()
            x = x.float()
            loss = criterion(x, target)
        loss_value = loss.item()
        
        original_min = 33
        original_max = 300
        new_min = 0.669
        new_max = 1.669
        inverse_scaled_targets = (x - new_min) / (new_max - new_min) * (original_max - original_min) + original_min
        mae_loss = torch.nn.L1Loss()
        mae = mae_loss(inverse_scaled_targets, target)

        wandb.log({"eval_loss": loss_value, 'Val_MAE': mae})

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}