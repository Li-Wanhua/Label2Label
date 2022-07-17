import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str


def batch_trainer(epoch, model, train_loader, criterion, optimizer, args):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[1]['lr']
    beta = torch.distributions.beta.Beta(args.beta, args.beta) if args.beta != 0 else 0
    # for step, (imgs, gt_label, imgname) in enumerate(train_loader):
    for step, batch in enumerate(train_loader):
        gt_label = batch['labels'].float().cuda()
        images = batch['image'].float().cuda()
        mask = batch['mask'].float().cuda()
        unk_mask = custom_replace(mask,1,0,0)
        mask_in = mask.clone()

        batch_time = time.time()
        # imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits1, train_logits2, mask, attns = model(images)
        train_loss = args.qratio * criterion(train_logits1, gt_label) + args.lratio * criterion(train_logits2, gt_label, mask)
        
        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits2)
        preds_probs.append(train_probs.detach().cpu().numpy())
        
        log_interval = 1000
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                  f'train_loss:{loss_meter.val:.4f}')
            # print(attns[0][0])
            

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


# @torch.no_grad()
def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            gt_label = batch['labels'].float().cuda()
            images = batch['image'].float().cuda()
            # mask = batch['mask'].float().cuda()
            # unk_mask = custom_replace(mask,1,0,0)
            # mask_in = mask.clone()
            # valid_logits,int_pred,attns = model(images,mask_in)
            # imgs = imgs.cuda()
            # gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            _, valid_logits, _, _ = model(images)
            # valid_logits = model(imgs)
            # valid_logits, _, _, _ = model(images)
            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs


def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res