# import `opts` first to set gpus
from opts import opt

from utils import set_seed
set_seed(opt.seed)

import os
import time
import shutil
from os.path import join, exists

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from loss import *
from utils import *
from model import get_model
from test import test_accuracy
from dataloader import get_dataloader

scaler = GradScaler()

model = get_model(opt, 'Model')

sim_loss = SimilarityLoss(
    rho=opt.loss_rho,
    gamma=opt.loss_gamma,
    reduction=opt.loss_reduction,
)

optimizer = optim.AdamW(
    [{'params': model.parameters()},],
    lr=opt.base_lr,
    weight_decay=opt.weight_decay,
)

if opt.resume_path:
    model, resume_epoch = load_from_ckpt(model,  opt.resume_path)
else:
    resume_epoch = -1
    if exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)

save_configs(opt)
logger = get_logger(opt.save_dir)
writer = SummaryWriter(opt.save_dir)

dataloader_train = get_dataloader('train', opt, 'RMOT_Dataset', show=True)
dataloader_test = get_dataloader('test', opt, 'RMOT_Dataset', show=False)

print(
    '========== Training (Text-Guided {}) =========='
        .format('ON' if opt.kum_mode else 'OFF')
)
iteration = 0
logger.info('Start training!')

for epoch in range(resume_epoch + 1, opt.max_epoch):
    model.train()
    BATCH_TIME = AverageMeter('Time', ':6.3f')
    LOSS = AverageMeter('Loss', ':.4e')
    lr = get_lr(opt, epoch)
    set_lr(optimizer, lr)
    meters = [BATCH_TIME, LOSS]
    PROGRESS = ProgressMeter(
        num_batches=len(dataloader_train),
        meters=meters,
        prefix="Epoch [{}/{}] ".format(epoch, opt.max_epoch),
        lr=lr
    )
    end = time.time()
    # train
    for batch_idx, data in enumerate(dataloader_train):
        # load
        expression = data['target_expressions']
        expression_ids = data['expression_id'].cuda()
        # forward
        inputs = dict(
            local_img=data['cropped_images'].cuda(),
            global_img=data['global_images'].cuda(),
            exp=tokenize(expression).cuda(),
        )
        targets = data['target_labels'].view(-1).cuda()
        logits = model(inputs, epoch)['logits']
        # loss
        loss = sim_loss(logits, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # write
        BATCH_TIME.update(time.time() - end)
        LOSS.update(loss.item(), opt.train_bs)
        end = time.time()
        iteration += 1
        writer.add_scalar('Train/LR', lr, iteration)
        writer.add_scalar('Loss/', loss.item(), iteration)
        if (batch_idx + 1) % opt.train_print_freq == 0:
            PROGRESS.display(batch_idx)
            logger.info(
                'Epoch:[{}/{}] [{}/{}] Loss:{:.5f}'
                    .format(epoch, opt.max_epoch, batch_idx, len(dataloader_train), loss.item())
            )

    # test
    torch.cuda.empty_cache()
    if (epoch + 1) % opt.eval_frequency == 0:
        p, r = test_accuracy(model, dataloader_test)
        log_info = 'precision: {:.2f}% / recall: {:.2f}%'.format(p, r)
        logger.info(log_info)
        print(log_info)
    if (epoch + 1) % opt.save_frequency == 0:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer,
            'epoch': epoch,
        }
        torch.save(state_dict, join(opt.save_dir, f'epoch{epoch}.pth'))
    torch.cuda.empty_cache()


logger.info('Finish training!')