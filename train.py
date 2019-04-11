# coding: utf-8

import argparse
import os
import time
from data.icdar import collate_fn, ICDAR
from loss import LossFunc
from utils import save_checkpoint, AverageMeter
from models import East

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

def train(dataloader, model, criterion, optimizer, scheduler, use_cuda, epoch):
    model.train()

    losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    for i, (img, score_map, geo_map, training_mask) in enumerate(dataloader):
        data_time.update(time.time() - end)

        if use_cuda:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), \
                                                     geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss.item(), img.size(0))

        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f}'.format(
            epoch, i, len(dataloader), loss=losses))
        print('DataTime {data_time.avg:.4f} BatchTime {batch_time.avg:.4f}'.format(
            data_time=data_time, batch_time=batch_time))


def main():
    parser = argparse.ArgumentParser('EAST')
    parser.add_argument('dataset', metavar='DIR',
        help='train dataset dir')
    parser.add_argument('pretrain', metavar='PTH',
        help='pretrain model')
    parser.add_argument('-b', '--batch-size', type=int, default=14,
        help='batch size per GPU(default=14)')
    parser.add_argument('-l', '--lr', type=float, default=0.0001,
        help='lr(default=0.0001)')
    parser.add_argument('-wd', type=float, default=1e-5,
        help='weight decay(default=1e-5)')
    parser.add_argument('--epochs', type=int, default=100,
        help='epochs(default=100)')
    parser.add_argument('-j', '--num-workers', type=int, default=16,
        help='dataloader workers(default=16)')
    parser.add_argument('-s', '--input-size', type=int, default=512,
        help='input image size(default=512)' +
             'INPUT SIZE is the image size used by training,' +
             'it should be compatible with TEXT_SCALE')
    parser.add_argument('--text-scale', type=int, default=512,
        help='text_scale is the max text length EAST can detect,' +
             'its restricted by the receptive field of CNN' +
             'default=512')
    parser.add_argument('--min-text-size', type=int, default=10,
        help='min text size(default=10)')
    parser.add_argument('--gpus', type=str, default='0',
        help='gpu ids(default=0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint',
        help='checkpoint dir(default=./checkpoint)')

    args = parser.parse_args()

    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    CUDA_COUNT = torch.cuda.device_count()

    # trainset
    trainset = ICDAR(args.dataset, input_size=args.input_size, min_text_size=args.min_text_size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size*CUDA_COUNT,
                             shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    # model
    model = East(args.pretrain)
    model = nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # criterion
    criterion = LossFunc()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    # resume
    start_epoch = 0

    # step
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        train(trainloader, model, criterion, optimizer, scheduler, True, epoch)

        is_best = True
        state = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'is_best': is_best
            }

        save_checkpoint(state, args.checkpoint)


if __name__ == "__main__":
    main()
