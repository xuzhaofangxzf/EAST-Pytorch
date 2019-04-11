# coding: utf-8

import argparse
import os
from data.icdar import collate_fn, ICDAR

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

def main():
    parser = argparse.ArgumentParser('EAST')
    parser.add_argument('dataset', metavar='DIR',
        help='train dataset dir')
    parser.add_argument('-b', '--batch-size', type=int, default=14,
        help='batch size per GPU')
    parser.add_argument('-l', '--lr', type=float, default=0.0001,
        help='lr')
    parser.add_argument('--epochs', type=int, default=500,
        help='epochs')
    parser.add_argument('-j', '--num-workers', default=4,
        help='dataloader workers')
    parser.add_argument('-s', '--input-size', type=int, default=512,
        help='input image size(default 512)' +
             'INPUT SIZE is the image size used by training,' +
             'it should be compatible with TEXT_SCALE')
    parser.add_argument('--text-scale', type=int, default=512,
        help='text_scale is the max text length EAST can detect,' +
             'its restricted by the receptive field of CNN')
    parser.add_argument('--min-text-size', type=int, default=10,
        help='min text size(default 10)')
    parser.add_argument('--gpus', type=str, default='0',
        help='gpu ids')

    args = parser.parse_args()

    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    CUDA_COUNT = torch.cuda.device_count()

    # trainset
    trainset = ICDAR(args.dataset, input_size=args.input_size, min_text_size=args.min_text_size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size*CUDA_COUNT,
                             shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    # model
    pass

    # criterion

    # optimizer

    # step



if __name__ == "__main__":
    main()
