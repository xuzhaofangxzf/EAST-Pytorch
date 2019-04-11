# EAST-Pytorch
EAST的Pytorch版本, paper:https://arxiv.org/abs/1704.03155

## Thanks
[argman/EAST](https://github.com/argman/EAST)
[songdejia](https://github.com/songdejia/EAST)

## Requirements
Python2 or Python3
Pytorch > 0.4.0
numpy
shapely

## Imagenet pretrain model
[resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)

## Dataset
TODO

## Train
```
➜ python train.py -h
usage: EAST [-h] [-b BATCH_SIZE] [-l LR] [-wd WD] [--epochs EPOCHS]
            [-j NUM_WORKERS] [-s INPUT_SIZE] [--text-scale TEXT_SCALE]
            [--min-text-size MIN_TEXT_SIZE] [--gpus GPUS]
            [--checkpoint CHECKPOINT]
            DIR PTH

positional arguments:
  DIR                   train dataset dir
  PTH                   pretrain model

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size per GPU(default=14)
  -l LR, --lr LR        lr(default=0.0001)
  -wd WD                weight decay(default=1e-5)
  --epochs EPOCHS       epochs(default=100)
  -j NUM_WORKERS, --num-workers NUM_WORKERS
                        dataloader workers(default=16)
  -s INPUT_SIZE, --input-size INPUT_SIZE
                        input image size(default=512)INPUT SIZE is the image
                        size used by training,it should be compatible with
                        TEXT_SCALE
  --text-scale TEXT_SCALE
                        text_scale is the max text length EAST can detect,its
                        restricted by the receptive field of CNNdefault=512
  --min-text-size MIN_TEXT_SIZE
                        min text size(default=10)
  --gpus GPUS           gpu ids(default=0
  --checkpoint CHECKPOINT
                        checkpoint dir(default=./checkpoint)
```

## Test
TODO
