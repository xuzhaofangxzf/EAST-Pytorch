import torch
import os
import shutil

def save_checkpoint(state, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = 'epoch_' + str(state['epoch']) + '_checkpoint.pth.tar'
    filefn = os.path.join(checkpoint_dir, filename)
    torch.save(state, filefn)

    if state['is_best']:
        src = filefn
        dst = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        shutil.copy(src, dst)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count