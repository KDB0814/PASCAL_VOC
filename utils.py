import torchvision.transforms.functional as F
import random
import torch


class Random_processing(object):
    def __init__(self, mean=False, std=False):
        '''
        mean : defalut=False, format : [0.0, 0.0, 0.0]
        std : default=False, format : [0.0, 0.0, 0.0]
        '''
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        img, lbl = samples['image'], samples['label']
        lbl = torch.where(lbl == 255, torch.tensor(21, dtype=lbl.dtype), lbl)

        img = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BILINEAR)
        lbl = F.resize(lbl, [224, 224], interpolation=F.InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img, lbl = F.hflip(img), F.hflip(lbl)

        if self.mean and self.std:
            img = F.normalize(img, self.mean, self.std)

        sample = {'image': img, 'label': lbl}

        return sample
