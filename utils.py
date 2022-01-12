import yaml
import argparse
import torch
import os
import shutil
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
from torchvision.utils import save_image

def get_config(config):
    with open(config, 'r') as content:
        return yaml.load(content)
    
def get_arguments():
    parser = argparse.ArgumentParser()
    # basic setting:
    parser.add_argument('--config', type=str, default='configs/textureless_COCO.yaml')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--manualseed', type=int, help='set seed', default=None)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', action='store_true', help='whether resume from the provided model')
    parser.add_argument('--job-name', type=str, default='')
    parser.add_argument('--savedir', type=str, default='', help='path to save')
    parser.add_argument('--net-savedir', type=str, default='', help='path to save')
    parser.add_argument('--eval', type=str, default='', help='path to evaluate the model')
    parser.add_argument('--func', type=str, default='main', help='which function to be used')
    parser.add_argument('--aug_num', type=int, default=100, help='number of images to be augmented in folder A')
    parser.add_argument('--pretrained_path', type=str, default='', help='the path for loading encoder')

    opts = parser.parse_args()
    return opts


class Single_Style_data(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, get_single_index=-1):
        self.img_dir = img_dir
        self.transforms = transform
        self.get_single_index = get_single_index
        self.names = self.get_all_img_names()

    def get_all_img_names(self):
        """ You should implement this method
        list all self.img_dir's images, stored in self.names
        and get each image' label, stored in self.labels
        """
        if self.get_single_index > -1:
            name = os.listdir(self.img_dir)[self.get_single_index]
        else:
            name = os.listdir(self.img_dir)[0]
        names = []
        for _ in range(len(self.transforms)):
            names.append(name)
        return names

    def __getitem__(self, index):
        name = self.names[index]
        fpath = os.path.join(self.img_dir, name)
        img = Image.open(fpath).convert('RGB')
        if self.transforms is not None:
            img = self.transforms[index](img)
        return img

    def __len__(self):
        return len(self.names)


def get_singleM_dataloaders(config, data_dir, shuffle=False):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    data_transforms = []

    data_size = 288

    num = 10
    for _ in range(num - 1):
        choice = np.random.randint(num)
        if choice == 0:
            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.Resize([data_size, data_size]), 
                transforms.ToTensor(),
                normalize,
            ])
            data_transforms.append(data_transform)
        elif choice == 1:
            big_data_size = int(data_size * 1.1)
            data_transform = transforms.Compose([
                transforms.Resize([big_data_size, big_data_size]), 
                transforms.CenterCrop(data_size),
                transforms.ToTensor(),
                normalize,
            ])
            data_transforms.append(data_transform)
        else:
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(data_size, scale=(0.8, 1)),
                transforms.ToTensor(),
                normalize,
            ])
            data_transforms.append(data_transform)

    data_transform = transforms.Compose([
            transforms.Resize([data_size, data_size]), 
            transforms.ToTensor(),
            normalize,
        ])
    data_transforms.append(data_transform) 

    print('Data loading: {} ...'.format(data_dir))
    dataset = Single_Style_data(data_dir, transform=data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle, drop_last=True, num_workers=config['num_workers'])

    return data_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.num_batches = num_batches

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def display_internet(self, batch, inner_it):
        num_batches = self.num_batches
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        entries = [self.prefix + '[' + str(inner_it) + '/' + str(batch) + '/' + fmt.format(num_batches) + ']']
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def write_display(display_images, display_size, saveroot, epoch, it):
    image_tensor = torch.cat([images[:display_size] for images in display_images], 0)
    image_grid = make_grid(image_tensor.data, nrow=display_size, padding=0, normalize=True)
    writepath = os.path.join(saveroot, 'epoch{}_{}.png'.format(epoch, it))
    save_image(image_grid, writepath)
    shutil.copyfile(writepath, os.path.join(saveroot, 'current.png'))
    return True
