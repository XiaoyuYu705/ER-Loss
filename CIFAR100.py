from torchvision import transforms, datasets
from torchvision.datasets import CIFAR100
import os
import torch
from PIL import Image
import scipy.io as scio

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']#wen jian kuo zhan ming

def ImageNetData(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }#shu ju zhuan huan 

    image_datasets = {}

    image_datasets['train'] = CIFAR100(root='./ImageData',train=True,transform=data_transforms['train'],download=True)
    image_datasets['val'] = CIFAR100(root='./ImageData',train=False,transform=data_transforms['val'],download=True)

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=8) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes
