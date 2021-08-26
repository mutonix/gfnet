import numpy as np
from torch.utils import data
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets

def construct_dataset(path, batch_size):
    '''
    ImageFolder方法根据文件夹命名label，可在下面迭代器中以label.data返回类别标签
    按随机顺序读取图片
    :param path:
    :param batch_size:
    :return:
    '''
    traindir = path + "/train"
    valdir = path + "/val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪成224个像素
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
            pin_memory=True)

    return train_loader, val_loader

def get_dataloader(data_path, batch_size, eval_batch_size, device, num_workers):

    """random data for testing"""
    # train_set = torch.randn((100, 3, 224, 224)).to(device)
    # train_labels = torch.randint(0, 9, (100,)).to(device)
    # test_set = torch.randn((32, 3, 224, 224)).to(device)
    # test_labels = torch.randint(0, 9, (32,)).to(device)

    # train_ds = TensorDataset(train_set, train_labels)
    # test_ds = TensorDataset(test_set, test_labels)
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=True, num_workers=num_workers)
   
    train_dl, test_dl = construct_dataset(data_path, batch_size=batch_size)
    print("************ Dataset loaded ************")

    return train_dl, test_dl
    
if __name__ == '__main__':
    pass