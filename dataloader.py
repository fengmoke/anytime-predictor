import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True, download=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False, download=True,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.data == 'caltech256':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        train_set = datasets.Caltech256(args.data_root, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        trans,
                                        normalize
                                      ]))
        val_set = datasets.Caltech256(args.data_root, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        trans,
                                        normalize
                                    ]))
    elif args.data == 'ImageNet':
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    # Har dataset
    elif args.data == 'UCI':
        data_feature_train = np.load(r'data(har)\UCI\x_train.npy')
        data_label_train = np.load(r'data(har)\UCI\y_train.npy')
        data_feature_test = np.load(r'data(har)\UCI\x_test.npy')
        data_label_test = np.load(r'data(har)\UCI\y_test.npy')
        data_feature_val = np.load(r'data(har)\UCI\x_valid.npy')
        data_label_val = np.load(r'data(har)\UCI\y_valid.npy')
        train_feature_tensor = torch.from_numpy(data_feature_train).float().reshape(data_feature_train.shape[0], 1, data_feature_train.shape[1], data_feature_train.shape[2])
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float().reshape(data_feature_test.shape[0], 1, data_feature_test.shape[1], data_feature_test.shape[2])
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float().reshape(data_feature_val.shape[0], 1, data_feature_val.shape[1], data_feature_val.shape[2])
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    elif args.data == 'UniMib':
        data_feature_train = np.load(r'data(har)\UniMib\x_train.npy')
        data_label_train = np.load(r'data(har)\UniMib\y_train.npy')
        if args.attack_mode is None:
            data_feature_test = np.load(r'data(har)\UniMib\x_test.npy')
            data_label_test = np.load(r'data(har)\UniMib\y_test.npy')
        else:
            data_feature_test = np.load(fr'samples\UniMib\UniMib_mynet_cnn\ours_linf_{args.attack_mode}_data.npy')
            data_label_test = np.load(fr'samples\UniMib\UniMib_mynet_cnn\ours_linf_{args.attack_mode}_label.npy')
            data_feature_test = np.squeeze(data_feature_test)
            data_label_test = np.squeeze(data_label_test)
        data_feature_val = np.load(r'data(har)\UniMib\x_valid.npy')
        data_label_val = np.load(r'data(har)\UniMib\y_valid.npy')
        train_feature_tensor = torch.from_numpy(data_feature_train).float().reshape(data_feature_train.shape[0], 1, data_feature_train.shape[1], data_feature_train.shape[2])
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float().reshape(data_feature_test.shape[0], 1, data_feature_test.shape[1], data_feature_test.shape[2])
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float().reshape(data_feature_val.shape[0], 1, data_feature_val.shape[1], data_feature_val.shape[2])
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    elif args.data == 'Wisdm':
        data_feature_train = np.load(r'data(har)\Wisdm\x_train.npy')
        data_label_train = np.load(r'data(har)\Wisdm\y_train.npy')
        data_feature_test = np.load(r'data(har)\Wisdm\x_test.npy')
        data_label_test = np.load(r'data(har)\Wisdm\y_test.npy')
        data_feature_val = np.load(r'data(har)\Wisdm\x_valid.npy')
        data_label_val = np.load(r'data(har)\Wisdm\y_valid.npy')
        train_feature_tensor = torch.from_numpy(data_feature_train).float().reshape(data_feature_train.shape[0], 1, data_feature_train.shape[1], data_feature_train.shape[2])
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float().reshape(data_feature_test.shape[0], 1, data_feature_test.shape[1], data_feature_test.shape[2])
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float().reshape(data_feature_val.shape[0], 1, data_feature_val.shape[1], data_feature_val.shape[2])
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    elif args.data == 'Pampa2':
        data_feature_train = np.load(r'data(har)\Pampa2\x_train.npy')
        data_label_train = np.load(r'data(har)\Pampa2\y_train.npy')
        data_feature_test = np.load(r'data(har)\Pampa2\x_test.npy')
        data_label_test = np.load(r'data(har)\Pampa2\y_test.npy')
        data_feature_val = np.load(r'data(har)\Pampa2\x_valid.npy')
        data_label_val = np.load(r'data(har)\Pampa2\y_valid.npy')
        train_feature_tensor = torch.from_numpy(data_feature_train).float().reshape(data_feature_train.shape[0], 1, data_feature_train.shape[1], data_feature_train.shape[2])
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float().reshape(data_feature_test.shape[0], 1, data_feature_test.shape[1], data_feature_test.shape[2])
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float().reshape(data_feature_val.shape[0], 1, data_feature_val.shape[1], data_feature_val.shape[2])
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    elif args.data == 'Usc':
        data_feature_train = np.load(r'data(har)\Usc\x_train.npy')
        data_label_train = np.load(r'data(har)\Usc\y_train.npy')
        data_feature_test = np.load(r'data(har)\Usc\x_test.npy')
        data_label_test = np.load(r'data(har)\Usc\y_test.npy')
        data_feature_val = np.load(r'data(har)\Usc\x_valid.npy')
        data_label_val = np.load(r'data(har)\Usc\y_valid.npy')
        train_feature_tensor = torch.from_numpy(data_feature_train).float().reshape(data_feature_train.shape[0], 1, data_feature_train.shape[1], data_feature_train.shape[2])
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float().reshape(data_feature_test.shape[0], 1, data_feature_test.shape[1], data_feature_test.shape[2])
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float().reshape(data_feature_val.shape[0], 1, data_feature_val.shape[1], data_feature_val.shape[2])
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    print('Number of training samples: ', len(train_set))
    print('Number of test samples: ', len(test_set))
    print('Number of valid samples: ', len(val_set))
    if args.use_valid:
        train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)


    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
