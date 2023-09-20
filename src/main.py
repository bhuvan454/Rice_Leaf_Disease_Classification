# Path: src/main.py


import os
import sys
import argparse
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as tf

from models.model import ResNet18, ResNet34, ResNet50
from data.CustomDataloader import CustomDataLoader
from data.CustomDataset import CustomDataset
from train_evaluation.train import train_model
from train_evaluation.eval import eval_model

import multiprocessing as mp

from utils.utils import set_logger


def parse_args():

    args = argparse.ArgumentParser(description='Leaf Disease Classification with PyTorch')

    args.add_argument('--data_dir',type = str, help='path to dataset')
    args.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    args.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 100)')
    args.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

    args.add_argument('--model', type=str, default='resnet18', help='model name, (default: resnet18) in [resnet18, resnet34, resnet50]')
    args.add_argument('--pretrained', type=bool, default=True, help='use pretrained model (default: True)')

    args.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args.add_argument('--save_interval', type=int, default=10, help='how many epochs to wait before saving model weights')

    args.add_argument('--save_dir', type=str, default='../models', help='path to save weights')
    args.add_argument('--log_dir', type=str, default='../logs', help='path to save logs')

    args.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args.add_argument('--gpu', type=int, default=0, help='GPU number to use (default: 0)')

    return args.parse_args()

# python profiler from standard library to profile the code

def main():
    
        args = parse_args()

        set_logger(args.log_dir, 'train.log')

        logging.info('Start training...')
    
        # set seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
    
        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Using device: {}'.format(device))


        # set dataset
        # define transforms
        transform = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor() ])
        dataset = CustomDataset(args.data_dir, transform=transform)
        logging.info('Dataset size: {}'.format(len(dataset)))
        train_size,val_size,test_size = int(0.8*len(dataset)),int(0.1*len(dataset)),int(0.1*len(dataset))
        train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])

        logging.info('Train dataset size: {}'.format(len(train_dataset)))
        logging.info('Val dataset size: {}'.format(len(val_dataset)))
        logging.info('Test dataset size: {}'.format(len(test_dataset)))

        # set dataloader
        num_workers = mp.cpu_count()
        logging.info('Number of workers: {}'.format(num_workers))
        with mp.Pool(processes=num_workers) as pool:
            train_dataloader = CustomDataLoader(train_dataset, args.batch_size, shuffle=True, num_workers= num_workers)
            val_dataloader = CustomDataLoader(val_dataset, args.batch_size, shuffle=False, num_workers= num_workers)
            test_dataloader = CustomDataLoader(test_dataset, args.batch_size, shuffle=False, num_workers= num_workers)

        logging.info('Train dataloader size: {}'.format(len(train_dataloader)))
        logging.info('Val dataloader size: {}'.format(len(val_dataloader)))
        logging.info('Test dataloader size: {}'.format(len(test_dataloader)))

        #set model
        if args.model == 'resnet18':
            model = ResNet18(num_classes=dataset.get_num_classes(), pretrained=args.pretrained)
        elif args.model == 'resnet34':
            model = ResNet34( num_classes=dataset.get_num_classes(), pretrained=args.pretrained).get_model()
        elif args.model == 'resnet50':
            model = ResNet50( num_classes=dataset.get_num_classes(), pretrained=args.pretrained).get_model()
        else:
            raise ValueError('Model name should be in [resnet18, resnet34, resnet50]')
    
        model.to(device)
    
        # set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
        # set loss function
        criterion = nn.CrossEntropyLoss()


        # def train_model(model,device, train_loader, criterion, optimizer, epochs, log_interval,save_model_path):
    
        train_loss = train_model(model,device, train_dataloader, criterion, optimizer, args.epochs, args.log_interval,save_model_path=None)
        val_loss,val_accuracies = eval_model(model,device, val_dataloader, criterion, args.log_interval)
        test_loss,test_accuracies = eval_model(model,device, test_dataloader, criterion,args.log_interval)

        logging.info('Train loss: {}'.format(train_loss))
        logging.info('Val loss: {}'.format(val_loss))
        logging.info('Test loss: {}'.format(test_loss))

        logging.info('Val accuracies: {}'.format(val_accuracies))
        logging.info('Test accuracies: {}'.format(test_accuracies))

        # close pool
        pool.close()
        pool.join()

        logging.info('End training...')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
