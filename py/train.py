import os
import argparse
import numpy as np
import pandas as pd

from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from steel_dataset import SteelDataset
from utils import get_transforms, get_model, get_loss, do_kaggle_metric

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='Res34Unetv4', type=str, help='Model version')
parser.add_argument('--n_classes', default=4, type=int, help='Number of classes')
parser.add_argument('--loss', default='BCEWithLogitsLoss', type=str, help='Loss function')
parser.add_argument('--fine_size', default=256, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=13, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=14, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=300, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=5, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(fine_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


train_df = pd.read_csv(os.path.join('..', 'input', 'preprocessed_train.csv'))

def test(test_loader, model, criterion):
    running_loss = 0.0
    predicts = []
    truths = []

    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, masks)

        predicts.append(F.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()
    epoch_loss = running_loss / val_data.__len__()
    return epoch_loss, precision

def train(train_loader, model, criterion):
    running_loss = 0.0
    data_size = train_data.__len__()

    model.train()
    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            logits= model(inputs)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / data_size
    return epoch_loss


if __name__ == '__main__':
    scheduler_step = args.epoch // args.snapshot
    # Get Model
    steel = get_model(args.model, args.n_classes)
    steel.to(device)
 
    # Setup optimizer
    optimizer = torch.optim.SGD(
                            steel.parameters(), 
                            lr=args.max_lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                        optimizer, 
                                                        scheduler_step, 
                                                        args.min_lr)

    # Load data
    train_idx, valid_idx, _, _ = train_test_split(train_df.index, train_df['split_label'], test_size=0.2, random_state=43)


    train_data = SteelDataset(
                        train_df.iloc[train_idx], 
                        mode='train', 
                        fine_size=args.fine_size, 
                        pad_left=args.pad_left, 
                        pad_right=args.pad_right,
                        transforms=get_transforms())
    train_loader = DataLoader(
                        train_data,
                        shuffle=RandomSampler(train_data),
                        batch_size=args.batch_size,
                        num_workers=0, #cpu_count(),
                        pin_memory=True)
    
    val_data = SteelDataset(
                        train_df.iloc[valid_idx], 
                        mode='valid', 
                        fine_size=args.fine_size, 
                        pad_left=args.pad_left,
                        pad_right=args.pad_right,
                        transforms=get_transforms())
    val_loader = DataLoader(
                        val_data,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=0, #cpu_count(),
                        pin_memory=True)

    num_snapshot = 0
    best_acc = 0

    criterion = get_loss(args.loss)
    for epoch in range(args.epoch):
        train_loss = train(train_loader, steel, criterion)
        val_loss, accuracy = test(val_loader, steel, criterion)
        lr_scheduler.step()

        if accuracy > best_acc:
            best_acc = accuracy
            best_param = steel.state_dict()

        if (epoch + 1) % scheduler_step == 0:
            torch.save(best_param, args.save_weight + args.weight_name + str(idx) + str(num_snapshot) + '.pth')
            optimizer = torch.optim.SGD(steel.parameters(), lr=args.max_lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)
            num_snapshot += 1
            best_acc = 0

        print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch + 1, train_loss,
                                                                                            val_loss, accuracy))
