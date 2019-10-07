# coding: utf-8
import argparse
import time
import shutil
import copy
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.optim
import torch.nn.functional as F
from torchvision.models import *
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from config.config import *
from networks.imageunet import init_network
from datasets.stage1_datasets import SteelDataset, BalanceClassSampler
from layers.loss_funcs.loss import *
from layers.scheduler import *
from utils.common_util import *
from utils.augment_util import *
from utils.albu_augment_util import *
from utils.log_util import Logger

loss_names = ['SymmetricLovaszLoss', 'BCEWithLogitsLoss', 'BCELoss', 'BCEDiceLoss']
split_types = ['split', 'cropv3_split', 'lung_split']
split_names = ['random_folds4', 'random_folds10']

parser = argparse.ArgumentParser(description='PyTorch Severstal Segmentation')
parser.add_argument('--out_dir', type=str, help='destination where trained network should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--arch', default='resnet3r', type=str, help='model architecture (default: resnet34)')
parser.add_argument('--loss', default='BCEWithLogitsLoss', choices=loss_names, type=str, help='loss function: ' + ' | '.join(loss_names) + ' (deafault: SymmetricLovaszLoss)')
parser.add_argument('--scheduler', default='Adam3', type=str, help='scheduler name')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run (default: 20)')
parser.add_argument('--img_size', default=(256, 1600), type=int, help='image size (default: (256, 1600))')
parser.add_argument('--batch_size', default=12, type=int, help='train mini-batch size (default: 12)')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--resume', default=None, type=str, help='name of the latest checkpoint (default: None)')
parser.add_argument('--crop_version', default=None, type=str, help='the cropped version (default: None)')
parser.add_argument('--train_transform', default='augment_default', type=str, help='train augmentation list (default: augement_default)')

def main():
    args = parser.parse_args()

    log_out_dir = opj(RESULT_DIR, 'logs', args.out_dir, f'fold{args.fold}')
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.train.txt'), mode='a')

    model_out_dir = opj(RESULT_DIR, 'models', args.out_dir, f'fold{args.fold}')
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(model_out_dir))
    if not ope(model_out_dir):
        os.makedirs(model_out_dir)

    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model = resnet34(pretrained=False, num_classes=5)

    # move network to gpu
    model = DataParallel(model)
    model.cuda()
    ema_model = None

    # define loss function (criterion)
    try:
        criterion = eval(args.loss)().cuda()
    except:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    start_epoch = 0
    best_epoch = 0
    best_dice = 0
    best_dice_arr = np.zeros(3)

    # define scheduler
    try:
        scheduler = eval(args.scheduler)()
    except:
        raise (RuntimeError("Scheduler {} not available!".format(args.scheduler)))
    optimizer = scheduler.schedule(model, start_epoch, args.epochs)[0]

    # optionally resume from a checkpoint
    if args.resume:
        model_fpath = os.path.join(model_out_dir, args.resume)
        if os.path.isfile(model_fpath):
            # load checkpoint weights and update model and optimizer
            log.write(">> Loading checkpoint:\n>> '{}'\n".format(model_fpath))

            checkpoint = torch.load(model_fpath)
            start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_dice_arr = checkpoint['best_dice_arr']
            best_dice = np.max(best_dice_arr)
            model.module.load_state_dict(checkpoint['state_dict'])

            optimizer_fpath = model_fpath.replace('.pth', '_optim.pth')
            if ope(optimizer_fpath):
                log.write(">> Loading checkpoint:\n>> '{}'\n".format(optimizer_fpath))
                optimizer.load_state_dict(torch.load(optimizer_fpath)['optimizer'])
            log.write(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})\n".format(model_fpath, checkpoint['epoch']))
        else:
            log.write(">> No checkpoint found at '{}'\n".format(model_fpath))

    # Data loading code
    train_transform = eval(args.train_transform)
    steel_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    steel_df['ImageId'], steel_df['ClassId'] = zip(*steel_df['ImageId_ClassId'].apply(lambda x: x.split('_')))
    steel_df = pd.pivot_table(steel_df, index='ImageId', columns='ClassId', values='EncodedPixels', aggfunc=lambda x: x, dropna=False)
    steel_df = steel_df.reset_index()
    steel_df.columns = [str(i) for i in steel_df.columns.values]
    steel_df['class_count'] = steel_df[['1', '2', '3', '4']].count(axis=1)
    steel_df['split_label'] = steel_df[['1', '2', '3', '4', 'class_count']].apply(lambda x: make_split_label(x), axis=1)
    steel_df['label'] = steel_df['split_label'].apply(lambda x: make_label(x))
    train_idx, valid_idx, _, _ = train_test_split(
                                            steel_df.index, 
                                            steel_df['split_label'], 
                                            test_size=0.2, 
                                            random_state=43)

    train_dataset = SteelDataset(
        steel_df.iloc[train_idx],
        img_size=args.img_size,
        mask_size=args.img_size,
        transform=train_transform,
        return_label=True,
        dataset='train',
    )
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_dataset = SteelDataset(
        steel_df.iloc[valid_idx],
        img_size=args.img_size,
        mask_size=args.img_size,
        transform=None,
        return_label=True,
        dataset='val',
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=max(int(args.batch_size // 2), 1),
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )

    log.write('** start training here! **\n')
    log.write('\n')
    log.write('epoch    iter      rate     | smooth_loss/dice | valid_loss/dice | best_epoch/best_score |  min \n')
    log.write('------------------------------------------------------------------------------------------------\n')
    start_epoch += 1
    for epoch in range(start_epoch, args.epochs + 1):
        end = time.time()

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        lr_list = scheduler.step(model, epoch, args.epochs)
        lr = lr_list[0]

        # train for one epoch on train set
        iter, train_loss, train_dice = train(train_loader, model, ema_model, criterion, optimizer, epoch, args, lr=lr)

        with torch.no_grad():
            valid_loss, valid_dice = validate(valid_loader, model, criterion, epoch)

        # remember best loss and save checkpoint
        is_best = valid_dice >= best_dice
        if is_best:
            best_epoch = epoch
            best_dice = valid_dice

        best_dice_arr = save_top_epochs(model_out_dir, model, best_dice_arr, valid_dice,
                                        best_epoch, epoch, best_dice, ema=False)

        print('\r', end='', flush=True)
        log.write('%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  |  %0.4f  %6.4f |  %6.1f     %6.4f    | %3.1f min \n' % \
                  (epoch, iter + 1, lr, train_loss, train_dice, valid_loss, valid_dice,
                   best_epoch, best_dice, (time.time() - end) / 60))

        model_name = '%03d' % epoch
        save_model(model, model_out_dir, epoch, model_name, best_dice_arr, is_best=is_best,
                   optimizer=optimizer, best_epoch=best_epoch, best_dice=best_dice, ema=False)

def train(train_loader, model, ema_model, criterion, optimizer, epoch, args, lr=1e-5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()

    # switch to train mode
    model.train()

    num_its = len(train_loader)
    end = time.time()
    iter = 0
    print_freq = 1
    for iter, iter_data in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        images, labels, indices = iter_data
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)

        print(outputs.shape)
        print(labels.shape)
        loss = criterion(outputs, labels)
        # loss = criterion(outputs, masks, epoch=epoch)

        losses.update(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clipnorm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        probs = F.sigmoid(outputs)
        dice = metric(probs, masks)
        dices.update(dice.item())

        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == num_its:
            print('\r%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / num_its, iter + 1, lr, losses.avg, dices.avg), \
                  end='', flush=True)

    return iter, losses.avg, dices.avg

def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for it, iter_data in enumerate(valid_loader, 0):
        images, masks, indices = iter_data
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())

        outputs = model(images)
        loss = criterion(outputs, masks)
        # loss = criterion(outputs, masks, epoch=epoch)
        probs = F.sigmoid(outputs)
        dice = metric(probs, masks)

        losses.update(loss.item())
        dices.update(dice.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, dices.avg

def save_model(model, model_out_dir, epoch, model_name, best_dice_arr, is_best=False,
               optimizer=None, best_epoch=None, best_dice=None, ema=False):
    if type(model) == DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
  
    model_fpath = opj(model_out_dir, '%s.pth' % model_name)
    torch.save({
        'state_dict': state_dict,
        'best_epoch': best_epoch,
        'epoch': epoch,
        'best_dice': best_dice,
        'best_dice_arr': best_dice_arr,
    }, model_fpath)

    optim_fpath = opj(model_out_dir, '%s_optim.pth' % model_name)
    if optimizer is not None:
        torch.save({
            'optimizer': optimizer.state_dict(),
        }, optim_fpath)

    if is_best:
        best_model_fpath = opj(model_out_dir, 'final.pth')
        shutil.copyfile(model_fpath, best_model_fpath)
        if optimizer is not None:
            best_optim_fpath = opj(model_out_dir, 'final_optim.pth')
            shutil.copyfile(optim_fpath, best_optim_fpath)

def metric(logit, truth, threshold=0.5):
    dice = dice_score(logit, truth, threshold=threshold)
    return dice

def accumulate(model1, model2, decay=0.99):
    par1 = model1.state_dict()
    par2 = model2.state_dict()

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_top_epochs(model_out_dir, model, best_dice_arr, valid_dice, best_epoch, epoch, best_dice, ema=False):
    best_dice_arr = best_dice_arr.copy()

    suffix = ''
    min_dice = np.min(best_dice_arr)
    last_ix = len(best_dice_arr) - 1

    def get_top_path(ix):
        return opj(model_out_dir, 'top%d%s.pth' % (ix + 1, suffix))

    if valid_dice > min_dice:
        min_ix = last_ix
        for ix, score in enumerate(best_dice_arr):
            if score < valid_dice:
                min_ix = ix
                break

        lowest_path = get_top_path(last_ix)
        if ope(lowest_path):
            os.remove(lowest_path)

        for ix in range(last_ix - 1, min_ix - 1, -1):
            score = best_dice_arr[ix]
            best_dice_arr[ix + 1] = score
            if score > 0 and ope(get_top_path(ix)):
                os.rename(get_top_path(ix), get_top_path(ix + 1))

        best_dice_arr[min_ix] = valid_dice

        model_name = 'top%d' % (min_ix + 1)
        save_model(model, model_out_dir, epoch, model_name, best_dice_arr, is_best=False,
                   optimizer=None, best_epoch=best_epoch, best_dice=best_dice, ema=False)

    return best_dice_arr

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
