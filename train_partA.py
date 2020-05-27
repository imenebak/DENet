import sys
import os

import warnings

from model import CSRNet,MCNN, SANet

from utils import save_checkpoint,save_net

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import pytorch_ssim
import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

class myloss(nn.Module):
    def __init__(self):
        super(myloss,self).__init__()
    def forward(self,GT_detection,target_sum):
        l=(GT_detection-target_sum)/(GT_detection+1)
        loss=l*l
        return torch.sum(loss)

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200 #800
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    args.train_json = '/content/train.json'
    args.test_json= '/content/test.json'
    args.gpu = '0'
    args.task = 'Visdrone2019'
    # args.pre = 'shanghaiAcheckpoint.pth.tar'
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    
    criterion = nn.MSELoss(size_average=False).cuda()
    criterion1 = nn.L1Loss().cuda()
    # criterion1 = myloss().cuda()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, criterion1, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
        # save_net('best.h5',model)

def train(train_list, model, criterion, criterion1, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                           # transforms.RandomCrop(300),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target,GT_detection,target_sum)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)


        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)


        GT_detection =GT_detection.type(torch.FloatTensor).unsqueeze(0).cuda()
        GT_detection = Variable(GT_detection)

        target_sum = target_sum.type(torch.FloatTensor).unsqueeze(0).cuda()
        target_sum = Variable(target_sum)

        loss = criterion(output, target)
        loss2 = criterion1(GT_detection,target_sum)

        loss=loss+loss2
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss2 {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=loss, loss2 = loss2))
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    mse  = 0

    for i,(img, target,GT_detection,target_sum) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        GT_detection = GT_detection.type(torch.FloatTensor).unsqueeze(0).cuda()
        GT_detection = Variable(GT_detection)
        
        mae += abs(output.data.sum()-GT_detection.data.sum())
        # mae += abs(output.detach().cpu().sum().numpy()-GT_detection.data.numpy())
        # mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - GT_detection.data.sum()) * (output.data.sum() - GT_detection.data.sum())
        # mse += np.square(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
    mae = mae / len(test_loader)
    mse = np.sqrt(mse / len(test_loader))
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    print(' * MSE {mse:.3f} '
          .format(mse=mse))

    return mae
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
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
    
if __name__ == '__main__':
    main()        
