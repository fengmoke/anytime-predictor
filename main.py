#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
import numpy as np
import torch.autograd.profiler as profiler
from anytime_predictors import anytime_product

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate, Entropy
from anytime_inference import anytime_evaluate
import models
from op_counter import measure_model
#
from laplace import get_hessian_efficient

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') 

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
elif args.data == 'caltech256':
    args.num_classes = 257
elif args.data == 'UCI':
    args.num_classes = 6
elif args.data == 'UniMib':
    args.num_classes = 17
elif args.data == 'Wisdm':
    args.num_classes = 6
elif args.data == 'Usc':
    args.num_classes = 12
elif args.data == 'Pampa2':
    args.num_classes = 12
else:
    args.num_classes = 1000

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from torchstat import stat
from torchviz import make_dot

torch.manual_seed(args.seed)
Laplase = False

def main():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        W_size = H_size = 32
    elif args.data == 'UCI':
        W_size = 128
        H_size = 9
    elif args.data == 'UniMib':
        W_size = 151
        H_size = 3
    elif args.data == 'Wisdm':
        W_size = 200
        H_size = 3
    elif args.data == 'Usc':
        W_size = 512
        H_size = 6
    elif args.data == 'Pampa2':
        W_size = 171
        H_size = 40
    else:
        W_size = H_size = 224
    model = getattr(models, args.arch)(args)
    # x = torch.randn(1, 1, W_size, H_size)
    # dot = make_dot(model(x), params=dict(model.named_parameters()))
    # dot.format = 'pdf'
    # dot.render("models\mynetstructure")
    # if args.arch in ['dcl', 'lstm']:
    #     n_flops = []
    #     input_data = torch.randn(1, 1, W_size, H_size)
    #     for i in range(args.nBlocks):
    #         with profiler.profile(record_shapes=False) as prof:
    #             with profiler.record_function("model_inference"):
    #                 output = model.predict_until(input_data, i+1)
    #                 # output = model(input_data)
    #         events = prof.function_events
    #         for event in events:
    #             if event.name == "model_inference":
    #                 flops = event.cpu_memory_usage 
    #                 n_flops.append(flops)
    # else:    
    n_flops, n_params = measure_model(model, W_size, H_size)
    args.n_flops = n_flops
    # print(args.save)    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)
    # dummy_input = torch.randn(1, 1, 200, 3)
    # torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)
    model = torch.nn.DataParallel(model).cuda()

    if args.arch == 'dcl':
        args.weight_decay = 0
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_PA = PACrossEntropyLoss().cuda()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    if args.resume:# 是否导入检查点
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.evalmode is not None:# 设置评估模式
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            anytime_evaluate(model, test_loader, val_loader, args)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    if not args.compute_only_laplace:
        initial_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
        
            model_filename = 'checkpoint_%03d.pth.tar' % epoch
            if 'FT' in args.save and epoch >= int(args.epochs * 2 / 3):
                W_m = [1] * args.nBlocks
                train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion_PA, optimizer, epoch, W_m)
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion_PA, W_m)
            else:
                # W_m = [i / args.nBlocks for i in range(1, args.nBlocks + 1)]
                W_m = [1] * args.nBlocks
                train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch, W_m)
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, W_m)

            scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                        .format(epoch, lr, train_loss, val_loss,
                                train_prec1, val_prec1, train_prec5, val_prec5))
            
            is_best = val_prec1 > best_prec1
            if is_best:
                best_prec1 = val_prec1
                best_epoch = epoch
                print('New best validation last_bloc_accuracy {}'.format(best_prec1))
            else:
                print('Current best validation last_bloc_accuracy {}'.format(best_prec1))
                    
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, args, is_best, model_filename, scores)

        print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
        print('Total training time: {}'.format(time.time() - initial_time))
        
    # Load the best model
    model_dir = os.path.join(args.save, 'save_models')
    best_filename = os.path.join(model_dir, 'model_best_acc.pth.tar')

    state_dict = torch.load(best_filename)['state_dict']
    model.load_state_dict(state_dict)

    ### Test the final model
    print('********** Final prediction results with the best model **********')
    W_m = [1] * args.nBlocks
    validate(test_loader, model, criterion, W_m)
    if Laplase:
        ### Test the final model + laplace
        print('********** Precalculate Laplace approximation **********')
        start_time = time.time()
        compute_laplace_efficient(args, model, train_loader)
        print('Laplace computation time: {}'.format(time.time() - start_time))
    return   

def compute_laplace_efficient(args, model, dset_loader):
    # compute the laplace approximations
    M_W, U, V = get_hessian_efficient(model, dset_loader)
    print(f'Saving the hessians...')
    M_W, U, V = [M_W[i].detach().cpu().numpy() for i in range(len(M_W))], \
                        [U[i].detach().cpu().numpy() for i in range(len(U))], \
                        [V[i].detach().cpu().numpy() for i in range(len(V))]
    np.save(os.path.join(args.save, "effL_llla.npy"), [M_W, U, V])


class PACrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(PACrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = anytime_product(logits, softplus=True)
        logits = logits.detach()
        logits = torch.squeeze(logits)
        logits = logits.gather(1, target)   # [NHW, 1]
        logits = torch.log(logits)
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        loss.requires_grad = True
        return loss


# class PACrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(PACrossEntropyLoss, self).__init__()

#     def forward(self, inputs, targets):
#         # 使用自定义的概率生成方法（示例中使用sigmoid）
#         probabilities, _ = anytime_product(inputs, softplus=True)
#         probabilities = probabilities.squeeze()
        
#         # 计算交叉熵损失
#         loss = -torch.mean(targets * torch.log(probabilities[:, targets]) + (1 - targets) * torch.log(1 - probabilities[:, targets]))
        
#         return loss

def train(train_loader, model, criterion, optimizer, epoch, w_m):# 训练函数
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()
    if args.optimizer == 'adam':
        running_lr = args.lr
    else:
        running_lr = None
    
    for i, (input, target) in enumerate(train_loader):
        total_block_counts = torch.zeros(args.nBlocks)
        if args.optimizer == 'sgd':
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                    nBatch=len(train_loader), method=args.lr_type)

            if running_lr is None:
                running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(device=None)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        loss = 0.0
        output = model(input_var)
        
        if not isinstance(output, list):# 检查output是否为list类型
            output = [output]
           
        for j in range(len(output)):
            loss += w_m[j] * criterion(output[j], target_var)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.2f}\t'
                  'Acc@1 {top1.val:.1f}\t'
                  'Acc@5 {top5.val:.1f}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion, w_m):# 验证函数

    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    flop_weights = torch.Tensor(flops)/flops[-1]
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5, entropy, softmax_value_list, correct_label = [], [], [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    
    n = len(val_loader.sampler)
    confs = torch.zeros(args.nBlocks,n)
    corrs = torch.zeros(args.nBlocks,n)

    model.eval()

    end = time.time()
    sample_ind = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.shape[0]
            correct_label.append(target)
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)

            output_e = torch.stack(output)
            # print(output_e)
            entropy.append(Entropy(output_e)[0].cpu())
            softmax_value_list.append(Entropy(output_e)[1].cpu())
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += w_m[j] * criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            sample_ind += batch_size

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
        entropy = torch.cat(entropy, dim=1).numpy()
        # softmax_value_numpy = torch.cat(softmax_value_list, dim=1).numpy()
        # correct_label_numpy = torch.cat(correct_label, dim=0).numpy()
        # print(entropy.shape)
        # print(softmax_value_numpy.shape)
        # print(correct_label_numpy.shape)
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))

    return losses.avg, top1[-1].avg, top5[-1].avg
    
def save_checkpoint(state, args, is_best, filename, result):
    print(args) # 打印当前参数设置
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best_acc.pth.tar')

    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    if args.compute_only_laplace:
        model_filename = os.path.join(model_dir, str(args.resume))
    else:
        latest_filename = os.path.join(model_dir, 'latest.txt')
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
        else:
            return None
    print("=> loading checkpoint '{}'".format(model_filename))
    # print(model_filename)
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state
    
def remove_checkpoints(args, model_filename):
    print("=> removing checkpoints")
    for epoch in range(args.epochs-1):
        filename = 'checkpoint_%03d.pth.tar' % epoch
        model_dir = os.path.join(args.save, 'save_models')
        model_filename = os.path.join(model_dir, filename)
        os.remove(model_filename)
    print("=> checkpoints removed")

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

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):# 调节学习率函数
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        elif args.data.startswith('caltech'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def Entropy_mean(x):
    # Calculates the mean entropy for a batch of output logits
    epsilon = 1e-5
    p = nn.functional.softmax(x, dim=1)
    Ex = torch.mean(-1*torch.sum(p*torch.log(p), dim=1))
    return Ex

if __name__ == '__main__':
    main()
