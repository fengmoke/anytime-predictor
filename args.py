import os
import glob
import time
import argparse

model_names = ['msdnet', 'mynet', 'lstm', 'dcl', 'CNN']
model = model_names[1]
arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')
# 添加实验设置参数组exp_group
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default=f'save/{model}/UniMib_4_32',
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory ')
exp_group.add_argument('--resume', default=None, type=str, help='Name of latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default='anytime', choices=['anytime', 'dynamic'], help='which mode to evaluate')
exp_group.add_argument('--evaluate-from', default=r'save/mynet/UniMib_4_32/save_models/model_best_acc.pth.tar', type=str, metavar='PATH', help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=42, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default='0', type=str, help='GPU available.')

# dataset related 数据集相关参数设置
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='UniMib',
                        choices=['cifar10', 'cifar100', 'ImageNet', 'caltech256', 'UCI', 'UniMib', 'Wisdm', 'Pampa2', 'Usc'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true', default=True, 
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related 模型架构参数设置
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default=f'{model}',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: msdnet)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')# 设置DenseNet的压缩率

# msdnet config  在MSDNet（Multi-Scale Dense Network）中，网络深度会随着训练的增加而动态增加。
arch_group.add_argument('--nBlocks', type=int, default=4)# 设置块的数量
arch_group.add_argument('--nChannels', type=int, default=32)# 设置块的通道数 # dcl: UCI 32, other 64
arch_group.add_argument('--base', type=int,default=4)# 指定基础值                                                                                                                                                                                 
arch_group.add_argument('--stepmode', type=str, default='even', choices=['even', 'lin_grow'])# 均匀增长，线性增长
arch_group.add_argument('--step', type=int, default=4)# 设置增长步长
arch_group.add_argument('--growthRate', type=int, default=16)# 设置增长率
arch_group.add_argument('--grFactor', default='1-2-4', type=str)# 设置增长因子
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])# 设置剪枝方式
arch_group.add_argument('--bnFactor', default='1-2-4')# 设置批量归一化因子
arch_group.add_argument('--bottleneck', default=True, type=bool)# 指定是否使用瓶颈结构


# training related 设置优化器参数
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')

optim_group.add_argument('--epochs', default=200, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')# 设置训练总epoch次数
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='adam',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])# 设置学习率调度策略
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')
                         
# new args
arg_parser.add_argument('--temperature', type=float, default=1.0, help="temperature scaling of softmax")
arg_parser.add_argument('--laplace_temperature', type=float, default=1.0,  help="temperature scaling of softmax for laplace predictions")
arg_parser.add_argument('--PA', action='store_true', default=False, help='Use PA in training')
arg_parser.add_argument('--PA_hoc', action='store_true', default=False, help='Use PA in hoc')
arg_parser.add_argument('--softplus', action='store_true', default=False, help='Use softplus in PA')
arg_parser.add_argument('--CA', action='store_true', default=False, help='Use CA')# 是否在模型内部使用CA
arg_parser.add_argument('--MIE', action='store_true', default=False, help='Use model-internal ensembling')# 是否在模型内部使用集成学习

arg_parser.add_argument('--optimize_temperature', action='store_true', default=False, help='Use the validation set to optimize temperature scaling individually for each block')# 是否使用验证集优化温度衰减参数（超参数）
arg_parser.add_argument('--optimize_var0', action='store_true', default=False, help='Use the validation set to optimize Laplace prior variance individually for each block')# 是否使用验证集优化拉普拉斯先验方差（超参数）

# Laplace arguments
arg_parser.add_argument('--compute_only_laplace', action='store_true', default=False, help='skip training and only fit laplace approximation')# action='store_true' - 当命令行中出现该参数时，其值为True 
arg_parser.add_argument('--var0', type=float, default=5e-4, help='Laplace prior variance(default =5e-4)')
arg_parser.add_argument('--laplace', action='store_true', default=False, help='test with MC integration and laplace approximation')# 指定是否使用蒙特卡洛积分和拉普拉斯近似来测试
arg_parser.add_argument('--n_mc_samples', type=int, default=50, help='number of samples to draw from laplace') # 从拉普拉斯分布抽取的样本数量

Deepsloth_experiment = arg_parser.add_argument_group('Attack',
                                           'Attack experiment setting')

# basic configurations
Deepsloth_experiment.add_argument('--best_model_path', type=str, default=r'save\Myresnet\UniMib_v1\save_models\model_best_acc.pth.tar')
Deepsloth_experiment.add_argument('--network', type=str, default='mynet',
                    help='location of the network (vgg16bn, resnet56, or mobilenet)')
Deepsloth_experiment.add_argument('--nettype', type=str, default='cnn',
                    help='location of the network (ex. cnn, or sdn_ic_only / PGD_10_8_2_cnn --- for AT nets)')
Deepsloth_experiment.add_argument('--runmode', type=str, default='analysis',
                    help='runmode of the script (attack - crafts the adversarial samples, or analysis - computes the efficacy)')

# attack configurations
Deepsloth_experiment.add_argument('--attacks', type=str, default='ours',
                    help='the attack that this script will use (PGD, PGD-avg, PGD-max, UAP, ours)')
Deepsloth_experiment.add_argument('--ellnorm', type=str, default='linf',
                    help='the norm used to bound the attack (default: linf - l1 and l2)')
Deepsloth_experiment.add_argument('--nsample', type=int, default=100,
                    help='the number of samples consider (for UAP)')
Deepsloth_experiment.add_argument('--attack_mode', type=str, default=None,choices=['persample', 'univ'],
                    help='the Mode of attack')
