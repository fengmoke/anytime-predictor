from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import os
import math
import time
import json

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from laplace import estimate_variance_efficient
import random
import sys
from sklearn.metrics import f1_score
from anytime_predictors import anytime_product, conditional_monotonicity_check, anytime_caching
# from DeepSloth_main.delay_attack_cost import convert_num_early_exits_at_each_ic_to_cumulative_dis, get_plot_data_and_auc
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total.item())

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist

def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    c_i = np.array(ic_costs)/np.array(ic_costs)[-1]
    
    c_i = np.insert(c_i, 0, 0)
    c_i = c_i.tolist()

    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve


def calc_ensemble_logits_new(logits, flop_weights_, rho):
    ens_logits = torch.zeros_like(logits)
    ens_logits[0,:,:] = logits[0,:,:].clone()
    for until in range(1, logits.shape[0]):
        p = flop_weights_[0]*rho[until, 0]
        summ = p*logits[0,:,:].clone()
        w = p
        for i in range(1,until+1):
            p = flop_weights_[i]*rho[until, i]
            summ += p*logits[i,:,:].clone()
            w += p
        if until > 0:
            ens_logits[until,:,:] = summ / w

    return ens_logits

def calc_ensemble_logits(logits, flop_weights):
    ens_logits = torch.zeros_like(logits)
    ens_logits[0,:,:] = logits[0,:,:].clone()
    
    p = flop_weights[0]
    summ = p*logits[0,:,:].clone()

    w = p
    for i in range(1,logits.shape[0]):
        p = flop_weights[i]
        summ += p*logits[i,:,:].clone()
        w += p
        ens_logits[i,:,:] = summ / w

    return ens_logits
        
def Entropy(p, islogits = False):
    # Calculates the sample entropies for a batch of output softmax values
    '''
        p: m * n * c
        m: Exits
        n: Samples
        c: Classes
    '''
    if not islogits:
        p = nn.functional.softmax(p, dim=2)
    softmax_value = p
    p = torch.where(p == 0, 1e-10, p)
    Ex = -1*torch.sum(p*torch.log(p), dim=2)
    return (Ex, softmax_value)
    
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
    
def calc_bins(confs, corrs):
    # confs and corrs are numpy arrays
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(confs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(confs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (corrs[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (confs[binned==bin]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes
    
def calculate_ECE(confs, corrs):
    # confs and corrs are numpy arrays
    ECE = 0
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(confs, corrs)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ECE


def anytime_evaluate(model, test_loader, val_loader, args, prints = False):
    
    # Expected computational cost of each block for the whole dataset             
    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    print(flops)
    flop_weights = np.array(flops)/np.array(flops)[-1] #.sum()
    print(flop_weights)
    tester = Tester(model, args, mie_flops_weights=flop_weights)
        
    ############ Set file naming strings based on options selected ############
    fname_ending = ''
    fname_ending += '_mie' if args.MIE else ''
    fname_ending += '_opttemp' if args.optimize_temperature else ''
    fname_ending += '_optvar' if args.optimize_var0 else ''
    fname_ending += '_PA_hoc' if args.PA_hoc else ''
    fname_ending += '_CA' if args.CA else ''
    fname_ending += '_best' if 'best' in args.evaluate_from else ''
    fname_ending += '_last' if 'checkpoint_199' in args.evaluate_from else ''
    ###########################################################################
    # Optimize the temperature scaling parameters
    if args.optimize_temperature:
        print('******* Optimizing temperatures scales ********')
        tester.args.laplace_temperature = [1.0 for i in range(args.nBlocks)]
        temp_grid = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    else:
        temp_grid = [args.laplace_temperature]
    if args.optimize_var0:
        print('******* Optimizing Laplace prior variance ********')
        var_grid = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
    else:
        var_grid = [args.var0]
    max_count = len(var_grid)*len(temp_grid)
    if max_count > 1:
        count = 1
        if not args.MIE:
            results = torch.zeros(args.nBlocks, len(temp_grid), len(var_grid))
            for j in range(len(temp_grid)):
                for i in range(len(var_grid)):
                    temp = temp_grid[j]
                    var0 = var_grid[i]
                    # print([temp for t in range(args.nBlocks)])
                    print('Optimizing setup {}/{}'.format(count, max_count))
                    tester.args.laplace_temperature = [temp for t in range(args.nBlocks)]
                    blockPrint()
                    if not args.PA_hoc:
                        val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=[temp for t in range(args.nBlocks)])
                    else:
                        val_pred_o, val_target_o = tester.calc_pa_logit(val_loader, temperature=[temp for t in range(args.nBlocks)])
                    enablePrint()
                    
                    for block in range(args.nBlocks):
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred_o[block,:,:]), val_target_o)
                        results[block,j,i] = -1*nlpd_o
                    count += 1
            optimized_vars, optimized_temps = [], []
            for block in range(args.nBlocks):
                max_ind = (results[block,:,:]==torch.max(results[block,:,:])).nonzero().squeeze()
                if max_ind.dim() > 1:
                    max_ind = max_ind[0]
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps.append(temp_o)
                optimized_vars.append(var_o)
                print('For block {}, best temperature is {} and best var0 is {}'.format(block+1, temp_o, var_o))
                print()
        else:
            optimized_temps, optimized_vars = [0 for t in range(args.nBlocks)],[0 for t in range(args.nBlocks)]
            current_temps = [0 for t in range(args.nBlocks)]
            current_vars = [0 for t in range(args.nBlocks)]
            for exit in range(args.nBlocks):
                count = 1
                results = torch.zeros(len(temp_grid), len(var_grid))
                print('Optimizing for exit {}'.format(exit+1))
                for j in range(len(temp_grid)):
                    for i in range(len(var_grid)):
                        temp = temp_grid[j]
                        var0 = var_grid[i]
                        print('Optimizing setup {}/{}'.format(count, max_count))
                        current_temps[0:exit+1] = optimized_temps[0:exit+1]
                        current_temps[exit] = temp
                        current_vars[0:exit+1] = optimized_vars[0:exit+1]
                        current_vars[exit] = var0
                        tester.args.laplace_temperature = current_temps
                        blockPrint()
                        if not args.PA_hoc:
                            val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=current_temps, until=exit+1)
                        else:
                            val_pred_o, val_target_o = tester.calc_pa_logit(val_loader, temperature=current_temps, until=exit+1)
                        enablePrint()
                        val_pred = calc_ensemble_logits(val_pred_o, flop_weights)
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred[exit,:,:]), val_target_o)# 计算神经网络预测值和目标值之间的负对数似然损失
                        results[j,i] = -1*nlpd_o
                        count += 1
                        
                max_ind = (results==torch.max(results)).nonzero().squeeze()
                if max_ind.dim() > 1:
                    max_ind = max_ind[0]
                # print(max_ind[0])
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps[exit] = temp_o  
                optimized_vars[exit] = var_o
                print('For block {}, best temperature is {} and best var0 is {}'.format(exit+1, temp_o, var_o))
                print()

        
        tester.args.laplace_temperature = optimized_temps
        args.laplace_temperature = optimized_temps
        vanilla_temps = optimized_temps
        args.var0 = optimized_vars
        print(optimized_temps)
        print(optimized_vars)
    else:
        vanilla_temps = None
        if not isinstance(args.var0, list):
            args.var0 = [args.var0]
        if not isinstance(args.laplace_temperature, list):
            tester.args.laplace_temperature = [args.laplace_temperature]
        else:
            tester.args.laplace_temperature = args.laplace_temperature
            vanilla_temps = args.laplace_temperature
        
    # Calculate validation and test predictions
    '''
    val_pred, test_pred are softmax outputs, shape (n_blocks, n_samples, n_classes)
    val_var, test_var are predicted class variances, shape (n_blocks, n_samples)
    '''
    if not args.PA_hoc:
        filename = os.path.join(args.save, 'dynamic%s.txt' % (fname_ending))
        val_pred, val_target = tester.calc_logit(val_loader, temperature=vanilla_temps)
        test_pred, test_target = tester.calc_logit(test_loader, temperature=vanilla_temps)  
    else:
        val_pred_find_rho, val_target_find_rho = tester.calc_logit(val_loader, temperature=vanilla_temps)
        _, argmax_val_find_rho = val_pred_find_rho.max(dim=2, keepdim=False)
        rho = np.zeros((val_pred_find_rho.shape[0], val_pred_find_rho.shape[0]))
        val_acc_find_rho = np.array([])
        for r in range(rho.shape[0]):
            val_acc_find_rho = np.append(val_acc_find_rho, (argmax_val_find_rho[r,:] == val_target_find_rho).sum()/val_pred_find_rho.shape[1])
        for h in range(1, rho.shape[0]):
            for o in range(h):
                if val_acc_find_rho[o] > val_acc_find_rho[h]:
                    rho[h, o] = 1
        if args.optimize_temperature:
            filename = os.path.join(args.save, 'dynamic_la_mc%s.txt' % (fname_ending))

        np.fill_diagonal(rho, 1)
        np.save(args.save, rho)
        val_pred, val_target = tester.calc_pa_logit(val_loader, rho, temperature=vanilla_temps)
        test_pred, test_target = tester.calc_pa_logit(test_loader, rho, temperature=vanilla_temps)
        
      
    # if args.MIE:
    #     val_pred = calc_ensemble_logits(val_pred, flop_weights)
    #     test_pred = calc_ensemble_logits(test_pred, flop_weights)  
    # if args.PA_hoc:
    #     L, N, C = val_pred.shape
    #     W = (torch.arange(1, L + 1, 1, dtype=float) / L)
    #     b = None
    #     val_pred, _ = anytime_product(val_pred, weights=W, thres_max=b, fall_back=True)
    #     test_pred, _ = anytime_product(test_pred, weights=W, thres_max=b, fall_back=True, softplus=True) # (torch.arange(1, L + 1, 1, dtype=float) / L)
    # elif args.CA:
    #     val_pred = anytime_caching(val_pred)
    #     test_pred = anytime_caching(test_pred) 
    # if not args.PA_hoc and not args.CA:
    #     val_pred = torch.softmax(val_pred, dim=2)
    #     test_pred = torch.softmax(test_pred, dim=2)     

    # Calculate validation and test set accuracies for each block
    _, argmax_val = val_pred.max(dim=2, keepdim=False) #predicted class confidences
    Thresold = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    probs_decrase = conditional_monotonicity_check(test_target, test_pred, thresholds=Thresold)
    maxpred_test, argmax_test = test_pred.max(dim=2, keepdim=False)
    data_save_path = os.path.join(args.save, 'data')
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    with open(os.path.join(data_save_path, f'data_perexit{fname_ending}.txt'), 'w') as fout0:
        fout0.write('test_acc\tF1_score\tECE\tnlpd\n')
        print('Val acc      Test acc    test_F1    test_ECE    test_nlpd')
        for e in range(val_pred.shape[0]):
            val_acc = (argmax_val[e,:] == val_target).sum()/val_pred.shape[1]
            test_acc = (argmax_test[e,:] == test_target).sum()/test_pred.shape[1]
            test_F1_score = f1_score(test_target, argmax_test[e,:], average='weighted')
            test_ECE = calculate_ECE(maxpred_test[e,:], (argmax_test[e,:] == test_target))
            test_nlpd = 1 * nn.functional.nll_loss(torch.log(test_pred[e,:]), test_target)
            print('{:.3f}       {:.3f}       {:.3f}       {:.3f}       {:.3f}'.format(val_acc*100, test_acc*100, test_F1_score*100, test_ECE, test_nlpd))
            fout0.write(f'{test_acc}\t{test_F1_score}\t{test_ECE}\t{test_nlpd}\n')
        fout0.close()
        print('')
    np.save(os.path.join(data_save_path, f'correct_label{fname_ending}.npy'), test_target.numpy())
    np.save(os.path.join(data_save_path, f'prob{fname_ending}.npy'), test_pred.numpy())
    np.save(os.path.join(data_save_path, f'test_label{fname_ending}.npy'), argmax_test.numpy())
    with open(os.path.join(data_save_path, f'Conditional_Monotonicity_dict{fname_ending}.txt'), 'w') as file:
        json.dump(probs_decrase, file)
    # save_T = np.array([])
    # save_plot_data = np.array([])
    # if args.attack_mode is not None:
    #     filename = os.path.splitext(filename)[0] + f'_{args.attack_mode}.txt'
    # with open(filename, 'w') as fout:
    #     fout.write('test_acc\tnlpd\tECE\tF1_score\texp_flops\tbudget_flops\n')
    #     for p in range(1, 41): # Loop over 40 different computational budget levels
    #         print("*********************")
    #         _p = torch.FloatTensor(1).fill_(p * 1.0 / 20) # 'Heaviness level' of the current computational budget
    #         probs = torch.exp(torch.log(_p) * torch.arange(1, args.nBlocks+1)) # Calculate proportions of computation for each DNN block
    #         probs /= probs.sum() # normalize
    #         computational_budget = 0
    #         for i in range(test_pred.shape[0]):
    #             computational_budget += test_pred.shape[1] * probs[i] * flops[i]
    #         val_t_metric_values, _ = val_pred.max(dim=2, keepdim=False) #predicted class confidences
    #         test_t_metric_values, _ = test_pred.max(dim=2, keepdim=False)
        
    #         # Find thresholds to determine which block handles each sample
    #         acc_val, _, T = tester.dynamic_find_threshold(val_pred, val_target, val_t_metric_values, probs, flops)
    #         save_T = np.append(save_T, T)
    #         # Calculate accuracy, expected computational cost, nlpd and ECE given thresholds in T
    #         acc_test, exp_flops, nlpd, ECE, acc5 , F1, AUC, plot_data = tester.dynamic_eval_threshold(test_pred, test_target, flops, T, test_t_metric_values, p)
    #         save_plot_data = np.append(save_plot_data, plot_data)
    #         print('valid acc: {:.3f}, test acc: {:.3f}, F1 score: {:.3f} nlpd: {:.3f}, ECE: {:.3f}, test flops: {:.2f}'.format(acc_val, acc_test, F1, nlpd, ECE,  exp_flops / 1e6))
    #         fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(acc_test, nlpd, ECE, F1, exp_flops.item(), computational_budget))
    #     # save_plot_data = save_plot_data.reshape(40, -1, 5)       
    #     np.save(os.path.splitext(filename)[0] + '_T', save_T.reshape(40, -1))
    #     if args.attack_mode is not None:
    #         np.save(os.path.splitext(filename)[0] + '_auc_plot_data', save_plot_data.reshape(40, -1, 5))

class Tester(object):
    def __init__(self, model, args=None, mie_flops_weights = None):
        self.args = args
        self.weights = mie_flops_weights
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader, temperature=None, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks
        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                # print(input_var.shape)
                #input_var = torch.autograd.Variable(input)
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until)
                else:
                    output, phi = self.model.module.predict(input_var)
                #output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_exit):
                    if temperature is not None:
                        _t = output[b]/temperature[b]
                    else:
                        _t = output[b]

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])
        # if self.args.MIE:
        #     ts_logits = calc_ensemble_logits(ts_logits, self.weights)
        if not self.args.CA:
            ts_logits = torch.softmax(ts_logits, dim=2)
        else:
            ts_logits = anytime_caching(ts_logits)

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Logits calculation time: {}'.format(time.time() - start_time))

        return ts_logits, targets
    def calc_pa_logit(self, dataloader, rho, temperature=None, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks
        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until)
                else:
                    output, phi = self.model.module.predict(input_var)
                if not isinstance(output, list):
                    output = [output]

                for b in range(n_exit):
                    logits[b].append(output[b])

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])
        if self.args.MIE:
            ts_logits = calc_ensemble_logits_new(ts_logits, flop_weights_=self.weights.copy(), rho=rho)
            # ts_logits = calc_ensemble_logits(ts_logits, self.weights)
        L, N, C = ts_logits.shape
        W = (torch.arange(1, L + 1, 1, dtype=float) / L)
        b = None
        ts_logits = anytime_product(ts_logits, weights=W, thres_max=b, fall_back=True, softplus=('FT' in self.args.save), Temperature=temperature)
        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Logits calculation time: {}'.format(time.time() - start_time))

        return ts_logits, targets
    
        
        

    def dynamic_find_threshold(self, logits, targets, t_metric_values, p, flops):
        """
            logits: m * n * c
            m: Exits
            n: Samples
            c: Classes
            
            t_metric_values: m * n
        """
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
            
        n_exit, n_sample, c = logits.size()
        _, argmax_preds = logits.max(dim=2, keepdim=False) # Predicted class index for each stage and sample
        _, sorted_idx = t_metric_values.sort(dim=1, descending=descend) # Sort threshold metric values for each stage

        filtered = torch.zeros(n_sample)
        
        # Initialize thresholds
        T = torch.Tensor(n_exit).fill_(1e8) if descend else torch.Tensor(n_exit).fill_(-1e8)

        for k in range(n_exit - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k]) # Number of samples that should be exited at stage k
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i] # Original index of the sorted sample
                if filtered[ori_idx] == 0: # Check if the sample has already been exited from an earlier stage
                    count += 1 # Add 1 to the count of samples exited at stage k
                    if count == out_n:
                        T[k] = t_metric_values[k][ori_idx] # Set threshold k to value of the last sample exited at exit k
                        break
            #Add 1 to filtered in locations of samples that were exited at stage k
            if descend:
                filtered.add_(t_metric_values[k].ge(T[k]).type_as(filtered))
            else:
                filtered.add_(t_metric_values[k].le(T[k]).type_as(filtered))

        # accept all of the samples at the last stage
        T[n_exit -1] = -1e8 if descend else 1e8

        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0 # Initialize accuracy and expected cumulative computational cost
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()): # check if prediction was correct
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0

        for k in range(n_exit):
            _t = 1.0 * exp[k] / n_sample # The fraction of samples that were exited at stage k
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T


    def dynamic_eval_threshold(self, logits, targets, flops, T, t_metric_values, p):
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
        
        n_exit, n_sample, n_class = logits.size()
        maxpreds, argmax_preds = logits.max(dim=2, keepdim=False) # predicted class indexes

        early_exit_counts = np.zeros((n_exit, 1))
        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0
        nlpd = 0 # Initialize cumulative nlpd
        final_confs = torch.zeros(n_sample) #Tensor for saving confidences for each sample based on which block was used
        final_corrs = torch.zeros(n_sample) #Prediction correctness of final preds
        final_logits = torch.zeros(n_sample, n_class)
        pre_insdex, act_index = [],[]
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    _g = int(gold_label.item())
                    pre_insdex.append(_g)
                    _pred = int(argmax_preds[k][i].item())
                    act_index.append(_pred)
                    if _g == _pred:
                        final_corrs[i] = 1
                        acc += 1
                        acc_rec[k] += 1
                    final_confs[i] = maxpreds[k][i]
                    exp[k] += 1
                    nlpd += -1*logits[k,i,_g].log()
                    final_logits[i,:] = logits[k,i,:]
                    early_exit_counts[k] += 1

                    break
        acc_all, sample_all = 0, 0
        for k in range(n_exit):
            _t = exp[k] * 1.0 / n_sample # The fraction of samples that were exited at stage k
            sample_all += exp[k]
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]
        
        layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts, n_sample)
        plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, flops)
        ECE = calculate_ECE(final_confs.numpy(), final_corrs.numpy())
        
        F_score = f1_score(pre_insdex, act_index,  average='weighted')
            
        prec5 = accuracy(final_logits, targets, topk=(5,))

        return acc * 100.0 / n_sample, expected_flops, nlpd / n_sample, ECE, prec5[0], F_score, early_exit_auc, plot_data
        
