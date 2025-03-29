# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:41:27 2024

@author: dalei
"""

import argparse
import torch
import os
import torch.optim as optim
import random
import time
import tempfile
from datetime import datetime
from cifar import dataset, Network, info, loss_func

dataset_name = 'GaN'
epoches_num = 1000000
lr = 1e-5
# decreasing_time = 2
update_period = 300
decreasing_rate = 2.0
num_classes = 6
Label = 'GaN-Ideal-480-480-VSHS-NEW'
# model_name = './resource/model_weights.pth'
model_name = ''
for i in range(1):
    seed = random.randint(0, 99999) # 基于时间的变化，确保每次不同
    random.seed(seed)  # 设置随机数种子
    # 在这里执行其他操作，比如生成随机数
    rand_seed = random.randint(1, 1000)
    print(f"Generated random number: {rand_seed}")
    # 系统参数设置
    parser = argparse.ArgumentParser(description = 'RHEED Ideal Model')
    parser.add_argument('--type', default = dataset_name)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=epoches_num, help='number of epochs to train (default: 32)') # epochs
    parser.add_argument('--grad_scale', type=float, default=lr, help='learning rate for wage delta calculation') # 这里的wage delta calculation是什么？
    parser.add_argument('--seed', type=int, default=rand_seed, help='random seed (default: 117)') # random seed 选取
    parser.add_argument('--log_interval', type=int, default=36,  help='how many batches to wait before logging training status default = 100') 
    parser.add_argument('--test_interval', type=int, default=8,  help='how many epochs to wait before another test (default = 1)')
    parser.add_argument('--logdir', default='log', help='folder to save to the log') #logdir是日志的存储地点
    parser.add_argument('--input_format', default=480, help='folder to save to the log') #logdir是日志的存储地点
    parser.add_argument('--modeldir', default='model', help='folder to save to the model') #logdir是日志的存储地点
    
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args = parser.parse_args()
    # decreasing_period = epoches_num//decreasing_time
    # args.decreasing_lr = ''
    # for i in range(decreasing_time-1):
    #     args.decreasing_lr+=str(decreasing_period*(i+1))
    #     if i<decreasing_time-2:
    #         args.decreasing_lr+=','
            
    # Setting logger 
    info.logger.init(args.logdir, 'train_log_' +current_time)
    logger = info.logger.info
    info.ensure_dir(args.logdir)
    
    # 加载CUDA资源
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = dataset.loading(datatype=args.type, batch_size=args.batch_size, num_workers=0, data_root=os.path.join(tempfile.gettempdir(), os.path.join('public_dataset','pytorch')))
    model = Network.construct(args=args, logger=logger, num_classes=num_classes, pretrained=None)
    
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    best_acc, old_file = 0, None
    t_begin = time.time()
    
    try:
        # Training Section
        update_point = 0
        decreasing_alert = False
        for epoch in range(args.epochs):
            if epoch-update_point >= update_period:
                if decreasing_alert:
                    print(f"\033[31m At epoch {epoch}, the training ends. \033[0m")
                    break
                else:
                    args.grad_scale = args.grad_scale / decreasing_rate
                    decreasing_alert = True
                    update_point = epoch
                    print(f"\033[36m Learning rate: {args.grad_scale:.3e} \033[0m")
            model.train()
            velocity = {}
            for name, layer in list(model.named_parameters())[::-1]:
                velocity[name] = torch.zeros_like(layer)
                
            # if (epoch-1) in decreasing_lr:
            #       args.grad_scale = args.grad_scale / decreasing_rate
            #       print(f"        \033[36m Learning rate: {args.grad_scale:.3e} \033[0m")
            # logger("training phase")
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
            for batch_idx, (data, target) in enumerate(train_loader):
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func.SSE(output,target)
                loss.backward()
                optimizer.step()
                
                # with torch.no_grad():
                #     for param in model.parameters():
                #         if param.requires_grad:  # 只对需要梯度的参数进行操作
                #             max_val = param.abs().max()  # 找到绝对值最大值
                #             param /= max_val
                # with torch.no_grad():
                #     param_stats = {"min": [], "max": [], "mean": [], "std": []}
                #     for name, param in model.named_parameters():
                #         if param.requires_grad:
                #             param_stats["min"].append(param.min().item())
                #             param_stats["max"].append(param.max().item())
                #             param_stats["mean"].append(param.mean().item())
                #             param_stats["std"].append(param.std().item())
                    
                    # # 打印分布统计
                    # print(f"Batch {batch_idx}:")
                    # print(f"Min: {min(param_stats['min']):.4f}, Max: {max(param_stats['max']):.4f}, "
                    #       f"Mean: {sum(param_stats['mean'])/len(param_stats['mean']):.4f}, "
                    #       f"Std: {sum(param_stats['std'])/len(param_stats['std']):.4f}")

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct = pred.cpu().eq(indx_target).sum()
                    acc = float(correct) * 1.0 / len(data)
                    
                    # logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.3f} lr: {:.20f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),loss.data, acc, args.grad_scale))
           
            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            # logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(elapse_time, speed_epoch, speed_batch, eta))      
            
            if epoch % args.test_interval == 0 and epoch != 0:
                model.eval()
                test_loss = 0
                correct = 0
                # logger("testing phase")
                for i, (data, target) in enumerate(test_loader):
                    indx_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    with torch.no_grad():
                        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
                        output = model(data)
                        test_loss_i = loss_func.SSE(output, target)
                        test_loss += test_loss_i.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.cpu().eq(indx_target).sum()
                
                test_loss = test_loss / len(test_loader) # average over number of mini-batch
                test_loss = test_loss.cpu().data.numpy()
                acc = 100. * correct / len(test_loader.dataset)
                # logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss, correct, len(test_loader.dataset), acc))
                accuracy = acc.cpu().data.numpy()
                if acc > best_acc:
                    update_point = epoch
                    decreasing_alert = False
                    path = os.path.join(args.modeldir, args.type, Label)
                    if not os.path.exists(path):
                        os.makedirs(path)
                        print(f"Directory '{path}' created.")
                    
                    new_file = os.path.join(args.modeldir, args.type, Label, str(int(accuracy*100)) + '-' + Label + '-' + str(rand_seed) + '-{}'.format(epoch)+'-'
                                    +'-{}.pth'.format(current_time))
                    info.model_save(model, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    print(f"Epoch {epoch}: New peak accuracy reaching {best_acc:.4f}......")
                    old_file = new_file
            if epoch % (epoches_num // 20) == 0 and epoch !=0:
                print(f"\033[32m    {epoch/epoches_num*100} percent of training has been completed!! \033[0m")
            
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))
