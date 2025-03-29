import argparse
import os
import time
import torch
import random
import torch.optim as optim
import numpy as np
import tempfile
from datetime import datetime
from util import email
from torch.autograd import Variable
from cifar import dataset, Network
from utee import misc, make_path, loss_func, wage_quantizer, hook, system_setting
from tqdm import tqdm
import warnings
env = system_setting.check_script_execution()
warnings.filterwarnings(
    "ignore",
    message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*",
    category=UserWarning,
)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Should be modified if the training environment changes@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
datachoice = 'N_VSHS'
dataset_name = 'GaN'
epoches_num = 10000
model_name = ''
log_perm = True
Sim_B = False
lr = 0.5
decreasing_rate = 2.0
update_period = 32 # 128
tolerance = 5
Label = f'{datachoice}-Linear'
trial_time = 12
report_best_acc = 99.0
t_begin = time.time()
#---------------------------------------------Parameter setting---------------------------------------------
accuracy_list = []

# 952
for _ in range(trial_time):    
    seed = random.randint(0, 99999) # 基于时间的变化，确保每次不同
    random.seed(seed)  # 设置随机数种子
    rand_seed = random.randint(1, 1000)
    print(f"Generated random number: {rand_seed}")
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
    # Selecting the dataset type
    parser.add_argument('--type', default=dataset_name, help='dataset for training') # 选择的数据集
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 200)') # Batch size, 所见即所得
    parser.add_argument('--epochs', type=int, default=epoches_num, help='number of epochs to train (default: 32)') # epochs
    parser.add_argument('--grad_scale', type=float, default=lr, help='learning rate for wage delta calculation') # 这里的wage delta calculation是什么？
    parser.add_argument('--seed', type=int, default=rand_seed, help='random seed (default: 117)') # random seed 选取
    parser.add_argument('--log_interval', type=int, default=1,  help='how many batches to wait before logging training status default = 100')
    parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test (default = 1)')
    parser.add_argument('--logdir', default='log', help='folder to save to the log') #logdir是日志的存储地点
    parser.add_argument('--modeldir', default='model', help='folder to save to the model') #logdir是日志的存储地点
    parser.add_argument('--decreasing_lr', default='', help='decreasing strategy') # 什么是decreasing strategy
    parser.add_argument('--wl_weight', type = int, default=2)
    parser.add_argument('--wl_grad', type = int, default=8)
    parser.add_argument('--wl_activate', type = int, default=8)
    parser.add_argument('--wl_error', type = int, default=8)
    parser.add_argument('--inference', default=0)
    parser.add_argument('--onoffratio', default=10)
    parser.add_argument('--cellBit', default=1)
    parser.add_argument('--subArray', default=128)
    parser.add_argument('--ADCprecision', default=5)
    parser.add_argument('--vari', default=0)
    parser.add_argument('--t', default=0)
    parser.add_argument('--v', default=0)
    parser.add_argument('--detect', default=0)
    parser.add_argument('--target', default=0)
    parser.add_argument('--nonlinearityLTPconv', default=0.01)      # Nonlinearity of LTP
    parser.add_argument('--nonlinearityLTDconv', default=-0.01)     # Nonlinearity of LTD
    parser.add_argument('--nonlinearityLTPconnect', default=0.01)   # Nonlinearity of LTP
    parser.add_argument('--nonlinearityLTDconnect', default=-0.01)  # Nonlinearity of LTD
    parser.add_argument('--max_level', default=100)
    parser.add_argument('--c2cVari', default=0) # cycle-to-cycle variance
    
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args = parser.parse_args()
    
    args.wl_weight = 6            # weight precision
    args.wl_grad = 6              # gradient precision
    args.cellBit = 6              # cell precision (in V2.0, we only support one-cell-per-synapse, i.e. cellBit==wl_weight==wl_grad)
    args.max_level = 64           # Maximum number of conductance states during weight update (floor(log2(max_level))=cellBit) 
    args.c2cVari = 0.0            # cycle-to-cycle variation
    args.d2dVari = 0.0            # device-to-device variation
    
    args.nonlinearityLTPconv = 1.49   # nonlinearity in LTP
    args.nonlinearityLTDconv = -2.31   # no3nlinearity in LTD (negative if LTP and LTD are asymmetric)
    args.nonlinearityLTPconnect = 1.49   # nonlinearity in LTP
    args.nonlinearityLTDconnect = -2.31   # nonlinearity in LTD (negative if LTP and LTD are asymmetric)
    # momentum
    gamma = 0.9
    alpha = 1.0 - gamma
        
    #************************************************************************************************************
    args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs'])
    misc.logger.init(args.logdir, 'train_log_' +current_time)
    logger = misc.logger.info
    
    # logger
    misc.ensure_dir(args.logdir)
    logger("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger('{}: {}'.format(k, v))
    logger("========================================")
    
    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # data loader and model
    if args.type == 'MNIST' or args.type == 'cifar10':
        num_classes = 10
    elif args.type == 'cifar100':
        num_classes = 100
    elif args.type == 'RHEED' or args.type == 'ScAlN_GaN':
        num_classes = 13
    else:
        num_classes = 6
    # processkernel = kernel_list[kernel_select]
    train_loader, test_loader = dataset.loading(datatype=args.type, batch_size=args.batch_size, label=datachoice, num_workers=0, data_root=os.path.join(tempfile.gettempdir(), os.path.join('public_dataset','pytorch')))
    max_iterations = len(train_loader.dataset) // args.batch_size
    if model_name == '':
        model = Network.construct(args=args, logger=logger, num_classes=num_classes)
        misc.model_save(model, 'Init.pth')
        # print(model)
    else:
        model_path = os.path.join(args.modeldir, model_name)
        model = torch.load(model_path)
    
    if args.cuda:
        model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=lr)
        
    best_acc, old_file = 0, None
    grad_scale = args.grad_scale
    
    train_log = np.zeros((args.epochs,3))
    test_log = np.zeros((int(np.ceil(args.epochs/args.test_interval)),3))
    if not os.path.exists(os.path.join(args.logdir, args.type)):
        os.makedirs(os.path.join(args.logdir, args.type))

    # ready to go
    if __name__ == '__main__':
        if args.cellBit != args.wl_weight:
            print("Warning: Weight precision should be the same as the cell precison !")
        # add d2dVari
        paramALTP = {}
        paramALTD = {}
# -------------------------------------------------------------------------------------------------------        
        for name, layer in list(model.named_parameters())[::-1]:
            d2dVariation = torch.normal(torch.zeros_like(layer), args.d2dVari*torch.ones_like(layer))
            layer_property = name.split('.')[0]
            if layer_property == 'classifier':
                NL_LTP = torch.ones_like(layer)*args.nonlinearityLTPconnect+d2dVariation
                NL_LTD = torch.ones_like(layer)*args.nonlinearityLTDconnect+d2dVariation
            elif layer_property == 'features':
                NL_LTP = torch.ones_like(layer)*args.nonlinearityLTPconv+d2dVariation
                NL_LTD = torch.ones_like(layer)*args.nonlinearityLTDconv+d2dVariation
            paramALTP[name] = wage_quantizer.GetParamA(NL_LTP.cpu().numpy())*args.max_level
            paramALTD[name] = wage_quantizer.GetParamA(NL_LTD.cpu().numpy())*args.max_level
# =======================================================================================================
        if log_perm:
            train_test_dir = './train_test_data'
            result_dir = os.path.join(train_test_dir, args.type, str(args.epochs)+str('_')+current_time)
            os.makedirs(result_dir)
            
        update_point = 0
        tol_counter = 0
        for epoch in range(args.epochs):
            if epoch-update_point >= update_period:
                if tol_counter >= tolerance:
                    print(f"\033[31m At epoch {epoch}, the training ends. \033[0m")
                    break
                else:
                    args.grad_scale = args.grad_scale / decreasing_rate
                    tol_counter += 1
                    update_point = epoch
                    print(f"\033[36m Learning rate: {args.grad_scale:.3e} \033[0m")
            model.train()
            path = os.path.join(args.modeldir, args.type, Label)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory '{path}' created.")

            velocity = {}
# -------------------------------------------------------------------------------------------------------
            for name, layer in list(model.named_parameters())[::-1]:
                velocity[name] = torch.zeros_like(layer)
# =======================================================================================================
            # logger("training phase")
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

            for batch_idx, (data, target) in enumerate(tqdm(train_loader,total=len(train_loader)//args.batch_size)):
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)

                loss = loss_func.SSE(output,target)
                loss.backward()
                # introduce non-ideal property
                for name, param in list(model.named_parameters())[::-1]:
# ------------------------------------------------------------------------------------------------------                                      
                    velocity[name] = gamma * velocity[name] + alpha * param.grad.data
                    param.grad.data = velocity[name]
                    param.grad.data = wage_quantizer.QG(param.data,args.wl_weight,param.grad.data,args.wl_grad,grad_scale,
                                  torch.from_numpy(paramALTP[name]).cuda(), torch.from_numpy(paramALTD[name]).cuda(), args.max_level, args.max_level)
# =======================================================================================================
                optimizer.step()
    
                for name, param in list(model.named_parameters())[::-1]:
                    param.data = wage_quantizer.W(param.data,param.grad.data,args.wl_weight,args.c2cVari)
    
                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct = pred.cpu().eq(indx_target).sum()
                    acc = float(correct) * 1.0 / len(data)
                    
            train_log[epoch, 0] = epoch
            train_log[epoch, 1] = loss.data
            train_log[epoch, 2] = acc
            print(f"Epoch {epoch}: Loss {loss.data:.4f} | accuracy: {acc*100:.2f}")
            
            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            path = os.path.join(args.modeldir, args.type, Label)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory '{path}' created.")
            if epoch % 50 == 0 or epoch < 1:
                new_file = os.path.join(args.modeldir, args.type, Label, args.type + '-' + Label + '-{}'.format(epoch)+ '.pth')
                misc.model_save(model, new_file)
            
#---------------------------------------Test part---------------------------------------
            if epoch % args.test_interval == 0 and epoch != 0:
                model.eval()
                test_loss = 0
                correct = 0
                # logger("testing phase")
                for i, (data, target) in enumerate(test_loader):
                    if i==0:
                        hook_handle_list = hook.hardware_evaluation(model,args.wl_weight,args.wl_activate,epoch,env)
                    indx_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    with torch.no_grad():
                        data, target = Variable(data), Variable(target)
                        output = model(data)
                        test_loss_i = loss_func.SSE(output, target)
                        test_loss += test_loss_i.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.cpu().eq(indx_target).sum()
                    if i==0:
                        hook.remove_hook_list(hook_handle_list)
                
                test_loss = test_loss / len(test_loader) # average over number of mini-batch
                test_loss = test_loss.cpu().data.numpy()
                acc = 100. * correct / len(test_loader.dataset)
                accuracy = acc.cpu().data.numpy()
                
                test_log[epoch // args.test_interval, 0] = epoch
                test_log[epoch // args.test_interval, 1] = test_loss
                test_log[epoch // args.test_interval, 2] = accuracy
                if acc > best_acc:
                    update_point = epoch
                    tol_counter = 0
                    path = os.path.join(args.modeldir, args.type, Label)
                    if not os.path.exists(path):
                        os.makedirs(path)
                        print(f"Directory '{path}' created.")
                    new_file = os.path.join(args.modeldir, args.type, Label, 'Best'+'-'+str(128)+'-'+str(rand_seed)+'-'+args.type + '-' + Label + '-{}'.format(epoch)+'-'+str(int(accuracy*100))+'-{}.pth'.format(current_time))
                    misc.model_save(model, new_file, old_file=old_file)
                    best_acc = acc
                    print(f"    Epoch {epoch}: New peak accuracy reaching {best_acc:.4f}......")
                    old_file = new_file
                if best_acc > report_best_acc:
                    report_best_acc = best_acc
                    body = f"Training Accuracy reaches {best_acc:.4f}"
                    subject = "Training acuuracy break throughs!"
                    email(body, subject)
                    
        
        #   Store all the npy files in related path
            if log_perm:
                np.save(os.path.join(result_dir, f'train_{best_acc}.npy'),  train_log)
                np.save(os.path.join(result_dir, f'test_{best_acc}.npy'),   test_log)
    accuracy_list.append(best_acc)
    
#****************************************************************************************
best_acc = max(accuracy_list)
average_acc = np.mean(np.array(accuracy_list))
body = "Training is complete!\nTotal Elapse: {:.2f} minutes\nBest Result: {:.4f}%\nAverage accuracy: {:.4f}%".format((time.time()-t_begin)/60.0, best_acc, average_acc)
subject = f"Notification: {datachoice} Training completed"
email(body, subject)

#******************************************************************************************
