# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:37:28 2024

@author: dalei
"""

import os
import numpy as np
from sys import exit
from Img_util import Towerimg_conv, PCA_visual, model_loading, epoch_counter, train_test_curve, model_hist, model_attention, Confusion_matrix 
import pandas as pd

if __name__ == '__main__':
    command = 1
    match command:
        # Print the convoluted image of college tower
        case 1:             
            Towerimg_conv()
        # PCA visualize of some dataset
        case 2:
            PCA_visual()
        # Model evaluation images (Confusion matrix, curve, )
        case 3:       
            material_label = 'GaN'
            label_type = input('Input the secondary class label: ')
            log_dir = 'train_test_data'
            
            data_dir = os.path.join(log_dir, material_label, label_type)
            test_log  = np.load(os.path.join(data_dir, 'test.npy'))
            train_log = np.load(os.path.join(data_dir,'train.npy'))
            epoches = epoch_counter(train_log)
            x = np.arange(epoches-1)
            train_loss = train_log[:epoches-1,1]
            valid_loss = test_log[1:epoches,1]
            valid_acc  = test_log[1:epoches,2]
            data = {
                'epoch': x,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc
            }
            
            df = pd.DataFrame(data)
            df.to_excel('train_test_data.xlsx', index=False)
            train_test_curve(test_log, train_log, epoches)
        
        # Parameter heating maps and hists 
        case 4:
            material_label = 'GaN'
            label_type = input('Input the secondary class label: ')
            index = input('Input the epoch number: ')
            if index == '':
                model_name = 'Best.pth'
            elif index == '-1':
                model_name = 'Init.pth'
            else:
                model_name = f'{material_label}-{label_type}-{index}.pth'
            model = model_loading(label_type, model_name)
            print(model)
            model_hist(model)
            model_attention(model)
        
        case 5:
            material_label = 'GaN'
            label_type = input('Input the secondary class label: ')
            index = input('Input the epoch number: ')
            if index == '':
                model_name = 'Best.pth'
            elif index == '-1':
                model_name = 'Init.pth'
            else:
                model_name = f'{material_label}-{label_type}-{index}.pth'
            model = model_loading(label_type, model_name)
            Confusion_matrix(model)
            
            
        case _:
            print(f'Command invalid: {command}.')