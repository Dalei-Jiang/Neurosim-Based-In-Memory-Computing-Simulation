# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:25:29 2024

@author: dalei
"""
import os
from cifar import dataset
import tempfile
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import numpy as np
import matplotlib.pyplot as plt

def model_loading(label_type, model_name):
    dir_path = '.\\model'
    model_path = os.path.join(dir_path, model_name)
    model = torch.load(model_path).cuda()
    model.eval()
    print(f'Model loaded: {model_path}')
    return model

def Confusion_matrix(model):
    dataset_name = 'GaN'
    datachoice = input('What is your dataset label choice? ')
    Label = input('What is the image label? ')
    _, test_loader = dataset.loading(
        datatype=dataset_name,
        batch_size=1,
        label=datachoice,
        num_workers=0,
        data_root=os.path.join(tempfile.gettempdir(), os.path.join('public_dataset', 'pytorch'))
    )
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Generate the 6x6 confusion matrix with labels fixed from 0 to 5
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=np.arange(6))
    print("Raw Confusion Matrix:\n", conf_matrix)  # Debug print
    conf_matrix_normalized = conf_matrix.astype('float')
    row_sums = conf_matrix.sum(axis=1, keepdims=True)  # Keep row dimension for safe division
    conf_matrix_normalized = np.divide(conf_matrix_normalized, row_sums, where=row_sums != 0) * 100  # Convert to percentages
    print("Normalized Confusion Matrix:\n", conf_matrix_normalized)  # Debug print
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix_normalized, cmap='Reds', vmin=0, vmax=100)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(conf_matrix_normalized):
        color = 'white' if i == j else 'black'  # 对角线上的字体颜色设为白色，其余的设为黑色
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color=color)
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(np.arange(6))
    ax.set_yticklabels(np.arange(6))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {Label}')
    plt.show()
    
if __name__ == '__main__':
    material_label = 'GaN'
    label_type = input('Input the secondary class label: ')
    index = input('Input the epoch number: ')
    if index == '':
        model_name = 'VSHS_ideal.pth'
    elif index == '-1':
        model_name = 'Init.pth'
    else:
        model_name = f'{material_label}-{label_type}-{index}.pth'
    model = model_loading(label_type, model_name)
    print(model)
    Confusion_matrix(model)