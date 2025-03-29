# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:01:38 2024

@author: dalei
"""
import os
import re
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from scipy.signal import convolve2d

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=int)
    
def softmax(x):
    # 计算每个样本的softmax
    exp_x = np.exp(8*(x - np.max(x, axis=1, keepdims=True)))  # 稳定性调整
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_final_scores(logits):
    # 应用softmax函数
    probabilities = softmax(logits)
    
    # 定义每个类别的分数
    scores = np.array([0.,0.,0.,0.,1.,0.])
    
    # 计算加权平均分数
    final_scores = np.sum(probabilities * scores, axis=1)
    
    return final_scores

# ================ Setting ================ 
mul_setting = True
dataset_type = 'GaN'
# model_name = '9913-GaN-Ideal-512-256-871-3584--2024_09_18_14_39_55.pth'
model_name = 'Ideal_normalize.pth'
label_name = 'GaN'
Xname = 'TEST_1506_revise'
# Xname = 'X_GaN_adv'
yname = 'y_GaN_hex'
# =========================================
num_classes = 6
weight = np.array([0.,15.,30.,45.,60.,75.])
center = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
smooth = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])/9.0
edge = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
enhance = np.array([[0.0,-1.0,0.0],[-1.0,5.0,-1.0],[0.0,-1.0,0.0]])
vsobel = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
hsobel = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
kernel_list = {'center':center,'smooth':smooth,'edge':edge,'enhance':enhance,'vsobel':vsobel,'hsobel':hsobel}

dir_path = './model/analyze'
model_path = os.path.join(dir_path, model_name)
model = torch.load(model_path).cuda()
model.eval()

step = 3
images = np.load(f"./resource/{Xname}.npy")
label = np.load(f"./resource/{yname}.npy")
edge = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
vsob = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
hsob = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
# Xedge = np.reshape(np.array([[[convolve2d(img[0], edge, mode='same') for img in images]]]),(images.shape[0],1,images.shape[2],images.shape[3]))
Xvsob = np.reshape(np.array([[[convolve2d(img[0], vsob, mode='same') for img in images]]]),(images.shape[0],1,images.shape[2],images.shape[3]))
Xhsob = np.reshape(np.array([[[convolve2d(img[0], hsob, mode='same') for img in images]]]),(images.shape[0],1,images.shape[2],images.shape[3]))
images = np.concatenate((Xvsob, Xhsob), axis=2)[:,:,::step,::step]
sample_number = images.shape[0]
ratio1 = 0.0
ratio2 = 1.0
images_train = images[:int(sample_number*ratio1)]
images_test = images[int(sample_number*ratio1):int(sample_number*ratio2)]
labels_train = label[:int(sample_number*ratio1)]
labels_test = label[int(sample_number*ratio1):int(sample_number*ratio2)]
dataset_test = CustomDataset(images_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=0)
print(model)
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        outputs = model(inputs).cpu().numpy()
        all_preds.append(outputs)
preds = np.vstack(all_preds)
result_array = calculate_final_scores(preds)
preds_arr = np.vstack(all_preds)
d1 = 2.5
d2 = 5
d3 = 10
d4 = 30
d = 2
k1 = int(144/60*d1)
k2 = int(144/60*d2)
k3 = int(144/60*d3)
k4 = int(144/60*d4)
k = int(144/60*d)
P = 60
baseline = np.array([np.mean(result_array[max(0,i-P):min(i+P,result_array.shape[0])]) for i in range(0,result_array.shape[0])])
simulate = np.array([np.mean(result_array[max(0,i-k//2):min(i+k//2,result_array.shape[0])]) for i in range(0,result_array.shape[0])])
# plt.figure(figsize=(15,9))
# plt.plot(baseline)
# plt.plot(simulate)
# print(labels.shape)
# plt.plot(weight[label])
score = np.sqrt(np.sum((baseline-simulate)**2)/result_array.shape[0])
t = np.arange(result_array.shape[0])/144.0
plt.figure(figsize=(15,9))
plt.xlabel("Time sequence(s)")
plt.ylabel("Score")
plt.grid()
plt.title(f'Image Flow analyze for {label_name} | Noise level: {score:.3f}')
plt.plot(t,baseline,'r',linewidth=25,alpha=0.3,label="Baseline")
plt.plot(t,np.array([np.mean(result_array[max(0,i-k1):max(k1,i)]) for i in range(0,result_array.shape[0])]),'y',linewidth=1,label=f"{d1} seconds delay")
plt.plot(t,np.array([np.mean(result_array[max(0,i-k2):max(k2,i)]) for i in range(0,result_array.shape[0])]),'g',linewidth=1,label=f"{d2} seconds delay")
plt.plot(t,np.array([np.mean(result_array[max(0,i-k3):max(k3,i)]) for i in range(0,result_array.shape[0])]),'b',linewidth=2,label=f"{d3} seconds delay")
plt.plot(t,np.array([np.mean(result_array[max(0,i-k4):max(k4,i)]) for i in range(0,result_array.shape[0])]),'r',linewidth=3,label=f"{d4} seconds delay")
plt.legend()