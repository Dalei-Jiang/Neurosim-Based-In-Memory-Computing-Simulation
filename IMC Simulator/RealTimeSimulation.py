# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:51:50 2024

@author: dalei
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
from util import calculate_final_scores
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pandas as pd

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
    
"""Plot expert score prediction"""
time_points = np.array([0, 165, 230, 265, 305, 330, 357, 410, 460, 470, 490, 565])
values =      np.array([0,   0,  10,  30,  40,  50,  60,  60,  30,  10,   0,   0])
total_time = 585
num_data_points = 1332
interp_function = interp1d(time_points, values, kind='linear', fill_value='extrapolate')
times = np.linspace(0, total_time, num_data_points)
y_values = interp_function(times)
noise = np.random.uniform(-5, 5, num_data_points)
y_values_with_noise = y_values + noise
y_values_with_noise = np.clip(y_values_with_noise, 0, 60)
smoothed_values = gaussian_filter1d(y_values_with_noise, sigma=30)
resource_path = '../../resource/RealTimeRecord.npy'
X = np.load(resource_path)[...]
label = np.load("../../resource/y_GaN_hex.npy")
Depth, _, H, W = X.shape
SM = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])/9.0
EG = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
HA = np.array([[0.0,-1.0,0.0],[-1.0,5.0,-1.0],[0.0,-1.0,0.0]])
VE = np.array([[0.0,-1.0,0.0],[0.0,2.0,0.0],[0.0,-1.0,0.0]])
HE = np.array([[0.0,0.0,0.0],[-1.0,2.0,-1.0],[0.0,0.0,0.0]])
VS = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
HS = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
RH = np.array([[-1.0,-1.0,-1.0],[-1.0,9.0,-1.0],[-1.0,-1.0,-1.0]])
RE = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])

vs = np.array([[1.0,0.0,-0.85],[1.74,0.0,-1.79],[1.0,0.0,-0.85]])
hs = np.array([[1.0,1.74,1.0],[0.0,0.0,0.0],[-0.85,-1.79,-0.85]])
sm = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])
re = np.array([[-0.85,-0.85,-0.85],[-0.85,8.45,-0.85],[-0.85,-0.85,-0.85]])
eg = np.array([[0.0,-0.85,0.0],[-0.85,3.78,-0.85],[0.0,-0.85,0.0]])
ce = np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
ve = np.array([[1.0,0.0,-0.85],[1.0,0.0,-0.85],[1.0,0.0,-0.85]])
he = np.array([[1.0,1.0,1.0],[0.0,0.0,0.0],[-0.85,-0.85,-0.85]])

""" Select the kernel to concatenate the dataset"""
X_1 = np.reshape(np.array([[[convolve2d(img[0], vs, mode='same', boundary='symm') for img in X]]]),(X.shape[0], 1, 36, 60))
X_2 = np.reshape(np.array([[[convolve2d(img[0], hs, mode='same', boundary='symm') for img in X]]]),(X.shape[0], 1, 36, 60))
Xnew = np.concatenate((X_1, X_2), axis=2)
images = Xnew[:, :, ::3, ::3]
sample_number = images.shape[0]
images_test = images
labels_test = label
dataset_test = CustomDataset(images_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    
"""Set the model"""
scores = [-15, 20, 25, 40, 70, 80]
model_name = 'GaN.pth'
dir_path = '../../model'
model_path = os.path.join(dir_path, model_name)
model = torch.load(model_path).cuda()

all_preds = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.cuda()
        outputs = model(inputs).cpu().numpy()
        all_preds.append(outputs)
apreds = np.vstack(all_preds)
result_array = calculate_final_scores(apreds, scores=scores)

"""Plot"""
noise_levels = []
for k in range(1,40):
    a = np.array([np.mean(result_array[max(0,i-k):max(k,i)]) for i in range(0,700)])
    noise_levels.append(np.std(np.diff(a)))
plt.figure(figsize=(20,6))
plt.title('Noise Level with different delay time')
plt.plot(np.arange(1, 40, 1), noise_levels)
plt.xticks(np.arange(1, 40, 1))
plt.xlabel('Frames')
plt.grid()

d = 15
k = int(144/60*d)
print(f'The delay period is {k} frames, {d} seconds.')
P = 60
t = np.arange(result_array.shape[0])/144*60
plt.figure(figsize=(20,12))
plt.xlabel("Time sequence(s)",fontsize=20)
plt.ylabel("Score",fontsize=20)
plt.grid()
plt.title('GaN Growth Video Grade Curve',fontsize=28)
plt.scatter(time_points, values, color='red', marker='X', s=300, label='Scored Points(By Dr.Ding Wang)', zorder=6)
plt.scatter(165, 0, color='blue', marker='X', s=300, label='Make GaN get spotty', zorder=6)
plt.scatter(330, 50, color='purple', marker='X', s=300, label='Change temperture to make it get streaky', zorder=6)
plt.plot(times, np.array([np.mean(result_array[max(0,i-k):max(k,i)]) for i in range(0,result_array.shape[0])]),'darkred',linewidth=6,label=f"{d} seconds delay")
plt.plot(times, smoothed_values,'blue', linewidth=25,alpha=0.2,label="Baseline(By Dr.Ding Wang)")
plt.ylim(-5.0, 65.0)
plt.xticks(np.arange(0, 600, 30))
plt.legend(
    loc='upper left', 
    fontsize=14, 
    handlelength=2.5, 
    handleheight=2.5, 
    labelspacing=1.5,  
    borderpad=2.0,     
    bbox_to_anchor=(0.0, 1)  
)
plt.savefig('GaN_growth_curve.png', format='png', dpi=300)
noise_levels = []
data = {
    'times': times,
    'Expert Score': smoothed_values,
    'Simulation Score': np.array([np.mean(result_array[max(0,i-k):max(k,i)]) for i in range(0,result_array.shape[0])])
}

df = pd.DataFrame(data)
df.to_excel('../Source/real-time-score.xlsx', index=False)
print("Saving Complete! Excel stored as 'real-time monitor.xlsx'")

