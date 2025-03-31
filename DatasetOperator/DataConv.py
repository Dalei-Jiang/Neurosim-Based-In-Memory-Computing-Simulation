# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:05:25 2025

@author: dalei
"""


import os
import numpy as np
from scipy.signal import convolve2d

resource_path = '../../resource/X_GaN.npy'
X = np.load(resource_path)

# Ideal Kernels
CE = np.array([[1.0]])
SM = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])/9.0
EG = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
HA = np.array([[0.0,-1.0,0.0],[-1.0,5.0,-1.0],[0.0,-1.0,0.0]])
VE = np.array([[0.0,-1.0,0.0],[0.0,2.0,0.0],[0.0,-1.0,0.0]])
HE = np.array([[0.0,0.0,0.0],[-1.0,2.0,-1.0],[0.0,0.0,0.0]])
VS = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
HS = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
RH = np.array([[-1.0,-1.0,-1.0],[-1.0,9.0,-1.0],[-1.0,-1.0,-1.0]])
RE = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])

# Experimental Kernels
vs = np.array([[1.0,0.0,-0.85], [1.74,0.0,-1.79], [1.0,0.0,-0.85]])
hs = np.array([[1.0,1.74,1.0], [0.0,0.0,0.0], [-0.85,-1.79,-0.85]])
sm = np.array([[1.0,1.0,1.0],  [1.0,1.0,1.0], [1.0,1.0,1.0]])
re = np.array([[-0.85,-0.85,-0.85],[-0.85,8.45,-0.85], [-0.85,-0.85,-0.85]])
eg = np.array([[0.0,-0.85,0.0],[-0.85,3.78,-0.85], [0.0,-0.85,0.0]])
ce = np.array([[0.0,0.0,0.0],  [0.0,1.0,0.0], [0.0,0.0,0.0]])
ve = np.array([[1.0,0.0,-0.85],[1.0,0.0,-0.85], [1.0,0.0,-0.85]])
he = np.array([[1.0,1.0,1.0],  [0.0,0.0,0.0], [-0.85,-0.85,-0.85]])

# Convolution
kernel_list = [vs,hs]
step = 3
X_1 = np.reshape(np.array([[[convolve2d(img[0], kernel_list[0], mode='same',boundary='symm') for img in X]]]),(X.shape[0], 1, 36, 60))
X_2 = np.reshape(np.array([[[convolve2d(img[0], kernel_list[1], mode='same',boundary='symm') for img in X]]]),(X.shape[0], 1, 36, 60))
Xnew = np.concatenate((X_1, X_2), axis=2)
Xstep = Xnew[:,:,::step,::step]
print(Xstep.shape)
path = '../../resource/X_GaNmul3_N_VSHS.npy'
# np.save(path, Xstep)
print(f'{path} saved!')