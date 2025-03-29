# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:25:50 2024

@author: dalei
"""

import torch
# Dalei Jiang
# This function returns the squared error/2

def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1.0
    out =  0.5*((logits-target)**2).sum()
    return out