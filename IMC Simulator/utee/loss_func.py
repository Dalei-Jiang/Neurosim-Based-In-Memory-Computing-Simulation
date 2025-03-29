import torch
# Dalei Jiang
# This function returns the squared error/2

def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1.0
    out =  0.5*((logits-target)**2).sum()
    return out

def SSE_weighted(logits, label, distance_list=[0.0,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,9.8,10.0]):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1.0
    offset =  logits-target
    distance_tensor = torch.tensor(distance_list)
    result = torch.empty_like(offset)
    for i in range(offset.size(0)):
        label_i = label[i].item()
        for j in range(offset.size(1)):
            diff = abs(distance_tensor[label_i] - distance_tensor[j])
            result[i, j] = offset[i, j] * diff
    out = (result**2).sum()
    return out