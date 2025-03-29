import torch.nn as nn
import torch

network_list = {
    'GaN':  [('L',480,480,True),
             ('L',480,480,True),
             ('ReLU',True),
             ('L',480,6,  True)]
}

class CIFAR(nn.Module):
    def __init__(self, args, network, num_classes,logger):
        super(CIFAR, self).__init__()   # Make sure that the function is initialized correctly
        self.classifier = network
        print(self.classifier)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, args, logger, in_channels):
    layers = []
    for i, v in enumerate(cfg):
        print(v)
        if v[0] == 'L':
            layers += [nn.Linear(v[1],v[2],bias=v[3])]
        if v[0] == 'ReLU':
            layers += [nn.ReLU(inplace=v[1])]    
    return nn.Sequential(*layers) 

def construct(args, logger, num_classes, pretrained=None):
    network = network_list[args.type]
    in_channels = 1
    layers = make_layers(network, args,logger, in_channels)
    model = CIFAR(args,layers, num_classes = num_classes, logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model
