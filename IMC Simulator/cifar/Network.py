from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
import torch

Linear_limitation = True
N = 32
class CIFAR(nn.Module):
    def __init__(self, args, features, num_classes, logger):
        super(CIFAR, self).__init__()   # Make sure that the function is initialized correctly
        assert isinstance(features, nn.Sequential), type(features) # features must be an nn.Sequen
        classtype = args.type
        if classtype == 'cifar10' or classtype == 'cifar100':
            input_channel = 8192
            hidden_channel = 1024
        elif classtype == 'MNIST':
            input_channel = 4608
            hidden_channel = 1024
        elif classtype == 'RHEED':
            input_channel = 240*N
            hidden_channel = min(input_channel//2,1024)
        else:
            input_channel = 240*N
            hidden_channel = min(input_channel//2,1024)           
        self.features = features
        if args.type == 'MNIST':
            self.classifier = nn.Sequential(
                QLinear(784, 400, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC0_'),                  
                QLinear(400, 100, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC0_'),                  
                QLinear(100, num_classes, logger=logger,
                        wl_input = args.wl_activate,wl_activate=-1.0,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC7_')
                )            
        elif Linear_limitation == False:
            self.classifier = nn.Sequential(
                QLinear(input_channel, hidden_channel, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC1_'),
                nn.ReLU(inplace=True), 
                QLinear(hidden_channel, num_classes, logger=logger,
                        wl_input = args.wl_activate,wl_activate=-1, wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,name='FC2_'))
        else:
            self.classifier = nn.Sequential(
                QLinear(480, 480, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC0_'),                  
                QLinear(480, 480, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC0_'),                  
                nn.ReLU(inplace=True),
                QLinear(480, num_classes, logger=logger,
                        wl_input = args.wl_activate,wl_activate=-1.0,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC7_')
                )
            
        # One fully connected layer 8192->1024
        # One activation layer ReLU 1024->1024
        # One fully connected layer 1024->num_classes
        
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args, logger, in_channels):
    layers = []
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                             logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                             wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name = 'Conv'+str(i)+'_' )
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
            
    return nn.Sequential(*layers) 

cfg_list = {
    'cifar10': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    'cifar100': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    # 'MNIST':    [('C', 128, 3, 'same', 2.0),
    #             ('C', 128, 3, 'same', 16.0),
    #             ('M', 2, 2),
    #             ('C', 256, 3, 'same', 16.0),
    #             ('C', 256, 3, 'same', 16.0),
    #             ('M', 2, 2),
    #             ('C', 512, 3, 'same', 16.0),
    #             ('C', 512, 3, 'same', 32.0),
    #             ('M', 2, 2)],
    'MNIST':    [],

    'RHEED':  [('C', N, 3, 'same', 2.0),
                ('C', N, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', N*2, 3, 'same', 16.0),
                ('C', N*2, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', N*4, 3, 'same', 16.0),
                ('C', N*4, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    'ScAlN_GaN':  [('C', N, 3, 'same', 2.0),
                ('C', N, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', N*2, 3, 'same', 16.0),
                ('C', N*2, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', N*4, 3, 'same', 16.0),
                ('C', N*4, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    'GaN':      [],
    'ScAlN':    []
}

def construct(args, logger, num_classes, pretrained=None):
    cfg = cfg_list[args.type]
    if args.type == 'cifar10' or args.type == 'cifar100':
        in_channels = 3
    elif args.type == 'MNIST' or args.type == 'RHEED' or args.type == 'GaN' or args.type == 'ScAlN' or args.type == 'ScAlN_GaN':
        in_channels = 1
    layers = make_layers(cfg, args,logger, in_channels)
    model = CIFAR(args, layers, num_classes = num_classes, logger = logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model
