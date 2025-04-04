import torch
from torch.nn import Module
from torch.autograd import Function
import numpy as np
# import matplotlib.pyplot as plt
def shift(x):
    return 2.**torch.round(torch.log2(x))

def S(bits):
    return 2.**(bits-1)

def SR(x):
    # r = torch.cuda.FloatTensor(*x.size()).uniform_()
    r = torch.empty(x.size(), dtype=torch.float, device='cuda').uniform_()
    return torch.floor(x+r)

def C(x, bits):
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1. / S(bits)
    upper = 1  - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper) # Any element which is outside of the boundary will be set as the bound value

def W(x, grad, bits_W, sigmaC2C):
    # add cycle-to-cycle variation here
    c2c = torch.normal(torch.zeros_like(x), sigmaC2C*2*torch.ones_like(x))
    x = x + c2c*torch.sign(torch.abs(grad))
    return C(x, bits_W)


def Q(x, bits):
    assert bits != -1
    if bits==1:
        return torch.sign(x)
    if bits > 15:
        return x
    return torch.round(x*S(bits))/S(bits)

def QW(x, bits, scale=1.0):
    y = Q(C(x, bits), bits)
    # per layer scaling
    if scale>1.8: y /= scale
    return y

def QE(x, bits):
    max_entry = x.abs().max()
    assert max_entry != 0, "QE blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    return Q(C(x, bits), bits)

# TODO Here the nonlinearity property is defined. 
# param.grad.data = wage_quantizer.QG(param.data,args.wl_weight,param.grad.data,args.wl_grad,grad_scale,
#                   torch.from_numpy(paramALTP[j]).cuda(), torch.from_numpy(paramALTD[j]).cuda(), args.max_level, args.max_level)

def QG(weight, bits_W, grad_weight, bits_G, lr, paramALTP, paramALTD, maxLevelLTP, maxLevelLTD):
    max_entry = grad_weight.abs().max()             # choose the maximum absolute entry
    assert max_entry != 0, "QG blow"                # The whole array should not be all zero
    grad_weight /= shift(max_entry)                 # Normalize the array to make maximum 1
    gradient = lr * grad_weight                     # apply learning rate to get gradient
    paramBLTP = GetParamB(paramALTP, maxLevelLTP)   # get B
    paramBLTD = GetParamB(paramALTD, maxLevelLTD)   # get B
    numLevel = max(maxLevelLTP, maxLevelLTD)        # Choose largest level number
    deltaPulse = torch.round((gradient)/2*numLevel) # Assume the linearity here
    paramA = torch.where(torch.sign(deltaPulse)<0, paramALTP, paramALTD).float() # based on the sign, choose relative A
    paramB = torch.where(torch.sign(deltaPulse)<0, paramBLTP, paramBLTD).float() # based on the sign, choose relative B
    xPulse = InvNonlinearWeight(weight, paramA, paramB) # assume max conductance is 1, and min conductance is -1. 
    xNew = NonlinearWeight(xPulse-deltaPulse, paramA, paramB) # Get new grad_weight
    gradient = weight - C(xNew, bits_W) # limit xNew in -1 to 1. weight is our original weight. We get the gradient.
    norm = SR(gradient)                 # stomatic rounding of gradient
    gradient = norm / S(bits_G)         
    return gradient

def NonlinearWeight(xPulse, A, B):
    return B*(1-torch.exp(-xPulse/A))-1

def InvNonlinearWeight(weight, A, B):
    return -A*torch.log(1 - (weight+1)/B)

def GetParamA(NL):
    index = (np.abs(NL)*100).astype(int)-1
    index = np.where(index<0, np.zeros_like(index), index)
    index = np.where(index>899, np.ones_like(index)*899, index)
    sign = np.sign(NL)
    # This normalized paramA table corresponds to nonlinearity label from 0.01 to 9, with step=0.01 
    data = np.array([   126.268958, 63.134314,  42.089359,  31.566827,  25.253264,  21.044185,  
                18.037668,  15.782754,  14.028906,  12.625807,  11.477796,  10.521102,  
                9.711575,   9.017679,   8.416288,   7.890057,   7.425722,   7.012968,   
                6.643650,   6.311253,   6.010503,   5.737083,   5.487429,   5.258571,   
                5.048012,   4.853642,   4.673662,   4.506529,   4.350915,   4.205668,   
                4.069785,   3.942387,   3.822704,   3.710055,   3.603836,   3.503513,   
                3.408606,   3.318688,   3.233376,   3.152324,   3.075221,   3.001784,   
                2.931757,   2.864909,   2.801026,   2.739916,   2.681402,   2.625322,   
                2.571526,   2.519877,   2.470249,   2.422526,   2.376600,   2.332370,   
                2.289745,   2.248638,   2.208970,   2.170666,   2.133656,   2.097877,   
                2.063266,   2.029769,   1.997332,   1.965904,   1.935441,   1.905897,   
                1.877232,   1.849406,   1.822384,   1.796131,   1.770614,   1.745803,   
                1.721669,   1.698184,   1.675322,   1.653059,   1.631371,   1.610236,   
                1.589634,   1.569544,   1.549947,   1.530826,   1.512163,   1.493941,   
                1.476145,   1.458761,   1.441774,   1.425170,   1.408937,   1.393062,   
                1.377534,   1.362340,   1.347472,   1.332917,   1.318666,   1.304709,   
                1.291038,   1.277644,   1.264518,   1.251653,   1.239040,   1.226672,   
                1.214542,   1.202643,   1.190969,   1.179512,   1.168268,   1.157230,   
                1.146393,   1.135750,   1.125297,   1.115029,   1.104940,   1.095027,   
                1.085284,   1.075707,   1.066292,   1.057034,   1.047930,   1.038976,   
                1.030168,   1.021503,   1.012977,   1.004586,   0.996328,   0.988199,   
                0.980196,   0.972317,   0.964558,   0.956917,   0.949390,   0.941976,   
                0.934671,   0.927474,   0.920382,   0.913393,   0.906504,   0.899713,   
                0.893018,   0.886417,   0.879908,   0.873489,   0.867158,   0.860914,   
                0.854754,   0.848677,   0.842681,   0.836765,   0.830926,   0.825164,   
                0.819477,   0.813863,   0.808320,   0.802849,   0.797446,   0.792111,   
                0.786843,   0.781640,   0.776500,   0.771424,   0.766409,   0.761455,   
                0.756560,   0.751723,   0.746944,   0.742221,   0.737553,   0.732939,   
                0.728378,   0.723870,   0.719413,   0.715006,   0.710649,   0.706341,   
                0.702081,   0.697867,   0.693700,   0.689579,   0.685502,   0.681470,   
                0.677480,   0.673533,   0.669628,   0.665764,   0.661941,   0.658157,   
                0.654413,   0.650707,   0.647039,   0.643409,   0.639815,   0.636257,   
                0.632736,   0.629249,   0.625796,   0.622378,   0.618994,   0.615642,   
                0.612322,   0.609035,   0.605779,   0.602555,   0.599361,   0.596197,   
                0.593062,   0.589957,   0.586881,   0.583833,   0.580814,   0.577822,   
                0.574857,   0.571919,   0.569007,   0.566121,   0.563262,   0.560427,   
                0.557618,   0.554833,   0.552072,   0.549336,   0.546623,   0.543934,   
                0.541267,   0.538624,   0.536002,   0.533403,   0.530826,   0.528270,   
                0.525735,   0.523222,   0.520729,   0.518256,   0.515804,   0.513372,   
                0.510959,   0.508565,   0.506191,   0.503836,   0.501499,   0.499181,   
                0.496881,   0.494599,   0.492335,   0.490088,   0.487859,   0.485647,   
                0.483451,   0.481273,   0.479111,   0.476965,   0.474835,   0.472721,   
                0.470623,   0.468541,   0.466473,   0.464421,   0.462384,   0.460362,   
                0.458354,   0.456361,   0.454382,   0.452418,   0.450467,   0.448530,   
                0.446607,   0.444697,   0.442801,   0.440917,   0.439047,   0.437190,   
                0.435345,   0.433514,   0.431694,   0.429887,   0.428092,   0.426310,   
                0.424539,   0.422780,   0.421033,   0.419297,   0.417573,   0.415860,   
                0.414158,   0.412467,   0.410787,   0.409118,   0.407460,   0.405812,   
                0.404175,   0.402549,   0.400932,   0.399326,   0.397730,   0.396143,   
                0.394567,   0.393000,   0.391443,   0.389896,   0.388358,   0.386830,   
                0.385310,   0.383800,   0.382299,   0.380807,   0.379324,   0.377850,   
                0.376385,   0.374928,   0.373479,   0.372040,   0.370608,   0.369185,   
                0.367770,   0.366363,   0.364965,   0.363574,   0.362192,   0.360817,   
                0.359450,   0.358090,   0.356738,   0.355394,   0.354057,   0.352728,   
                0.351406,   0.350091,   0.348784,   0.347484,   0.346190,   0.344904,   
                0.343625,   0.342352,   0.341087,   0.339828,   0.338576,   0.337330,   
                0.336091,   0.334859,   0.333632,   0.332413,   0.331200,   0.329993,   
                0.328792,   0.327597,   0.326409,   0.325226,   0.324050,   0.322879,   
                0.321715,   0.320556,   0.319403,   0.318256,   0.317114,   0.315979,   
                0.314848,   0.313724,   0.312605,   0.311491,   0.310382,   0.309280,   
                0.308182,   0.307090,   0.306003,   0.304921,   0.303844,   0.302772,   
                0.301705,   0.300644,   0.299587,   0.298536,   0.297489,   0.296447,   
                0.295410,   0.294377,   0.293350,   0.292327,   0.291308,   0.290295,   
                0.289285,   0.288281,   0.287281,   0.286285,   0.285294,   0.284307,   
                0.283325,   0.282347,   0.281373,   0.280403,   0.279438,   0.278477,   
                0.277520,   0.276567,   0.275618,   0.274673,   0.273733,   0.272796,   
                0.271863,   0.270935,   0.270010,   0.269089,   0.268171,   0.267258,   
                0.266349,   0.265443,   0.264541,   0.263642,   0.262747,   0.261856,   
                0.260969,   0.260085,   0.259205,   0.258328,   0.257454,   0.256585,   
                0.255718,   0.254855,   0.253996,   0.253140,   0.252287,   0.251437,   
                0.250591,   0.249748,   0.248908,   0.248072,   0.247239,   0.246409,   
                0.245582,   0.244758,   0.243937,   0.243120,   0.242305,   0.241494,   
                0.240685,   0.239880,   0.239077,   0.238278,   0.237481,   0.236687,   
                0.235897,   0.235109,   0.234324,   0.233541,   0.232762,   0.231985,   
                0.231212,   0.230440,   0.229672,   0.228906,   0.228143,   0.227383,   
                0.226626,   0.225871,   0.225118,   0.224368,   0.223621,   0.222877,   
                0.222134,   0.221395,   0.220658,   0.219923,   0.219191,   0.218462,   
                0.217734,   0.217010,   0.216287,   0.215568,   0.214850,   0.214135,   
                0.213422,   0.212711,   0.212003,   0.211297,   0.210594,   0.209892,   
                0.209193,   0.208496,   0.207802,   0.207109,   0.206419,   0.205731,   
                0.205045,   0.204361,   0.203680,   0.203000,   0.202323,   0.201648,   
                0.200975,   0.200303,   0.199634,   0.198967,   0.198302,   0.197639,   
                0.196978,   0.196319,   0.195662,   0.195007,   0.194354,   0.193703,   
                0.193054,   0.192406,   0.191761,   0.191117,   0.190476,   0.189836,   
                0.189198,   0.188562,   0.187928,   0.187295,   0.186664,   0.186036,   
                0.185409,   0.184783,   0.184160,   0.183538,   0.182918,   0.182300,   
                0.181683,   0.181068,   0.180455,   0.179843,   0.179234,   0.178625,   
                0.178019,   0.177414,   0.176811,   0.176209,   0.175609,   0.175011,   
                0.174414,   0.173819,   0.173226,   0.172634,   0.172043,   0.171454,   
                0.170867,   0.170281,   0.169697,   0.169114,   0.168533,   0.167953,   
                0.167375,   0.166798,   0.166222,   0.165649,   0.165076,   0.164505,   
                0.163936,   0.163368,   0.162801,   0.162236,   0.161672,   0.161109,   
                0.160548,   0.159989,   0.159430,   0.158873,   0.158318,   0.157764,   
                0.157211,   0.156659,   0.156109,   0.155560,   0.155013,   0.154466,   
                0.153921,   0.153378,   0.152835,   0.152294,   0.151755,   0.151216,   
                0.150679,   0.150143,   0.149608,   0.149075,   0.148542,   0.148011,   
                0.147481,   0.146953,   0.146425,   0.145899,   0.145374,   0.144850,   
                0.144328,   0.143806,   0.143286,   0.142767,   0.142249,   0.141732,   
                0.141217,   0.140702,   0.140189,   0.139676,   0.139165,   0.138655,   
                0.138147,   0.137639,   0.137132,   0.136627,   0.136122,   0.135619,   
                0.135117,   0.134616,   0.134116,   0.133617,   0.133119,   0.132622,   
                0.132126,   0.131631,   0.131138,   0.130645,   0.130153,   0.129663,   
                0.129173,   0.128685,   0.128197,   0.127711,   0.127225,   0.126741,   
                0.126258,   0.125775,   0.125294,   0.124813,   0.124334,   0.123855,   
                0.123378,   0.122901,   0.122426,   0.121951,   0.121478,   0.121005,   
                0.120533,   0.120063,   0.119593,   0.119124,   0.118656,   0.118189,   
                0.117723,   0.117258,   0.116794,   0.116331,   0.115869,   0.115407,   
                0.114947,   0.114487,   0.114029,   0.113571,   0.113114,   0.112659,   
                0.112204,   0.111750,   0.111296,   0.110844,   0.110393,   0.109942,   
                0.109493,   0.109044,   0.108596,   0.108149,   0.107703,   0.107258,   
                0.106813,   0.106370,   0.105927,   0.105486,   0.105045,   0.104605,   
                0.104166,   0.103727,   0.103290,   0.102853,   0.102417,   0.101982,   
                0.101548,   0.101115,   0.100683,   0.100251,   0.099820,   0.099390,   
                0.098961,   0.098533,   0.098105,   0.097679,   0.097253,   0.096828,   
                0.096404,   0.095981,   0.095558,   0.095136,   0.094715,   0.094295,   
                0.093876,   0.093458,   0.093040,   0.092623,   0.092207,   0.091792,   
                0.091377,   0.090964,   0.090551,   0.090139,   0.089728,   0.089317,   
                0.088907,   0.088498,   0.088090,   0.087683,   0.087276,   0.086871,   
                0.086466,   0.086062,   0.085658,   0.085256,   0.084854,   0.084453,   
                0.084052,   0.083653,   0.083254,   0.082856,   0.082459,   0.082062,   
                0.081667,   0.081272,   0.080878,   0.080484,   0.080092,   0.079700,   
                0.079309,   0.078919,   0.078529,   0.078140,   0.077752,   0.077365,   
                0.076979,   0.076593,   0.076208,   0.075824,   0.075440,   0.075057,   
                0.074675,   0.074294,   0.073914,   0.073534,   0.073155,   0.072777,   
                0.072400,   0.072023,   0.071647,   0.071272,   0.070897,   0.070524,   
                0.070151,   0.069778,   0.069407,   0.069036,   0.068666,   0.068297,   
                0.067929,   0.067561,   0.067194,   0.066827,   0.066462,   0.066097,   
                0.065733,   0.065370,   0.065007,   0.064645,   0.064284,   0.063924,   
                0.063564,   0.063206,   0.062847,   0.062490,   0.062133,   0.061777,   
                0.061422,   0.061068,   0.060714,   0.060361,   0.060009,   0.059657,   
                0.059306,   0.058956,   0.058607,   0.058259,   0.057911,   0.057564,   
                0.057217,   0.056871,   0.056527,   0.056182,   0.055839,   0.055496,   
                0.055154,   0.054813,   0.054472,   0.054132,   0.053793,   0.053455,   
                0.053117,   0.052781,   0.052444,   0.052109,   0.051774,   0.051440,   
                0.051107,   0.050774,   0.050443,   0.050112,   0.049781,   0.049452,   
                0.049123,   0.048795,   0.048467,   0.048141,   0.047815,   0.047489,   
                0.047165,   0.046841,   0.046518,   0.046196,   0.045874,   0.045553,   
                0.045233,   0.044914,   0.044595,   0.044277,   0.043960,   0.043643,   
                0.043328,   0.043013,   0.042698,   0.042385,   0.042072,   0.041760,   
                0.041449,   0.041138,   0.040828,   0.040519,   0.040211,   0.039903,   
                0.039596,   0.039290,   0.038984,   0.038680,   0.038376,   0.038072,   
                0.037770,   0.037468,   0.037167,   0.036867,   0.036567,   0.036268,   
                0.035970,   0.035673,   0.035376,   0.035080,   0.034785,   0.034491,   
                0.034197,   0.033904,   0.033612,   0.033321,   0.033030,   0.032740,   
                0.032451,   0.032163,   0.031875,   0.031588,   0.031302,   0.031016,   
                0.030732,   0.030448,   0.030165,   0.029882,   0.029601,   0.029320,   
                0.029040,   0.028760,   0.028482,   0.028204,   0.027927,   0.027651,   
                0.027375,   0.027101,   0.026827,   0.026553,   0.026281,   0.026009,   
                0.025738,   0.025468,   0.025199,   0.024930,   0.024663,   0.024396,   
                0.024129,   0.023864,   0.023599,   0.023336,   0.023073,   0.022810])

    # extend A table to 2d or 4d
    ADim = np.append(np.delete(index.shape,-1),1)
    lookupdata = np.tile(data,ADim)
    # find a value according to index from the extend table
    y = np.take_along_axis(lookupdata, index, axis=-1)
    A = sign * y
    # print(f"The A is {A}")
    return A


def GetParamB(A, maxLevel):
    return 2 / (1 - torch.exp(-maxLevel/A))

# =============================================================================================


def Retention(x, t, v, detect, target):
    lower = torch.min(x).item()
    upper = torch.max(x).item()
    target = (torch.max(x).item() - torch.min(x).item())*target
    if detect == 1: # need to define the sign of v 
        sign = torch.zeros_like(x)
        truncateX = (x+1)/2
        truncateTarget = (target+1)/2
        sign = torch.sign(torch.add(torch.zeros_like(x),truncateTarget)-truncateX)
        ratio = t**(v*sign)
    else :  # random generate target for each cell
        sign = torch.randint_like(x, -1, 2)
        truncateX = (x+1)/2
        ratio = t**(v*sign)

    return torch.clamp((2*truncateX*ratio-1), lower, upper)

def NonLinearQuantizeOut(x, bit):
    minQ = torch.min(x)
    delta = torch.max(x) - torch.min(x)
    #print(minQ)
    #print(delta)
    if (bit == 3) :
        # 3-bit ADC
        y = x.clone()
        base = torch.zeros_like(y)

        bound = np.array([0.02, 0.08, 0.12, 0.18, 0.3, 0.5, 0.7, 1])
        out = np.array([0.01, 0.05, 0.1, 0.15, 0.24, 0.4, 0.6, 0.85])

        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        
    elif (bit == 4):
        y = x.clone()
        # 4-bit ADC
        base = torch.zeros_like(y)
        
        # good for 2-bit cell
        bound = np.array([0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1])
        out = np.array([0.01, 0.035, 0.065, 0.1, 0.14, 0.18, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.55, 0.65, 0.775, 0.925])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y.data<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y.data) & (y.data<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y.data) & (y.data<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y.data) & (y.data<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y.data) & (y.data<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y.data) & (y.data<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y.data) & (y.data<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y.data) & (y.data<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        y = torch.where(((minQ+ref[7]*delta)<=y.data) & (y.data<(minQ+ref[8]*delta)), torch.add(base,(minQ+quant[8]*delta)), y)
        y = torch.where(((minQ+ref[8]*delta)<=y.data) & (y.data<(minQ+ref[9]*delta)), torch.add(base,(minQ+quant[9]*delta)), y)
        y = torch.where(((minQ+ref[9]*delta)<=y.data) & (y.data<(minQ+ref[10]*delta)), torch.add(base,(minQ+quant[10]*delta)), y)
        y = torch.where(((minQ+ref[10]*delta)<=y.data) & (y.data<(minQ+ref[11]*delta)), torch.add(base,(minQ+quant[11]*delta)), y)
        y = torch.where(((minQ+ref[11]*delta)<=y.data) & (y.data<(minQ+ref[12]*delta)), torch.add(base,(minQ+quant[12]*delta)), y)
        y = torch.where(((minQ+ref[12]*delta)<=y.data) & (y.data<(minQ+ref[13]*delta)), torch.add(base,(minQ+quant[13]*delta)), y)
        y = torch.where(((minQ+ref[13]*delta)<=y.data) & (y.data<(minQ+ref[14]*delta)), torch.add(base,(minQ+quant[14]*delta)), y)
        y = torch.where(((minQ+ref[14]*delta)<=y.data) & (y.data<(minQ+ref[15]*delta)), torch.add(base,(minQ+quant[15]*delta)), y)
        
    elif (bit == 5):
        y = x.clone()
        # 5-bit ADC
        base = torch.zeros_like(y)
        """
        # good for 2-bit cell
        bound = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        out = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975])
        """
        
        # 4-bit cell
        bound = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1])
        out = np.array([0.001, 0.003, 0.007, 0.010, 0.015, 0.020, 0.030, 0.040, 0.055, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.75, 0.85, 0.95])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,minQ+quant[0]*delta), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,minQ+quant[1]*delta), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,minQ+quant[2]*delta), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,minQ+quant[3]*delta), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,minQ+quant[4]*delta), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,minQ+quant[5]*delta), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,minQ+quant[6]*delta), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,minQ+quant[7]*delta), y)
        y = torch.where(((minQ+ref[7]*delta)<=y) & (y<(minQ+ref[8]*delta)), torch.add(base,minQ+quant[8]*delta), y)
        y = torch.where(((minQ+ref[8]*delta)<=y) & (y<(minQ+ref[9]*delta)), torch.add(base,minQ+quant[9]*delta), y)
        y = torch.where(((minQ+ref[9]*delta)<=y) & (y<(minQ+ref[10]*delta)), torch.add(base,minQ+quant[10]*delta), y)
        y = torch.where(((minQ+ref[10]*delta)<=y) & (y<(minQ+ref[11]*delta)), torch.add(base,minQ+quant[11]*delta), y)
        y = torch.where(((minQ+ref[11]*delta)<=y) & (y<(minQ+ref[12]*delta)), torch.add(base,minQ+quant[12]*delta), y)
        y = torch.where(((minQ+ref[12]*delta)<=y) & (y<(minQ+ref[13]*delta)), torch.add(base,minQ+quant[13]*delta), y)
        y = torch.where(((minQ+ref[13]*delta)<=y) & (y<(minQ+ref[14]*delta)), torch.add(base,minQ+quant[14]*delta), y)
        y = torch.where(((minQ+ref[14]*delta)<=y) & (y<(minQ+ref[15]*delta)), torch.add(base,minQ+quant[15]*delta), y)
        y = torch.where(((minQ+ref[15]*delta)<=y) & (y<(minQ+ref[16]*delta)), torch.add(base,minQ+quant[16]*delta), y)
        y = torch.where(((minQ+ref[16]*delta)<=y) & (y<(minQ+ref[17]*delta)), torch.add(base,minQ+quant[17]*delta), y)
        y = torch.where(((minQ+ref[17]*delta)<=y) & (y<(minQ+ref[18]*delta)), torch.add(base,minQ+quant[18]*delta), y)
        y = torch.where(((minQ+ref[18]*delta)<=y) & (y<(minQ+ref[19]*delta)), torch.add(base,minQ+quant[19]*delta), y)
        y = torch.where(((minQ+ref[19]*delta)<=y) & (y<(minQ+ref[20]*delta)), torch.add(base,minQ+quant[20]*delta), y)
        y = torch.where(((minQ+ref[20]*delta)<=y) & (y<(minQ+ref[21]*delta)), torch.add(base,minQ+quant[21]*delta), y)
        y = torch.where(((minQ+ref[21]*delta)<=y) & (y<(minQ+ref[22]*delta)), torch.add(base,minQ+quant[22]*delta), y)
        y = torch.where(((minQ+ref[22]*delta)<=y) & (y<(minQ+ref[23]*delta)), torch.add(base,minQ+quant[23]*delta), y)
        y = torch.where(((minQ+ref[23]*delta)<=y) & (y<(minQ+ref[24]*delta)), torch.add(base,minQ+quant[24]*delta), y)
        y = torch.where(((minQ+ref[24]*delta)<=y) & (y<(minQ+ref[25]*delta)), torch.add(base,minQ+quant[25]*delta), y)
        y = torch.where(((minQ+ref[25]*delta)<=y) & (y<(minQ+ref[26]*delta)), torch.add(base,minQ+quant[26]*delta), y)
        y = torch.where(((minQ+ref[26]*delta)<=y) & (y<(minQ+ref[27]*delta)), torch.add(base,minQ+quant[27]*delta), y)
        y = torch.where(((minQ+ref[27]*delta)<=y) & (y<(minQ+ref[28]*delta)), torch.add(base,minQ+quant[28]*delta), y)
        y = torch.where(((minQ+ref[28]*delta)<=y) & (y<(minQ+ref[29]*delta)), torch.add(base,minQ+quant[29]*delta), y)
        y = torch.where(((minQ+ref[29]*delta)<=y) & (y<(minQ+ref[30]*delta)), torch.add(base,minQ+quant[30]*delta), y)
        y = torch.where(((minQ+ref[30]*delta)<=y) & (y<(minQ+ref[31]*delta)), torch.add(base,minQ+quant[31]*delta), y)
        
        
    else:
        y = x.clone()
    return y


def LinearQuantizeOut(x, bit):
    minQ = torch.min(x)
    delta = torch.max(x) - torch.min(x)
    y = x.clone()

    stepSizeRatio = 2.**(-bit)
    stepSize = stepSizeRatio*delta.item()
    index = torch.clamp(torch.floor((x-minQ.item())/stepSize), 0, (2.**(bit)-1))
    y = index*stepSize + minQ.item()

    return y


class WAGERounding(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        if self.bits_E == -1: return grad_output, None, None, None

        if self.needs_input_grad[0]:
            try:
                grad_input = QE(grad_output, self.bits_E)
            except AssertionError as e:
                print("="*80)
                print("Error backward:%s"%self.optional)
                print("-"*80)
                print(grad_output.max())
                print(grad_output.min())
                print("="*80)
                raise e
        else:
            grad_input = grad_output

        return grad_input, None, None, None

class WAGERounding_forward(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None


quantize_wage = WAGERounding.apply

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E, name="", writer=None):
        super(WAGEQuantizer, self).__init__()
        self.bits_A = bits_A
        self.bits_E = bits_E
        self.name = name
        self.writer = writer

    def forward(self, x):
        if self.bits_A != -1:
            x = C(x, self.bits_A) #  keeps the gradients
        #print(x.std())
        y = quantize_wage(x, self.bits_A, self.bits_E, self.name)
        if self.writer is not None:
            self.writer.add_histogram(
                    "activation-before/%s"%self.name, x.clone().cpu().data.numpy())
            self.writer.add_histogram(
                    "activation-after/%s"%self.name, y.clone().cpu().data.numpy())
        return y

def WAGEQuantizer_f(x, bits_A, bits_E, name=""):
        if bits_A != -1:
            x = C(x, bits_A) #  keeps the gradients
        y = quantize_wage(x, bits_A, bits_E, name)
        return y

# if __name__ == "__main__":
#     import numpy as np
#     np.random.seed(10)
#     shape = (5,5)
#     # test QG
#     test_data = np.random.rand(*shape)
#     r = np.random.rand(*shape)
#     print(test_data*10)
#     print(r*10)
#     test_tensor = torch.from_numpy(test_data).float()
#     rand_tensor = torch.from_numpy(r).float()
#     lr = 2
#     bits_W = 2
#     bits_G = 8
#     bits_A = 8
#     bits_E = 8
#     bits_R = 16
#     print("="*80)
#     print("Gradient")
#     print("="*80)
#     quant_data = QG(test_tensor, bits_G, bits_R, lr, rand_tensor).data.numpy()
#     print(quant_data)
#     # test QA
#     print("="*80)
#     print("Activation")
#     print("="*80)
#     quant_data = QA(test_tensor, bits_A).data.numpy()
#     print(quant_data)
#     # test QW
#     print("="*80)
#     print("Weight")
#     print("="*80)
#     quant_data = QW(test_tensor, bits_W, scale=16.0).data.numpy()
#     print(quant_data)
#     # test QW
#     print("="*80)
#     print("Error")
#     print("="*80)
#     quant_data = QE(test_tensor, bits_E).data.numpy()
#     print(quant_data)

