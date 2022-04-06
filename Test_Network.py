from torchvision.models import resnet50
import torch
import torchvision.models as models
from torch.autograd import Variable

import time
import argparse
from ptflops import get_model_complexity_info
from Network import MPAPNet_res50, MPAPNet_res18, MPAPNet_vgg19, MPAPNet_res101, MPAPNet_res152, MPAPNet_rawmobilenetv2, MPAPNet_vgg16
 

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batch', default=128, type=int) 
parser.add_argument('--flops', action='store_true', default=False, help='test flops and params')

args = parser.parse_args()

if args.flops:
    print('#### Test Case ###')
    b = args.batch 
    x = Variable(torch.rand(b, 3, 224, 224)).cuda()
    #model = MPAPNet_res50(num_classes=47, feature_index=[4]).cuda()
    model = resnet50().cuda()
    model_name = "Our"

    t = time.time()
    y = model(x)
    t = (time.time() - t) / b
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Output shape:', y[0].size())
    print('Time cost:', t)
    print("%s |%s |%s" % (model_name, flops, params))