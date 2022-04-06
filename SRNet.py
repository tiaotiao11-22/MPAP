from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from classification_image_loader import ClassificationImageLoader, ClassificationImageLoaderTriplet
import numpy as np
import torchvision.models as models
import time
import torchvision
import math
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import confusion_matrix
import random
#import transforms
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
import cv2
from Network import DSRNet
from Network import LabelSmoothLoss
from Util import AUG

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#    We need sh script
#    DTD = 47,batch = 256  
#    FMD = 10,batch = 256  
#    MINC-2500 = 24,batch = 64 
#    MIT-indoor = 67,batch = 256 
#    GTOSM = 31,batch = 128 
#    GTOS = 40,batch = 128 
#    KTH = 11,batch = 128
#    CAR = 196,batch = 256  
#    CUB = 200,batch = 256  

##--The number of folds--
#   'DTD,10'   
#   'FMD,10'      
#   'MINC,5'      
#   'INDOOR,1'  
#   'GTOSM,1'  
#   'GTOS,5'    
#   'KTH,10'    
#   'CAR,1'
#   'CUB,1'

basedir = os.getcwd()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', default='DTD', type=str) 
parser.add_argument('--tag', default='1', type=str) 
parser.add_argument('--fold', default=1, type=int) 
parser.add_argument('--classes', default=47, type=int) 
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpuname', default=[0,1,2,3], type=int) #
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 2)')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--accumulation_steps', type=int, default=2)

# Augmentation
parser.add_argument('--mixup', action='store_true', default=False, help='enables mixup')
parser.add_argument('--alpha', default=0.5, type=float, help='interpolation strength (uniform=1., ERM=0.)')

parser.add_argument('--labelsmoothing', action='store_true', default=False, help='enables label smoothing')
parser.add_argument('--smooth', default=0.5, type=float, help='smooth factor')

parser.add_argument('--randearP', default=0, type=float, help='none')
parser.add_argument('--randhorP', default=0, type=float, help='none')
parser.add_argument('--randverP', default=0, type=float, help='none')
parser.add_argument('--randperP', default=0, type=float, help='none')
parser.add_argument('--randaugP', default=0, type=float, help='none')
parser.add_argument('--randcutP', default=0, type=float, help='none')
parser.add_argument('--mixupP', default=0.4, type=float, help='none')

#parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
#parser.add_argument('--cutmix_prob', default=0, type=float, help='cutmix probability')

parser.add_argument('--seed', action='store_true', default=False,
                    help='hold seed')
parser.add_argument('--saveweight', action='store_true', default=False,
                    help='enables save model')
parser.add_argument('--distributedsampler', action='store_true', default=False,
                    help='enables distributed sampler')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument("--local_rank", type=int)

parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
parser.add_argument('--rank', default=0,
                    help='rank of current process')
parser.add_argument('--word_size', default=1,
                    help="word size")


args = parser.parse_args()

if args.seed:
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    setup_seed(666)
else:
    torch.backends.cudnn.benchmark = True

num_class = 10
if(args.dataset=='DTD'):
    num_class = 47
elif(args.dataset=='FMD'):
    num_class = 10
elif(args.dataset=='KTH'):
    num_class = 11
elif(args.dataset=='MINC'):
    num_class = 24
elif(args.dataset=='GTOS'):
    num_class = 40
elif(args.dataset=='GTOSM'):
    num_class = 31
elif(args.dataset=='CUB'):
    num_class = 200
elif(args.dataset=='CAR'):
    num_class = 196
else:        #INDOOR
    num_class = 67

global best_acc
best_acc = 0.0
best_epoch = 0

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#torch.distributed.init_process_group(backend='nccl', init_method=args.init_method)
#torch.distributed.init_process_group(backend='gloo', init_method='env://')
#torch.cuda.set_device(args.local_rank)

def main():    
    global best_acc
    best_acc = 0.0

    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    n_fold = args.fold    
    Data_set = args.dataset
        
    trainset_name = 'train_' + str(n_fold) + '.txt'
    testset_name = 'test_' + str(n_fold) + '.txt'
    file_root = 'Datasets/' + Data_set +  ' dataset need/classification_train_test/'
    dataset_source = basedir + '/DatasetsSource/' + Data_set + ' dataset/'

    transform = [transforms.RandomHorizontalFlip(), 
                    transforms.RandomVerticalFlip(),
                    transforms.RandomPerspective()]
    train_dataset = ClassificationImageLoader(file_root, 
                           trainset_name, 
                           dataset_source,
                       transform=AUG.Compose([
                           AUG.Scale(256),
                           #transforms.RandomCrop(224),
                           AUG.RandomSizedCrop(224), 
                           #transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(p=args.randhorP),
                           transforms.RandomVerticalFlip(p=args.randverP),
                           transforms.RandomPerspective(p=args.randperP),
                           transforms.RandomApply(transform, p=args.randaugP),
                           AUG.cutout(p=args.randcutP),
                           AUG.ToTensor(),
                           transforms.RandomErasing(p=args.randearP),
                           AUG.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                        std = [ 0.229, 0.224, 0.225 ]), 
                       ]))
    if args.distributedsampler:
        # 2.0 sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        # 1.0 sampler
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    
    def worker_init_fn(worker_id):   
        np.random.seed(7 + worker_id)
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn, pin_memory=True)                       

    test_loader = torch.utils.data.DataLoader(
        ClassificationImageLoader(file_root, 
                           testset_name, 
                           dataset_source,
                       transform=AUG.Compose([
                           AUG.Scale(256),
                           #transforms.RandomCrop(224),
                           AUG.CenterCrop(224),
                           AUG.ToTensor(),
                           AUG.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                        std = [ 0.229, 0.224, 0.225 ]), 
                        
                       ])),
        batch_size=args.test_batch_size, shuffle=False, worker_init_fn=worker_init_fn, **kwargs) 
        '''

    train_loader = DataLoaderX(
        train_dataset, batch_size=args.batch_size, shuffle=None, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn, pin_memory=True)                       

    test_loader = DataLoaderX(
        ClassificationImageLoader(file_root, 
                           testset_name, 
                           dataset_source,
                       transform=AUG.Compose([
                           AUG.Scale(256),
                           #transforms.RandomCrop(224),
                           AUG.CenterCrop(224),
                           AUG.ToTensor(),
                           AUG.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                        std = [ 0.229, 0.224, 0.225 ]), 
                        
                       ])),
        batch_size=args.test_batch_size, shuffle=False, worker_init_fn=worker_init_fn, **kwargs)

    Net_ = DSRNet(pretrained=True, Num_C=num_class)
    #Net_ = torch.nn.parallel.DistributedDataParallel(Net_, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    tnet = Net_ 
    
    if args.cuda:
        tnet.cuda()
    
    lr = args.lr

    if args.labelsmoothing:
        criterion = LabelSmoothLoss(smoothing=args.smooth)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
    #optimizer = optim.AdamW(tnet.parameters(), lr=lr*10, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

    if args.cuda:
        criterion.cuda()

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    params_final = list(tnet.classifier.parameters())
    #params_final = list(tnet.module.classifier.parameters())
    params_final = params_final[0]
    
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        # train for one epoch
        if t in {50, 100, 150}:
            lr *= 0.1
            optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
        train(train_loader, test_loader, tnet, criterion, optimizer, epoch, params_final)
        print(time.time()-t)
        if args.saveweight:
            # save weights
            save_name = 'Model/' + str(args.dataset) + '/' + str(args.dataset) + '_' + str(args.fold) + '.pkl'
            torch.save(Net_.state_dict(), save_name)  
    
    filename = os.getcwd() + '/Result/' + args.dataset + '_' + args.tag + '_result.txt'
    file = open(filename, 'a')
    file.write(str(best_acc)+'\n')
    #file.close()

def setbest_acc(temp_acc):
    global best_acc
    best_acc = temp_acc   
def getbest_acc():
    return best_acc 
def setbest_epoch(temp_epoch):
    global best_epoch
    best_epoch = temp_epoch   
def getbest_epoch():
    return best_epoch  

def train(train_loader, test_loader, tnet, criterion, optimizer, epoch, params_final):
    
    losses = AverageMeter()
    total_all = 0.0
    correct_all_summary = 0.0
    
    tnet.train()
    for batch_idx, (data, label) in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        
        data = Variable(data)
        label = Variable(label)

        if(batch_idx==0):
            optimizer.zero_grad()
        
        if args.mixup:
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = AUG.mixup_data(data, label, args.alpha, args.cuda)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = tnet(inputs)
            loss_func = AUG.mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
            '''
            inputs, targets = AUG.mixup_data_v2(data, label, alpha=args.mixupP, half=True)
            outputs = tnet(inputs)
            loss = AUG.mixup_criterion_v2(criterion, outputs, targets[0], targets[1], targets[2]).mean()
            '''
        else:
            outputs = tnet(data)
            loss = criterion(outputs, label)
        '''
        elif args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = tnet(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        '''
 
        losses.update(loss.data, data.data)
        
        '''
        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        '''

        loss.backward()
        # 3. update parameters of net
        if((batch_idx+1)%args.accumulation_steps)==0:
            # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient

        
        total = 0.0
        summary = outputs
        _1_summary, predicted_summary = torch.max(summary.data, 1)
        predicted_summary = predicted_summary.cpu()
        label = label.cpu()
        total = label.size(0)
        total_all += total
        correct_summary = (predicted_summary.numpy() == label.data.numpy()).sum()
        correct_all_summary += correct_summary
        
        #accuracy compute
        total = 0.0
        correct = 0.0
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        label = label.cpu()
        total = label.size(0)
        correct = (predicted.numpy() == label.data.numpy()).sum()
        accuracy_classification = 100 * correct / total
        
        del outputs
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.2f}\t'
                  'Acc_classification: {:.2f}%\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                losses.val, 
                accuracy_classification))
        
    print('Accuracy of trainset: %.2f %%' % (100 * correct_all_summary / total_all))   
    
    total_all = 0.0
    correct_all = 0.0
    tnet.eval()
    for batch_idx, (data, label) in enumerate(test_loader):

        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        
        data = Variable(data)
        label = Variable(label)
        
        with torch.no_grad():
            classification = tnet(data)
        
        #compute loss
        loss_softmax = criterion(classification, label)
        
        total = 0.0
        correct = 0.0

        _, predicted = torch.max(classification.data, 1)

        predicted = predicted.cpu()
        label = label.cpu()
        
        total = label.size(0)
        correct = (predicted.numpy() == label.data.numpy()).sum()
        
        total_all += total
        correct_all += correct
            
    print('Accuracy of testset: %.2f %%' % (100 * correct_all / total_all))
    temp_acc = 100 * correct_all / total_all
    best = getbest_acc()
    if(temp_acc >= best):
        setbest_acc(temp_acc)
        setbest_epoch(epoch)
    best_ep = getbest_epoch()
    print('(Best)Accuracy of testset: %.2f %% in epoch: %d' %(best_acc, best_ep))

    del classification

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()  
