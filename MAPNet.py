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
from Network import MAPNet
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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

setup_seed(666)

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
parser.add_argument('--mixup', action='store_true', default=False,
                    help='enables mixup')
parser.add_argument('--alpha', default=0.5, type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--labelsmoothing', action='store_true', default=False,
                    help='enables label smoothing')
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
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomPerspective()]
    train_dataset = ClassificationImageLoaderTriplet(file_root, 
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
                           #transforms.cutout(),
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
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn, pin_memory=True)                       

    test_loader = DataLoader(
        ClassificationImageLoaderTriplet(file_root, 
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

    Net_ = MAPNet(pretrained=True, Num_C=num_class)
    
    tnet = Net_ 
    
    if args.cuda:
        tnet.cuda()
    
    lr = args.lr

    if args.labelsmoothing:
        criterion = LabelSmoothLoss(smoothing=0.1)
    else:
        criterion_t = torch.nn.MarginRankingLoss(margin=0.2)
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
    
    if args.cuda:
        criterion.cuda()
        criterion_t.cuda()

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        # train for one epoch
        if t in {30, 40, 80}:
            lr *= 0.1
            optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
        train(train_loader, test_loader, tnet, criterion, criterion_t, optimizer, epoch)
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

def train(train_loader, test_loader, tnet, criterion, criterion_t, optimizer, epoch):
    
    losses = AverageMeter()
    loss_triplet_show = AverageMeter()
    emb_norms = AverageMeter()
    loss_softmax_show = AverageMeter()
    total_all = 0.0
    
    for batch_idx, (data1, data2, data3, label1, label2, label3) in enumerate(train_loader):    # [+, +, -]
        
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            label1, label2, label3 = label1.cuda(), label2.cuda(), label3.cuda()
        
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
        label1, label2, label3 = Variable(label1), Variable(label2), Variable(label3)

        if(batch_idx==0):
            optimizer.zero_grad()

        # compute output
        dist_a_b1, dist_b_b1, dist_a_b2, dist_b_b2, dist_a_b3, dist_b_b3, embedding_b1_1, embedding_b1_2, embedding_b1_3, embedding_b2_1, embedding_b2_2, embedding_b2_3, embedding_b3_1, embedding_b3_2, embedding_b3_3, classification_b1_1, classification_b1_2, classification_b1_3, classification_b2_1, classification_b2_2, classification_b2_3, classification_b3_1, classification_b3_2, classification_b3_3, C1, C2, C3 = tnet(data1, data2, data3)

        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dist_a_b1.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
    
        loss_triplet_b1 = criterion_t(dist_a_b1, dist_b_b1, target)
        loss_triplet_b2 = criterion_t(dist_a_b2, dist_b_b2, target)
        loss_triplet_b3 = criterion_t(dist_a_b3, dist_b_b3, target)
        loss_triplet = loss_triplet_b1 + loss_triplet_b2 + loss_triplet_b3
        loss_embedd = embedding_b1_1.norm(2) + embedding_b1_2.norm(2) + embedding_b1_3.norm(2) + embedding_b2_1.norm(2) + embedding_b2_2.norm(2) + embedding_b2_3.norm(2) + embedding_b3_1.norm(2) + embedding_b3_2.norm(2) + embedding_b3_3.norm(2)
        loss1 = loss_triplet + 0.01 * loss_embedd

        loss_softmax_1 = criterion(classification_b1_1, label1) + criterion(classification_b2_1, label1) + criterion(classification_b3_1, label1)
        loss_softmax_2 = criterion(classification_b1_2, label2) + criterion(classification_b2_2, label2) + criterion(classification_b3_2, label2)
        loss_softmax_3 = criterion(classification_b1_3, label3) + criterion(classification_b2_3, label3) + criterion(classification_b3_3, label3)

        loss_softmax1 = criterion(C1, label1)
        loss_softmax2 = criterion(C2, label2)
        loss_softmax3 = criterion(C3, label3)
        loss2_1 = loss_softmax1 + loss_softmax2 + loss_softmax3
        loss2_2 = loss_softmax_1 + loss_softmax_2 + loss_softmax_3
        loss2 = loss2_1 + loss2_2
        #print([loss1, loss2_1, loss2_2])
        loss = 0.2*loss1 + loss2
        
        '''
        loss = loss_softmax_1 + loss_softmax_2 + loss_softmax_3
        loss1 = loss
        loss2 = loss
        loss_embedd = loss
        '''

        loss.backward()
        # 3. update parameters of net
        if((batch_idx+1)%args.accumulation_steps)==0:
            # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient

        # measure accuracy and record loss
        losses.update(loss.data, data1.data)
        loss_softmax_show.update(loss2.data, data1.data)
        loss_triplet_show.update(loss1.data, data1.data)
        emb_norms.update(loss_embedd.data/3, data1.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f}\t'
                  'Loss_triplet: {:.4f}\t'
                  'Loss_softmax: {:.4f}\t'
                  'Emb_Norm: {:.2f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, loss_triplet_show.val, loss_softmax_show.val,
                emb_norms.val)) 
    
    del dist_a_b1, dist_b_b1, dist_a_b2, dist_b_b2, dist_a_b3, dist_b_b3, embedding_b1_1, embedding_b1_2, embedding_b1_3, embedding_b2_1, embedding_b2_2, embedding_b2_3, embedding_b3_1, embedding_b3_2, embedding_b3_3, classification_b1_1, classification_b1_2, classification_b1_3, classification_b2_1, classification_b2_2, classification_b2_3, classification_b3_1, classification_b3_2, classification_b3_3, C1, C2, C3

    total_all = 0.0
    correct_all = 0.0
    tnet.eval()
    for batch_idx, (data1, data2, data3, label1, label2, label3) in enumerate(test_loader):

        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            label1, label2, label3 = label1.cuda(), label2.cuda(), label3.cuda()
        
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
        label1, label2, label3 = Variable(label1), Variable(label2), Variable(label3)
        label = label1
        # compute output
        with torch.no_grad():
            dist_a_b1, dist_b_b1, dist_a_b2, dist_b_b2, dist_a_b3, dist_b_b3, embedding_b1_1, embedding_b1_2, embedding_b1_3, embedding_b2_1, embedding_b2_2, embedding_b2_3, embedding_b3_1, embedding_b3_2, embedding_b3_3, classification_b1_1, classification_b1_2, classification_b1_3, classification_b2_1, classification_b2_2, classification_b2_3, classification_b3_1, classification_b3_2, classification_b3_3, C1, C2, C3 = tnet(data1, data2, data3)
            classification = classification_b1_1
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

    del dist_a_b1, dist_b_b1, dist_a_b2, dist_b_b2, dist_a_b3, dist_b_b3, embedding_b1_1, embedding_b1_2, embedding_b1_3, embedding_b2_1, embedding_b2_2, embedding_b2_3, embedding_b3_1, embedding_b3_2, embedding_b3_3, classification_b1_1, classification_b1_2, classification_b1_3, classification_b2_1, classification_b2_2, classification_b2_3, classification_b3_1, classification_b3_2, classification_b3_3, C1, C2, C3

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
