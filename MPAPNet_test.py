from __future__ import print_function
import argparse
import os
import shutil
import cv2
import random
import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.autograd.function import Function
from torch.nn.parameter import Parameter

from Network import MPAPNet_res50, MPAPNet_res18, MPAPNet_vgg19, MPAPNet_res101, MPAPNet_res152, MPAPNet_rawmobilenetv2, MPAPNet_vgg16
from Network import LabelSmoothLoss
from Util import AUG
from classification_image_loader import ClassificationImageLoader, ClassificationImageLoaderTriplet
from ECE_test import ece_score
from FGSM import fgsm_attack
from PGD import pgd_attack

import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import wandb
wandb.init(project="My Project")
wandb.watch_called = False

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
parser.add_argument('--backbone', default='resnet50', type=str) 
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

parser.add_argument('--a', default=0.5, type=float, help='none')
parser.add_argument('--b', default=0.3, type=float, help='none')
parser.add_argument('--c', default=0.2, type=float, help='none')

parser.add_argument('--seed', action='store_true', default=False,
                    help='hold seed')
parser.add_argument('--saveweight', action='store_true', default=False,
                    help='enables save model')
parser.add_argument('--savebestweight', action='store_true', default=False,
                    help='enables save best model')
parser.add_argument('--distributedsampler', action='store_true', default=False,
                    help='enables distributed sampler')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

#Visualization
parser.add_argument('--tsne', action='store_true', default=False,
                    help='tsne')
parser.add_argument('--cm', action='store_true', default=False,
                    help='CM')
parser.add_argument('--ece', action='store_true', default=False,
                    help='ecebm')
parser.add_argument('--cam', action='store_true', default=False,
                    help='cam')

parser.add_argument("--local_rank", type=int)
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
parser.add_argument('--rank', default=0,
                    help='rank of current process')
parser.add_argument('--word_size', default=1,
                    help="word size")

parser.add_argument('--FGSM', action='store_true',
                    help="evalute FGSM robustness")
parser.add_argument('--PGD', action='store_true',
                    help="evalute PGD robustness")
parser.add_argument('--adv_iter', default=5, type=int,
                    help='PGD iters')
parser.add_argument('--bins', default=10, type=int) 
parser.add_argument('--randcutPtest', default=0, type=float, help='Random Cutout Test')
parser.add_argument('--cutoutsizetest', default=10, type=int, help='The mask size of random cutout test') 
parser.add_argument('--noncenter', action='store_false')

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
best_ece = 0.0

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
                           AUG.cutout(mask_size=args.cutoutsizetest, p=args.randcutPtest, center=args.noncenter),
                           AUG.ToTensor(),
                           AUG.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                        std = [ 0.229, 0.224, 0.225 ]), 
                        
                       ])),
        batch_size=args.test_batch_size, shuffle=False, worker_init_fn=worker_init_fn, **kwargs)

    classes = ['scaly', 'matted', 'chequered', 'zigzagged', 'pitted', 'blotchy', 'smeared', 'cobwebbed', 'lacelike', 'lined', 'polka-dotted', 'gauzy', 'striped', 'pleated', 'crystalline', 'porous', 'honeycombed', 'grid', 'knitted', 'banded', 'braided', 'stained', 'studded', 'potholed', 'bubbly', 'cracked', 'wrinkled', 'perforated', 'freckled', 'spiralled', 'crosshatched', 'bumpy', 'woven', 'meshed', 'veined', 'stratified', 'dotted', 'flecked', 'frilly', 'interlaced', 'waffled', 'swirly', 'marbled', 'fibrous', 'sprinkled', 'paisley', 'grooved']

    if(args.backbone=='resnet50'):
        tnet = MPAPNet_res50(num_classes=num_class, feature_index=[4])
    elif(args.backbone=='resnet18'):
        tnet = MPAPNet_res18(num_classes=num_class, feature_index=[4])
    elif(args.backbone=='vgg19'):
        tnet = MPAPNet_vgg19(num_classes=num_class)
    elif(args.backbone=='resnet101'):
        tnet = MPAPNet_res101(num_classes=num_class, feature_index=[4])
    elif(args.backbone=='resnet152'):
        tnet = MPAPNet_res152(num_classes=num_class, feature_index=[4])
    elif(args.backbone=='mobilenet'):
        tnet = MPAPNet_rawmobilenetv2(num_classes=num_class)
    elif(args.backbone=='vgg16'):
        tnet = MPAPNet_vgg16(num_classes=num_class)

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

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()]) / 1024 /1024
    print('  + Number of params: {}'.format(n_parameters))
    
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        # train for one epoch
        if t in {50, 100, 150}:
            lr *= 0.1
            optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
        train(train_loader, test_loader, tnet, criterion, optimizer, epoch, classes)
        print(time.time()-t)
        if args.saveweight:
            # save weights
            save_name = 'Model/' + str(args.dataset) + '/' + str(args.dataset) + '_' + str(args.fold) + '.pkl'
            torch.save(tnet.state_dict(), save_name)  
    
    filename = os.getcwd() + '/Result/' + args.dataset + '_' + args.tag + '_result.txt'
    file = open(filename, 'a')
    file.write(str(best_acc)+'\n')
    #file.close()
    wandb.watch(tnet, log="all")

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
def setbest_ece(temp_ece):
    global best_ece
    best_ece = temp_ece 
def getbest_ece():
    return best_ece

def train(train_loader, test_loader, tnet, criterion, optimizer, epoch, classes):
    losses = AverageMeter()
    total_all = 0.0
    correct_all_summary = 0.0
    
    pre_train = []
    tar_train = []

    tnet.train()
    for batch_idx, (data, label) in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        
        data = Variable(data)
        label = Variable(label)
        
        tart_label = label ##############################################

        if(batch_idx==0):
            optimizer.zero_grad()
        
        if args.mixup:
            inputs, targets_a, targets_b, lam = AUG.mixup_data(data, label, args.alpha, args.cuda)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = tnet(inputs)
            loss_func = AUG.mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs[0]) + loss_func(criterion, outputs[1]) + loss_func(criterion, outputs[2])
        else:
            outputs = tnet(data)
            loss = args.a * criterion(outputs[0], label) + args.b * criterion(outputs[1], label) + args.c * criterion(outputs[2], label)
 
        losses.update(loss.data, data.data)

        loss.backward()
        # 3. update parameters of net
        if((batch_idx+1)%args.accumulation_steps)==0:
            # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient

        total = 0.0
        summary = outputs[0]
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
        _, predicted = torch.max(outputs[0].data, 1)
        predicted = predicted.cpu()
        label = label.cpu()
        total = label.size(0)
        correct = (predicted.numpy() == label.data.numpy()).sum()
        accuracy_classification = 100 * correct / total
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.2f}\t'
                  'Acc_classification: {:.2f}%\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                losses.val, 
                accuracy_classification))
        
        #data collection for visualization
        pred_train = outputs[0] + outputs[1] + outputs[2]
        pre_train.append(pred_train.data.cpu().numpy())
        tar_train.append(tart_label.data.cpu().numpy())
    
    pre_train = np.concatenate(pre_train, 0)
    tar_train = np.concatenate(tar_train, 0)

    total_all = 0.0
    correct_all = 0.0
    example_images = []
    cam_images = []
    cam_imagesori = []
    tnet.eval()

    pre = []
    pre_oh = []
    tar = []

    for batch_idx, (data, label) in enumerate(test_loader):
        #cam_images = []
        #cam_imagesori = []
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        
        data = Variable(data)
        label = Variable(label)
        
        if args.FGSM:
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(shape=(3, 1, 1)).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(shape=(3, 1, 1)).cuda()
            data.requires_grad = True
            output = tnet(data)
            cla = args.a * output[0] + args.b * output[1] + args.c * output[2]
            loss = torch.nn.functional.nll_loss(cla, label)
            tnet.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            epsilon = 1 / std / 255.0 * 16.0
            upper_bound = (1 - mean) / std
            lower_bound = (0 - mean) / std
            data = fgsm_attack(data, epsilon, data_grad, upper_bound, lower_bound)
        
        if args.PGD:
            data = pgd_attack(tnet, data, label, criterion, eps=0.3, alpha=16/255, iters=args.adv_iter) 

        with torch.no_grad():
            classification = tnet(data)
        
        pred = F.softmax(classification[0] + classification[1] + classification[2], dim=1)
        tart = label
        cla = args.a * classification[0] + args.b * classification[1] + args.c * classification[2]

        total = 0.0
        correct = 0.0

        label = label.cpu()
        total = label.size(0)
        total_all += total

        _, predicted = torch.max(cla.data, 1)
        predicted = predicted.cpu()
        correct = (predicted.numpy() == label.data.numpy()).sum()
        correct_all += correct

        pre.append(pred.data.cpu().numpy())
        pre_oh.append(predicted.numpy())
        tar.append(tart.data.cpu().numpy())

        example_images.append(wandb.Image(
        data[0], caption="Pred: {} Truth: {}".format(classes[predicted[0].item()], classes[label[0]])))

        params_final = list(tnet.our.classifier1.parameters())
        params_final = params_final[0]
        CAM_temp, CAM_image = visualization(params_final, classification[3], predicted, data, label)

        if args.cam:
            for cam in CAM_temp:
                #cam = np.transpose(cam, (2, 0, 1))
                #cam_images.append(wandb.Image(cam), caption="Pred: {} Truth: {}".format(classes[predicted[0].item()], classes[label[0]]))
                #print(np.shape(cam))
                #print(type(cam))
                cam_images.append(wandb.Image(cam))
            for img in CAM_image:
                cam_imagesori.append(wandb.Image(img))

    pre = np.concatenate(pre, 0)
    pre_oh = np.concatenate(pre_oh, 0)
    tar = np.concatenate(tar, 0)
    #CAM_temp = np.concatenate(CAM_temp, 0)

    ece, acc, conf = ece_score(pre, tar, n_bins=args.bins)
    
    print('ECE of testset: %.2f %%' % (100 * ece))
    print('Accuracy of testset: %.2f %%' % (100 * correct_all / total_all))
    temp_acc = 100 * correct_all / total_all
    best = getbest_acc()
    if(temp_acc >= best):
        setbest_acc(temp_acc)
        setbest_epoch(epoch)
        setbest_ece(ece)
        if args.saveweight:
            # save weights
            print('Saveing Model...')
            save_name = 'Model/' + str(args.dataset) + '/' + str(args.dataset) + '_' + str(args.fold) + '_' + str(args.tag) + '.pkl'
            torch.save(tnet.state_dict(), save_name)  
        if args.tsne:
            #draw TSNE
            plt.close('all')
            plt.rcParams['figure.dpi'] = 500
            tsne = TSNE(n_components=2, init='random', random_state=666, learning_rate=10)
            pre_Y = tsne.fit_transform(pre_train)
            plt.scatter(pre_Y[:, 0], pre_Y[:, 1], c=tar_train, alpha=0.3, s=3, cmap='hsv')
            #cmap='twilight'
            plt.tick_params(bottom=False, top=False, left=False, right=False)
            plt.axis('off')
        if args.cm:
            # draw CM
            plt.close('all')
            plt.rcParams['figure.dpi'] = 500 
            plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
            cm = confusion_matrix(pre_oh, tar) 
            plt.imshow(cm, cmap='YlGn')
            #plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=90)
            plt.yticks(tick_marks, classes)
            plt.tick_params(labelsize=5, bottom=False, top=False, left=False, right=False, length=0, width=7)
        if args.savebestweight:
            # save best weights
            save_name = 'Model/' + str(args.dataset) + '/' + str(args.dataset) + '_' + str(args.fold) + '_best.pkl'
            torch.save(tnet.state_dict(), save_name)

    if args.ece:
        #draw ECE
        plt.close('all')
        plt.rcParams['figure.dpi'] = 500
        x = np.linspace(0,1,100)
        y = x
        plt.plot(x, y, '-r', linewidth=3, linestyle='--', label='Expectation')
        plt.plot(conf, acc, '-b', linewidth=3, label='Outputs')
        plt.axis([0, 1, 0, 1])
        plt.bar(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)+0.1, edgecolor='orange', linewidth=3, width=0.1, align='edge', color='orange', alpha=0.6, label='Expectation')
        plt.bar(np.arange(0, 1, 0.1), acc, edgecolor='skyblue', linewidth=3, width=0.1, align='edge', color='skyblue', alpha=0.6, label='Outputs')
        plt.text(0.38, 0.01, "Confidence", fontsize=20, rotation=0)
        plt.text(0.01, 0.38, "Accuracy", fontsize=20, rotation=270)
        #plt.text(0.38, 0.9, "ECE: "+str(np.around(100*ece, 2))+"%", fontsize=20, rotation=0)

        plt.legend()
            
    best_ep = getbest_epoch()
    best_ece = getbest_ece()
    print('(Best)Accuracy of testset: %.2f %% in epoch: %d and ECE is %.3f %%' %(best_acc, best_ep, best_ece*100))

    wandb.log({
        "Best_ECE": best_ece*100,
        "Best_Acc": best_acc,
        "Best_Epoch": best_ep,
        "CM": wandb.Image(plt),
        "CAM": cam_images,
        "Images": cam_imagesori,
        "Test Accuracy": 100 * correct_all / total_all,
        "ECE": 100 * ece})

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

def CAM(out, weight):
    sum_map = 0.0
    count = float(len(out))
    for i in range(len(out)):
        sum_map = sum_map + weight[i] * out[i,:,:]
        
    sum_map = sum_map / count
    return sum_map

def visualization(params_final, feature_map, classification, images, labels):
    size = 224
    
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    feature_map_temp = feature_map.data.cpu().numpy()
    classification_temp = classification.data.cpu().numpy()
    
    CAM_temp = []
    CAM_image = []
    for i in range(classification.size()[0]):

        cla = classification_temp[i]
        feature_map = feature_map_temp[i]
        
        weight_cla_final = params_final[cla]
        weight_cla_final = weight_cla_final.data.cpu().numpy()
        
        CAM_final = CAM(feature_map, weight_cla_final)
        
        image = images.cpu().numpy()[i]
        image = np.transpose(image, (1, 2, 0))
        image = image * std + mean
        b = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        g = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        r = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        b[: ,:] = image[: ,: ,0]
        g[: ,:] = image[: ,: ,1]
        r[: ,:] = image[: ,: ,2]
        image = np.dstack([r, g, b])
        image = np.uint8(image * 255)
        
        CAM_final = cv2.resize(CAM_final, (size,size), interpolation=cv2.INTER_LANCZOS4) 
        CAM_final = (CAM_final - np.min(CAM_final)) / (np.max(CAM_final) - np.min(CAM_final))  
        CAM_final = np.uint8(CAM_final * 240)
        CAM_final = cv2.applyColorMap(CAM_final, cv2.COLORMAP_JET)
        #CAM_fuseimg = cv2.addWeighted(CAM_final, 0.4, image, 0.6, 0)

        image = image[:,:,::-1]
        CAM_final = CAM_final[:,:,::-1]
        CAM_temp.append(CAM_final)
        CAM_image.append(image)

    return CAM_temp, CAM_image

if __name__ == '__main__':
    main()  
