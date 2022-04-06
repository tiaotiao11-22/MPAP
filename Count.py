import os
import sys
import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--dataset', default='DTD', type=str) 
parser.add_argument('--tag', default='1', type=str) 
parser.add_argument('--fold', default=1, type=int) 
args = parser.parse_args()

filename = os.getcwd() + '/Result/' + args.dataset + '_' + args.tag + '_result.txt'

result=[]
c = 1
with open(filename,'r') as f:
    for line in f:
        result.append(float(line))
print(len(result))
print(result)

Acc = 0.0
for i in result:
    Acc = Acc + i
Acc_Ave = Acc / len(result)

print(Acc_Ave)

file = open(filename, 'a')
file.write('Average_Acc: '+str(Acc_Ave)+'% \r')
file.write('TAG: [-'+args.tag+'-] \r\r\r\r')