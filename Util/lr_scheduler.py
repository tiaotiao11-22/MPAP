import math
import torch.optim as optim

def adjust_learning_rate(optimizer, epoch, args, net, batch=None, nBatch=None):

    T_total = args.epochs * nBatch
    T_cur = (epoch % args.epochs) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
    return optimizer