import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)
        self.l_bn1x1 = nn.BatchNorm2d(channels // m)
        self.l_relu1x1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.l_relu1x1(self.l_bn1x1(self.l(x)))

class TriConv(torch.nn.Module):
    def __init__(self, channels, m):
        super(TriConv, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)
        self.l_bn1x1 = nn.BatchNorm2d(channels // m)
        self.l_relu1x1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.l_relu1x1(self.l_bn1x1(self.l(x)))

class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
                    key_map.shape[0], key_map.shape[1],
                    -1,
                    key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
                    query_map.shape[0], query_map.shape[1],
                    -1,
                    query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]

class DepenL(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(DepenL, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
                    key_map.shape[0], key_map.shape[1],
                    -1,
                    key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
                    query_map.shape[0], query_map.shape[1],
                    -1,
                    query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1], key_map_unfold[:, :, :, k**2//2:k**2//2+1] * query_map_unfold

def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel, dim=-1)

def two_way(A, B):
    return torch.nn.functional.softmax(A + B, dim=-1)

class DirPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5, dir_id='1'):
        super(DirPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.dir_id = dir_id
        self.position = Dir_Gen(self.dir_id, self.k)
        self.l1 = torch.nn.Conv2d(1, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position.cuda())))
        return x.view(1, self.channels, 1, self.k ** 2)
    
class DirectionConvLayer(torch.nn.Module):
    def __init__(self, channels, o_channels, k=3, stride=1, m=1, padding=1, dir_id='1'):
        super(DirectionConvLayer, self).__init__()
        self.channels = channels//64
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.dir_id = dir_id
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = DirPrior(k, channels//m, dir_id=self.dir_id)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, o_channels, 1)
        self.final_bn1x1 = nn.BatchNorm2d(o_channels)
        self.final_relu1x1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1], -1, x_unfold.shape[-2] // x_unfold.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1    
        pre_output = torch.sum(pre_output, axis=-2).view(x.shape[0], x.shape[1], h_out, w_out)
        return self.final_relu1x1(self.final_bn1x1(self.final1x1(pre_output)))

def Dir_Gen(dir_id, k):
    
    if dir_id == '1':
        U_ones = torch.ones([1, k], dtype=torch.int)
        U_ones = U_ones.unsqueeze(-1)    
        U_range = torch.arange(k, dtype=torch.int).unsqueeze(0)
        U_range = U_range.unsqueeze(1)    
        U_channel = torch.matmul(U_ones, U_range)
        U_channel = U_channel.unsqueeze(0)
        U_channel = U_channel.float() / torch.max(U_channel)
        U_channel = U_channel * 2 - 1
        
        return U_channel
    
    elif dir_id == '2':
        D_ones = torch.ones([1, k], dtype=torch.int)
        D_ones = D_ones.unsqueeze(-1)    
        D_range = torch.arange(k - 1, -1, -1, dtype=torch.int).unsqueeze(0)
        D_range = D_range.unsqueeze(1)    
        D_channel = torch.matmul(D_ones, D_range)
        D_channel = D_channel.unsqueeze(0)
        D_channel = D_channel.float() / torch.max(D_channel)
        D_channel = D_channel * 2 - 1
        
        return D_channel
    
    elif dir_id == '3':
        L_ones = torch.ones([1, k], dtype=torch.int)
        L_ones = L_ones.unsqueeze(1) 
        L_range = torch.arange(k, dtype=torch.int).unsqueeze(0)
        L_range = L_range.unsqueeze(-1)    
        L_channel = torch.matmul(L_range, L_ones) 
        L_channel = L_channel.unsqueeze(0)
        L_channel = L_channel.float() / torch.max(L_channel)
        L_channel = L_channel * 2 - 1
        
        return L_channel
        
    elif dir_id == '4':
        R_ones = torch.ones([1, k], dtype=torch.int)
        R_ones = R_ones.unsqueeze(1) 
        R_range = torch.arange(k - 1, -1, -1, dtype=torch.int).unsqueeze(0)
        R_range = R_range.unsqueeze(-1)    
        R_channel = torch.matmul(R_range, R_ones)  
        R_channel = R_channel.unsqueeze(0)
        R_channel = R_channel.float() / torch.max(R_channel)
        R_channel = R_channel * 2 - 1
        
        return R_channel
        
    elif dir_id == '5':
        LU_ones_1 = torch.ones([k, k], dtype=torch.int)
        LU_ones_1 = torch.triu(LU_ones_1)
        LU_ones_2 = torch.ones([k, k], dtype=torch.int)
        LU_change = torch.arange(k - 1, -1, -1, dtype=torch.int)
        LU_ones_2[k - 1, :] = LU_change
        LU_channel = torch.matmul(LU_ones_1, LU_ones_2) 
        LU_channel = LU_channel.unsqueeze(0)
        LU_channel = LU_channel.unsqueeze(0)
        LU_channel = LU_channel.float() / torch.max(LU_channel)
        LU_channel = LU_channel * 2 - 1
        
        return LU_channel
        
    elif dir_id == '6':
        RD_ones_1 = torch.ones([k, k], dtype=torch.int)
        RD_ones_1 = torch.triu(RD_ones_1)
        RD_ones_1 = torch.t(RD_ones_1)
        RD_ones_2 = torch.ones([k, k], dtype=torch.int)
        RD_change = torch.arange(k, dtype=torch.int)
        RD_ones_2[0, :] = RD_change
        RD_channel = torch.matmul(RD_ones_1, RD_ones_2) 
        RD_channel = RD_channel.unsqueeze(0)
        RD_channel = RD_channel.unsqueeze(0)
        RD_channel = RD_channel.float() / torch.max(RD_channel)
        RD_channel = RD_channel * 2 - 1
        
        return RD_channel
        
    elif dir_id == '7':
        RU_ones_1 = torch.ones([k, k], dtype=torch.int)
        RU_ones_1 = torch.triu(RU_ones_1)
        RU_ones_2 = torch.ones([k, k], dtype=torch.int)
        RU_change = torch.arange(k, dtype=torch.int)
        RU_ones_2[k - 1, :] = RU_change
        RU_channel = torch.matmul(RU_ones_1, RU_ones_2) 
        RU_channel = RU_channel.unsqueeze(0)
        RU_channel = RU_channel.unsqueeze(0)
        RU_channel = RU_channel.float() / torch.max(RU_channel)
        RU_channel = RU_channel * 2 - 1
        
        return RU_channel
        
    elif dir_id == '8':
        LD_ones_1 = torch.ones([k, k], dtype=torch.int)
        LD_ones_1 = torch.triu(LD_ones_1)
        LD_ones_1 = torch.t(LD_ones_1)
        LD_ones_2 = torch.ones([k, k], dtype=torch.int)
        LD_change = torch.arange(k - 1, -1, -1, dtype=torch.int)
        LD_ones_2[0, :] = LD_change
        LD_channel = torch.matmul(LD_ones_1, LD_ones_2)
        LD_channel = LD_channel.unsqueeze(0)
        LD_channel = LD_channel.unsqueeze(0)
        LD_channel = LD_channel.float() / torch.max(LD_channel)
        LD_channel = LD_channel * 2 - 1
        
        return LD_channel


class Dependence_Learning(torch.nn.Module):
    def __init__(self, channels, o_channels, k=3, stride=1, m=1, padding=1):
        super(Dependence_Learning, self).__init__()
        self.channels = channels//64
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.Tri1 = TriConv(channels, self.m)
        self.Tri2 = TriConv(channels, self.m)
        self.DL = DepenL(k, padding, stride)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, o_channels, 1)
        self.bn1x1 = nn.BatchNorm2d(o_channels)
        self.relu1x1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        tri1 = self.Tri1(x)
        tri2 = self.Tri2(x)
        dlA, dlB = self.DL((tri1, tri2))
        dl_weight = two_way(dlA, dlB)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (dl_weight * x_unfold).view(x.shape[0], x.shape[1], -1, x_unfold.shape[-2] // x_unfold.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1    
        pre_output = torch.sum(pre_output, axis=-2).view(x.shape[0], x.shape[1], h_out, w_out)
        return self.relu1x1(self.bn1x1(self.final1x1(pre_output)))


# Deep SR Module
class SRModule(torch.nn.Module):
    def __init__(self, channels, o_channels):
        super(SRModule, self).__init__()
        self.DirL1 = DirectionConvLayer(channels, o_channels, dir_id='1')
        self.DirL2 = DirectionConvLayer(channels, o_channels, dir_id='2')
        self.DirL3 = DirectionConvLayer(channels, o_channels, dir_id='3')
        self.DirL4 = DirectionConvLayer(channels, o_channels, dir_id='4')
        self.DirL5 = DirectionConvLayer(channels, o_channels, dir_id='5')
        self.DirL6 = DirectionConvLayer(channels, o_channels, dir_id='6')
        self.DirL7 = DirectionConvLayer(channels, o_channels, dir_id='7')
        self.DirL8 = DirectionConvLayer(channels, o_channels, dir_id='8')
        
        self.final1x1 = torch.nn.Conv2d(o_channels*8, channels, 1)
        self.bn1x1 = nn.BatchNorm2d(channels)
        self.relu1x1 = nn.ReLU(inplace=True)
        
        self.DL = Dependence_Learning(channels, channels)
        
    def forward(self, x):
        dir1 = self.DirL1(x)
        dir2 = self.DirL2(x)
        dir3 = self.DirL3(x)
        dir4 = self.DirL4(x)
        dir5 = self.DirL5(x)
        dir6 = self.DirL6(x)
        dir7 = self.DirL7(x)
        dir8 = self.DirL8(x)
        
        out = torch.cat((dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8), 1)
        out = self.relu1x1(self.bn1x1(self.final1x1(out)))
        out = self.DL(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.srm = SRModule(2048, 32)        
        self.classifier = nn.Linear(512 * block.expansion * 1, num_classes)  
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #Spatial Ordered
        SOB = self.avgpool(x)
        
        #Structure Revealed
        SRB = self.srm(x)
        SRB = self.avgpool(SRB)
        
        x = SOB + SRB
        
        x = x.view(x.size(0), -1)

        classification = self.classifier(x)

        return classification

def DSRNet(pretrained=True, Num_C=1):

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=Num_C).cuda()
    if pretrained:
        pretrained_dict = torch.load('resnet50-19c8e357.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model