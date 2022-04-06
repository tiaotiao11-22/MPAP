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
        #self.position = 2 * torch.rand(1, 2, k, k, requires_grad=False) - 1
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

'''
class Dependence_Learning(torch.nn.Module):
    def __init__(self, channels, out_channels=None, mid_channels=None, kernel_size=3, stride=1, padding=None):
        super(Dependence_Learning, self).__init__()
        self.channels = channels
        self.k = kernel_size
        self.stride = stride
        if padding is None:
            padding = self.k // 2
        if out_channels is None:
            self.out_channels = self.channels
        else:
            self.out_channels = out_channels
        self.mid_channels = self.channels // 128
        self.padding = padding
        self.unfold = torch.nn.Unfold(kernel_size=(self.k, self.k), stride=1, padding=self.padding)
        
        self.tri_conv1 = torch.nn.Conv2d(self.channels, self.mid_channels, 1)
        self.tri_conv2 = torch.nn.Conv2d(self.channels, self.mid_channels, 1)

        self.final1x1 = torch.nn.Conv2d(self.channels, self.channels, 1)
    
    def forward(self, x):
        x1 = self.tri_conv1(x)
        x2 = self.tri_conv2(x)

        x1_unf = self.unfold(x1)
        x2_unf = self.unfold(x2)
        print(x1_unf.size())

        x1_unf = x1_unf.view(
                    x1.shape[0], x1.shape[1],
                    -1,
                    x1_unf.shape[-2] // x1.shape[1])
        print(x1_unf.size())
        x2_unf = x2_unf.view(
                    x2.shape[0], x2.shape[1],
                    -1,
                    x2_unf.shape[-2] // x2.shape[1])
        dl_weight = x1_unf * x2_unf
        dl_weight = torch.nn.functional.softmax(dl_weight, dim=-1)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = (dl_weight*x_unfold).view(x.shape[0], 1, x.shape[1],
                                 -1, x_unfold.shape[-2] // x.shape[1])
        
        pre_output = x_unfold.view(x.shape[0], x.shape[1], -1, x_unfold.shape[-2] // x_unfold.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1    
        print(pre_output.size())
        pre_output = torch.sum(pre_output, axis=-2).view(x.shape[0], x.shape[1], h_out, w_out)

        print(pre_output.size())
        return self.final1x1(pre_output)
'''

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
        #dl_weight = torch.nn.functional.softmax(dl, dim=-1)[:, None, :, :, :]
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

'''
class Dependence_Learning(torch.nn.Module):
    def __init__(self, channels, out_channels=None, mid_channels=None, kernel_size=3, stride=1, padding=None):
        super(Dependence_Learning, self).__init__()
        self.channels = channels
        if out_channels is None:
            self.out_channels = self.channels
        else:
            self.out_channels = out_channels
        self.mid_channels = self.channels // 128
        self.k = kernel_size
        self.stride = stride
        if padding is None:
            padding = self.k // 2
        self.padding = padding
        
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.mid_channels, self.k, self.k), requires_grad=True)
        #self.weight.data.fill_(0.25)
        
        self.tri_conv1 = torch.nn.Conv2d(self.channels, self.mid_channels, 1)
        self.tri_conv2 = torch.nn.Conv2d(self.channels, self.mid_channels, 1)
        
        self.unfold = torch.nn.Unfold(kernel_size=(self.k, self.k), stride=1, padding=self.padding)


    def forward(self, x):

        N, c, w, h = x.size()
        x1 = self.tri_conv1(x)
        #x2 = self.tri_conv2(x)
        #unf_x1 = self.unfold(x1)
        inp_unf = torch.nn.functional.unfold(x1, kernel_size=(self.k, self.k))
        #unf_x2 = self.unfold(x2)

        #unf_x1 = torch.transpose(unf_x1, 1, 2)
        #unf_x2 = torch.transpose(unf_x2, 1, 2)
        

        #compute DL weight
        unf_x1 = unf_x1.view(N, w*h, self.mid_channels, self.k*self.k)
        unf_x2 = unf_x2.view(N, w*h, self.mid_channels, self.k*self.k)

        dl_weight = torch.matmul(torch.transpose(unf_x2, 2, 3).contiguous(), unf_x1) / (self.k*self.k)
        #\dl_weight = torch.nn.functional.normalize(dl_weight)
        dl_weight = torch.nn.functional.softmax(dl_weight, -1)

        print(dl_weight)

        #two-way collaborative
        dl_weight_t = torch.transpose(dl_weight, 1, 2).contiguous()
        dl_weight = dl_weight / (torch.exp(dl_weight) * torch.exp(dl_weight_t) + 1e-6)
        dl_weight = torch.nn.functional.normalize(dl_weight)


        #unf_x_dl = torch.matmul(unf_x1, dl_weight)
        #unf_x_dl = unf_x1
        #unf_x_dl = torch.transpose(unf_x_dl, 2, 3).contiguous()
        #unf_x_dl = unf_x_dl.view(N, w*h, self.mid_channels*self.k*self.k).contiguous()

        #out_unf = unf_x_dl.matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        print(out_unf.size())
        out = torch.nn.functional.fold(out_unf, (w+1-self.k, h+1-self.k), (1, 1))
        
        print((torch.nn.functional.conv2d(x1, self.weight) - out).abs().max())
        #out = out_unf.view(N, c, w, h)
        #out = torch.nn.functional.fold(out_unf, (w, h), kernel_size=(1, 1))


        inp = torch.randn(1, 3, 10, 12)
        w = torch.randn(2, 3, 4, 5)
        inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))

        
        return out
'''
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

'''
class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)
    
    def forward(self, x):
        return self.l(x)

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

def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel, dim=-1)

class DirPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5, dir_id='1'):
        super(DirPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.dir_id = dir_id
        #self.position = 2 * torch.rand(1, 2, k, k, requires_grad=False) - 1
        self.position = Dir_Gen(self.dir_id, self.k)
        self.l1 = torch.nn.Conv2d(1, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position.cuda())))
        return x.view(1, self.channels, 1, self.k ** 2)
    
class DirectionConvLayer(torch.nn.Module):
    def __init__(self, channels, o_channels, k=3, stride=1, m=1, padding=1, dir_id='1'):
        super(DirectionConvLayer, self).__init__()
        self.channels = channels
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
        return self.final1x1(pre_output)

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
'''
'''
class DepenLearning(torch.nn.Module):
    def __init__(self, channels, out_channels=None, mid_channels=None, kernel_size=3, stride=1, padding=None):
        super(DepenLearning, self).__init__()
        self.channels = channels
        if out_channels is None:
            self.out_channels = self.channels
        else:
            self.out_channels = out_channels
        self.mid_channels = self.channels // 64
        self.k = kernel_size
        self.stride = stride
        if padding is None:
            padding = self.k // 2
        self.padding = padding
        
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.mid_channels, self.k, self.k), requires_grad=True)
        
        self.tri_conv = torch.nn.Conv2d(self.channels, self.mid_channels, 1)
        
        self.unfold = torch.nn.Unfold(self.k, 1, self.padding)
        
    def forward(self, x):
        N, _, w, h = x.size()
        x = self.tri_conv(x)
        unf_x = self.unfold(x)

        unf_x = torch.transpose(unf_x, 1, 2).contiguous()
        
        #compute LR weight
        unf_x = unf_x.view(N, w*h, self.mid_channels, self.k*self.k)
        
        bl_weight = torch.matmul(unf_x, torch.transpose(unf_x, 2, 3).contiguous()) / (self.channels*self.channels)
        bl_weight = torch.nn.functional.normalize(bl_weight)
        
        unf_x_bl = torch.matmul(torch.transpose(unf_x, 2, 3).contiguous(), bl_weight)
        unf_x_bl = torch.transpose(unf_x_bl, 2, 3).contiguous()
        unf_x_bl = unf_x_bl.view(N, w*h, self.mid_channels*self.k*self.k).contiguous()
        out_unf = unf_x_bl.matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2).contiguous()
        out = torch.nn.functional.fold(out_unf, (w, h), kernel_size=(1, 1))
        return out
'''
'''
class Dependence_Learning(torch.nn.Module):
    def __init__(self, channels, out_channels=None, mid_channels=None, kernel_size=3, stride=1, padding=None):
        super(Dependence_Learning, self).__init__()
        self.channels = channels
        if out_channels is None:
            self.out_channels = self.channels
        else:
            self.out_channels = out_channels
        self.mid_channels = self.channels // 64
        self.k = kernel_size
        self.stride = stride
        if padding is None:
            padding = self.k // 2
        self.padding = padding
        
        self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.mid_channels, self.k, self.k), requires_grad=True)
        
        self.tri_conv = torch.nn.Conv2d(self.channels, self.mid_channels, 1)
        
        self.unfold = torch.nn.Unfold(self.k, 1, self.padding)
        
    def forward(self, x):
        N, _, w, h = x.size()
        x = self.tri_conv(x)
        unf_x = self.unfold(x)

        unf_x = torch.transpose(unf_x, 1, 2).contiguous()
        
        #compute DL weight
        unf_x = unf_x.view(N, w*h, self.mid_channels, self.k*self.k)
        dl_weight = torch.matmul(torch.transpose(unf_x, 2, 3).contiguous(), unf_x) / (self.k*self.k)
        dl_weight = torch.nn.functional.normalize(dl_weight)
        #two-way collaborative
        dl_weight_t = torch.transpose(dl_weight, 2, 3).contiguous()
        dl_weight_res = torch.exp(dl_weight) / (torch.exp(dl_weight) * torch.exp(dl_weight_t) + 1e-6)
        dl_weight = dl_weight_res + dl_weight
        #dl_weight = torch.nn.functional.normalize(dl_weight)

        unf_x_dl = torch.matmul(unf_x, dl_weight)
        unf_x_dl = torch.transpose(unf_x_dl, 2, 3).contiguous()
        unf_x_dl = unf_x_dl.view(N, w*h, self.mid_channels*self.k*self.k).contiguous()
        out_unf = unf_x_dl.matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2).contiguous()
        out = torch.nn.functional.fold(out_unf, (w, h), kernel_size=(1, 1))
        return out

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
        
        self.DL = Dependence_Learning(channels)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = x
        dir1 = self.DirL1(x)
        dir2 = self.DirL2(x)
        dir3 = self.DirL3(x)
        dir4 = self.DirL4(x)
        dir5 = self.DirL5(x)
        dir6 = self.DirL6(x)
        dir7 = self.DirL7(x)
        dir8 = self.DirL8(x)
        
        out = torch.cat((dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8), 1)
        out = self.relu(self.final1x1(out))

        out = self.DL(out)
        #out = out + res

        return out

'''

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

        #self.tri_conv = nn.Conv2d(3072, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
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
        #x = self.tri_conv(torch.cat((x3, self.up(x4)), 1))
        SRB = self.srm(x)
        SRB = self.avgpool(SRB)
        
        x = SOB + SRB

        #x = SOB * (1 + SRB)
        
        x = x.view(x.size(0), -1)
        #SOB = SOB.view(SOB.size(0), -1)
        #SRB = SRB.view(SRB.size(0), -1)
        #x = torch.cat((SOB, SRB), 1)

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