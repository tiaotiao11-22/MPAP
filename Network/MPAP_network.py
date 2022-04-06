import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from .Deform_Conv import ConvOffset2D

_default_anchors_setting = (
    dict(layer='p3', stride=1, size=1, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)

def generate_default_anchor_maps(anchors_setting=None, input_shape=(7, 7)):

    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:

        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

    return center_anchors, edge_anchors, anchor_areas

def hard_nms(cdds, topn=10, iou_thresh=0.25):
    if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
        raise TypeError('edge_box_map should be N * 5+ ndarray')

    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
            cdd[4] - cdd[2]) - intersec_map)
        res = res[iou_map_cur < iou_thresh]

    return np.array(cdd_results)

class ProposalNet(nn.Module):
    def __init__(self, inp):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(inp, inp//2, 3, 1, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(inp//2, 6, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        t1 = self.tidy1(d1).view(batch_size, -1)
        return t1

class Roi(torch.nn.Module):
    def __init__(self, inp, topN=1, edge_size=7):
        super(Roi, self).__init__()
        self.inp_tri = nn.Conv2d(inp, inp//2, kernel_size=1, stride=1, padding=0)
        self.inp_tri_bn = nn.BatchNorm2d(inp//2)
        self.inp_tri_relu = nn.ReLU(inplace=True)

        self.proposal_net = ProposalNet(inp//2)
        self.topN = topN
        self.inp = inp
        _, edge_anchors, _ = generate_default_anchor_maps(input_shape=(edge_size, edge_size))
        self.pad_side = 1
        self.edge_anchors = (edge_anchors + 1).astype(np.int)

    def forward(self, x):
        x = self.inp_tri_relu(self.inp_tri_bn(self.inp_tri(x)))
        rpn_feature = x.detach()
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='replicate', value=0)
        batch = x.size(0)
        rpn_score = self.proposal_net(rpn_feature)
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        part_imgs = torch.zeros([batch, self.topN, self.inp//2, 1, 1]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.adaptive_avg_pool2d(x_pad[i:i + 1, :, y0:y1, x0:x1], 1)
        part_imgs = part_imgs.view(batch * self.topN, self.inp//2, 1, 1)

        return part_imgs

#ATM module
class ATM(torch.nn.Module):    
    def __init__(self, inp=2048, oup=2048, tri_rate=64, edge_size=7):
        super(ATM, self).__init__()

        self.inp_vec = inp
        self.inp = inp
        self.oup = oup
        self.ROI = Roi(2*inp//tri_rate, edge_size=edge_size)
        self.roi_mapping = nn.Linear(self.inp_vec, 3)     #location [x, y] and length [s]
        self.offset = ConvOffset2D(2*inp//tri_rate)

        self.inp_vec_tri = nn.Conv2d(inp, inp//tri_rate, kernel_size=1, stride=1, padding=0)
        self.inp_vec_tri_bn = nn.BatchNorm2d(inp//tri_rate)
        self.inp_vec_tri_relu = nn.ReLU(inplace=True)

        self.conv_offset = nn.Conv2d(2*inp//tri_rate, inp//tri_rate, kernel_size=3, stride=1, padding=1)
        self.conv_offset_bn = nn.BatchNorm2d(inp//tri_rate)
        self.conv_offset_relu = nn.ReLU(inplace=True)

        self.tri_in_tensor = nn.Conv2d(inp, inp//tri_rate, kernel_size=1, stride=1, padding=0)
        self.tri_in_bn = nn.BatchNorm2d(inp//tri_rate)
        self.tri_in_relu = nn.ReLU(inplace=True)

        self.tri_out_tensor = nn.Conv2d(3*inp//tri_rate, oup, kernel_size=1, stride=1, padding=0)
        self.tri_out_bn = nn.BatchNorm2d(oup)
        self.tri_out_relu = nn.ReLU(inplace=True)
        
    def forward(self, feature_tensor, feature_vector):     # [1024]  [128]

        b, c = feature_vector.size()
        feature_vector = feature_vector.view([b, c, 1, 1])
        feature_vector = self.inp_vec_tri_relu(self.inp_vec_tri_bn(self.inp_vec_tri(feature_vector)))

        feature_tensor = self.tri_in_relu(self.tri_in_bn(self.tri_in_tensor(feature_tensor)))

        b, _, w, h = feature_tensor.size()

        feature_vector = feature_vector.repeat(1, 1, w, h)
        fusion = torch.cat((feature_tensor, feature_vector),1)

        roi_pool = self.ROI(fusion)
        roi_pool = roi_pool.repeat(1, 1, w, h)
        fusion_roi = torch.cat((feature_tensor, roi_pool),1)       # 2 * (inp // 64)
        fusion_roi = self.offset(fusion_roi)
        fusion_roi = self.conv_offset_relu(self.conv_offset_bn(self.conv_offset(fusion_roi)))

        new_feature = torch.cat((fusion, fusion_roi),1)   # 2 * (inp // 64) + inp_vec

        new_feature = self.tri_out_relu(self.tri_out_bn(self.tri_out_tensor(new_feature)))

        return new_feature

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
        self.channels = channels
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
    def __init__(self, channels, mid=32):
        super(SRModule, self).__init__()
        self.tri_conv = torch.nn.Conv2d(channels, mid, 1)
        self.tri_bn = nn.BatchNorm2d(mid)
        self.tri_relu = nn.ReLU(inplace=True)

        self.DirL1 = DirectionConvLayer(mid, mid, dir_id='1')
        self.DirL2 = DirectionConvLayer(mid, mid, dir_id='2')
        self.DirL3 = DirectionConvLayer(mid, mid, dir_id='3')
        self.DirL4 = DirectionConvLayer(mid, mid, dir_id='4')
        self.DirL5 = DirectionConvLayer(mid, mid, dir_id='5')
        self.DirL6 = DirectionConvLayer(mid, mid, dir_id='6')
        self.DirL7 = DirectionConvLayer(mid, mid, dir_id='7')
        self.DirL8 = DirectionConvLayer(mid, mid, dir_id='8')
        
        self.final1x1 = torch.nn.Conv2d(mid*8, mid, 1)
        self.bn1x1 = nn.BatchNorm2d(mid)
        self.relu1x1 = nn.ReLU(inplace=True)
        
        self.DL = Dependence_Learning(mid, channels)
        
    def forward(self, x):
        x = self.tri_relu(self.tri_bn(self.tri_conv(x)))
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

class Fusion(nn.Module):
    expansion = 1

    def __init__(self, inp=2048, oup=2048, rate=32):
        super(Fusion, self).__init__()
        self.conv1x1_tensor = nn.Conv2d(inp, inp//rate, kernel_size=1, stride=1, padding=0)
        self.bn_conv1x1_tensor = nn.BatchNorm2d(inp//rate)

        self.conv1x1_vector = nn.Conv2d(inp, inp//rate, kernel_size=1, stride=1, padding=0)
        self.bn_conv1x1_vector = nn.BatchNorm2d(inp//rate)

        self.conv3x3_trans = nn.Conv2d(inp//rate*2, oup, kernel_size=3, stride=1, padding=1)
        self.bn_conv3x3_trans = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_tensor, feature_vector):
        b, _, w, h = feature_tensor.size()
        _, L, _, _ = feature_vector.size()

        feature_tensor = self.relu(self.bn_conv1x1_tensor(self.conv1x1_tensor(feature_tensor)))
        feature_vector = self.relu(self.bn_conv1x1_vector(self.conv1x1_vector(feature_vector)))

        feature_vector = feature_vector.repeat(1, 1, w, h)
        fusion = torch.cat((feature_tensor, feature_vector), 1)

        out = self.relu(self.bn_conv3x3_trans(self.conv3x3_trans(fusion)))

        return out

class Feature_Pool(nn.Module):
    def __init__(self, feature_index=[3, 4], out_channel=2048, net='resnet50'):
        super(Feature_Pool, self).__init__()
        self.feature_index = feature_index
        if net=='resnet50':
            self.channel = [256, 512, 1024, 2048]
        elif net=='resnet18':
            self.channel = [64, 128, 256, 512]
        self.size = [56, 28, 14, 7]
        channels = 0
        for i in feature_index:
            channels += self.channel[i-1]

        self.tri_in = torch.nn.Conv2d(channels, channels//64, 1)
        self.tri_in_bn = nn.BatchNorm2d(channels//64)
        self.tri_in_relu = nn.ReLU(inplace=True)

        self.tri_out = torch.nn.Conv2d(channels//64, out_channel, 1)
        self.tri_out_bn = nn.BatchNorm2d(out_channel)
        self.tri_out_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        feature_pool = []
        for i in self.feature_index:
            temp = F.interpolate(x[i-1], size=[self.size[4-len(self.feature_index)], self.size[4-len(self.feature_index)]], mode='bilinear', align_corners=True)
            feature_pool.append(temp)

        feature = torch.cat(feature_pool, 1)

        feature_in = self.tri_in_relu(self.tri_in_bn(self.tri_in(feature)))
        feature_out = self.tri_out_relu(self.tri_out_bn(self.tri_out(feature_in)))
        
        return feature_out

class TPAM(nn.Module):
    expansion = 1

    def __init__(self, inp=2048, oup=2048, rate=32):
        super(TPAM, self).__init__()
        self.conv1x1_tensor = nn.Conv2d(inp, inp//rate, kernel_size=1, stride=1, padding=0)
        self.bn_conv1x1_tensor = nn.BatchNorm2d(inp//rate)

        self.conv1x1_vector = nn.Conv2d(inp, inp//rate, kernel_size=1, stride=1, padding=0)
        self.bn_conv1x1_vector = nn.BatchNorm2d(inp//rate)

        self.conv1x1_trans = nn.Conv2d(inp//rate, oup, kernel_size=1, stride=1, padding=0)
        self.bn_conv1x1_trans = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1) 

    def forward(self, feature_tensor, feature_vector):
        b, _, w, h = feature_tensor.size()
        _, L, _, _ = feature_vector.size()

        feature_tensor = self.relu(self.bn_conv1x1_tensor(self.conv1x1_tensor(feature_tensor)))
        feature_vector = self.relu(self.bn_conv1x1_vector(self.conv1x1_vector(feature_vector)))

        vector = feature_vector
        feature_vector = feature_vector.repeat(1, 1, w, h)
        
        fusion = feature_tensor * feature_vector.expand_as(feature_tensor)

        fusion = vector + self.avgpool(fusion)

        out = self.relu(self.bn_conv1x1_trans(self.conv1x1_trans(fusion)))

        return out

class Ours(nn.Module):
    def __init__(self, channel=2048, num_classes=1, edge_size=7):
        super(Ours, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.srm1 = SRModule(channel)     
        self.srm2 = SRModule(channel)
        self.srm3 = SRModule(channel)   

        self.atm1 = ATM(inp=channel, oup=channel, edge_size=edge_size)
        self.atm2 = ATM(inp=channel, oup=channel, edge_size=edge_size)

        self.classifier1 = nn.Linear(channel, num_classes)  
        self.classifier2 = nn.Linear(channel, num_classes)  
        self.classifier3 = nn.Linear(channel, num_classes)

        #self.fusion1 = Fusion(inp=channel, oup=channel)
        #self.fusion2 = Fusion(inp=channel, oup=channel)
        
        self.fusion1 = TPAM(inp=channel, oup=channel)
        self.fusion2 = TPAM(inp=channel, oup=channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x, x_base, x_ori):
        
        #Spatial Ordered
        SOB = self.avgpool(x_ori)
        
        SRB1 = self.srm1(x)
        SRB1_v = self.avgpool(SRB1)
        x1 = SOB + SRB1_v
        x1_down = self.fusion1(SRB1, x1)
        #x1_down = self.avgpool(x1_down)
        x1_down = x1_down.view(x1.size(0), -1)
        ATM1 = self.atm1(x_base, x1_down)

        SRB2 = self.srm2(ATM1)
        SRB2_v = self.avgpool(SRB2)
        x2 = SOB + SRB2_v
        x2_down = self.fusion2(SRB2, x2)
        #x2_down = self.avgpool(x2_down)
        x2_down = x2_down.view(x2_down.size(0), -1)
        ATM2 = self.atm2(x_base, x2_down)

        SRB3 = self.srm3(ATM2)
        SRB3 = self.avgpool(SRB3)
        x3 = SOB + SRB3
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        classification1 = self.classifier1(x1)
        classification2 = self.classifier2(x2)
        classification3 = self.classifier3(x3)

        return [classification1, classification2, classification3, x_ori]

class Encoding_rawres50(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawres50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrain)
        #model_dir = '/public/data0/users/zhaiwei16/ResNet-50.pth'
        #self.resnet.load_state_dict(torch.load(model_dir))
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        return [x1, x2, x3, x4]

class MPAPNet_res50(nn.Module):
    def __init__(self, num_classes=1, feature_index=[3, 4]):
        super(MPAPNet_res50, self).__init__()
        self.encoding = Encoding_rawres50()  
        self.size = [56, 28, 14, 7]
        self.our = Ours(channel=2048, num_classes=num_classes, edge_size=self.size[feature_index[0]-1])
        self.fp = Feature_Pool(feature_index=feature_index, out_channel=2048, net='resnet50')

    def forward(self, x):

        [x1, x2, x3, x4] = self.encoding(x)
        x_fp = self.fp([x1, x2, x3, x4])

        x_ori = x4
        x = x_fp
        x_base = x_fp

        return self.our(x, x_base, x_ori)

class Encoding_rawres18(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawres18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        return [x1, x2, x3, x4]

class MPAPNet_res18(nn.Module):
    def __init__(self, num_classes=1, feature_index=[3, 4]):
        super(MPAPNet_res18, self).__init__()
        self.encoding = Encoding_rawres18()  
        self.size = [56, 28, 14, 7]
        self.our = Ours(channel=512, num_classes=num_classes, edge_size=self.size[feature_index[0]-1])
        self.fp = Feature_Pool(feature_index=feature_index, out_channel=512, net='resnet18')

    def forward(self, x):

        [x1, x2, x3, x4] = self.encoding(x)
        x_fp = self.fp([x1, x2, x3, x4])

        x_ori = x4
        x = x_fp
        x_base = x_fp

        return self.our(x, x_base, x_ori)

class Encoding_rawVGG19(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawVGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=pretrain)
    
    def forward(self, x):
        
        x = self.vgg19.features(x)
        
        return [x]

class MPAPNet_vgg19(nn.Module):
    def __init__(self, num_classes=1):
        super(MPAPNet_vgg19, self).__init__()
        self.encoding = Encoding_rawVGG19()  
        self.our = Ours(channel=512, num_classes=num_classes)

    def forward(self, x):

        x_fp = self.encoding(x)

        x_ori = x_fp[0]
        x = x_fp[0]
        x_base = x_fp[0]

        return self.our(x, x_base, x_ori)

class Encoding_rawVGG16(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrain)
    
    def forward(self, x):
        
        x = self.vgg16.features(x)
        
        return [x]

class MPAPNet_vgg16(nn.Module):
    def __init__(self, num_classes=1):
        super(MPAPNet_vgg16, self).__init__()
        self.encoding = Encoding_rawVGG16()  
        self.our = Ours(channel=512, num_classes=num_classes)

    def forward(self, x):

        x_fp = self.encoding(x)

        x_ori = x_fp[0]
        x = x_fp[0]
        x_base = x_fp[0]

        return self.our(x, x_base, x_ori)

class Encoding_rawres101(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawres101, self).__init__()
        self.resnet = models.resnet101(pretrained=pretrain)
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        return [x1, x2, x3, x4]

class MPAPNet_res101(nn.Module):
    def __init__(self, num_classes=1, feature_index=[3, 4]):
        super(MPAPNet_res101, self).__init__()
        self.encoding = Encoding_rawres101()  
        self.size = [56, 28, 14, 7]
        self.our = Ours(channel=2048, num_classes=num_classes, edge_size=self.size[feature_index[0]-1])
        self.fp = Feature_Pool(feature_index=feature_index, out_channel=2048, net='resnet50')

    def forward(self, x):

        [x1, x2, x3, x4] = self.encoding(x)
        x_fp = self.fp([x1, x2, x3, x4])

        x_ori = x4
        x = x_fp
        x_base = x_fp

        return self.our(x, x_base, x_ori)

class Encoding_rawres152(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawres152, self).__init__()
        self.resnet = models.resnet152(pretrained=pretrain)
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        return [x1, x2, x3, x4]

class MPAPNet_res152(nn.Module):
    def __init__(self, num_classes=1, feature_index=[3, 4]):
        super(MPAPNet_res152, self).__init__()
        self.encoding = Encoding_rawres152()  
        self.size = [56, 28, 14, 7]
        self.our = Ours(channel=2048, num_classes=num_classes, edge_size=self.size[feature_index[0]-1])
        self.fp = Feature_Pool(feature_index=feature_index, out_channel=2048, net='resnet50')

    def forward(self, x):

        [x1, x2, x3, x4] = self.encoding(x)
        x_fp = self.fp([x1, x2, x3, x4])

        x_ori = x4
        x = x_fp
        x_base = x_fp

        return self.our(x, x_base, x_ori)

class Encoding_rawmobilenetv2(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoding_rawmobilenetv2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
    
    def forward(self, x):
        
        x = self.mobilenetv2.features(x)
        
        return [x]

class MPAPNet_rawmobilenetv2(nn.Module):
    def __init__(self, num_classes=1):
        super(MPAPNet_rawmobilenetv2, self).__init__()
        self.encoding = Encoding_rawmobilenetv2()  
        self.our = Ours(channel=1280, num_classes=num_classes)

    def forward(self, x):

        x_fp = self.encoding(x)

        x_ori = x_fp[0]
        x = x_fp[0]
        x_base = x_fp[0]

        return self.our(x, x_base, x_ori)
