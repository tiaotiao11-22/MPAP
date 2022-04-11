import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .Deform_Conv import ConvOffset2D

_default_anchors_setting = (
    dict(layer='p3', stride=1, size=3, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)

def generate_default_anchor_maps(anchors_setting=None, input_shape=(14, 14)):

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
        self.down = nn.Conv2d(inp, inp//2, 3, 1, 1)
        self.ReLU = nn.ReLU()
        self.tidy = nn.Conv2d(inp//2, 6, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d = self.ReLU(self.down(x))
        t = self.tidy(d).view(batch_size, -1)
        return t

class Roi(torch.nn.Module):
    def __init__(self, inp, topN=1, inp_size=7):
        super(Roi, self).__init__()
        self.inp_tri = nn.Conv2d(inp, inp//2, kernel_size=1, stride=1, padding=0)
        self.inp_tri_bn = nn.BatchNorm2d(inp//2)
        self.inp_tri_relu = nn.ReLU(inplace=True)

        self.proposal_net = ProposalNet(inp//2)
        self.topN = topN
        self.inp = inp
        _, edge_anchors, _ = generate_default_anchor_maps(input_shape=(inp_size, inp_size))
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
    def __init__(self, inp=2048, oup=2048, tri_rate=32, inp_size=7):
        super(ATM, self).__init__()

        self.inp_vec = inp
        self.inp = inp
        self.oup = oup
        self.ROI = Roi(inp=2*inp//tri_rate, inp_size=inp_size)
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
        
    def forward(self, feature_tensor, feature_vector):     # [1024]  [1024]

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

class Fusion(nn.Module):
    expansion = 1

    def __init__(self, inp=2048, oup=2048, rate=16):
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

def conv3x3(in_planes, out_planes, stride=1):

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

    def __init__(self, block, layers, num_classes, tri_rate=64, emb=2048):
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
        
        self.classifier_b1 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier_b2 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier_b3 = nn.Linear(512 * block.expansion, num_classes)
        self.embedding_b1 = nn.Linear(512 * block.expansion, emb)
        self.embedding_b2 = nn.Linear(512 * block.expansion, emb)
        self.embedding_b3 = nn.Linear(512 * block.expansion, emb)

        self.atm12 = ATM(inp=2048, oup=2048, inp_size=7)
        self.atm23 = ATM(inp=2048, oup=2048, inp_size=7)
        
        self.atm12 = Fusion()
        self.atm23 = Fusion()

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

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_base = x
        
        x_b1 = self.avgpool(x)
        x_b1 = x_b1.view(x_b1.size(0), -1)
        classification_b1 = self.classifier_b1(x_b1)
        embedding_b1 = self.embedding_b1(x_b1)
        
        x_b1 = x_b1.view([x_b1.size(0), x_b1.size(1), 1, 1])
        x_b2 = self.atm12(x_base, x_b1)
        x_b2 = self.avgpool(x_b2)
        x_b2 = x_b2.view(x_b2.size(0), -1)
        classification_b2 = self.classifier_b2(x_b2)
        embedding_b2 = self.embedding_b2(x_b2)
        
        x_b2 = x_b2.view([x_b2.size(0), x_b2.size(1), 1, 1])
        x_b3 = self.atm23(x_base, x_b2)
        x_b3 = self.avgpool(x_b3)
        x_b3 = x_b3.view(x_b3.size(0), -1)
        classification_b3 = self.classifier_b3(x_b3)
        embedding_b3 = self.embedding_b3(x_b3)
        
        classification = (classification_b1 + classification_b2 + classification_b3) / 3
        
        return embedding_b1, embedding_b2, embedding_b3, classification_b1, classification_b2, classification_b3, classification
    
    def forward(self, input1, input2, input3):               # [+ , + , -]
        embedding_b1_1, embedding_b2_1, embedding_b3_1, classification_b1_1, classification_b2_1, classification_b3_1, C1 = self.forward_once(input1)
        embedding_b1_2, embedding_b2_2, embedding_b3_2, classification_b1_2, classification_b2_2, classification_b3_2, C2 = self.forward_once(input2)
        embedding_b1_3, embedding_b2_3, embedding_b3_3, classification_b1_3, classification_b2_3, classification_b3_3, C3 = self.forward_once(input3)
        dist_a_b3 = F.pairwise_distance(embedding_b3_1, embedding_b3_2, 2)
        dist_b_b3 = F.pairwise_distance(embedding_b3_1, embedding_b3_3, 2)
        dist_a_b2 = F.pairwise_distance(embedding_b2_1, embedding_b2_2, 2)
        dist_b_b2 = F.pairwise_distance(embedding_b2_1, embedding_b2_3, 2)
        dist_a_b1 = F.pairwise_distance(embedding_b1_1, embedding_b1_2, 2)
        dist_b_b1 = F.pairwise_distance(embedding_b1_1, embedding_b1_3, 2)
        return dist_a_b1, dist_b_b1, dist_a_b2, dist_b_b2, dist_a_b3, dist_b_b3, embedding_b1_1, embedding_b1_2, embedding_b1_3, embedding_b2_1, embedding_b2_2, embedding_b2_3, embedding_b3_1, embedding_b3_2, embedding_b3_3, classification_b1_1, classification_b1_2, classification_b1_3, classification_b2_1, classification_b2_2, classification_b2_3, classification_b3_1, classification_b3_2, classification_b3_3, C1, C2, C3

def MAPNet(pretrained=True, Num_C=1):

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=Num_C).cuda()
    if pretrained:
        pretrained_dict = torch.load('resnet50-19c8e357.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model