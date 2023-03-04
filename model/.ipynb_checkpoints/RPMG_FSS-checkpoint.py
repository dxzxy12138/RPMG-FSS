import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
import math
import copy

import model.resnet as models
import model.vgg as vgg_models
from model.loss import WeightedDiceLoss
from model.non_local_embedded_gaussian import NONLocalBlock2D, NONLocalBlock1D
from model.non_channel_embedded import NONLocalChannelBlock2D, CAM_Module

# from model.ms_deform_attn import MSDeformAttn
# from model.positional_encoding import SinePositionalEncoding
# from model.self_attn import CyCTransformer


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class RPMG(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
                 criterion=WeightedDiceLoss(), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(RPMG, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        
        self.criterion_2 = nn.CrossEntropyLoss(ignore_index=255)
        
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                # nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        
        self.target_avgpool_list = []
        self.target_init_merge = []
        self.target_beta_conv = []
        self.target_inner_cls = []
        self.target_alpha_conv = []
        
        self.target_res1 = None
        self.target_res2 = None
        
        self.allocation_intra_conv = nn.Sequential(
            nn.Conv2d(27, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(128, inter_channels=64, sub_sample=False, bn_layer=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.allocation_cross_down_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            NONLocalBlock1D(128),
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.fushion_conv = []
        for idx in range(4):
            self.fushion_conv.append(nn.Sequential(
                nn.Conv2d(128 * 4, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
        self.fushion_conv = nn.ModuleList(self.fushion_conv)
        
        self.fushion_down_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.online_predictor = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )
    
    
    def _get_target_model(self):
        target_avgpool_list = []
        target_init_merge = []
        target_beta_conv = []
        target_inner_cls = []
        target_alpha_conv = []
        
        target_res1 = None
        target_res2 = None
        for idx in range(len(self.pyramid_bins)):
            init_merge_layer = copy.deepcopy(self.init_merge[idx])
            set_requires_grad(init_merge_layer, False)
            beta_conv_layer = copy.deepcopy(self.beta_conv[idx])
            set_requires_grad(beta_conv_layer, False)
            inner_cls_layer = copy.deepcopy(self.inner_cls[idx])
            set_requires_grad(inner_cls_layer, False)
            avgpool_layer = copy.deepcopy(self.avgpool_list[idx])
            set_requires_grad(avgpool_layer, False)
            
            target_init_merge.append(init_merge_layer)
            target_beta_conv.append(beta_conv_layer)
            target_inner_cls.append(inner_cls_layer)
            target_avgpool_list.append(avgpool_layer)
        
        for idx in range(len(self.pyramid_bins) - 1):
            alpha_conv_layer = copy.deepcopy(self.alpha_conv[idx])
            set_requires_grad(alpha_conv_layer, False)
            
            target_alpha_conv.append(alpha_conv_layer)
        
        target_res1 = copy.deepcopy(self.res1)
        set_requires_grad(target_res1, False)
        target_res2 = copy.deepcopy(self.res2)
        set_requires_grad(target_res2, False)
        
        target_init_merge = nn.ModuleList(target_init_merge)
        target_beta_conv = nn.ModuleList(target_beta_conv)
        target_inner_cls = nn.ModuleList(target_inner_cls)
        target_alpha_conv = nn.ModuleList(target_alpha_conv)
        
        return target_init_merge, target_beta_conv, target_inner_cls, target_alpha_conv, target_avgpool_list, target_res1, target_res2
        
    
    def cal_contrast_loss(self, query_feat_list, flag):
        bsize, ch_sz, _, _ = query_feat_list[0].shape
        # print(query_feat_list[0].shape)
        query_feat = self.GAP(query_feat_list[0]).view(bsize, ch_sz)
        query_feat_1, query_feat_2, query_feat_3 = self.GAP(query_feat_list[1]), self.GAP(query_feat_list[2]), self.GAP(query_feat_list[3])
        query_feat_1, query_feat_2, query_feat_3 = query_feat_1.view(bsize, ch_sz), query_feat_2.view(bsize, ch_sz), query_feat_3.view(bsize, ch_sz)
        if flag:
            query_feat = self.online_predictor(query_feat)
        
        loss_1 = torch.sum(torch.sqrt(torch.abs(query_feat - query_feat_1) ** 2) / (ch_sz * bsize))
        loss_2 = torch.sum(torch.sqrt(torch.abs(query_feat - query_feat_2) ** 2) / (ch_sz * bsize))
        loss_3 = torch.sum(torch.sqrt(torch.abs(query_feat - query_feat_3) ** 2) / (ch_sz * bsize))
        return (loss_1 + loss_2 + loss_3) / 3 * 0.1
    
    
    def get_support_feat(self, s_x, s_y, mask_list, final_supp_list, supp_feat_list):
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                 align_corners=True)                
                supp_feat_4 = (self.layer4(supp_feat_3 * mask))
        
            final_supp_list.append(supp_feat_4)
            if self.vgg:
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),mode='bilinear', align_corners=True)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            # print('supp_feat shape is', supp_feat.shape)
            # print('mask shape is', mask.shape)
            supp_feat = Weighted_GAP(supp_feat, mask)
            # print('supp_feat shape is', supp_feat.shape)
            supp_feat_list.append(supp_feat)
        return mask_list, final_supp_list, supp_feat_list
    
    
    def get_aug_supp_feat(self, supp_feat_, aug_supp_feat_list):
        aug_supp_feat = None
        for i in range(len(supp_feat_)):
            supp_feat_i = supp_feat_[i]
            if i == 0:
                aug_supp_feat = supp_feat_i
            else:
                aug_supp_feat += supp_feat_i
        aug_supp_feat = aug_supp_feat / len(supp_feat_)
        aug_supp_feat_list.append(aug_supp_feat)
        return aug_supp_feat_list
    
    
    def get_prior_mask(self, final_supp_list, mask_list, query_feat_4, query_feat_3, corr_query_mask_list):
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)
            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]
            cosine_eps = 1e-7
            
            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            
            similarity_index  = similarity.topk(9, dim=1)[1]
            similarity_index_raw = similarity_index // sp_sz
            similarity_index_col = similarity_index % sp_sz
            similarity_index_raw = similarity_index_raw / sp_sz
            similarity_index_col = similarity_index_col / sp_sz
            similarity_value = similarity.topk(9, dim=1)[0]
            
            similarity_map = list()
            for j in range(similarity_index.shape[1]):
                similarity_map.append(similarity_value[:, j, :].view(bsize, 1, sp_sz*sp_sz))
                similarity_map.append(similarity_index_raw[:, j, :].view(bsize, 1, sp_sz*sp_sz))
                similarity_map.append(similarity_index_col[:, j, :].view(bsize, 1, sp_sz*sp_sz))
            similarity_map = torch.cat(similarity_map, dim=1)
            
            similarity = similarity_map.contiguous().view(bsize, 27, sp_sz, sp_sz)
            similarity = self.allocation_intra_conv(similarity)
            
            corr_query = F.interpolate(similarity, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        return corr_query_mask_list
    
    
    def transform_mask_0(self, corr_query_mask_list_i, bsize, ch_sz, sp_sz):
        corr_query_mask_list_i = corr_query_mask_list_i.contiguous().view(bsize, ch_sz, sp_sz*sp_sz)
        corr_query_mask_list_i = corr_query_mask_list_i.contiguous().permute(0, 2, 1)
        corr_query_mask_list_i = corr_query_mask_list_i.contiguous().view(-1, ch_sz)
        return corr_query_mask_list_i
    
    def transform_mask_1(self, corr_query_mask_list_i, bsize, ch_sz, sp_sz):
        corr_query_mask_i = corr_query_mask_list_i.contiguous().view(bsize, sp_sz*sp_sz, ch_sz)
        corr_query_mask_i = corr_query_mask_i.contiguous().permute(0, 2, 1)
        corr_query_mask_i = corr_query_mask_i.contiguous().view(bsize, ch_sz, sp_sz, sp_sz)
        return corr_query_mask_i

    
    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(), s_y=torch.FloatTensor(1, 1, 473, 473).cuda(), y=None, padding_mask=None, s_padding_mask=None, s_x_view_1=None, s_y_view_1=None, s_x_view_2=None, s_y_view_2=None, s_x_view_3=None, s_y_view_3=None, s_x_view_4=None, s_y_view_4=None, epoch=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        
        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = (self.layer4(query_feat_3))
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        query_feat_clone = query_feat

        # Support Feature
        supp_feat_list, final_supp_list, mask_list = [], [], []
        mask_list, final_supp_list, supp_feat_list = self.get_support_feat(s_x, s_y, mask_list, final_supp_list, supp_feat_list)
        
        # select the views of support features randomly
        s_x_view_list = [s_x_view_1, s_x_view_2, s_x_view_3, s_x_view_4]
        s_y_view_list = [s_y_view_1, s_y_view_2, s_y_view_3, s_y_view_4]
        view_list = random.sample(range(0,4), 3)
        s_x_list_view_1, s_y_list_view_1 = s_x_view_list[view_list[0]], s_y_view_list[view_list[0]]
        s_x_list_view_2, s_y_list_view_2 = s_x_view_list[view_list[1]], s_y_view_list[view_list[1]]
        s_x_list_view_3, s_y_list_view_3 = s_x_view_list[view_list[2]], s_y_view_list[view_list[2]]

        with torch.no_grad():
            # View 1 Support Feature
            supp_feat_list_view_1, final_supp_list_view_1, mask_list_view_1 = [], [], []
            mask_list_view_1, final_supp_list_view_1, supp_feat_list_view_1 = self.get_support_feat(s_x_list_view_1, s_y_list_view_1, mask_list_view_1, final_supp_list_view_1, supp_feat_list_view_1)

            # View 2 Support Feature
            supp_feat_list_view_2, final_supp_list_view_2, mask_list_view_2  = [], [], []
            mask_list_view_2, final_supp_list_view_2, supp_feat_list_view_2 = self.get_support_feat(s_x_list_view_2, s_y_list_view_2, mask_list_view_2, final_supp_list_view_2, supp_feat_list_view_2)

            # View 3 Support Feature
            supp_feat_list_view_3, final_supp_list_view_3, mask_list_view_3  = [], [], []
            mask_list_view_3, final_supp_list_view_3, supp_feat_list_view_3 = self.get_support_feat(s_x_list_view_3, s_y_list_view_3, mask_list_view_3, final_supp_list_view_3, supp_feat_list_view_3)

        
        aug_supp_feat_list = []
        supp_feat_ = supp_feat_list
        aug_supp_feat_list = self.get_aug_supp_feat(supp_feat_, aug_supp_feat_list)
        aug_supp_feat_list = self.get_aug_supp_feat(supp_feat_list_view_1, aug_supp_feat_list)
        aug_supp_feat_list = self.get_aug_supp_feat(supp_feat_list_view_2, aug_supp_feat_list)
        aug_supp_feat_list = self.get_aug_supp_feat(supp_feat_list_view_3, aug_supp_feat_list)
        
        # Support Feature generate mask
        cosine_eps = 1e-7
        corr_query_mask_list, ori_corr_query_mask_list = [], []
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)
            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            mask_similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            mask_similarity = (mask_similarity - mask_similarity.min(1)[0].unsqueeze(1))/(mask_similarity.max(1)[0].unsqueeze(1) - mask_similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            mask_corr_query = mask_similarity.view(bsize, 1, sp_sz, sp_sz)
            # corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            ori_corr_query_mask_list.append(mask_corr_query)
            
            similarity_index  = similarity.topk(9, dim=1)[1]
            similarity_index_raw = similarity_index // sp_sz
            similarity_index_col = similarity_index % sp_sz
            similarity_index_raw = similarity_index_raw / sp_sz
            similarity_index_col = similarity_index_col / sp_sz
            similarity_value = similarity.topk(9, dim=1)[0]
            
            similarity_map = list()
            for j in range(similarity_index.shape[1]):
                similarity_map.append(similarity_value[:, j, :].view(bsize, 1, sp_sz*sp_sz))
                similarity_map.append(similarity_index_raw[:, j, :].view(bsize, 1, sp_sz*sp_sz))
                similarity_map.append(similarity_index_col[:, j, :].view(bsize, 1, sp_sz*sp_sz))
            similarity_map = torch.cat(similarity_map, dim=1)
            
            similarity = similarity_map.contiguous().view(bsize, 27, sp_sz, sp_sz)
            similarity = self.allocation_intra_conv(similarity)
            
            # similarity = similarity_map.contiguous().view(bsize, 27, sp_sz*sp_sz)
            # similarity = similarity.contiguous().permute(0, 2, 1)
            # similarity = self.allocation_intra_linear(similarity)
            # similarity = similarity.contiguous().permute(0, 2, 1)
            # similarity = similarity.view(bsize, -1, sp_sz, sp_sz)
            # similarity = self.allocation_intra_conv(similarity)
            
            corr_query = F.interpolate(similarity, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        
        
        corr_query_view_1_mask_list, corr_query_view_2_mask_list, corr_query_view_3_mask_list = [], [], []
        corr_query_view_4_mask_list = []
        corr_query_view_1_mask_list = self.get_prior_mask(final_supp_list_view_1, mask_list_view_1, query_feat_4, query_feat_3, corr_query_view_1_mask_list)
        corr_query_view_2_mask_list = self.get_prior_mask(final_supp_list_view_2, mask_list_view_2, query_feat_4, query_feat_3, corr_query_view_2_mask_list)
        corr_query_view_3_mask_list = self.get_prior_mask(final_supp_list_view_3, mask_list_view_3, query_feat_4, query_feat_3, corr_query_view_3_mask_list)
        
        for i in range(len(corr_query_mask_list)):
            bsize, ch_sz, sp_sz, _ = corr_query_mask_list[i].size()[:]
            corr_query_mask_list_i = corr_query_mask_list[i]
            corr_query_mask_list_i = self.transform_mask_0(corr_query_mask_list_i, bsize, ch_sz, sp_sz)
           
            corr_query_view_1_mask_list_i = self.transform_mask_0(corr_query_view_1_mask_list[i], bsize, ch_sz, sp_sz)
            corr_query_view_2_mask_list_i = self.transform_mask_0(corr_query_view_2_mask_list[i], bsize, ch_sz, sp_sz)
            corr_query_view_3_mask_list_i = self.transform_mask_0(corr_query_view_3_mask_list[i], bsize, ch_sz, sp_sz)
            
            # corr_query_mask_ = torch.cat([corr_query_mask_list_i.unsqueeze(2), corr_query_view_1_mask_list_i.unsqueeze(2), corr_query_view_2_mask_list_i.unsqueeze(2), corr_query_view_3_mask_list_i.unsqueeze(2)], dim=2)
            
            corr_query_mask_ = torch.cat([corr_query_mask_list_i.unsqueeze(2), corr_query_view_1_mask_list_i.unsqueeze(2), corr_query_view_2_mask_list_i.unsqueeze(2), corr_query_view_3_mask_list_i.unsqueeze(2)], dim=2) 
            corr_query_mask_ = self.allocation_cross_down_conv(corr_query_mask_)
            
            corr_query_mask_i = corr_query_mask_[:, :, 0]
            corr_query_mask_i = self.transform_mask_1(corr_query_mask_[:, :, 0], bsize, ch_sz, sp_sz)
            corr_query_view_1_mask_i = self.transform_mask_1(corr_query_mask_[:, :, 1], bsize, ch_sz, sp_sz)
            corr_query_view_2_mask_i = self.transform_mask_1(corr_query_mask_[:, :, 2], bsize, ch_sz, sp_sz)
            corr_query_view_3_mask_i = self.transform_mask_1(corr_query_mask_[:, :, 3], bsize, ch_sz, sp_sz)
    
            corr_query_mask_fushion_list = [corr_query_mask_i, corr_query_view_1_mask_i, corr_query_view_2_mask_i, corr_query_view_3_mask_i]
            corr_query_mask_fushion = torch.cat(corr_query_mask_fushion_list, dim=1)
            
            for j in range(4):
                w_j = self.fushion_conv[j](corr_query_mask_fushion)
                # print('w_i.grad_fn is ', w_i.grad_fn)
                w_j = w_j.view(bsize, 1, sp_sz, sp_sz)
                # print('corr_query_mask_ is ', corr_query_mask_)
                # print()
                # print('w_i is', w_i)
                corr_query_mask_fushion_list[j] = corr_query_mask_fushion_list[j] * w_j
            
            # corr_query_mask_fushion = sum(corr_query_mask_fushion_list)
            # ori_corr_query_mask = corr_query_mask_fushion_list[0].clone()
            corr_query_mask_fushion = corr_query_mask_fushion_list[0] + 0.1 * (corr_query_mask_fushion_list[1] + corr_query_mask_fushion_list[2] + corr_query_mask_fushion_list[3])
                
            corr_query_mask_fushion_down = self.fushion_down_conv(corr_query_mask_fushion)
            
            corr_query_mask_list[i] = corr_query_mask_fushion_down
            
        corr_query_mask = None
        for i in range(len(corr_query_mask_list)):
            if i == 0:
                corr_query_mask = corr_query_mask_list[i]
            else:
                corr_query_mask += corr_query_mask_list[i]
        corr_query_mask = corr_query_mask / len(corr_query_mask_list)
        
        # print(corr_query_mask)
        # corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        
        '''归一化'''
        bsize, _, sp_sz, _ = corr_query_mask.size()[:]
        corr_query_mask = corr_query_mask.view(bsize, sp_sz*sp_sz)   
        corr_query_mask = (corr_query_mask - corr_query_mask.min(1)[0].unsqueeze(1))/(corr_query_mask.max(1)[0].unsqueeze(1) - corr_query_mask.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query_mask = corr_query_mask.view(bsize, 1, sp_sz, sp_sz)
        
        # corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)
        
        contrast_query_feat_list = []
        query_feat_clone = query_feat.clone()
        # final_out_list = list()
        out_list = []
        
        contrast_query_feat_list = []
        out_list = []
        pyramid_feat_list = []
        merge_feat_bin_list = []
        # query_feat = query_feat_clone
        aug_supp_feat = aug_supp_feat_list[0]

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat_clone.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat_clone)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat_clone)
            
            supp_feat_bin = aug_supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat_clone = torch.cat(pyramid_feat_list, 1)
        
        query_feat_clone = self.res1(query_feat_clone)
        
        contrast_query_feat_list.append(query_feat_clone)
        
        query_feat_clone = self.res2(query_feat_clone) + query_feat_clone
        out = self.cls(query_feat_clone)
        
        if epoch > 50:
        # if epoch >= 40:
            with torch.no_grad():
                target_init_merge, target_beta_conv, target_inner_cls, target_alpha_conv, target_avgpool_list, target_res1, target_res2 = self._get_target_model()
                for i in range(1, 4):
                    # out_list = []
                    pyramid_feat_list = []
                    merge_feat_bin_list = []
                    query_feat_clone = query_feat.clone()
                    aug_supp_feat = aug_supp_feat_list[i]
                    for idx, tmp_bin in enumerate(self.pyramid_bins):
                        if tmp_bin <= 1.0:
                            bin = int(query_feat_clone.shape[2] * tmp_bin)
                            query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat_clone)
                        else:
                            bin = tmp_bin
                            query_feat_bin = target_avgpool_list[idx](query_feat_clone)
                        supp_feat_bin = aug_supp_feat.expand(-1, -1, bin, bin)
                        corr_mask_bin = F.interpolate(corr_query_mask.clone(), size=(bin, bin), mode='bilinear', align_corners=True)
                        merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
                        merge_feat_bin = target_init_merge[idx](merge_feat_bin)
                        if idx >= 1:
                            pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                            pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                            rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                            merge_feat_bin = target_alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin
                        merge_feat_bin = target_beta_conv[idx](merge_feat_bin) + merge_feat_bin
                        inner_out_bin = target_inner_cls[idx](merge_feat_bin)
                        merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                                       mode='bilinear', align_corners=True)
                        pyramid_feat_list.append(merge_feat_bin)
                        # out_list.append(inner_out_bin)
                    query_feat_clone = torch.cat(pyramid_feat_list, 1)
                    query_feat_clone = target_res1(query_feat_clone)
                    query_feat_clone.detach_()
                    contrast_query_feat_list.append(query_feat_clone)
                    flag = True
        
        else:
            a = torch.randn(bsize, 10, 10, 10).cuda()
            contrast_query_feat_list = [a, a, a, a]
            flag = False
        contrast_loss = self.cal_contrast_loss(contrast_query_feat_list, flag)
        # print('contrast_loss is ', contrast_loss)
        
        
        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            # main_loss =  0.5 * (self.criterion(out, y.long())) + 0.5 * (self.criterion_2(out, y.long()))
            main_loss =  self.criterion_2(out, y.long())
            # main_loss =  self.criterion_2(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                # aux_loss = aux_loss + 0.5 * (self.criterion(inner_out, y.long())) + 0.5 * (self.criterion_2(inner_out, y.long()))
                aux_loss = aux_loss + self.criterion_2(inner_out, y.long())
                # aux_loss = aux_loss + self.criterion_2(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss, contrast_loss
        else:
            return out
