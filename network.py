# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 2021

@author: Aline Sindel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import functools
import os
from data_utils import extract_regions_from_landmarks_batch, resample_landmarks_from_regions_batch
 
class ArtFacePoints(nn.Module):
    def __init__(self, opt, device, with_augmentation=False):
        super().__init__() 
        self.opt = opt
        self.device = device
        self.with_augmentation = with_augmentation
        #networks
        self.netG = defineHeatmapNetwork(opt.which_network, opt.input_nc, opt.output_nc, opt)
        self.netG_E = defineHeatmapNetwork(opt.which_network, opt.input_nc+11, 11, opt)    
        self.netG_N = defineHeatmapNetwork(opt.which_network, opt.input_nc+9, 9, opt) 
        self.netG_M = defineHeatmapNetwork(opt.which_network, opt.input_nc+20, 20, opt)    
        self.soft_argmax = SoftArgmax2D()
        self.netG = self.netG.to(self.device) 
        self.netG_E = self.netG_E.to(self.device)
        self.netG_N = self.netG_N.to(self.device)
        self.netG_M = self.netG_M.to(self.device)     
    
    def forward(self, images):        
        LR_images = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=False)
        #global forward
        pred_global_featmaps, pred_global_heatmaps, global_predictions = self.global_forward(LR_images)
        #extract regions based on predicted landmarks
        regions, global_region_featmaps, region_rois = extract_regions_from_landmarks_batch(images, global_predictions, pred_global_featmaps, self.opt.patch_size, self.opt.region_size, self.with_augmentation, self.device)        
        #apply region facial networks for refinement
        region_predictions, pred_region_heatmaps = self.region_forward(regions, global_region_featmaps)
        #resample from regions
        prediction = resample_landmarks_from_regions_batch(images, global_predictions, region_predictions, region_rois, self.device)
        return prediction, global_predictions, LR_images
        
    def global_forward(self, x):
        pred_featmaps = self.netG(x)
        predictions, pred_heatmaps = self.soft_argmax(pred_featmaps)        
        return pred_featmaps, pred_heatmaps, predictions
    
    def region_forward(self, x1, x2):
        pred1, heat1 = self.forward_netG_r(self.netG_E, x1[0], x2[0])
        pred2, heat2 = self.forward_netG_r(self.netG_E, x1[1], x2[1])
        pred3, heat3 = self.forward_netG_r(self.netG_N, x1[2], x2[2])
        pred4, heat4 = self.forward_netG_r(self.netG_M, x1[3], x2[3])        
        return [pred1, pred2, pred3, pred4], [heat1, heat2, heat3, heat4]
    
    def forward_netG_r(self, netG_r, x1, x2):
        #channel fusion of region image and feature maps
        x = torch.cat((x1, x2), dim=1) 
        x = netG_r(x)
        predictions, pred_heatmaps = self.soft_argmax(x)        
        return predictions, pred_heatmaps          
            
    def load_weights(self, model_dir_path, epoch, checkpoint = True):
        weights_path = os.path.join(model_dir_path, "ArtFacePoints_{}_net_G.pth".format(epoch))
        weights_path_E = os.path.join(model_dir_path, "ArtFacePoints_{}_net_GEye.pth".format(epoch))
        weights_path_N = os.path.join(model_dir_path, "ArtFacePoints_{}_net_GNose.pth".format(epoch))
        weights_path_M = os.path.join(model_dir_path, "ArtFacePoints_{}_net_GMouth.pth".format(epoch))
        if checkpoint == True:
            self.netG.load_state_dict(torch.load(weights_path)['state_dict'])
            self.netG_E.load_state_dict(torch.load(weights_path_E)['state_dict'])
            self.netG_N.load_state_dict(torch.load(weights_path_N)['state_dict'])
            self.netG_M.load_state_dict(torch.load(weights_path_M)['state_dict'])
        else:                
            self.netG.load_state_dict(torch.load(weights_path))
            self.netG_E.load_state_dict(torch.load(weights_path_E))
            self.netG_N.load_state_dict(torch.load(weights_path_N))
            self.netG_M.load_state_dict(torch.load(weights_path_M))

class SoftArgmax2D(nn.Module):
    "Soft-Argmax: extract landmark coordinates from heatmap. Code from: https://github.com/lext/deep-pipeline/blob/master/deeppipeline/keypoints/models/modules.py"
    def __init__(self, beta=1):
        super(SoftArgmax2D, self).__init__()
        self.beta = beta

    def forward(self, hm):
        hm = hm.mul(self.beta)
        bs, nc, h, w = hm.size()
        hm = hm.squeeze()

        softmax = F.softmax(hm.view(bs, nc, h * w), dim=2).view(bs, nc, h, w)

        weights = torch.ones(bs, nc, h, w).float().to(hm.device)
        w_x = torch.arange(w).float().div(w)
        w_x = w_x.to(hm.device).mul(weights)

        w_y = torch.arange(h).float().div(h)
        w_y = w_y.to(hm.device).mul(weights.transpose(2, 3)).transpose(2, 3)

        approx_x = softmax.mul(w_x).view(bs, nc, h * w).sum(2).unsqueeze(2)
        approx_y = softmax.mul(w_y).view(bs, nc, h * w).sum(2).unsqueeze(2)

        res_xy = torch.cat([approx_x, approx_y], 2)
        res_xy -= 0.5 #added to shift coords to [-0.5,0.5]
        
        # scale to 1
        mval,mind = softmax.view(bs,nc,h*w).max(dim=2)
        pred_heatmap = softmax/(mval.view(bs,nc,1,1)+1e-8)

        return res_xy, pred_heatmap

"""
Code for heatmap encoder-decoder (including helper functions) adapted from the 
generator network of CycleGAN and Pix2Pix: 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
    
def defineHeatmapNetwork(which_network, input_nc, output_nc, opt):
    if opt.norm == 'batch':
        norm = nn.BatchNorm2d
    else:
        print('norm not defined')

    if which_network == 'resnet_6blocks':
        net = ResNetEncoderDecoder(input_nc, output_nc, ngf=64,
                              norm_layer=norm, use_dropout= not opt.no_dropout, n_blocks=6, up_type=opt.up_type) 
    elif which_network == 'resnet_9blocks':
        net = ResNetEncoderDecoder(input_nc, output_nc, ngf=64,
                              norm_layer=norm, use_dropout= not opt.no_dropout, n_blocks=9, up_type=opt.up_type) 
        
    init_weights(net, init_type=opt.init_type)    
    return net
                   
def defineLRscheduler(optimizer,lr_policy,lr_step_size, lr_iter_const, start_epoch,num_epochs):
    if lr_policy == 'step':
        # Decay LR by a factor of 0.1 every x epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1) #25
    elif lr_policy == 'lambda':
        #Lambda decay: first x epochs with LR, then linear decay to zero for remaining epochs    
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + start_epoch - lr_iter_const) / float(num_epochs - lr_iter_const + 1)
            return lr_l
        exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return exp_lr_scheduler
    
class ResNetEncoderDecoder(nn.Module):
    """
    Defines the encoder-decoder architecture that consists of Resnet blocks between a few downsampling/upsampling operations.
    Code and idea originally from Justin Johnson's architecture: https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', up_type='transposed'):
        assert(n_blocks >= 0)
        super(ResNetEncoderDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if up_type=='transposed':
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else: #new: replace convTranspose with normal conv and upsampling
                 model += [nn.Conv2d(in_channels=ngf * mult, out_channels=int(ngf * mult / 2), kernel_size=3, padding=1, bias=use_bias),
                           nn.Upsample(scale_factor=2, mode='bicubic'),
                           norm_layer(int(ngf * mult / 2)),
                           nn.ReLU(True)]                   
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    """
    Define a resnet block.
    Code and idea originally from Justin Johnson's architecture: https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out 
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    