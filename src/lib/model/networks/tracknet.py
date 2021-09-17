from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class Tracknet(nn.Module):
    def __init__(self, heads, head_convs, opt=None):
        super(Tracknet, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.heads = heads
        last_channel=64
        self.num_stacks=1
        self.opt=opt
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes, 
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel, 
                               padding=head_kernel // 2, bias=True)
              convs = [conv]
              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True), 
                    convs[1], nn.ReLU(inplace=True), 
                    convs[2], nn.ReLU(inplace=True), 
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        
        self.pre_hm= nn.Sequential(
            nn.Conv2d(1,64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        dropout=0.1
        self.first_down_sampling_layer = nn.Sequential(
                nn.Conv2d(7, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=0.1),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True))
            
        self.pre_second_layer= nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1))

        self.second_down_sampling_layer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))
        
        self.pre_third_layer= nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, padding=1))

        self.third_down_sampling_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))
        
        self.fourth_down_sampling_layer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))
        
        self.sqex  = SqEx(256)

        self.bottle_neck_layer = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        self.first_up_sampling_layer= nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True))

        self.pre_first_up_sampling_layer=nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.Dropout(dropout))

        self.pre_second_up_sampling_layer=nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Dropout(dropout))

        self.pre_third_up_sampling_layer=nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2))

        self.pre_fourth_up_sampling_layer=nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1))

        self.first_decoder_layer= nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))
        
        self.second_up_sampling_layer= nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        self.second_decoder_layer= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        self.third_up_sampling_layer= nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        self.third_decoder_layer= nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        self.fourth_decoder_layer= nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True))

        
    def forward(self, x, pre_img, pre_hm, 
        x_half_scaled, x_quarter_scaled, 
        pre_img_half_scaled, pre_img_quarter_scaled):

        concat_x_pre_img_feats=torch.cat([x,pre_img, pre_hm], dim=1)
        
        first_layer_feats= self.first_down_sampling_layer(concat_x_pre_img_feats)
        del concat_x_pre_img_feats

        concat_x_half_pre_img_half_feats= torch.cat([x_half_scaled, pre_img_half_scaled], dim=1)
        
        half_scaled_feats_pre = self.pre_second_layer(concat_x_half_pre_img_half_feats)
        del concat_x_half_pre_img_half_feats

        concat_x_quarter_pre_img_quarter_feats= torch.cat([x_quarter_scaled, pre_img_quarter_scaled], dim=1)
        
        quarter_scaled_feats_pre = self.pre_third_layer(concat_x_quarter_pre_img_quarter_feats)
        del concat_x_quarter_pre_img_quarter_feats

        concat_first_layer_with_half_scaled_pre= torch.cat(
                    [first_layer_feats, half_scaled_feats_pre], dim=1)
        
        second_layer_encoded_features= self.second_down_sampling_layer(
                    concat_first_layer_with_half_scaled_pre)
        del concat_first_layer_with_half_scaled_pre

        concat_second_layer_feats_with_quarter_scaled_pre=torch.cat(
                    [second_layer_encoded_features, quarter_scaled_feats_pre], dim=1)
        
        third_layer_encoded_features= self.third_down_sampling_layer(
                    concat_second_layer_feats_with_quarter_scaled_pre)
        del concat_second_layer_feats_with_quarter_scaled_pre

        fourth_layer_encoded_features= self.fourth_down_sampling_layer(third_layer_encoded_features)

        bottle_neck_feats= self.bottle_neck_layer(fourth_layer_encoded_features)

        first_upsampling_layer_feats= self.first_up_sampling_layer(bottle_neck_feats)

        first_decoder_layer_feats= self.first_decoder_layer(first_upsampling_layer_feats
                     + self.pre_first_up_sampling_layer(fourth_layer_encoded_features))

        second_upsampling_layer_feats= self.second_up_sampling_layer(first_decoder_layer_feats)

        second_decoder_layer_feats= self.second_decoder_layer(second_upsampling_layer_feats
                         + self.pre_second_up_sampling_layer(third_layer_encoded_features)
                         )
        squeeze_feats= self.sqex(second_decoder_layer_feats) 

        third_upsampling_layer_feats= self.third_up_sampling_layer(squeeze_feats)

        third_decoder_layer_feats=self.third_decoder_layer(third_upsampling_layer_feats
                        + self.pre_third_up_sampling_layer(second_layer_encoded_features))
        
        fourth_decoder_layer_feats= self.fourth_decoder_layer(
                        self.pre_fourth_up_sampling_layer(third_decoder_layer_feats)
                        + first_layer_feats)
                      
        feats= [fourth_decoder_layer_feats]
        
        out = []
        if self.opt.model_output_list:
            for s in range(self.num_stacks):
                z = []
                for head in sorted(self.heads):
                    z.append(self.__getattr__(head)(feats[s]))
                out.append(z)
        else:
            for s in range(self.num_stacks):
                z = {}
                for head in self.heads:
                    z[head] = self.__getattr__(head)(feats[s])
                out.append(z)
        return out