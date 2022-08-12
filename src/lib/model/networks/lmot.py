from .dla_backbone import dla34
import torch
from torch import nn
import torch.nn.functional as F
from .transformer import linear_tiny

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class LMOT(nn.Module):
    def __init__(self, heads, head_convs, opt=None):
        super(LMOT, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.heads = heads
        last_channel = 64
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

        self.backbone = dla34()

        self.trans = linear_tiny(511, 4, 2040)

        dropout=0.07

        self.depths =3

        self.res_layer = ResidualBlock(64,64)
        
        self.first_layer= nn.Sequential(
            nn.Conv2d(320, 256, 3,stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

       

        self.second_layer= nn.Sequential(
            nn.Conv2d(256, 128, 3,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        

    def forward(self, x, pre_img, pre_hm):
        image_features = self.backbone(x)

        all_inputs = torch.cat([pre_img, pre_hm], dim=1)

        trans_feats = self.trans(all_inputs)

        feats = trans_feats.reshape(trans_feats.shape[0],trans_feats.shape[1], 34, 60)

        for x in range(self.depths):
          feats = self.res_layer(feats)

        feats = torch.cat([feats, image_features[2]], dim=1)

        first_layer_feats = self.first_layer(feats)

        feats = torch.cat([first_layer_feats, image_features[1]], dim=1)
        
        second_layer_feats = self.second_layer(feats)

        feats = torch.cat([second_layer_feats, image_features[0]], dim=1)

        last_layer = self.last_layer(feats)

        feats=[last_layer]
        
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
