from .backbone import build_backbone
import torch
from torch import nn
import torch.nn.functional as F
from .transformer import linear_tiny

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class TrackResnet(nn.Module):
    def __init__(self, heads, head_convs, opt=None):
        super(TrackResnet, self).__init__()
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

        self.backbone = build_backbone().cuda()

        self.trans = linear_tiny(511, 4, 2040)

        dropout=0.1
        
        self.first_layer= nn.Sequential(
            nn.Conv2d(320, 256, 3,stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

       

        self.second_layer= nn.Sequential(
            nn.Conv2d(256, 128, 3,stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.bottle_neck_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 64, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3 ,stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        

    def forward(self, x, pre_img, pre_hm):
        image_features = self.backbone(x)

        # all_inputs = torch.cat([x, pre_img, pre_hm], dim=1)
        all_inputs = torch.cat([pre_img, pre_hm], dim=1)

        trans_feats = self.trans(all_inputs)

        feats = trans_feats.reshape(trans_feats.shape[0],trans_feats.shape[1], 34, 60)

        feats = torch.cat([feats, image_features[2]], dim=1)

        first_layer_feats = self.first_layer(feats)

        feats = torch.cat([first_layer_feats, image_features[1]], dim=1)
        
        second_layer_feats = self.second_layer(feats)

        feats = torch.cat([second_layer_feats, image_features[0]], dim=1)

        bottle_neck_layer = self.bottle_neck_layer(feats)

        feats=[bottle_neck_layer]
        
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
