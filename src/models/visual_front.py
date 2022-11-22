import torch
from torch import nn
from src.models.resnet import BottleNeck_IR, ResNet, BottleNeck_IR_udp, ResNet_udp

class Visual_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNet(BottleNeck_IR, [2, 2, 2, 2])

    def forward(self, x):
        x = self.frontend(x)    #B,C,T,H,W
        B, C, T, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B*T, C, H, W)
        x = self.resnet(x)  # B*T, 512             #0.20 sec (5 frames)
        x = x.view(B, T, -1)   # B, T, 512
        return x

class Visual_front_udp(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(  # 112, 112
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 0, 0), bias=False),    #2, 3, 3
            nn.BatchNorm3d(64), #56, 56
            nn.PReLU(64),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )   #28, 28

        self.resnet = ResNet_udp(BottleNeck_IR_udp, [2, 2, 2, 2])

    def forward(self, x, udp):
        x = attach_udp_input_3d(2, 3, x, udp[0].cuda())
        x = self.frontend(x)    #B,C,T,H,W
        B, C, T, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B*T, C, H, W)
        x = self.resnet(x, udp)  # B*T, 512             #0.20 sec (5 frames)
        x = x.view(B, T, -1)   # B, T, 512
        return x

def attach_udp_input_3d(pad_t, pad_s, x, udp):
    # x: B, C, T, H, W
    feat_size_t, feat_size_h, feat_size_w = [*x.size()[-3:]]
    meta_frame = torch.zeros([x.size(1), feat_size_t + pad_t * 2, feat_size_h + pad_s * 2, feat_size_w + pad_s * 2]).cuda()
    index = torch.ones_like(meta_frame)
    index[:, :, pad_s:feat_size_h + pad_s, pad_s:feat_size_w + pad_s] = 0.
    index = index.int().bool()
    meta_frame[index] = udp.repeat(x.size(2) + pad_t * 2)
    meta_framed = meta_frame.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)
    meta_framed[:, :, pad_t:feat_size_t + pad_t, pad_s:feat_size_h + pad_s, pad_s:feat_size_w + pad_s] = x
    meta_framed[:, :, :pad_t, :, :] = 0.
    meta_framed[:, :, feat_size_t + pad_t:, :, :] = 0.
    return meta_framed