import math
import torch
import torch.nn as nn

class BottleNeck1D_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck1D_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm1d(in_channel),
                                       nn.Conv1d(in_channel, out_channel, 3, 1, 1, bias=False),
                                       nn.BatchNorm1d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv1d(out_channel, out_channel, 3, stride, 1, bias=False),
                                       nn.BatchNorm1d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_udp(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_udp, self).__init__()
        self.res_layer = mySequential_bottleneck(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 0, bias=False),    # 1
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 0, bias=False),  # 1
                                       nn.BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x, udp1, udp2):
        shortcut = x
        res = self.res_layer(x, udp1, udp2)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class mySequential_bottleneck(nn.Sequential):
    def forward(self, *input):
        x = input[0]
        for i, module in enumerate(self._modules.values()):
            if i == 1:
                x = attach_udp_input(x, input[1])
                x = module(x)
            elif i == 4:
                x = attach_udp_input(x, input[2])
                x = module(x)
            else:
                x = module(x)
        return x

class mySequential_resnet(nn.Sequential):
    def forward(self, *input):
        x = input[0]
        for i, module in enumerate(self._modules.values()):
            if i == 0:
                x = module(x, *input[1:3])
            else:
                x = module(x, *input[3:5])
        return x

def attach_udp_input(x, udp, pad=1):
    feat_size = x.size(2)
    meta_frame = torch.zeros([x.size(1), feat_size + pad * 2, feat_size + pad * 2]).cuda()
    index = torch.ones_like(meta_frame)
    index[:, pad:feat_size + pad, pad:feat_size + pad] = 0
    index = index.int().bool()
    meta_frame[index] = udp
    meta_framed = meta_frame.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
    meta_framed[:, :, pad:feat_size + pad, pad:feat_size + pad] = x
    return meta_framed

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.out_layer = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Dropout(0.4),
                                       Flatten(),
                                       nn.Linear(512 * 4 * 4, 512),
                                       nn.LayerNorm(512)
                                       )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_layer(x)
        return x    #BT,512

class ResNet_udp(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_udp, self).__init__()

        self.layer1 = self._make_layer(block, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.out_layer = nn.Sequential(nn.BatchNorm2d(512),
                                       nn.Dropout(0.4),
                                       Flatten(),
                                       nn.Linear(512 * 4 * 4, 512),
                                       nn.LayerNorm(512)
                                       )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return mySequential_resnet(*layers)

    def forward(self, x, udp):
        x = self.layer1(x, udp[1].cuda(), udp[2].cuda(), udp[3].cuda(), udp[4].cuda())
        x = self.layer2(x, udp[5].cuda(), udp[6].cuda(), udp[7].cuda(), udp[8].cuda())
        x = self.layer3(x, udp[9].cuda(), udp[10].cuda(), udp[11].cuda(), udp[12].cuda())
        x = self.layer4(x, udp[13].cuda(), udp[14].cuda(), udp[15].cuda(), udp[16].cuda())
        x = self.out_layer(x)
        return x    #BT,512