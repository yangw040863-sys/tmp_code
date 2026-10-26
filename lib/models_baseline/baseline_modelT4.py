import logging
import torchvision
from torch.nn.utils import spectral_norm

from .networks import *

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


class Baseline_Net(nn.Module):
    def __init__(self, args):  # 1046
        print("baseline T4")
        super(Baseline_Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(11, 64, 7),
            # nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            # nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(8)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            # nn.ReLU(True),
            UpConv(128, 64),
            # nn.ReLU(True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )
    def forward(self, x_rn, x_guid, x_Nd, x_msk, msk10, x_Vis, x_Quant):
        x = torch.cat((x_rn,x_guid,x_Nd,x_Quant,x_Vis,msk10), dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# class DNet(nn.Module):
#     def __init__(
#         self,
#     ):
#         super(DNet, self).__init__()
#         inc = 12
#         self.conv = nn.Sequential(
#             spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, 4, stride=1, padding=1),
#         )
#
#     def forward(self, output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant):
#         x_In = torch.cat((output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant), 1)
#         feat = self.conv(x_In)
#         return feat
class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()


        self.block1 = Conv2dLayer(12, 64, 7, 1, 3, pad_type='zero', activation='lrelu', norm='none', sn=True)
        self.block2 = Conv2dLayer(64, 64 * 2, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block3 = Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block4 = Conv2dLayer(64 * 4, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block5 = Conv2dLayer(64 * 4, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block6 = Conv2dLayer(64 * 4, 64, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block_merge_1 = Conv2dLayer(64, 32, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block_merge_2 = Conv2dLayer(32, 1, 3, 1, 1, pad_type='zero', activation='none', norm='none', sn=True)

    def forward(self, output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant):

        x_In = torch.cat((output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant), 1)
        x_In = self.block1(x_In)
        x_In = self.block2(x_In)
        x_In = self.block3(x_In)
        x_In = self.block4(x_In)
        x_In = self.block5(x_In)
        x_In = self.block6(x_In)
        x_In = self.block_merge_1(x_In)
        x_In = self.block_merge_2(x_In)
        return x_In


def weights_init(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    net.apply(init_func)

def get_baseline_net(config, **kwargs):
    model = Baseline_Net(config, **kwargs)
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    weights_init(net=model, init_type='xavier', init_gain=0.02)
    return model


def D_net():
    model = DNet()
    weights_init(net=model, init_type='xavier', init_gain=0.02)
    return model