import torch
from torch import nn, optim
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from utils.pytorch_prototyping.pytorch_prototyping import Unet
import math
from utils.propagation_ASM import propagation_ASM

class CDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CDown2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, x):
        out1 = (self.COV1(x))
        return out1

class CUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1

class CUp2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1=nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
    def forward(self, x):
        out1 =self.COV1(x)
        return out1

class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=2):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase

class CCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)
        self.netdown4 = CDown(16, 32)

        self.netup4 = CUp(32, 16)
        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)
        out4 = self.netdown4(out3)

        out17 = self.netup4(out4)
        out18 = self.netup3(out17+out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        return out20

class holonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_phase = InitialPhaseUnet(4, 16)
        self.final_phase_only = FinalPhaseOnlyUnet(4, 16)
        self.ccnn3 = CCNN3()

    def forward(self, amp, H):
        predict_phase = self.initial_phase(amp)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))
        slmfield_ = propagation_ASM(u_in=predict_complex, precomped_H=H)

        res_cpx = self.ccnn3(predict_complex)
        slmfield = slmfield_ + res_cpx

        slmamp=torch.pow(slmfield.real**2 + slmfield.imag**2, 0.5)
        slmphase=torch.atan2(slmfield.imag, slmfield.real)

        slm_amp_phase = torch.cat((slmamp, slmphase), -3)
        holophase = self.final_phase_only(slm_amp_phase)

        return holophase, slmfield