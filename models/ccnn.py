import torch
from torch import nn, optim
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
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



class CCNN1(nn.Module):
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

        predictphase = torch.atan2(out20.imag, out20.real)


        return predictphase

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

class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase

class ccnncgh(nn.Module):
    def __init__(self):
        super().__init__()
        self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()
        self.ccnn3 = CCNN3()
    def forward(self, amp, H):

        # target_complex = torch.complex(amp, torch.tensor(0.))
        target_complex = amp*torch.exp(1j*torch.zeros_like(amp))

        predict_phase = self.ccnn1(target_complex)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))

        res_cpx = self.ccnn3(predict_complex)

        slmfield = propagation_ASM(u_in=predict_complex, precomped_H=H)

        slmfield = slmfield + res_cpx
        holophase = self.ccnn2(slmfield)

        return holophase, slmfield