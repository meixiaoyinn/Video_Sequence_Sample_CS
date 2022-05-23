import torch
import torch.nn as nn
from data.utils import TimesPhiTPhi,TimesPhix,TimesPhiT


class CPMB(nn.Module):
    def __init__(self, res_scale_linear, nf=32):
        super(CPMB, self).__init__()
        conv_bias = True
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)  # 3*3 conv kernel
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.res_scale = res_scale_linear
        self.act = nn.ReLU(inplace=True)


    def forward(self,x):

        out = self.act(self.conv1(x))  # first conv layer and ReLu
        out = self.conv2(out)  # second conv layer

        return x+out


class CPMM(nn.Module):
    def __init__(self,res_scale_linear):
        super(CPMM, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(4, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 4, 3, 1, 1, bias=True)

    def forward(self,x,Phi,block,channels):
        phix = TimesPhix(x, Phi, channels)
        x=x-self.lambda_step*TimesPhiTPhi(x,Phi,channels)   #[1,4,264,264]
        x=x+self.lambda_step*TimesPhiT(phix,Phi,channels)   #[1,4,264,264]
        # x_input = x.view(-1, 4, block, block)  # [n,4,33,33]
        x_input=x

        x_mid = self.head_conv(x_input)  #[n,4,33,33]
        x_mid= self.ResidualBlocks(x_mid)  # 近端算子
        x_mid = self.tail_conv(x_mid)   #[n,16,33,33]

        x_pred = x_input + x_mid
        x_pred = x_pred.contiguous().view(-1, 4, 264, 264)
        return x_pred


class Reconstructor(nn.Module):
    def __init__(self,LayerNo):
        super(Reconstructor, self).__init__()
        onelayer=[]
        self.LayerNo=LayerNo
        nf = 32
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)
        for i in range(LayerNo):
            onelayer.append(CPMM(res_scale_linear))
        self.fcs=nn.ModuleList(onelayer)
        # self.weight=nn.Parameter(torch.zeros((1,4,264,264)))

    def forward(self,data,Phi,block,channels):
        x=data
        # noisy=torch.randn([1,4,264,264],dtype=torch.float32,requires_grad=False)
        # noisy=noisy.to("cuda:0" if torch.cuda.is_available() else "cpu")
        # x = x + self.weight * noisy
        for i in range(self.LayerNo):
            x=self.fcs[i](x,Phi,block,channels)
        x_final = x
        return x_final
