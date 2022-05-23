import torch

from subnet.Sampler import Sampler
from subnet.Reconstructor import Reconstructor
import torch.nn as nn
from data.utils import TimesPhiTPhi,TimesPhix,TimesPhiT
import cv2
import numpy as np


class VSS(nn.Module):
    def __init__(self):
        super(VSS, self).__init__()
        mask = torch.empty(4, 264, 264, requires_grad=True, device='cuda').type(torch.cuda.FloatTensor)
        # mask = torch.empty(4, 264, 264, requires_grad=True).type(torch.FloatTensor)
        torch.nn.init.kaiming_normal_(mask)
        self.mask = nn.Parameter(mask)
        self.sampler = Sampler()
        self.reconstructor = Reconstructor(7)

    def _init_mask(self, data):
        data = (0.5 * data) + 0.5
        rounded = torch.round(data)
        return torch.clamp((data + (rounded - data)), min=0, max=1)


    def forward(self,x_input,channels,block,use_sampler):
        Phi=self._init_mask(self.mask)
        Phi=Phi.unsqueeze(0)
        x_out=[]
        for i in range(4):
            if use_sampler:
                Phi=self.sampler(xi_input,Phi)
            xi = x_input[:, i*channels:(i + 1) * channels, :, :]  # i 4-frames, [1,4,264,264]

            meas = TimesPhix(xi, Phi, channels)
            meas=meas.squeeze()
            mask_s=torch.sum(Phi.squeeze(),dim=0)
            loc=torch.where(mask_s==0)
            mask_s[loc]=1
            meas_re = torch.div(meas, mask_s)
            xi_init=TimesPhiT(meas_re.unsqueeze(0).unsqueeze(0),Phi,channels)

            # xi_init=TimesPhiTPhi(xi,Phi,channels)
            xi_input=self.reconstructor(xi_init,Phi,block,channels)    #xi_input[1,4,264,264],Phi:[1,4,264,264]
            use_sampler=True
            x_out.append(xi_input)
        x_out=torch.cat(x_out,dim=1)
        return x_out











    # p=Phi.clone()
    # p=p.squeeze().cpu().detach().numpy()
    # p= np.clip(p[0], 0, 255)[:256, :256]
    # cv2.imwrite('E:/data/Datasets/ss-video/phi/phi_{}.jpg'.format(i),p*255)