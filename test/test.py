import os
import torch
import numpy as np

import cv2
from tqdm import tqdm
from time import time
import torch.nn as nn
from argparse import ArgumentParser
from skimage.metrics import structural_similarity as sk_cpt_ssim

from data.utils import RandomDataset, imread_CS_py, img2col_py, col2im_CS_py, psnr, write_data
from data.dataloader import operateData,imgloader,writeImgfortrain

from VSS import VSS

parser=ArgumentParser(description='VSS')
'''dir setting'''
parser.add_argument('--dataset_dir',type=str,default='E:/data/Datasets/ss-video',help='video dataset dir')
parser.add_argument('--dataset_name',type=str,default='zjm',help='dataset_name')
parser.add_argument('--model_dir', type=str, default='E:/data/net', help='trained or pre-trained model directory')
parser.add_argument('--model_name', type=str, default='VSS_5.pkl', help='trained or pre-trained model name')
# parser.add_argument('--video_transform',type=bool,default=False,help='decide if transform video to frame')
parser.add_argument('--frame_dir',type=str,default='E:/data/Datasets/ss-video/test_frames',help='frame dir')
'''net parameter set'''
parser.add_argument('--frame_num',type=int,default=4,help='cs frames')
parser.add_argument('--blocksize', type=int, default=0, help='image block size of each train cycle')

args=parser.parse_args()
test_dir=os.sep.join([args.frame_dir,args.dataset_name])
img_dir=test_dir
result=os.sep.join([args.dataset_dir,'result/2'])
if not os.path.exists(result):
    os.makedirs(result)
model_dir=os.path.join(args.model_dir,args.model_name)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=VSS()
model.load_state_dict(torch.load(model_dir),False)
# print(model.state_dict())
model = model.to("cpu")

def get_cond(cs_ratio, sigma, cond_type):    #设置条件向量z
    para_noise = sigma / 5.0
    if cond_type == 'org_ratio':
        para_cs = cs_ratio / 100.0
    else:
        para_cs = cs_ratio * 2.0 / 100.0

    para_cs_np = np.array([para_cs])
    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)
    para_cs = para_cs.to("cpu")

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.to("cpu")
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)

    return para


def vsstest(filepath,channels):
    a1 = writeImgfortrain(filepath)
    imgs = torch.tensor(a1[0]).unsqueeze(0)
    block_num_row = a1[2]
    block_num_col=a1[3]
    for img, index in imgloader(imgs, channels):
        psnr_1 = 0
        ssim_1 = 0
        psnr_2 = 0
        ssim_2 = 0
        img = img.to("cpu")
        x_input = img
        start_time = time()
        a = model(x_input, channels, 33, False)
        end_time = time()
        for i in range(a.shape[1]):
            X_rec = a[0, i, :, :]
            X_irog = img[0, i, :, :]
            X_irog = X_irog.detach().numpy() * 255
            X_rec = X_rec.detach().numpy() * 255
            im_rec_rgb = np.clip(X_rec, 0, 255)[:a1[4], :a1[5]]
            r_path = os.sep.join([result, 'test_second{}.jpg'.format(index+i)])
            cv2.imwrite(r_path, im_rec_rgb)
            rec_PSNR = psnr(X_rec, X_irog)
            psnr_1 += rec_PSNR
            rec_SSIM = sk_cpt_ssim(X_rec, X_irog, data_range=255)
            ssim_1 += rec_SSIM
            if i>3:
                psnr_2 += rec_PSNR
                ssim_2 += rec_SSIM
        print(
            '(Train)CS reconstruction,avg cost time is %.4f second(s),frams number of model is %d,avg PSNR/SSIM is %.2f/%.4f\n'
            % ((end_time - start_time), index, psnr_1 / 16, ssim_1 / 16))
        print("psnr_2/ssim_2 is %.2f/%.4f\n"%(psnr_2/12,ssim_2/12))

vsstest(img_dir,4)