import os
import torch
import platform
import numpy as np

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
parser.add_argument('--dataset_name',type=str,default='video',help='dataset_name')
parser.add_argument('--model_dir', type=str, default='E:/data/net', help='trained or pre-trained model directory')
parser.add_argument('--model_name', type=str, default='VSS_3.pkl', help='trained or pre-trained model name')
# parser.add_argument('--video_transform',type=bool,default=False,help='decide if transform video to frame')
parser.add_argument('--frame_dir',type=str,default='E:/data/Datasets/ss-video/video_frames',help='frame dir')
'''net parameter set'''
parser.add_argument('--frame_num',type=int,default=4,help='cs frames')
parser.add_argument('--train_epoch',type=int,default=20,help='epoch number of testing')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of model')
parser.add_argument('--blocksize', type=int, default=0, help='image block size of each train cycle')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

args=parser.parse_args()
train_epoch=args.train_epoch
learning_rate=args.learning_rate
# video_transform=args.video_transform
gpu_list=args.gpu_list
train_dir=os.sep.join([args.dataset_dir,args.dataset_name])
img_dir=args.frame_dir
# fps=operateData(train_dir,img_dir)

model_dir=os.path.join(args.model_dir,args.model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# ImgNum=len(train_dir)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_cond(cs_ratio, sigma, cond_type):    #设置条件向量z
    para_noise = sigma / 5.0
    if cond_type == 'org_ratio':
        para_cs = cs_ratio / 100.0
    else:
        para_cs = cs_ratio * 2.0 / 100.0

    para_cs_np = np.array([para_cs])
    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)
    para_cs = para_cs.to(device)

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.to(device)
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)

    return para

model=VSS()
model = model.to(device)
# model.load_state_dict(torch.load(model_dir),False)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_func=nn.MSELoss()


def train(filepath,epochs,channels):
    videopath = os.listdir(filepath)
    VideoNum=len(videopath)
    for epoch in tqdm(range(epochs)):
        for path in  videopath:
            p=os.path.join(filepath,path)
            a1=writeImgfortrain(p)
            imgs=torch.tensor(a1[0]).unsqueeze(0)
            for img,index in imgloader(imgs,channels):
                psnr_ = 0
                ssim_ = 0
                img=img.to(device)
                x_input = img.clone()
                start_time = time()
                a = model(x_input,channels,33,False)
                loss = loss_func(a, img)

                optimizer.zero_grad()
                loss = loss.to(torch.float32)
                loss.backward()
                optimizer.step()
                end_time = time()
                for i in range(a.shape[1]):
                    X_rec=a[0,i,:,:]
                    X_gt=img[0,i,:,:]
                    X_gt=X_gt.cpu().detach().numpy()*255
                    X_rec=X_rec.cpu().detach().numpy() * 255
                    rec_PSNR = psnr(X_rec, X_gt)
                    psnr_+=rec_PSNR
                    rec_SSIM = sk_cpt_ssim(X_rec, X_gt, data_range=255)
                    ssim_+=rec_SSIM
                print(
                    '(Train)CS reconstruction,epoch number of model is %d,avg cost time is %.4f second(s),frams number of %s is %d,avg PSNR/SSIM is %.2f/%.4f,and loss is %.4f\n'
                    % (epoch, (end_time - start_time), path,index, psnr_ / 16, ssim_ / 16, loss.data))
                if index / 200 == 0:
                    torch.save(model.state_dict(), 'E:/data/net/VSS_6.pkl')
    torch.save(model.state_dict(), 'E:/data/net/VSS_6.pkl')


train(img_dir,train_epoch,args.frame_num)