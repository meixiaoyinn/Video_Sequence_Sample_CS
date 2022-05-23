import glob
import os

import cv2
import torch
import platform
import numpy as np

from tqdm import tqdm
from time import time
import torch.nn as nn
from argparse import ArgumentParser
from skimage.metrics import structural_similarity as sk_cpt_ssim
from data.utils import imread_CS_py,TimesPhix,TimesPhiT,psnr
from data.dataloader import writeimg2,imgloader
from VSS import VSS

parser=ArgumentParser(description='VSS')
'''dir setting'''
parser.add_argument('--dataset_dir',type=str,default='E:/data/Datasets/ss-video/DAVIS/JPEGImages',help='video dataset dir')
parser.add_argument('--dataset_name',type=str,default='low-Resolution',help='dataset_name')
parser.add_argument('--model_dir', type=str, default='E:/data/net', help='trained or pre-trained model directory')
parser.add_argument('--model_name', type=str, default='VSS_davis1.pkl', help='trained or pre-trained model name')
# parser.add_argument('--video_transform',type=bool,default=False,help='decide if transform video to frame')
# parser.add_argument('--frame_dir',type=str,default='E:/data/Datasets/ss-video/video_frames',help='frame dir')
'''net parameter set'''
parser.add_argument('--frame_num',type=int,default=4,help='cs frames')
parser.add_argument('--train_epoch',type=int,default=100,help='epoch number of testing')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of model')
parser.add_argument('--blocksize', type=int, default=0, help='image block size of each train cycle')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')


args=parser.parse_args()
train_epoch=args.train_epoch
learning_rate=args.learning_rate
# video_transform=args.video_transform
gpu_list=args.gpu_list
train_dir=os.path.join(args.dataset_dir,args.dataset_name)
# img_dir=args.frame_dir

model_dir=os.path.join(args.model_dir,args.model_name)
dir_names=os.listdir(train_dir)

# for dir_name in dir_names:    #operate dataset
#     img_dir = os.path.join(train_dir, dir_name)
#     result_name=os.path.join(args.dataset_dir,'low-Resolution',dir_name)
#     if not os.path.exists(result_name):
#         os.makedirs(result_name)
#     writeimg2(img_dir,result_name)


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# ImgNum=len(train_dir)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mask = torch.empty(4, 264, 264, requires_grad=True, device='cuda').type(torch.cuda.FloatTensor)
torch.nn.init.kaiming_normal_(mask)


def img_operate(img):
    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(img, 33)
    Icol = Ipad / 255.0
    block_num_row = int(row_new // 33)
    block_num_col = int(col_new // 33)
    return Icol,block_num_row,block_num_col

model=VSS()
model = model.to(device)
# model.load_state_dict(torch.load(model_dir),False)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_func=nn.MSELoss()
def train(train_dir,dir_names,train_epoch,learning_rate,model_dir,mask,channels=4):
    for epoch in tqdm(range(train_epoch)):
        for img_name in dir_names:
            img_path=os.path.join(train_dir,img_name)
            path=glob.glob(img_path+'/*.jpg')
            new_frames=[]
            for ipath in path:
                img=cv2.imread(ipath,0)
                Icol, block_num_row, block_num_col=img_operate(img)
                Icol=np.reshape(Icol,(1,264,264))
                new_frames.append(Icol)
            gt = np.concatenate(new_frames, axis=0).astype(np.float32)
            gt=torch.tensor(gt).unsqueeze(0)
            for img,index in imgloader(gt,4):
                psnr_=0
                ssim_=0
                img=img.to(device)
                x_input = img.clone()
                start_time = time()
                a = model(x_input, channels, 33, False)
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
                    % (epoch, (end_time - start_time), img_name,index, psnr_ / 16, ssim_ / 16, loss.data))
                if index / 200 == 0:
                    torch.save(model.state_dict(), 'E:/data/net/VSS_davis1.pkl')
    torch.save(model.state_dict(), 'E:/data/net/VSS_davis1.pkl')


train(train_dir,dir_names,train_epoch,learning_rate,model_dir,mask)

# for i in range(channels):
#     xi = img[:, i * channels:(i + 1) * channels, :, :]
#     meas = TimesPhix(xi, mask, channels)
#     meas = meas.squeeze()
#     mask_s = torch.sum(mask.squeeze(), dim=0)
#     loc = torch.where(mask_s == 0)
#     mask_s[loc] = 1
#     meas_re = torch.div(meas, mask_s)
#     xi_init = TimesPhiT(meas_re.unsqueeze(0).unsqueeze(0), mask, channels)
#     print(xi_init)
#     break