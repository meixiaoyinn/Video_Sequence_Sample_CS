import glob

import numpy as np
import cv2
import os
from data.utils import imread_CS_py


def split(filepaths):    #split image name and sort image by right index
    # filepaths = glob.glob('result\\videobyfps' + '/*.png')
    img_num=len(filepaths)
    filepaths_copy=[]
    for i in range(img_num):
        f = int(filepaths[i].split('_')[-2])   #split image name to get index
        filepaths_copy.append(f)
    z=sorted(enumerate(filepaths_copy),key=lambda x: x[1])   #sort by index
    result_index=[]
    for i in range(img_num):
        result_index.append(z[i][0])
    b=np.array(filepaths)
    a=b[result_index]
    return a


def operateData(path,oppath):   #path:video path,oppath:result path
    videopath=os.listdir(path)
    fps=[]
    for i in range(len(videopath)):
        video_name=videopath[i].split('_')[0]
        result_path=os.sep.join([oppath,video_name])
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        p=os.path.sep.join([path,videopath[i]])
        # print(p,result_path,video_name)
        fps.append(writeImg(p,result_path,video_name))
    return fps


def writeImg(path,img_paths,iname):   #path:video dir,img_paths:result path,iname:video name
    vi=cv2.VideoCapture(path)
    fps=int(vi.get(5))
    n_frames = int(vi.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(n_frames):
        ret,fram=vi.read()
        fram = cv2.resize(fram, (256, 256), interpolation=cv2.INTER_LINEAR)
        imgname='{}_{}_.jpg'.format(iname,i)
        imagepath = os.sep.join([img_paths, imgname])
        cv2.imwrite(imagepath, fram)
    vi.release()
    return fps


def writeimg2(imgs_name,result_name):
    img_dir=glob.glob(imgs_name+'/*.jpg')
    for img_name in img_dir:
        r_name=result_name
        name=img_name.split('/')[-1].split('\\')[-1]
        img=cv2.imread(img_name,1)
        Iorg_y = img[:, :, 0]
        fram = cv2.resize(Iorg_y, (256, 256), interpolation=cv2.INTER_LINEAR)
        r_name=os.sep.join([r_name,name])
        cv2.imwrite(r_name, fram)
    # print(img_dir)


def writeImgfortrain(path):
    new_frame = []
    gt=[]
    frampaths = glob.glob(path + '/*.jpg')
    frampaths = split(frampaths)
    for i in range(len(frampaths)):
        img_path = frampaths[i]
        img = cv2.imread(img_path, 1)
        Img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = img[:, :, 0]
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y, 33)
        Icol = Ipad / 255.0
        block_num_row = int(row_new // 33)
        block_num_col = int(col_new // 33)
        new_frame.append(Icol)
        gt.append(Img_rec_yuv)
    gt=np.concatenate(gt,axis=2)
    new_frame=np.array(new_frame).astype(np.float32)
    return new_frame,gt,block_num_row,block_num_col,row, col


# a=operateData('E:/data/Datasets/ss-video/test','E:/data/Datasets/ss-video/test_frames')
# print(a)


def operatedata1(path):
    imgs_path=glob.glob(path)
    if len(imgs_path)/16!=0:
        imgs_path=imgs_path[:(int(len(imgs_path)/16))*16]
    print(len(imgs_path))
    for img_path in imgs_path:
        img=cv2.imread(img_path,1)

# operatedata1('E:/data\Datasets/ss-video\DAVIS-2017-trainval-Full-Resolution\DAVIS/JPEGImages/Full-Resolution/bear'+'/*.jpg')


def imgloader(img, channels):
    [n, c, h, w] = img.shape
    step=channels*channels
    if c / step != 0: c -= step
    for i in range(0, c, step):
        # print(i)
        data = img[:,i:i + step, :, :]
        yield data,i