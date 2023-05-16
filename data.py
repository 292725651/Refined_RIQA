"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import sys

import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image
from my_imgprocess_lib import *
import cv2
import numpy as np
import os
import scipy.misc as misc
import torchvision

Img_preprocess_lib=Img_preprocess_lib()

    
    
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
#        print('################################')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
#        print(type(h))
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
#        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#    else:
#        print('*******************************')
    return image



def randomShiftScaleRotate(image, mask,
                           shift_limit1=(-0.0, 0.0),
                           shift_limit2=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit1[0], shift_limit1[1]) * width)
        dy = round(np.random.uniform(shift_limit2[0], shift_limit2[1]) * height)
#        print(dx, dy)
        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask



def default_ori_eh_maskencode_loader(img_path, eh_path, mask_path, sz=0, if_eh=1, if_HSV=1, if_SSR=1, if_flip=1, 
                          pad_value=0, 
                          eh_pad_value=0.5,
                          hue_limit=30,
                          sat_limit=0.3,
                          val_limit=0.1,
                          shift_lmt=0.3,
                          scale_lmt=0.3,
                          aspect_lmt=0.1,
                          rotate_lmt=20,
                          mask_code_list = [1, 5, 3, 7],
                          bkgrd = 0
                          ):
    img = cv2.imread(img_path)#[:,:,::-1]
    img_eh = cv2.imread(eh_path)#[:,:,::-1]
    mask = cv2.imread(mask_path)
    if sz!=0:
        img = cv2.resize(img, (sz, sz))    
        img_eh = cv2.resize(img_eh, (sz, sz))
        mask = np.float32(cv2.resize(mask, (sz, sz)))
    
    sz =  img.shape[0]
    
    img = np.float32(img.copy())/255
    img_eh = np.float32(img_eh.copy())/255
    if not if_eh:
        if_HSV=0
        if_SSR=0
        if_flip=0
        u = np.zeros(10,np.bool)
    else:
        
        hue_shift_limit = np.random.randint(-hue_limit, hue_limit+1)
        sat_shift_limit = np.random.uniform(-sat_limit, sat_limit)
        val_shift_limit = np.random.uniform(-val_limit, val_limit)
        u = np.float32(np.random.random(10)<0.5)
#        u1 = float(np.random.random()<0.5)
        shift_limit1 = np.random.uniform(-shift_lmt, shift_lmt)
        shift_limit2 = np.random.uniform(-shift_lmt, shift_lmt)
        scale_limit = np.random.uniform(-scale_lmt, scale_lmt)
        aspect_limit = np.random.uniform(-aspect_lmt, aspect_lmt)
        rotate_limit = np.random.uniform(-rotate_lmt, rotate_lmt)
    
    if if_HSV:
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(hue_shift_limit, hue_shift_limit),
                                       sat_shift_limit=(sat_shift_limit, sat_shift_limit),
                                       val_shift_limit=(val_shift_limit, val_shift_limit),
                                       u=u[0])
        
        img_eh = randomHueSaturationValue(img_eh,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.1, 0.1),
                                       u=0.5)
    if if_SSR:
        img, mask = randomShiftScaleRotate(img-pad_value, mask,
                                           shift_limit1=(shift_limit1, shift_limit1),
                                           shift_limit2=(shift_limit2, shift_limit2),
#                                           scale_limit=(scale_limit, scale_limit),
#                                           aspect_limit=(val_shift_limit, val_shift_limit),
                                           scale_limit=(scale_limit, scale_limit),
                                           aspect_limit=(aspect_limit, aspect_limit),
                                           rotate_limit=(rotate_limit, rotate_limit),
                                           u=u[1])
        img_eh, _ = randomShiftScaleRotate(img_eh-eh_pad_value, mask,
                                           shift_limit1=(shift_limit1, shift_limit1),
                                           shift_limit2=(shift_limit2, shift_limit2),
#                                           scale_limit=(scale_limit, scale_limit),
#                                           aspect_limit=(val_shift_limit, val_shift_limit),
                                           scale_limit=(scale_limit, scale_limit),
                                           aspect_limit=(aspect_limit, aspect_limit),
                                           rotate_limit=(rotate_limit, rotate_limit),
                                           u=u[1]) 
        img = img+pad_value
        img_eh = img_eh+eh_pad_value
    if if_flip: 
        img, mask = randomHorizontalFlip(img, mask, u[2])
        img, mask = randomVerticleFlip(img, mask, u[3])
        # img, mask = randomRotate90(img, mask, u[4])
        img_eh,_ = randomHorizontalFlip(img_eh, mask, u[2])
        img_eh, _ = randomVerticleFlip(img_eh, mask, u[3])
        # img_eh, _ = randomRotate90(img_eh, mask, u[4])        
  
    h_flip = if_flip * u[2]
    
    img = np.array(img[:,:,::-1], np.float32).transpose(2, 0, 1) #* 3.2 - 1.6
    img_eh = np.array(img_eh[:,:,::-1], np.float32).transpose(2, 0, 1) #* 3.2 - 1.6    
    mask = np.array(mask[:,:,::-1], np.float32).transpose(2, 0, 1)
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    mask_code = mask[0]*1+mask[1]*2+mask[2]*4
    if not bkgrd:
        mask_decode = np.zeros((len(mask_code_list),sz,sz),np.float32)        
        for kk in range(len(mask_code_list)):
            mask_decode[kk,:,:] = mask_code == mask_code_list[kk]
    else:
        mask_decode = np.zeros((len(mask_code_list) + 1, sz, sz), np.float32)        
        for kk in range(len(mask_code_list)):
            mask_decode[kk + 1,:,:] = mask_code == mask_code_list[kk]                
        mask_decode[0,:,:]=1-mask_decode.sum(0)
        
    # mask = abs(mask-1)
    return img, img_eh, mask_decode, h_flip


def default_ori_maskencode_loader(img_path, mask_path, sz=0, if_eh=1, if_HSV=1, if_SSR=1, if_flip=1, 
                          pad_value=0, 
                          hue_limit=30,
                          sat_limit=0.3,
                          val_limit=0.1,
                          shift_lmt=0.3,
                          scale_lmt=0.3,
                          aspect_lmt=0.1,
                          rotate_lmt=20,
                          mask_code_list = [1, 5, 3, 7],
                          bkgrd = 0
                          ):
    img = cv2.imread(img_path)#[:,:,::-1]
    mask = cv2.imread(mask_path)
    if sz!=0:
        img = cv2.resize(img, (sz, sz))    
        mask = np.float32(cv2.resize(mask, (sz, sz)))
    
    sz =  img.shape[0]
    
    img = np.float32(img.copy())/255
    if not if_eh:
        if_HSV=0
        if_SSR=0
        if_flip=0
        u = np.zeros(10,np.bool)
    else:
        
        hue_shift_limit = np.random.randint(-hue_limit, hue_limit+1)
        sat_shift_limit = np.random.uniform(-sat_limit, sat_limit)
        val_shift_limit = np.random.uniform(-val_limit, val_limit)
        u = np.float32(np.random.random(10)<0.5)
#        u1 = float(np.random.random()<0.5)
        shift_limit1 = np.random.uniform(-shift_lmt, shift_lmt)
        shift_limit2 = np.random.uniform(-shift_lmt, shift_lmt)
        scale_limit = np.random.uniform(-scale_lmt, scale_lmt)
        aspect_limit = np.random.uniform(-aspect_lmt, aspect_lmt)
        rotate_limit = np.random.uniform(-rotate_lmt, rotate_lmt)
    
    if if_HSV:
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(hue_shift_limit, hue_shift_limit),
                                       sat_shift_limit=(sat_shift_limit, sat_shift_limit),
                                       val_shift_limit=(val_shift_limit, val_shift_limit),
                                       u=u[0])
        
    if if_SSR:
        img, mask = randomShiftScaleRotate(img-pad_value, mask,
                                           shift_limit1=(shift_limit1, shift_limit1),
                                           shift_limit2=(shift_limit2, shift_limit2),
#                                           scale_limit=(scale_limit, scale_limit),
#                                           aspect_limit=(val_shift_limit, val_shift_limit),
                                           scale_limit=(scale_limit, scale_limit),
                                           aspect_limit=(aspect_limit, aspect_limit),
                                           rotate_limit=(rotate_limit, rotate_limit),
                                           u=u[1])

        img = img+pad_value
    if if_flip: 
        img, mask = randomHorizontalFlip(img, mask, u[2])
        img, mask = randomVerticleFlip(img, mask, u[3])
  
    h_flip = if_flip * u[2]
    
    img = np.array(img[:,:,::-1], np.float32).transpose(2, 0, 1) #* 3.2 - 1.6
    mask = np.array(mask[:,:,::-1], np.float32).transpose(2, 0, 1)
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    mask_code = mask[0]*1+mask[1]*2+mask[2]*4
    if not bkgrd:
        mask_decode = np.zeros((len(mask_code_list),sz,sz),np.float32)        
        for kk in range(len(mask_code_list)):
            mask_decode[kk,:,:] = mask_code == mask_code_list[kk]
    else:
        mask_decode = np.zeros((len(mask_code_list) + 1, sz, sz), np.float32)        
        for kk in range(len(mask_code_list)):
            mask_decode[kk + 1,:,:] = mask_code == mask_code_list[kk]                
        mask_decode[0,:,:]=1-mask_decode.sum(0)
        
    # mask = abs(mask-1)
    return img, mask_decode, h_flip



class ImageFolder_maskencode_mtlb(data.Dataset):

    def __init__(self, img_list, eh_list, lb_list, mtlbnp, w_list, 
                 sz=448,
                 if_eh=1, 
                 if_HSV=1, 
                 if_SSR=1, 
                 if_flip=1, 
                 pad_value=0, 
                 eh_pad_value=0.5,
                 bkgrd=0,
                 mask_code_list = [1, 5, 3, 7]
                 ):
        self.img_list = img_list
        self.eh_list = eh_list
        self.lb_list = lb_list
        self.w_list = w_list
        self.mtlbnp = mtlbnp
        self.sz = sz
        self.if_eh = if_eh
        self.if_HSV = if_HSV
        self.if_SSR = if_SSR
        self.if_flip = if_flip
        self.pad_value = pad_value
        self.eh_pad_value = eh_pad_value
        
        self.bkgrd = bkgrd
        self.mask_code_list = mask_code_list
        
        
    def __getitem__(self, index):
            
        rand_np=np.random.uniform(size=(20,)) 
        img, img_eh, mask, h_flip = default_ori_eh_maskencode_loader(self.img_list[index], self.eh_list[index], self.lb_list[index], 
                                                             self.sz, self.if_eh, self.if_HSV, 
                                                             self.if_SSR, self.if_flip, self.pad_value, self.eh_pad_value,
                                                             mask_code_list = self.mask_code_list,
                                                             bkgrd = self.bkgrd)
        w_tensor = torch.Tensor(self.w_list[index]).unsqueeze(-1).unsqueeze(-1)
#        print(img.shape, mask.shape)
#        plt.imshow(img.transpose(1,2,0))
        img = torch.Tensor(img)
        img_eh = torch.Tensor(img_eh)
        mask = torch.Tensor(mask)
        mtlb = torch.LongTensor(self.mtlbnp[index])
        
        return img, img_eh, mask, w_tensor, mtlb, h_flip
    def __len__(self):
        assert len(self.img_list) == len(self.lb_list) == len(self.eh_list) == len(self.mtlbnp), 'The number of images must be equal to labels'
        return len(self.img_list)


class ImageFolder_ori_maskencode_mtlb(data.Dataset):

    def __init__(self, img_list, lb_list, mtlbnp, w_list, 
                 sz=448,
                 if_eh=1, 
                 if_HSV=1, 
                 if_SSR=1, 
                 if_flip=1, 
                 pad_value=0, 
                 bkgrd=0,
                 mask_code_list = [1, 5, 3, 7]
                 ):
        self.img_list = img_list
        self.lb_list = lb_list
        self.w_list = w_list
        self.mtlbnp = mtlbnp
        self.sz = sz
        self.if_eh = if_eh
        self.if_HSV = if_HSV
        self.if_SSR = if_SSR
        self.if_flip = if_flip
        self.pad_value = pad_value     
        self.bkgrd = bkgrd
        self.mask_code_list = mask_code_list
        
        
    def __getitem__(self, index):

        img, mask, h_flip = default_ori_maskencode_loader(self.img_list[index], self.lb_list[index], 
                                                             self.sz, self.if_eh, self.if_HSV, 
                                                             self.if_SSR, self.if_flip, self.pad_value, 
                                                             mask_code_list = self.mask_code_list,
                                                             bkgrd = self.bkgrd)
        w_tensor = torch.Tensor(self.w_list[index]).unsqueeze(-1).unsqueeze(-1)
#        print(img.shape, mask.shape)
#        plt.imshow(img.transpose(1,2,0))
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        mtlb = torch.LongTensor(self.mtlbnp[index])
        
        return img, mask, w_tensor, mtlb, h_flip
    def __len__(self):
        assert len(self.img_list) == len(self.lb_list) == len(self.mtlbnp), 'The number of images must be equal to labels'
        return len(self.img_list)


 
class ImageFolder_maskencode(data.Dataset):

    def __init__(self, img_list, eh_list, lb_list, w_list,
                 sz=448,
                 if_eh=1, 
                 if_HSV=1, 
                 if_SSR=1, 
                 if_flip=1, 
                 pad_value=0, 
                 eh_pad_value=0.5,
                 bkgrd=0,
                 mask_code_list = [1, 5, 3, 7]
                 ):
        self.img_list = img_list
        self.eh_list = eh_list
        self.lb_list = lb_list
        self.w_list = w_list
        self.sz = sz
        self.if_eh = if_eh
        self.if_HSV = if_HSV
        self.if_SSR = if_SSR
        self.if_flip = if_flip
        self.pad_value = pad_value
        self.eh_pad_value = eh_pad_value
        
        self.bkgrd = bkgrd
        self.mask_code_list = mask_code_list
    def __getitem__(self, index):
            
        rand_np=np.random.uniform(size=(20,)) 
        img, img_eh, mask = default_ori_eh_maskencode_loader(self.img_list[index], self.eh_list[index], self.lb_list[index], 
                                                             self.sz, self.if_eh, self.if_HSV, 
                                                             self.if_SSR, self.if_flip, self.pad_value, self.eh_pad_value,
                                                             mask_code_list = self.mask_code_list,
                                                             bkgrd = self.bkgrd)
        w_tensor = torch.Tensor(self.w_list[index]).unsqueeze(-1).unsqueeze(-1)
#        print(img.shape, mask.shape)
#        plt.imshow(img.transpose(1,2,0))
        img = torch.Tensor(img)
        img_eh = torch.Tensor(img_eh)
        mask = torch.Tensor(mask)
        
        return img, img_eh, mask, w_tensor


    def __len__(self):
        assert len(self.img_list) == len(self.lb_list) == len(self.eh_list), 'The number of images must be equal to labels'
        return len(self.img_list)
    

class ImageFolder_maskencode_loc(data.Dataset):

    def __init__(self, img_list, eh_list, lb_list, w_list,
                 sz=448,
                 if_eh=1, 
                 if_HSV=1, 
                 if_SSR=1, 
                 if_flip=1, 
                 pad_value=0, 
                 eh_pad_value=0.5,
                 bkgrd=0,
                 mask_code_list = [1, 5, 3, 7]
                 ):
        self.img_list = img_list
        self.eh_list = eh_list
        self.lb_list = lb_list
        self.w_list = w_list
        self.sz = sz
        self.if_eh = if_eh
        self.if_HSV = if_HSV
        self.if_SSR = if_SSR
        self.if_flip = if_flip
        self.pad_value = pad_value
        self.eh_pad_value = eh_pad_value
        
        self.bkgrd = bkgrd
        self.mask_code_list = mask_code_list
        self.linespace = np.linspace(0,sz-1,sz)
    def __getitem__(self, index):
            
        rand_np=np.random.uniform(size=(20,)) 
        img, img_eh, mask = default_ori_eh_maskencode_loader(self.img_list[index], self.eh_list[index], self.lb_list[index], 
                                                             self.sz, self.if_eh, self.if_HSV, 
                                                             self.if_SSR, self.if_flip, self.pad_value, self.eh_pad_value,
                                                             mask_code_list = self.mask_code_list,
                                                             bkgrd = self.bkgrd)
        
        # c_thick = (mask.sum(-1)==np.matmul(mask.sum(-1).max(-1)[:,np.newaxis], np.ones((1,sz))))
        # c_loc = (c_thick * np.linspace(0,sz-1,sz)[np.newaxis,:]).sum(-1)/c_thick.sum(-1) * (mask.sum(-1).max(-1)>0)
        # r_thick = (mask.sum(-2)==np.matmul(mask.sum(-2).max(-1)[:,np.newaxis], np.ones((1,sz))))
        # r_loc = (r_thick * np.linspace(0,sz-1,sz)[np.newaxis,:]).sum(-1)/r_thick.sum(-1) * (mask.sum(-1).max(-1)>0)
        # tt_loc = (np.concatenate((r_loc[:,np.newaxis], c_loc[:,np.newaxis]),1)).flatten()        
        
        y_thick = (mask.sum(-1)==np.matmul(mask.sum(-1).max(-1)[:,np.newaxis], np.ones((1,self.sz))))
        y_loc = (y_thick * self.linespace[np.newaxis,:]).sum(-1)/y_thick.sum(-1) * (mask.sum(-1).max(-1)>0)
        x_thick = (mask.sum(-2)==np.matmul(mask.sum(-2).max(-1)[:,np.newaxis], np.ones((1,self.sz))))
        x_loc = (x_thick * self.linespace[np.newaxis,:]).sum(-1)/x_thick.sum(-1) * (mask.sum(-1).max(-1)>0)
        tt_loc = (np.concatenate((x_loc[:,np.newaxis], y_loc[:,np.newaxis]),1)).flatten()
        
        w_tensor = torch.Tensor(self.w_list[index]).unsqueeze(-1).unsqueeze(-1)
        
#        print(img.shape, mask.shape)
#        plt.imshow(img.transpose(1,2,0))
        img = torch.Tensor(img)
        img_eh = torch.Tensor(img_eh)
        mask = torch.Tensor(mask)
        tt_loc = torch.Tensor(tt_loc)
        return img, img_eh, mask, w_tensor, tt_loc


    def __len__(self):
        assert len(self.img_list) == len(self.lb_list) == len(self.eh_list), 'The number of images must be equal to labels'
        return len(self.img_list)
    
if __name__=="__main__":  
    max_r = 224
    max_theta=896 
    eh_list=listdir_nfi('./dataset/IDRID/eh')
    ori_list=listdir_nfi('./dataset/IDRID/ori')
    lb_list=listdir_nfi('./dataset/IDRID/lb')
    trans_mat_x, trans_mat_y = trans_ori2p_mat(max_r, max_theta)
#    mask_polar = cv2.imread(lb_list[0])[:,:,2:]
#    img_polar = cv2.imread(ori_list[0])
#    img, img_eh, mask, img_polar, img_eh_polar, mask_polar=default_ori_eh_loader_polar(ori_list[0],eh_list[0],lb_list[0],trans_mat_x, trans_mat_y)
#    pad_value=0
#    eh_pad_value=0.5
#    hue_limit=30
#    sat_limit=0.3
#    val_limit=0.1
#    shift_lmt=0.3
#    scale_lmt=0.3
#    aspect_lmt=0.1    
##    tt=randomHueSaturationValue(img.transpose(1,2,0), hue_shift_limit=(-30, 30), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.1, 0.1), u=1)
#    img_polar, mask_polar=randomShiftScaleRotate(img_polar-pad_value, mask_polar,
#                                               shift_limit=(shift_lmt, shift_lmt),
#                                               scale_limit=(scale_lmt, scale_lmt),
#                                               aspect_limit=(val_limit, val_limit),
#                                               rotate_limit=(-0, 0),
#                                               u=1)