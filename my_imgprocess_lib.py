# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:03:39 2019

@author: gtj
"""
from scipy import misc
import sys, os
#from scipy import misc
import numpy as np
from PIL import Image, ImageFilter
import math
import scipy.io as sio
import cv2
import time
import h5py
import warnings
import matplotlib.pyplot as plt
import random
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import shutil
def strlist_add(input_list,str_add,direction=0):
    output_list=input_list.copy()
    for kk in range(len(input_list)):
        if direction==0:
            output_list[kk]=str_add+input_list[kk]
        elif direction==1:
            output_list[kk]=input_list[kk]+str_add
        else:
            raise Exception("wrong input")
    return output_list


def fileparts(fileName):
    path_only, name_type = os.path.split(fileName)
    filename_only,filetype=os.path.splitext(name_type)
    return path_only, filename_only, filetype


def visualize(data, filename,imgtype='bmp'):
#    assert (len(data.shape) == 3)  #height*width*channels
    img = None
    if len(data.shape) == 2:  #in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8)
                             )  #the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8)
                             )  #the image is between 0-1
    img.save(filename + '.'+imgtype)
    return img

def auto_number(i,n=5):
    output=str(i)
    for ii in range(n):
        if len(output)<n:
            output='0'+output
    return output
    
    
def listdir(path, list_name,showinfo=0):  #
    files=os.listdir(path)    
    files.sort()
    if showinfo:
        print(files)
    for file in files:
        if file!='Thumbs.db':          
            file_path = os.path.join(path, file)
            flag=(file_path in list_name)
            if os.path.isdir(file_path):  
                listdir(file_path, list_name)  
            else:
                if not flag:            
                    list_name.append(file_path)  
                    
                    
def listdir_nfi(path,showinfo=0):
    files=os.listdir(path)    
    files.sort()
    if showinfo:
        print(files)
    files_list_return=[]
    total_num=len(files)
    for kk in range(total_num):
        if files[kk]!='Thumbs.db':
            files_list_return.append(os.path.join(path, files[kk]))
    return files_list_return   
             
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        print(path+'already exsit')
        return False
        
        
class Mycvlib(object):
   
    def fillHole(self,im_in):        
        im_floodfill = im_in.copy()            
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_in.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)          
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)         
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)           
        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv           
        return im_out     
 
        
        
class Img_preprocess_lib(object):    
        
    def img_padding_up(self,img_ori,up,pad_val):
        if len(img_ori.shape)==3:
            im_pad=np.zeros([img_ori.shape[0]+up,img_ori.shape[1],img_ori.shape[2]],dtype=img_ori.dtype)+pad_val
        else:
            im_pad=np.zeros([img_ori.shape[0]+up,img_ori.shape[1]],dtype=img_ori.dtype)+pad_val
#        print(im_pad.shape)
        if up>=0:
            im_pad[up:,:,]=img_ori
        else:
            im_pad=img_ori[-up:,:,]
        return im_pad

        
    def img_padding_down(self,img_ori,down,pad_val):
        if len(img_ori.shape)==3:
            im_pad=np.zeros([img_ori.shape[0]+down,img_ori.shape[1],img_ori.shape[2]],dtype=img_ori.dtype)+pad_val
        else:
            im_pad=np.zeros([img_ori.shape[0]+down,img_ori.shape[1]],dtype=img_ori.dtype)+pad_val
#        print(im_pad.shape)
        if down>=0:
            im_pad[:img_ori.shape[0],:,]=img_ori
        else:
            im_pad=img_ori[:img_ori.shape[0]+down,:,]
        return im_pad 
        
        
    def img_padding_left(self,img_ori,left,pad_val):
        if len(img_ori.shape)==3:
            im_pad=np.zeros([img_ori.shape[0],img_ori.shape[1]+left,img_ori.shape[2]],dtype=img_ori.dtype)+pad_val
        else:
            im_pad=np.zeros([img_ori.shape[0],img_ori.shape[1]+left],dtype=img_ori.dtype)+pad_val
#        print(im_pad.shape)
        if left>=0:
            im_pad[:,left:,]=img_ori
        else:
            im_pad=img_ori[:,-left:,]
        return im_pad 
        
        
    def img_padding_right(self,img_ori,right,pad_val):
        if len(img_ori.shape)==3:
            im_pad=np.zeros([img_ori.shape[0],img_ori.shape[1]+right,img_ori.shape[2]],dtype=img_ori.dtype)+pad_val
        else:
            im_pad=np.zeros([img_ori.shape[0],img_ori.shape[1]+right],dtype=img_ori.dtype)+pad_val
#        print(im_pad.shape)
        if right>=0:
            im_pad[:,:img_ori.shape[1],]=img_ori
        else:
            im_pad=img_ori[:,:img_ori.shape[1]+right,]
        return im_pad 
        
        
    def img_padding_udlr(self,img_ori,up,down,left,right,pad_val=0):
        im_ori=img_ori.copy()
        im_ori_pad_up=self.img_padding_up(im_ori,up,pad_val)
        im_ori_pad_down=self.img_padding_down(im_ori_pad_up,down,pad_val)
        im_ori_pad_left=self.img_padding_left(im_ori_pad_down,left,pad_val)
        im_ori_pad_right=self.img_padding_right(im_ori_pad_left,right,pad_val)
        return im_ori_pad_right
        
    def img_rot_fill(self,img,angle,fill=0):
        rows=img.shape[0]
        cols=img.shape[1]
        M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst=cv2.warpAffine(img,M,(cols,rows)) 
        if fill!=0:
            blk=img*0+1
            dst_blk=cv2.warpAffine(blk,M,(cols,rows))
            dst=dst+(1-dst_blk)*fill
        return dst



    def img_dteh(self,ori_img,nameonly_ori,save_dir,save_type_ori,flip_d_list=[],rot_list=[],bk=0):
        
        tp_img_flip=ori_img.copy()
        str1='FN'
        if len(rot_list)!=0:
            for rr in range(len(rot_list)):
                tp_rot_img=self.img_rot_fill(tp_img_flip,rot_list[rr],fill=bk)
                str2='R'+str(int(rot_list[rr]))
                save_ori_name=save_dir+nameonly_ori+'-'+str1+'-'+str2+save_type_ori
                cv2.imwrite(save_ori_name,tp_rot_img)
        else:
            save_ori_name=save_dir+nameonly_ori+'-'+str1+save_type_ori
            cv2.imwrite(save_ori_name,tp_img_flip)
        
        if len(flip_d_list)!=0:    
            for kk in range(len(flip_d_list)):
                tp_img_flip=cv2.flip(ori_img, flip_d_list[kk])
                str1='F'+str(flip_d_list[kk])
                if len(rot_list)!=0:
                    for rr in range(len(rot_list)):
                        tp_rot_img=self.img_rot_fill(tp_img_flip,rot_list[rr],fill=bk)
                        str2='R'+str(int(rot_list[rr]))
                        save_ori_name=save_dir+nameonly_ori+'-'+str1+'-'+str2+save_type_ori
                        cv2.imwrite(save_ori_name,tp_rot_img)
                else:
                    save_ori_name=save_dir+nameonly_ori+'-'+str1+save_type_ori
                    cv2.imwrite(save_ori_name,tp_img_flip)

def show_dif_2map(map1,map2):
    assert(len(map1.shape)==2)
    assert(len(map2.shape)==2)
    fusion_map=np.zeros((map1.shape)+(3,),dtype=map1.dtype)
    fusion_map[:,:,0]=map1.copy()
    fusion_map[:,:,1]=map2.copy()
    return fusion_map


def regionprops(bwlabel):
    
    n_areas=bwlabel.max()
    
    areas=np.int32(np.zeros((n_areas,1)))
    b_box=np.int32(np.zeros((n_areas,4)))
    center=np.float32(np.zeros((n_areas,2)))
    for idx in range(n_areas):
        tp_map=(bwlabel==(idx+1))
        
        
        areas[idx]=tp_map.sum()
        
        r_min=np.where(tp_map.sum(-1)>=1)[0].min()
        r_max=np.where(tp_map.sum(-1)>=1)[0].max()
        c_min=np.where(tp_map.sum(-2)>=1)[0].min()
        c_max=np.where(tp_map.sum(-2)>=1)[0].max()        
        
        b_box[idx,0]=c_min
        b_box[idx,1]=r_min
        b_box[idx,2]=c_max-c_min
        b_box[idx,3]=r_max-r_min        
        
        center[idx,0]=(c_min+c_max)/2
        center[idx,1]=(r_min+r_max)/2
     
    
    class Return_struct(object):
        def __init__(self, areas, b_box, center):
            self.areas=areas
            self.b_box=b_box
            self.center=center
    return_struct=Return_struct(areas, b_box, center)
    
    return return_struct    

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
	if (scale is None) and (center is None):
		return image.rotate(angle=angle, resample=resample)
	nx,ny = x,y = center
	sx=sy=1.0
	if new_center:
		(nx,ny) = new_center
	if scale:
		(sx,sy) = (scale, scale)
	cosine = math.cos(angle)
	sine = math.sin(angle)
	a = cosine/sx
	b = sine/sx
	c = x-nx*a-ny*b
	d = -sine/sy
	e = cosine/sy
	f = y-nx*d-ny*e
	return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)  ,a        

           
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)      
        
        
if __name__=="__main__":
    image=Image.fromarray(np.array(Image.open('E:/retinal_patch_les/eh_1024//IDRiD-IDRiD_01.jpg'))-128)
    
#    image2=image.transform(image.size,Image.AFFINE, ( 1, 0.1, 0,
#                                                      -0.1, 1, 0), resample=Image.BICUBIC) 
    img_siz=1024
    img_center=(img_siz/2)
    
    start_time=time.time()
    for kk in range(100):
        resiz_x_rt=1.1-random.random()*0.2
        resiz_y_rt=1.1-random.random()*0.2
        resiz_mat=np.array([[resiz_x_rt,0,0],[0,resiz_y_rt,0],[0,0,1]])
        
        rot_ang=(random.random())*2*math.pi
        cos_ang=math.cos(rot_ang)
        sin_ang=math.sin(rot_ang)    
        tp_rot_mat=np.array([[cos_ang,-sin_ang,0],[sin_ang,cos_ang,0],[0,0,1]])
        center_shift_mat1=np.array([[1,0,img_center],[0,1,img_center],[0,0,1]])
        center_shift_mat2=np.array([[1,0,-img_center],[0,1,-img_center],[0,0,1]])
        rot_mat=np.matmul(np.matmul(center_shift_mat1, tp_rot_mat),center_shift_mat2)
        
        dx_shear=0.2-random.random()*0.4
        dy_shear=0.2-random.random()*0.4    
        dxy_mat=np.array([[1,dx_shear,0],[dy_shear,1,0],[0,0,1]])
        
        shift_x=img_siz/20-random.random()*img_siz/10
        shift_y=img_siz/20-random.random()*img_siz/10
        shift_mat=np.array([[1,0,0],[0,1,0],[shift_x,shift_y,1]])
        
        flip_rd=(0.5>random.random())
        flip_rd_mat=np.eye(3)
        if flip_rd:
            flip_rd_mat=np.array([[1,0,0],[0,-1,img_siz],[0,0,1]])
        
        
        
        total_affine_mat=np.matmul(np.matmul(shift_mat,np.matmul(np.matmul(rot_mat,resiz_mat),dxy_mat)),flip_rd_mat)
    #    total_affine_mat=np.matmul(shift_mat,np.matmul(np.matmul(rot_mat,resiz_mat),dxy_mat))
        a=total_affine_mat[0,0]
        b=total_affine_mat[0,1]
        c=total_affine_mat[0,2]
        d=total_affine_mat[1,0]
        e=total_affine_mat[1,1]
        f=total_affine_mat[1,2]    
        image3=Image.fromarray(np.array(image.transform(image.size,Image.AFFINE, (a,b,c,d,e,f)))+128)
        
        
        left=int(random.random()*img_siz)
        up=int(random.random()*img_siz)
        right=left+20+int(random.random()*img_siz)
        down=up+20+int(random.random()*img_siz)
        rand_radius=random.random()*10
        image3=image3.filter(MyGaussianBlur(radius=rand_radius,bounds=(left,up,right,down)
                                           ))
        print(kk)
    end_time=time.time()
#    image2,a=ScaleRotateTranslate(image, 30, center = (512,512), new_center = (550,550))
    plt.imshow(image3)
    
#    ss=np.ones((10000,))
#    for kk in range(10000):
#        ss[kk]=random.random()
#    img=cv2.imread('E://Model_992//ori_1024//13_left.jpg')
#    rows,cols,ch=img.shape
#    pts1=np.float32([[10,10],[110,10],[10,110]])
#    pts2=np.float32([[0,10],[120,10],[0,130]])
#    M=cv2.getAffineTransform(pts1,pts2)
#    dst=cv2.warpAffine(img,M,(1035,1099))
#    
#    pltimg=img*0
#    pltimg[:,:,0]=dst[:,:,2]
#    pltimg[:,:,1]=dst[:,:,1]
#    pltimg[:,:,2]=dst[:,:,0]
#    plt.imshow(pltimg)
