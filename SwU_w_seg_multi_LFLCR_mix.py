# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:57:33 2019

@author: 29272
"""

import sys

from my_imgprocess_lib import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import random
import os
import argparse
from loss import *
from data import *
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from eval_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='unnamed', help='experiment name (default: unname)')
parser.add_argument('--times', type=str, default='1', help='experiment times (default: 1)')
parser.add_argument('--gpu', type=str, default='0', help='gpu ID (default: 0)')
parser.add_argument('--bsz', type=int, default=16, help='batch size (default: 16)')
parser.add_argument('--n_w', type=int, default=12, help='number of workers (default: 16)')
parser.add_argument('--folds', type=int, default=1, help='number of workers (default: 16)')
parser.add_argument('--idx', type=int, default=1, help='number of workers (default: 16)')
parser.add_argument('--lr', type=float, default=1e-4, help='number of workers (default: 16)')
parser.add_argument('--ep', type=int, default=1000, help='number of workers (default: 16)')
parser.add_argument('--om', type=str, default='adam', help='optimizer')


args = parser.parse_args()
exp_times=args.times
gpu_id=args.gpu
exp_name=args.exp_name
total_folds=args.folds
valid_fold_idx=args.idx
lr=args.lr    
epochs=args.ep
om=args.om


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
#torch.cuda.set_device(gpu_id)  


seed = int(exp_times)-1    
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Smooth_CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(Smooth_CrossEntropyLoss, self).__init__()
    def forward(self, pred, truth):
        return  torch.mean(-(torch.log(pred+1e-8)*truth).sum(1)+(torch.log(truth+1e-8)*truth).sum(1))

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def forward(self, pred, truth):
        return  torch.mean(-(torch.log(pred+1e-8)*truth).sum(1))

def load_from(swin_unet, pretrained_path, map_location='cpu'):
    import copy
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=map_location)
        if "model"  not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = swin_unet.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return swin_unet
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        msg = swin_unet.load_state_dict(full_dict, strict=False)
        return swin_unet
        # print(msg)
    else:
        print("none pretrain")


class SwU_multi(nn.Module):
    def __init__(self, num_classes = 3, seg_classes = 2):
        super(SwU_multi, self).__init__()
        self.SwU_cls = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=seg_classes,
                                embed_dim=96,
                                depths=[2, 2, 2, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        
        self.fc1 = nn.Linear(in_features=int(96 * 2 ** 3), out_features=num_classes, bias=True)
        self.fc2 = nn.Linear(in_features=int(96 * 2 ** 3), out_features=num_classes, bias=True)
        self.fc3 = nn.Linear(in_features=int(96 * 2 ** 3), out_features=num_classes, bias=True)
        self.fc4 = nn.Linear(in_features=int(96 * 2 ** 3), out_features=num_classes, bias=True)
        self.fc5 = nn.Linear(in_features=int(96 * 2 ** 3), out_features=num_classes, bias=True)
        
    def forward(self, x):
        x, x_emb = self.SwU_cls(x)
        x_emb_pool = x_emb.mean(1)
        
        x_fc1 = self.fc1(x_emb_pool)
        x_fc2 = self.fc2(x_emb_pool)
        x_fc3 = self.fc3(x_emb_pool)
        x_fc4 = self.fc4(x_emb_pool)
        x_fc5 = self.fc5(x_emb_pool)

        return x, x_fc1, x_fc2, x_fc3, x_fc4, x_fc5


    
if __name__=="__main__":  

    log_interval=1
#    epochs=400
    epo_start=1
#    gpu_device_ID=[0]
    LR = lr
    batch_size=args.bsz
    num_workers=args.n_w
    if sys.platform=='win32':
        num_workers=0
    
    num_classes = 3
    seg_classes = 2
    imgsz=224
    RR = 150*imgsz/2140
    mkdir(exp_name+'//')
    
    
    ori_list=[]; lb_list=[]; q2_idx = None
    df1 = pd.read_csv('./dataset/MMCCFP/MMC20220607.csv')
    fnm_list = list(df1.iloc[:,0])
    
    for kk in range(len(fnm_list)):
        ori_list.append('./dataset/MMCCFP/ori_224/'+fnm_list[kk])
        lb_list.append('./dataset/MMCCFP/lb_png_encode_224/'+fnm_list[kk][:-4]+'.png')   
        
    df2 = pd.read_csv('./dataset/SHDRS/q_fixed_0717.csv')
    fnm_list = list(df2.iloc[:,0])
    
    for kk in range(len(fnm_list)):
        ori_list.append('./dataset/SHDRS/ori_224/'+fnm_list[kk])
        lb_list.append('./dataset/SHDRS/lb_png_encode_224/'+fnm_list[kk][:-4]+'.png')               
     
    
    mtlbnp = np.concatenate((np.int32(df1.iloc[:,1:]), (np.int32(df2.iloc[:,1:]))))
    weight_np = np.zeros((len(ori_list), seg_classes),np.float32)
    weight_np[:, 0]=1
    weight_np[:, 1:]=1
    weight_np_list = weight_np.tolist()
    mtlbnp_list = mtlbnp.tolist()
    
    t_data_idx = (np.where((mtlbnp[:,10]==0)+(mtlbnp[:,10]==2))[0]); v_data_idx = (np.where(mtlbnp[:,10]==1)[0])
    train_ori_list = (np.array(ori_list)[t_data_idx]).tolist(); valid_ori_list = (np.array(ori_list)[v_data_idx]).tolist()
    #train_eh_list = (np.array(eh_list)[t_data_idx]).tolist(); valid_eh_list = (np.array(eh_list)[v_data_idx]).tolist()
    train_lb_list = (np.array(lb_list)[t_data_idx]).tolist(); valid_lb_list = (np.array(lb_list)[v_data_idx]).tolist()
    train_mtlbnp = mtlbnp[t_data_idx]; valid_mtlbnp = mtlbnp[v_data_idx]
    train_w_list = np.array(weight_np_list)[t_data_idx].tolist(); valid_w_list = np.array(weight_np_list)[v_data_idx].tolist()    
    
    
    train_dataset = ImageFolder_ori_maskencode_mtlb(train_ori_list[:], train_lb_list[:], train_mtlbnp, train_w_list[:], sz=imgsz,
                                 if_eh=1,                                 
                                 if_HSV=0, 
                                 if_SSR=0, 
                                 if_flip=1, pad_value=0, 
                                 bkgrd = 0,
                                 mask_code_list = [2,4]
                                 )
   
    train_data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    total_N_img=train_dataset.__len__()

    valid_dataset=ImageFolder_ori_maskencode_mtlb(valid_ori_list[:], valid_lb_list[:], valid_mtlbnp, valid_w_list[:], sz=imgsz, 
                                                  if_eh=0, 
                                                  pad_value=0, 
                                                  bkgrd = 0,
                                                  mask_code_list = [2,4]                                             
                                                 )
    valid_data_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_N_v_img=valid_dataset.__len__()  
    
    test_dataset = valid_dataset
    test_data_loader = valid_data_loader
    total_N_t_img=test_dataset.__len__()     
    
    model_class = SwU_multi(num_classes = num_classes, seg_classes = seg_classes)
 
    model_class = model_class.cuda()


    
    
    loss_function = dice_bce_loss().cuda()
    loss_function_CE=Smooth_CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop([
                                  {'params': model_class.parameters()},
                                  ], lr=LR)   # optimize all cnn parameters

    model_class = torch.nn.DataParallel(model_class)
    max_JD=np.zeros((num_classes,),np.float32)
    max_AUPR=np.zeros((num_classes,),np.float32)
    
    max_JD[:]=0.2
    max_acc1=0; max_kp1 = 0;max_acc2=0; max_kp2 = 0
    max_acc3=0; max_kp3 = 0;max_acc4=0; max_kp4 = 0;max_acc5=0; max_kp5 = 0
    
    newest_path='0.0.0.0.0.0'
    best_JD_model=[]
    for idx in range(num_classes): 
        best_JD_model.append('0.0.0.0.0.0')
    best_loss_model='0.0.0.0.0.0'
    best_kp_model='0.0.0.0.0.0'
    min_v_loss=1.0
    max_auc=0
    step_np=np.zeros((epochs,7),dtype=np.float32)
    tp_lbidx1 = 7
    tp_lbidx2 = 8

    tp_lbidx3 = 6
    
    tp_lbidx4 = 4
    tp_lbidx5 = 5

#    sys.exit(0)
    
    for epoch in range(epo_start, epochs + 1):

        model_class.train()
        N_proceed_imgs=1e-10
        train_loss = 0
        tq = tqdm(train_data_loader, desc='loss', leave=True, ncols=100)
        batch_idx=0
        for bx_ori, by, w_tensor, by_mtlb, h_flip in tq:
#            break
         
            batch_idx=batch_idx+1
            b_siz=bx_ori.shape[0]
            
            by_mtlb_oh1 = torch.zeros(b_siz,num_classes).scatter_(1,by_mtlb[:,tp_lbidx1:tp_lbidx1+1]-1,1)
            by_mtlb_oh2 = torch.zeros(b_siz,num_classes).scatter_(1,by_mtlb[:,tp_lbidx2:tp_lbidx2+1]-1,1)
            by_mtlb_oh3 = torch.zeros(b_siz,num_classes).scatter_(1,by_mtlb[:,tp_lbidx3:tp_lbidx3+1]-1,1)
            
            
            tp_lb4 = by_mtlb[:,tp_lbidx4:tp_lbidx4+1] - 1
            tp_lb4_w_flip = tp_lb4^h_flip.unsqueeze(-1).long()
            tp_lb4_w_flip[tp_lb4_w_flip==3]=2
            
            by_mtlb_oh4 = torch.zeros(b_siz,num_classes).scatter_(1,tp_lb4_w_flip,1)            
            by_mtlb_oh5 = torch.zeros(b_siz,num_classes).scatter_(1,by_mtlb[:,tp_lbidx5:tp_lbidx5+1]-1,1)
            
#            sys.exit(0)
            
            bx = bx_ori#torch.cat((bx_eh,bx_ori),1)
            
            b_x=bx.cuda()#.cuda(gpu_device_ID)
            b_y=by.cuda()#.cuda(gpu_device_ID)
            
            b_y_mtlb_oh1=by_mtlb_oh1.cuda()
            b_y_mtlb_oh2=by_mtlb_oh2.cuda()
            b_y_mtlb_oh3=by_mtlb_oh3.cuda()
            b_y_mtlb_oh4=by_mtlb_oh4.cuda()            
            b_y_mtlb_oh5=by_mtlb_oh5.cuda() 
            
            w_tensor = w_tensor.cuda()
                   
                
            b_seg_out,b_loc_out1, b_loc_out2, b_loc_out3, b_loc_out4, b_loc_out5 = model_class(b_x) 
            b_seg_out_sg = F.sigmoid(b_seg_out)
            
            b_loc_out_sm1 = F.softmax(b_loc_out1,1)
            b_loc_out_sm2 = F.softmax(b_loc_out2,1)
            b_loc_out_sm3 = F.softmax(b_loc_out3,1)
            b_loc_out_sm4 = F.softmax(b_loc_out4,1)
            b_loc_out_sm5 = F.softmax(b_loc_out5,1)            
            
            seg_loss = loss_function(b_y * w_tensor, b_seg_out_sg* w_tensor)*1.0 + \
            Dice_loss(b_seg_out_sg* w_tensor, b_y* w_tensor, if_sq=1, if_bias=1).mean()* 1.0 
            
            loss1 = loss_function_CE(b_loc_out_sm1,b_y_mtlb_oh1)
            loss2 = loss_function_CE(b_loc_out_sm2,b_y_mtlb_oh2)
            
            loss3 = loss_function_CE(b_loc_out_sm3,b_y_mtlb_oh3)
            loss4 = loss_function_CE(b_loc_out_sm4,b_y_mtlb_oh4)
            loss5 = loss_function_CE(b_loc_out_sm5,b_y_mtlb_oh5)
            
            loss = loss1 * 0.5 + loss2 + loss3 + loss4 + loss5 + seg_loss
#            loss = seg_loss
            
            if not(math.isnan(loss.data.item())):            
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.item()*b_siz
                optimizer.step()
                N_proceed_imgs=N_proceed_imgs+b_siz
            else:
                train_loss = (train_loss/(N_proceed_imgs))*(N_proceed_imgs+b_siz)
                N_proceed_imgs=N_proceed_imgs+b_siz
                del b_x, b_y, loss, b_seg_out
                torch.cuda.empty_cache()
                continue
               
#            break
            tq.set_description("epoch %i loss %f" % (epoch, loss.data.item()))
#            tq.set_description("epoch %i" % (epoch))
            tq.refresh() # to show immediately the update
        tq.set_description("epoch %i loss %f" % (epoch, train_loss/(N_proceed_imgs)))
        tq.close()
        train_ave_loss=train_loss / total_N_img
     
        model_class.eval()            
        total_v_loss=0
        N_proceed_v_img=0
    

        total_v_JD=np.zeros((total_N_v_img, seg_classes),dtype=np.float32)
        total_v_AUPR=np.zeros((total_N_v_img, seg_classes),dtype=np.float32)
        
        valid_tq = tqdm(valid_data_loader, desc='valid loss', leave=True, ncols=100)
        v_idx=0
        
        t_acc1 = np.zeros((total_N_v_img,2),dtype=np.float32)
        t_acc1[:,1]=-1
        pred_all1 = np.zeros((total_N_v_img,3),dtype=np.float32)
        
        t_acc2 = np.zeros((total_N_v_img,2),dtype=np.float32)
        t_acc2[:,1]=-1
        pred_all2 = np.zeros((total_N_v_img,3),dtype=np.float32)
        
        t_acc3 = np.zeros((total_N_v_img,2),dtype=np.float32)
        t_acc3[:,1]=-1
        pred_all3 = np.zeros((total_N_v_img,3),dtype=np.float32)        
        
        t_acc4 = np.zeros((total_N_v_img,2),dtype=np.float32)
        t_acc4[:,1]=-1
        pred_all4 = np.zeros((total_N_v_img,3),dtype=np.float32)        

        t_acc5 = np.zeros((total_N_v_img,2),dtype=np.float32)
        t_acc5[:,1]=-1
        pred_all5 = np.zeros((total_N_v_img,3),dtype=np.float32)
        
        for vx_ori,vy, w_tensor, vy_mtlb, vh_flip in valid_tq:
            
            vb_siz=vx_ori.shape[0]
            vy_mtlb_oh1 = torch.zeros(vb_siz,num_classes).scatter_(1,vy_mtlb[:,tp_lbidx1:tp_lbidx1+1]-1,1)
            vy_mtlb_oh2 = torch.zeros(vb_siz,num_classes).scatter_(1,vy_mtlb[:,tp_lbidx2:tp_lbidx2+1]-1,1)
            vy_mtlb_oh3 = torch.zeros(vb_siz,num_classes).scatter_(1,vy_mtlb[:,tp_lbidx3:tp_lbidx3+1]-1,1)
            vy_mtlb_oh4 = torch.zeros(vb_siz,num_classes).scatter_(1,vy_mtlb[:,tp_lbidx4:tp_lbidx4+1]-1,1)
            vy_mtlb_oh5 = torch.zeros(vb_siz,num_classes).scatter_(1,vy_mtlb[:,tp_lbidx5:tp_lbidx5+1]-1,1)
            
            vx=vx_ori#torch.cat((vx_eh,vx_ori),1)            
            v_x=vx[:,:,:,:].cuda()#.cuda(gpu_device_ID)
            v_y=vy[:,:,:,:].cuda()#.cuda(gpu_device_ID)   
            v_y_mtlb_oh1 = vy_mtlb_oh1.cuda()
            v_y_mtlb_oh2 = vy_mtlb_oh2.cuda()
            v_y_mtlb_oh3 = vy_mtlb_oh3.cuda()
            v_y_mtlb_oh4 = vy_mtlb_oh4.cuda()
            v_y_mtlb_oh5 = vy_mtlb_oh5.cuda()
            
            
            w_tensor = w_tensor.cuda()
        
            
            with torch.no_grad():
                v_seg_out, v_loc_out1, v_loc_out2, v_loc_out3, v_loc_out4, v_loc_out5 = model_class(v_x)
                v_seg_out_sg = F.sigmoid(v_seg_out)
                v_loc_out_sm1 = F.softmax(v_loc_out1, 1)
                v_loc_out_sm2 = F.softmax(v_loc_out2, 1)
                v_loc_out_sm3 = F.softmax(v_loc_out3, 1)
                v_loc_out_sm4 = F.softmax(v_loc_out4, 1)
                v_loc_out_sm5 = F.softmax(v_loc_out5, 1)
                
                
                
                v_seg_loss = loss_function(v_y , v_seg_out_sg) + \
                Dice_loss(v_seg_out_sg, v_y, if_sq=1, if_bias=1).mean()  
                
                v_loss1 = loss_function_CE(v_loc_out_sm1,v_y_mtlb_oh1)
                v_loss2 = loss_function_CE(v_loc_out_sm2,v_y_mtlb_oh2)
                v_loss3 = loss_function_CE(v_loc_out_sm3,v_y_mtlb_oh3)
                v_loss4 = loss_function_CE(v_loc_out_sm4,v_y_mtlb_oh4)
                v_loss5 = loss_function_CE(v_loc_out_sm5,v_y_mtlb_oh5)
                
                
                v_loss = v_loss1 + v_loss2 + v_loss3 + v_loss4 + v_loss5 + v_seg_loss


            val_JD=cal_JD((v_seg_out_sg.detach().cpu()).float(), v_y.detach().cpu(),if_norm=1, if_soft_bias=0, if_d_bias=0)
            val_AUPR = cal_AUPR((v_seg_out_sg.detach().cpu()).float(), v_y.detach().cpu())
            total_v_JD[v_idx:v_idx+vb_siz,]=val_JD 
            total_v_AUPR[v_idx:v_idx+vb_siz,]=val_AUPR 
                
            pred_all1[v_idx:v_idx+vb_siz,:] = v_loc_out_sm1.detach().cpu().numpy()
            pred_v1 = v_loc_out_sm1.detach().cpu().numpy().argmax(1)            
            t_acc1[v_idx:v_idx+vb_siz,0] = pred_v1
            t_acc1[v_idx:v_idx+vb_siz,1] = vy_mtlb[:,tp_lbidx1].detach().cpu().numpy()-1

            pred_all2[v_idx:v_idx+vb_siz,:] = v_loc_out_sm2.detach().cpu().numpy()
            pred_v2 = v_loc_out_sm2.detach().cpu().numpy().argmax(1)            
            t_acc2[v_idx:v_idx+vb_siz,0] = pred_v2
            t_acc2[v_idx:v_idx+vb_siz,1] = vy_mtlb[:,tp_lbidx2].detach().cpu().numpy()-1

            pred_all3[v_idx:v_idx+vb_siz,:] = v_loc_out_sm3.detach().cpu().numpy()
            pred_v3 = v_loc_out_sm3.detach().cpu().numpy().argmax(1)            
            t_acc3[v_idx:v_idx+vb_siz,0] = pred_v3
            t_acc3[v_idx:v_idx+vb_siz,1] = vy_mtlb[:,tp_lbidx3].detach().cpu().numpy()-1

            pred_all4[v_idx:v_idx+vb_siz,:] = v_loc_out_sm4.detach().cpu().numpy()
            pred_v4 = v_loc_out_sm4.detach().cpu().numpy().argmax(1)            
            t_acc4[v_idx:v_idx+vb_siz,0] = pred_v4
            t_acc4[v_idx:v_idx+vb_siz,1] = vy_mtlb[:,tp_lbidx4].detach().cpu().numpy()-1

            pred_all5[v_idx:v_idx+vb_siz,:] = v_loc_out_sm5.detach().cpu().numpy()
            pred_v5 = v_loc_out_sm5.detach().cpu().numpy().argmax(1)            
            t_acc5[v_idx:v_idx+vb_siz,0] = pred_v5
            t_acc5[v_idx:v_idx+vb_siz,1] = vy_mtlb[:,tp_lbidx5].detach().cpu().numpy()-1
                
            if not(math.isnan(v_loss.data.item())):            
                total_v_loss += v_loss.data.item()*vb_siz
            else:
                total_v_loss = (total_v_loss/(N_proceed_v_img+1e-10))*(N_proceed_v_img+vb_siz)
                del v_x, v_y, v_loss, v_cl_out, v_cl_out_sm
                torch.cuda.empty_cache()
            v_idx=v_idx+vb_siz
            valid_tq.set_description("epoch %i valid loss %f" % (epoch, v_loss.data.item()))
            valid_tq.refresh()
        v_loss_ave=total_v_loss/(total_N_v_img)
        
        N_acc1=(t_acc1[:,0]==t_acc1[:,1]).sum()
        Kappa1 = quadratic_weighted_kappa(t_acc1[:,0],t_acc1[:,1])
        cfm1 = confusion_matrix(np.int64(t_acc1[:,0]),np.int64(t_acc1[:,1]),0,2)
        acc_v_ave1=N_acc1/total_N_v_img
        
        N_acc2=(t_acc2[:,0]==t_acc2[:,1]).sum()
        Kappa2 = quadratic_weighted_kappa(t_acc2[:,0],t_acc2[:,1])
        cfm2 = confusion_matrix(np.int64(t_acc2[:,0]),np.int64(t_acc2[:,1]),0,2)
        acc_v_ave2=N_acc2/total_N_v_img

        N_acc3=(t_acc3[:,0]==t_acc3[:,1]).sum()
        Kappa3 = quadratic_weighted_kappa(t_acc3[:,0],t_acc3[:,1])
        cfm3 = confusion_matrix(np.int64(t_acc3[:,0]),np.int64(t_acc3[:,1]),0,2)
        acc_v_ave3=N_acc3/total_N_v_img
        
        N_acc4=(t_acc4[:,0]==t_acc4[:,1]).sum()
        Kappa4 = quadratic_weighted_kappa(t_acc4[:,0],t_acc4[:,1])
        cfm4 = confusion_matrix(np.int64(t_acc4[:,0]),np.int64(t_acc4[:,1]),0,2)
        acc_v_ave4=N_acc4/total_N_v_img

        N_acc5=(t_acc5[:,0]==t_acc5[:,1]).sum()
        Kappa5 = quadratic_weighted_kappa(t_acc5[:,0],t_acc5[:,1])
        cfm5 = confusion_matrix(np.int64(t_acc5[:,0]),np.int64(t_acc5[:,1]),0,2)
        acc_v_ave5=N_acc5/total_N_v_img
        
        mean_JD = total_v_JD.mean(0)[:]
        mean_AUPR = (total_v_AUPR * (total_v_AUPR>-1)).sum(0)/(total_v_AUPR>-1).sum(0)
        
        print('\ntrain loss %.4f, valid loss %.4f, valid JD, valid AUPR'% (train_loss/N_proceed_imgs,total_v_loss/total_N_v_img), mean_JD, mean_AUPR)
        
        print('\ntrain loss %.4f, valid loss %.4f,  \
              valid acc1 %.4f, valid kp1 %.4f, \
              valid acc2 %.4f, valid kp2 %.4f,\
              valid acc3 %.4f, valid kp3 %.4f,\
              valid acc4 %.4f, valid kp4 %.4f,\
              valid acc5 %.4f, valid kp5 %.4f,\
              ' \
              % (train_loss/N_proceed_imgs,total_v_loss/total_N_v_img, \
                 acc_v_ave1, Kappa1, \
                 acc_v_ave2, Kappa2, \
                 acc_v_ave3, Kappa3, \
                 acc_v_ave4, Kappa4, \
                 acc_v_ave5, Kappa5, \
                 )
              )
        
        min_v_loss=min(min_v_loss,v_loss_ave)
        max_acc1=max(max_acc1,acc_v_ave1)
        max_kp1=max(max_kp1,Kappa1)
        max_acc2=max(max_acc2,acc_v_ave2)
        max_kp2=max(max_kp2,Kappa2)
        max_acc3=max(max_acc3,acc_v_ave3)
        max_kp3=max(max_kp3,Kappa3)
        max_acc4=max(max_acc4,acc_v_ave4)
        max_kp4=max(max_kp4,Kappa4)
        max_acc5=max(max_acc5,acc_v_ave5)
        max_kp5=max(max_kp5,Kappa3)


        step_np[epoch-1,0]=epoch
        step_np[epoch-1,1]=train_ave_loss
        step_np[epoch-1,2]=v_loss_ave
        step_np[epoch-1,3]=acc_v_ave1
        step_np[epoch-1,4]=Kappa1
        step_np[epoch-1,5]=acc_v_ave2
        step_np[epoch-1,6]=Kappa2

        
        print('max acc1 %.4f, max kp1 %.4f,max acc2 %.4f, max kp2 %.4f'% (max_acc1, max_kp1,max_acc2, max_kp2))
        print('max acc3 %.4f, max kp3 %.4f,max acc4 %.4f, max acc5 %.4f'% (max_acc3, max_kp3,max_acc4, max_acc5))     
        
#        sys.exit(0)
        savestr=exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'-epo-'+auto_number(epoch,4) +'-t_l-'+'{:.4f}'.format(train_ave_loss)\
        +'-v_loss-'+'{:.4f}'.format(v_loss_ave)\
        +'-v_JD'
        for idx in range(len(mean_JD)):
            max_JD[idx] = max(mean_JD[idx],max_JD[idx])    
            savestr = savestr + '-{:.4f}'.format(mean_JD[idx])
        savestr = savestr + '.pkl'
        
        np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_valid_im.txt', np.array(valid_lb_list),fmt='%s') 

#        if epoch!=epo_start-1:
        np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'.txt', step_np)               
        if acc_v_ave1==max_acc1:
            np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_t_acc1.txt', t_acc1)
            torch.save(model_class.module.state_dict(), exp_name+'//'+exp_name+'_max_acc1.pkl')
        if acc_v_ave2==max_acc2:
            np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_t_acc2.txt', t_acc2)
            torch.save(model_class.module.state_dict(), exp_name+'//'+exp_name+'_max_acc2.pkl')
        if acc_v_ave3==max_acc3:
            np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_t_acc3.txt', t_acc3)
            torch.save(model_class.module.state_dict(), exp_name+'//'+exp_name+'_max_acc3.pkl')
        if acc_v_ave4==max_acc4:
            np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_t_acc4.txt', t_acc4)
            torch.save(model_class.module.state_dict(), exp_name+'//'+exp_name+'_max_acc4.pkl')
        if acc_v_ave5==max_acc5:
            np.savetxt(exp_name+'//'+exp_name+'_'+str(valid_fold_idx)+'-'+str(total_folds)+'_'+exp_times+'_t_acc5.txt', t_acc5)
            torch.save(model_class.module.state_dict(), exp_name+'//'+exp_name+'_max_acc5.pkl')            

        for idx in range(0,len(mean_JD)):
            if mean_JD[idx]==max_JD[idx]:
                if os.path.exists(best_JD_model[idx]):        
                    os.remove(best_JD_model[idx])
                torch.save(model_class.module.state_dict(), exp_name+'//max_JD'+str(idx)+'-'+savestr)
                best_JD_model[idx]=exp_name+'//max_JD'+str(idx)+'-'+savestr

    