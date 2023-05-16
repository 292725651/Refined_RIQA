#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:58:59 2020

@author: gtj
"""

import sys


import numpy as np
from my_imgprocess_lib import *
import sklearn.metrics as metrics
import torch
#torch.manual_seed(1)
#pred=torch.rand(8,1, 200, 100)
#pred_np=pred.numpy()
#
#truth = (torch.rand(8,1, 200, 100)>0.5).float()
#truth_np = truth.numpy()

def cal_JD(pred, truth, if_norm=0, if_soft_bias=0, if_d_bias=1):
    
    assert pred.shape==truth.shape
    assert len(pred.shape)==4
    
    pred=torch.Tensor(pred)
    truth=torch.Tensor(truth)
    
    bsz = pred.shape[0]
    classes = pred.shape[1]
    r=pred.shape[2]
    c=pred.shape[3]
    
    pred_flat=pred.reshape([bsz,classes,r*c])
    truth_flat=truth.reshape([bsz,classes,r*c])
    if if_norm:
        pred_flat_norm = (pred_flat-pred_flat.min(-1)[0].unsqueeze(-1)+1e-16)/\
        ((pred_flat-pred_flat.min(-1)[0].unsqueeze(-1)).max(-1)[0].unsqueeze(-1)+1e-16)
    else:
        pred_flat_norm = pred_flat
#    (lb_I*pred_I_norm).sum()/(lb_I+pred_I_norm-lb_I*pred_I_norm).sum()
    JD=((truth_flat*pred_flat_norm).sum(-1)+if_d_bias)/((truth_flat+pred_flat_norm-truth_flat*pred_flat_norm).sum(-1)+if_d_bias)
    if if_soft_bias:
        JD_soft_bias=((truth_flat*truth_flat).sum(-1)+if_d_bias)/((truth_flat+truth_flat-truth_flat*truth_flat).sum(-1)+if_d_bias)
    else:
        JD_soft_bias=JD*0+1.0
    
    return (JD/JD_soft_bias).numpy()
    

def cal_AUC(pred, truth):
    
    assert pred.shape==truth.shape
    assert len(pred.shape)==4
    
    pred=torch.Tensor(pred)
    truth=(torch.Tensor(truth)>0.5).int()
    
    bsz = pred.shape[0]
    classes = pred.shape[1]
    r=pred.shape[2]
    c=pred.shape[3]
    
    pred_flat=(pred.reshape([bsz,classes,r*c])).numpy()
    truth_flat=(truth.reshape([bsz,classes,r*c])).numpy()
    
    rt_AUC = np.zeros((bsz, classes),dtype=np.float32)
    rt_AUPR = np.zeros((bsz, classes),dtype=np.float32)
    
    
    for bb in range(bsz):
        for cc in range(classes):
            if (truth_flat[bb,cc,].max()==0) or (truth_flat[bb,cc,].min()==1):
                rt_AUC[bb,cc] = -1
                rt_AUPR[bb,cc] = -1
            else:
                rt_AUC[bb,cc] = metrics.roc_auc_score(truth_flat[bb,cc,], pred_flat[bb,cc,])
                precision, recall, _thresholds = metrics.precision_recall_curve(truth_flat[bb,cc,], np.round(pred_flat[bb,cc,]*32)/32)
                rt_AUPR[bb,cc] = metrics.auc(recall, precision)
            
    return rt_AUC, rt_AUPR


def cal_AUPR(pred, truth):
    
    assert pred.shape==truth.shape
    assert len(pred.shape)==4
    
    pred=torch.Tensor(pred)
    truth=(torch.Tensor(truth)>0.5).int()
    
    bsz = pred.shape[0]
    classes = pred.shape[1]
    r=pred.shape[2]
    c=pred.shape[3]
    
    pred_flat=pred.reshape([bsz,classes,r*c])
    truth_flat=truth.reshape([bsz,classes,r*c])
    
    rt_AUPR = np.zeros((bsz, classes),dtype=np.float32)
    
    
    for bb in range(bsz):
        for cc in range(classes):
            if (truth_flat[bb,cc,].max()==0) or (truth_flat[bb,cc,].min()==1):
                rt_AUPR[bb,cc] = -1
            else:
                precision, recall, _thresholds = metrics.precision_recall_curve(truth_flat[bb,cc,], np.round(pred_flat[bb,cc,]*32)/32)
                rt_AUPR[bb,cc] = metrics.auc(recall, precision)
            
    return rt_AUPR


def Dice_loss(pred, truth, if_sq=0, if_bias=1):
    assert pred.shape==truth.shape
    assert len(pred.shape)==4
    
    p = pred
    g = truth
    
    if if_sq:
        Dice = (2*(p*g).sum(-1).sum(-1)+if_bias)/((p*p+g*g).sum(-1).sum(-1)+if_bias)
    else:
        Dice = (2*(p*g).sum(-1).sum(-1)+if_bias)/((p+g).sum(-1).sum(-1)+if_bias)
    loss = 1 - Dice     
    return loss    
#    (lb_I*pred_I_norm).sum()/(lb_I+pred_I_norm-lb_I*pred_I_norm).sum()
#    JD=((truth_flat*pred_flat_norm).sum(-1)+1)/((truth_flat+pred_flat_norm-truth_flat*pred_flat_norm).sum(-1)+1)
#    if if_soft_bias:
#        JD_soft_bias=((truth_flat*truth_flat).sum(-1)+1)/((truth_flat+truth_flat-truth_flat*truth_flat).sum(-1)+1)
#    else:
#        JD_soft_bias=JD*0+1.0
    
#    return JD/JD_soft_bias