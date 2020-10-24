#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 01:43:52 2020

@author: meti
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

import random
from matplotlib import pyplot



def FPRat95TPR(D1, D2):
    tot = 0
    fpr = 0
    separation_range = np.linspace(np.min(D1), np.max(D2), num=1000)

    for sep in separation_range:
        #get tpr and fpr
        tpr = np.sum(np.sum(D1 >= sep)) / len(D1)
        fpr = np.sum(np.sum(D2 > sep)) / len(D2)
        
        if tpr <= 0.9550 and tpr >= 0.9450:      
            return fpr


def auroc(D1, D2):
    #calculate the AUROC
    auroc_score = 0
    prev_fpr= 1
    separation_range = np.linspace(np.min(D1), np.max(D2), num=1000)
    
    for sep in separation_range:
        #get tpr and fpr
        tpr = np.sum(np.sum(D1 >= sep)) / len(D1)
        fpr = np.sum(np.sum(D2 > sep)) / len(D2)
        
        auroc_score += (-fpr+prev_fpr)*tpr
        prev_fpr = fpr
    auroc_score += fpr * tpr

    return auroc_score

def plot_dist(x,y):
    #x=x.detach().numpy()
    #y=y.detach().numpy()
    
    #cooment lines below for ood and ind
    x = [random.gauss(4,2) for _ in range(400)]   
 
    bins = np.linspace(-10, 10, 100)
    
    pyplot.title("OOD score distributions")
    pyplot.hist(x, bins, alpha=0.5, label='ind data')
    pyplot.hist(y, bins, alpha=0.5, label='ood data')
    pyplot.legend(loc='upper right')
    pyplot.show()


def resultss(ood_sc_in, ood_sc_ood, ood_name, net_name = "DenseNet"):
        
        indis = "cifar-10"
        tpr95=FPRat95TPR(ood_sc_in,ood_sc_ood)
        auroc_score=auroc(ood_sc_in,ood_sc_ood)
        
        #plot table
        print('Results')
        print("{:20}{:>13}".format("Net:", net_name))
        print("{:20}{:>13}".format("ind data:", indis))
        print("{:20}{:>13}".format("ood data:", ood_name))
        print("{:20}{:13.1f} ".format("FPR at TPR 95%:",tpr95*100))
        print("{:20}{:13.1f} ".format("auroc:",auroc_score*100))
        
        #plot distribution plot
        plot_dist(ood_sc_in,ood_sc_ood)
        
        