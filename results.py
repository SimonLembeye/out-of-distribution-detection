#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 01:43:52 2020

@author: meti
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def resultss(ood_sc_in, ood_sc_ood, ood_name):
        
        ood_scores = np.append(ood_sc_in,ood_sc_ood)
        labels_in = np.zeros(len(ood_sc_in))
        labels_ood = np.ones(len(ood_sc_ood))
        labels = np.append(labels_in,labels_ood)
        
        #testing with roc_auc (0.5 is bad 1 is good)
        res = roc_auc_score(labels, ood_scores)
        test = 'roc_auc_score'
        print(test + ' on ' + ood_name + ' is: ' +  str(res))
        
      

       