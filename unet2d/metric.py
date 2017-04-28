'''
Created on Apr 28, 2017

@author: xiagai
'''
import numpy as np

def jc(pred, label):
    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))
    
    intersection = np.count_nonzero(pred & label)
    union = np.count_nonzero(pred | label)
    if union == 0:
        jc = 1
    else:
        jc = float(intersection) / float(union)
    return jc