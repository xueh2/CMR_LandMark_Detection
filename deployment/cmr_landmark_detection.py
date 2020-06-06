
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import *
import numpy as np
import scipy
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#from PIL import Image
import imp
import os
import sys
import math
import time
import random
import shutil
import scipy.misc
from scipy.io import loadmat, savemat
from glob import glob
#import sklearn
import logging

from adaptive_thresh import get_landmark_from_prob
from adaptive_thresh import cpad_2d

def load_model_cmr_landmark_detection(model_dir, model_file):

    m = None

    try:
        curr_dir = os.getcwd()
        os.chdir(model_dir)

        print("Load aif model : %s" % os.path.join(model_dir, model_file), file=sys.stderr)
        t0 = time.time()
        m = torch.jit.load(os.path.join(model_dir, model_file))
        t1 = time.time()
        print("Model loading took %f seconds " % (t1-t0), file=sys.stderr)

        os.chdir(curr_dir)

        sys.stderr.flush()
    except Exception as e:
        print("Error happened in load_model_cmr_landmark_detection for %s" % model_file, file=sys.stderr)
        print(e)

    return m

def perform_cmr_landmark_detection(im, model, p_thresh=0.1, oper_RO=352, oper_E1=352):
    """
    Perform CMR landmark detection

    Input :
    im : [RO, E1],image
    model : loaded model
    p_thres: if max(prob)<p_thres, then no landmark is found
    oper_RO, oper_E1: expected array size of model. Image will be padded.

    Output:
    pts: [N, 2], landmark points, if no landmark, it is -1
    probs: [RO, E1, N], probability of detected landmark points
    """

    RO, E1 = im.shape

    try:
        im_used, s_ro, s_e1 = cpad_2d(im, oper_RO, oper_E1)

        im_used = im_used / np.max(im_used)

        im_used = np.reshape(im_used, (1, 1, oper_RO, oper_E1))

        t0 = time.time()
        images = torch.from_numpy(im_used).float()
        model.eval() 

        t0 = time.time()
        with torch.no_grad():
            scores = model(images)
            probs = torch.softmax(scores, dim=1)
        t1 = time.time()
        print("perform_cmr_landmark_detection, model runs in %.2f seconds " % (t1-t0))
        print(probs.shape)

        probs = probs.numpy()
        probs = probs.astype(np.float32)
        probs = np.squeeze(probs)

        probs = np.transpose(probs, (1,2,0))

        if(s_ro>=0):
            probs = probs[s_ro:s_ro+RO, :, :]
        else:
            probs, s_ro_p, s_e1_p = cpad_2d(probs, RO, probs.shape[1])

        if(s_e1>=0):
            probs = probs[:, s_e1:s_e1+E1, :]
        else:
            probs, s_ro_p, s_e1_p = cpad_2d(probs, probs.shape[0], E1)

        N = probs.shape[2]-1

        pts = np.zeros((N,2))-1.0

        for p in range(N):
            prob = probs[:,:,p+1]
            pt = get_landmark_from_prob(prob, thres=p_thresh, mode="mean", binary_mask=False)
            if(pt is not None):
                pts[p, 0] = pt[0]
                pts[p, 1] = pt[1]

        probs = probs.astype(np.float32)
        pts = pts.astype(np.float32)

        sys.stderr.flush()

    except Exception as e:
        print("Error happened in perform_cmr_landmark_detection ...", file=sys.stderr)
        print(e)
        probs = np.zeros((RO,E1,3), dtype=np.float32)
        pts = np.zeros((3,2), dtype=np.float32) - 1.0
        pts = pts.astype(np.float32)
        sys.stderr.flush()

    return pts, probs

if __name__ == "__main__":

    GT_HOME = os.environ['GADGETRON_HOME']
    model_dir = os.path.join(GT_HOME, 'share/gadgetron/python')
    print("GT_HOME is", GT_HOME)

    GT_CMR_ML_UT_HOME = os.environ['GT_CMR_ML_UNITTEST_DIRECTORY']
    print("GT_CMR_ML_UT_HOME is", GT_CMR_ML_UT_HOME)

    # Unit Test
    print("=================================================================")
    print("Test RetroCine, CH4")

    data_file = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'cmr_landmark_detection', 'RetroCine', 'CH4', '20180104_1462193_ch4_ED.npy')
    print("load - ", data_file)
    im = np.load(data_file)
    print(im.shape)

    model = load_model_cmr_landmark_detection(model_dir, 'CMR_landmark_network_ch2_ch3_ch4_myo_pts_LossMultiSoftProb_KLD_Dice_CMR_View__Pytorch_1.5.0_2020-06-06_20200606_034214.pts')
    print(model)

    pts, probs = perform_cmr_landmark_detection(im, model, p_thresh=0.1)
    print('PTs ', pts.shape)
    print('Probs ', probs.shape)

    res_dir = os.path.join(GT_CMR_ML_UT_HOME, 'result', 'cmr_landmark_detection')
    if(os.path.isdir(res_dir)==False):
        os.mkdir(res_dir)

    res_file = os.path.join(res_dir, 'RetronCine_CH4_ED_pts.npy')
    print("save - ", res_file)
    np.save(res_file, pts)

    res_file = os.path.join(res_dir, 'RetronCine_CH4_ED_probs.npy')
    print("save - ", res_file)
    np.save(res_file, probs)

    print("=================================================================")
    print("Test RetroCine, CH2")

    data_file = os.path.join(GT_CMR_ML_UT_HOME, 'data', 'cmr_landmark_detection', 'RetroCine', 'CH2', '20180323_425511678_ch2_ES.npy')
    print("load - ", data_file)
    im = np.load(data_file)
    print(im.shape)

    pts, probs = perform_cmr_landmark_detection(im, model, p_thresh=0.1)
    print('PTs ', pts.shape)
    print('Probs ', probs.shape)

    res_dir = os.path.join(GT_CMR_ML_UT_HOME, 'result', 'cmr_landmark_detection')
    if(os.path.isdir(res_dir)==False):
        os.mkdir(res_dir)

    res_file = os.path.join(res_dir, 'RetronCine_CH2_ES_pts.npy')
    print("save - ", res_file)
    np.save(res_file, pts)

    res_file = os.path.join(res_dir, 'RetronCine_CH2_ES_probs.npy')
    print("save - ", res_file)
    np.save(res_file, probs)
