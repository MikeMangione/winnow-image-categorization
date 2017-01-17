#!/usr/bin/env python

import cv2
import glob
import numpy as np
from os import path
import skimage
import skimage.segmentation as segmentation
import matplotlib.pyplot as plt
from hou_saliency import Saliency
from itti_saliency import pySaliencyMap
import manifold_ranking_saliency
import time
import imageio

start = time.clock()
cv_img = []
#collect training images
for img in glob.glob("data/waves/*.jpg"):
    cv_img.append(cv2.imread(img))
for img in glob.glob("data/waves/*.png"):
    cv_img.append(cv2.imread(img))
for img in glob.glob("data/waves/*.jpeg"):
    cv_img.append(cv2.imread(img))
count = 0
models = ['hou','itti','mani']
sals = [[] for _ in models]
crops = [[] for _ in models]
for img in cv_img:
    imgsize  = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    #compute saliencies with the various models
    sals[0] = Saliency(img, use_numpy_fft=True, gauss_kernel=(3, 3)).get_proto_objects_map()
    sals[1] = pySaliencyMap(img_width, img_height).SMGetBinarizedSM(img)
    sals[2] = manifold_ranking_saliency.MR_saliency().saliency(img).astype(np.uint8)*255
    #resize the saliencies and write them to file
    for i in range(0,len(models)):
        blank_image = cv2.resize(sals[i],(100,100))
        name = "outs/"+models[i]+"_waves_out/test%d.png" % (count)
        cv2.imwrite(name,blank_image)
        print("Finished at time ",time.clock() - start)
    count+=1
