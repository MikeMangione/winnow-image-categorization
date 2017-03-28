#!/usr/bin/env python
import cv2
import glob
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
import hou_saliency
import itti_saliency
import manifold_ranking_saliency
import time
import scipy.stats as stats

#collects precomputed training saliencies
def getModels():
    sals = [[[] for i in categories] for i in models]
    for i in range(0,len(models)):
        for j in range(0,len(categories)):
            for img in glob.glob("outs/"+models[i]+"_"+categories[j]+"_out/*.png"):
                sals[i][j].append(cv2.imread(img, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    return sals

#calculates saliencies for a test image
def getSals(img):
    sals = []
    sals.append(hou_saliency.Saliency(img, use_numpy_fft = True, gauss_kernel = (3, 3)).get_proto_objects_map())

    sm = itti_saliency.pySaliencyMap(img.shape[0],img.shape[1])
    sals.append(sm.SMGetBinarizedSM(img))

    sals.append(manifold_ranking_saliency.MR_saliency().saliency(img).astype(np.uint8))

    for i in range(0,len(sals)):
        sals[i] = cv2.resize(sals[i], (100, 100))
    return sals

#winnow2 for a set of results
def reg_winnow(results):
    summation = 0
    for w in range(0,len(results)):
        summation += results[w]
    if summation > len(results)/2:
        m_predictor = 1
    else:
        m_predictor = 0
    if results[len(results)-1] != m_predictor:
        if m_predictor is 0:
            for w in range(0,len(results)):
                results[w] = 2 * results[w]
                #pow(4,ci[w])
        else:
            for w in range(0,len(results)):
                results[w] = results[w] / 2
                #pow(8,ci[w])
    print "learned guess accuracy ",sum(results)*1.0/len(results)

#confidence level winnow
def ci_winnow(results,ci):
    model_predictor = [0 for _ in models]
    for w in range(0,len(results)):
        summation = 0
        for w_res in results[w]:
            summation += w_res
        if summation > len(results[w])/2:
            model_predictor[w] = 1
        else:
            model_predictor[w] = 0
        if results[w][len(results[w])-1] != model_predictor[w]:
            if model_predictor[w] is 0:
                for w_ in range(0,len(results[w])):
                    results[w][w_] = pow(4,ci[w]) * results[w][w_]
                    #pow(4,ci[w])
            else:
                for w_ in range(0,len(results[w])):
                    results[w][w_] = results[w][w_] / pow(8,ci[w])
                    #pow(8,ci[w])
    #print(model_predictor)
    return model_predictor

#computes the confidence level for the minimum of a set
def arrayToCI(c_,ci):
    mean = np.mean(ci)
    std = np.std(ci)
    z = abs((min(c_) - mean)/std)
    return (stats.norm.cdf(z) - 0.5) * 2

#prints the results of the winnow algorithms, in a pretty way
def prettyPrintResults():
    for i in range(0,len(w_results)):
        print models[i]
        for j in range(0,len(w_results[i])):
            if len(w_results[i][j]) > 0:
                print categories[j], np.round(sum(w_results[i][j])/len(w_results[i][j]),2)
            else:
                print categories[j], 0
    for i in range(0,len(a_results)):
        print models[i]," average"
        for j in range(0,len(a_results[i])):
            if len(a_results[i][j]) > 0:
                print categories[j], np.round(sum(a_results[i][j],2)/len(a_results[i][j]),2)
            else:
                print categories[j], 0

#start of execution
start = time.clock()

INFINITY = 999999999
models = ["hou","itti","mani"]
categories = ['waves','cat','car']

test_images = []
for img in glob.glob("test_images/*.jpeg"):
    test_images.append(img)
print "Images collected in ", time.clock() - start," seconds"
#manually labeled inputs
test_categories = ['cat','cat','waves','car','car',
                    'cat','cat','cat','waves','waves',
                    'waves','waves','waves','waves','waves',
                    'car','waves','waves','waves','waves',
                    'waves','waves','waves','waves','waves',
                    'cat','car','cat','cat','cat',
                    'car','car','car','car','car',
                    'car','car','car','car','cat',
                    'cat','cat','cat','cat','cat',
                    'cat','cat','car','car','car',
                    'waves','waves','waves','waves','waves','waves',
                    'waves','waves','waves','waves','waves',
                    'waves','waves','waves','waves','waves',
                    'waves','waves','waves','waves','waves',
                    'cat','cat','cat','cat','cat',
                    'cat','cat','cat','cat','cat',
                    'cat','cat','cat','cat',
                    'cat','cat','cat','cat','cat',
                    'car','car','car','car','car',
                    'car','car','car','car','car',
                    'car','car','car','car','car',
                    'car','car','car','car','car']

#randomization of input data
rando = []
for i in range(0,len(test_categories)):
    i_ = i
    rando.append(i_)
np.random.shuffle(rando)
new_test_images = []
new_test_categories = []
for i in rando:
    new_test_images.append(test_images[i])
    new_test_categories.append(test_categories[i])
test_images = new_test_images
test_categories = new_test_categories


start = time.clock()

w_results = [[[] for _ in range(0,len(categories))] for __ in range(0,len(models))]
a_results = [[[] for _ in range(0,len(categories))] for __ in range(0,len(models))]
w_predictor = [[[] for _ in range(0,len(categories))] for __ in range(0,len(models))]
a_predictor = [[[] for _ in range(0,len(categories))] for __ in range(0,len(models))]
winner = []
#collect models
sals = getModels()

print "Training saliencies collected in ", time.clock() - start," seconds"

for x in range(0,len(test_images)):
    start = time.clock()
    #print "/\/\/\/\/\/"
    #compute each saliency model for the image
    testing_sals = getSals(cv2.imread(test_images[x]))
    #print("Saliency Computed at time ", time.clock() - start)

    #print "#-------#"
    ci_w = [[0 for _ in range(0,len(categories))] for __ in range(0,len(models))]
    ci_a = [[0 for _ in range(0,len(categories))] for __ in range(0,len(models))]
    weight_ = [[INFINITY for _ in range(0,len(categories))] for __ in range(0,len(models))]
    avg_ = [[0 for _ in range(0,len(categories))] for __ in range(0,len(models))]
    for i in range(0,len(models)):
        temp_ = [0 for _ in range(0,len(categories))]
        for j in range(0,len(categories)):
            for sali in sals[i][j]:
                temp_[j] = np.linalg.norm(sali - testing_sals[i],ord=1)
                if (weight_[i][j] > temp_[j]):
                    weight_[i][j] = temp_[j]
                avg_[i][j] += temp_[j]
            temp_[j] = 0
            avg_[i][j] = avg_[i][j]/len(sals[i][j])
            ci_w[i][j] = weight_[i][j]
            ci_a[i][j] = avg_[i][j]

        ci_w[i] = arrayToCI(weight_[i],ci_w[i])
        #print models[i]," weight ci: ",ci_w[i]

        ci_a[i] = arrayToCI(avg_[i],ci_a[i])
        #print models[i]," avg ci: ",ci_a[i]

    predictions = [0 for __ in range(0,2*len(models))]
    pre_per = [0 for __ in range(0,2*len(models))]

    for i in range(0,len(models)):
        predictions[i] = weight_[i].index(min(weight_[i]))
        predictions[i+len(models)] = avg_[i].index(min(avg_[i]))
        if len(w_results[i][predictions[i]]) != 0:
            pre_per[i] = sum(np.round(w_results[i][predictions[i]],2))/len(w_results[i][predictions[i]])
        else:
            pre_per[i] = 0
        if len(a_results[i][predictions[i]]) != 0:
            pre_per[i+len(models)] = sum(np.round(a_results[i][predictions[i]],2))/len(a_results[i][predictions[i]])
        else:
            pre_per[i+len(models)] = 0

    winner_temp = pre_per.index(max(pre_per))%len(models)
    if(categories[winner_temp] == test_categories[x]):
        winner.append(1)
    else:
        winner.append(0)

    for i in range(0,len(models)):
        if categories[weight_[i].index(min(weight_[i]))] == test_categories[x]:
            w_results[i][categories.index(test_categories[x])].append(1)
        else:
            w_results[i][categories.index(test_categories[x])].append(0)
            w_results[i][weight_[i].index(min(weight_[i]))].append(0)

        if categories[avg_[i].index(min(avg_[i]))] == test_categories[x]:
            a_results[i][categories.index(test_categories[x])].append(1)
        else:
            a_results[i][categories.index(test_categories[x])].append(0)
            w_results[i][avg_[i].index(min(avg_[i]))].append(0)

    w_pre = ci_winnow(np.asarray(w_results).T[categories.index(test_categories[x])],ci_w)
    for i in range(0,len(w_pre)):
        w_predictor[i][categories.index(test_categories[x])].append(w_pre[i])

    a_pre = ci_winnow(np.asarray(a_results).T[categories.index(test_categories[x])],ci_a)
    for i in range(0,len(a_pre)):
        a_predictor[i][categories.index(test_categories[x])].append(a_pre[i])

    reg_winnow(winner)
    #print "Classified and evaluated in ", time.clock() - start," seconds"
    prettyPrintResults()
