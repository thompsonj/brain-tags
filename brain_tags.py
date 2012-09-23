#!/usr/bin/env python
# encoding: utf-8
"""
brain_tags.py

Created by Jessica Thompson in May 2012 for COSC 174 - Machine Learning and Statistical Data Analysis 

Predict brain activity to music given a vector of tags describing the music that was heard. 
Algorithm adapted from Weston 2011

"""

# these are all external code not written by me
import os, sys
import numpy as np
import bregman 
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
#import matplotlib.pyplot as plt
#from multiprocessing import Process, Queue
import pprocess
from pylab import *
import random
import time


def WARP_opt_weighted3(stim_ind, Xs_train,Ys_train,k,C,D,learning_rate=.2,Xs_test=None,Ys_test=None,stop_pre=None):
    """
    Input:
        stim_ind - unused, 1 based
        Xs - images
        Ys - annotations
        k - for evaluating precision@k
        C - constraint for max norm regularization
        D - dimensionality of joint space
        gamma - learning rate
        Xs_test - test images
        Ys_test - test annotation
        stop_pre - training precision value at which to stop training
        
        Either (Xs_test AND Ys_test) OR stop_pre must be specified 
        if Xs_test and Ys_test are given, stopping condition is based on validation error (test precision)
        if stop_pre is specified, the algorithm will stop iterating over random examples once a training 
        precision of stop_pre has been achieved. This value should be selected via cross validation on 
        the tag prediction task.

    """
    print 'learning rate:', learning_rate
    n = Xs_train.shape[1]
    Y = Ys_train.shape[0] # size of dictionary
    d = Xs_train.shape[0]# dimensionality of images
    if Xs_test!=None and Ys_test !=None:
        select_model=True
    else:
        select_model=False
        pre_test = 0
    stop_train_pre=0
    # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
    W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
    W_best = W
    V= np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
    V_best = V
    if select_model:
        pre_init,dist_init = validate(W,V,Xs_test,Ys_test,k)
        pre_test = [pre_init]
        dist_test = [dist_init]
        print 'initial test precision: ',pre_test[-1]
        stop_pre = .95 # almost 100%, won't increase much more past this value, best test error is surely before we get this
    pre_tr, dist_tr=validate(W,V,Xs_train,Ys_train,k)
    pre_train = [pre_tr]
    dist_train = [dist_tr]
    best_test_dist = None
    print 'initial train precision:',pre_train[-1]
    c=0
    while pre_train[-1]<stop_pre: #improving: #pre_new <.90): or count < 100:
        c+=1
        # pick random labeled example (out of a number less than 600, depending on number of folds in cv)
        i = random.randrange(0,n,1)
        #print i
        x = Xs_train[:,i] # d x 1
        y = Ys_train[:,i] # Y x 1
        # get index of positive labels 
        positive = [po for po in range(0,Y,1) if y[po]>0]
        # pick a random positive label from y
        j = random.choice(positive)
        phi_x = V*x
        phi_y = W[:,j]
        fyi=np.transpose(phi_y)*phi_x
        N=0
        #print 'choosing negative examples until rank(neg) < rank(pos)-1...'
        while N==0 or not(f_hat > (fyi-1)):
            if N>= (Y-1):
                break;
            # instead of just random, pick the most offending negative annotation
            # calculate the rank of all tags whose prob is less than
            #negative_ranks = [np.transpose(W[:,ne])*phi_x for ne in range(0,Y,1) if y[ne]<y[j]]
            #ind = [ne for ne in range(0,Y,1) if y[ne]<y[j]]
            #print negative_ranks
            #negative_ranks = np.matrix(negative_ranks)
            # index of largest rank
            #y_hat_ind = ind[np.argmax(negative_ranks)]
            # pick a random negative annotation 
            negative = [ne for ne in range(0,Y,1) if y[ne]<y[j]]
            y_hat_ind = random.choice(negative)
            phi_y_hat = W[:,y_hat_ind]
            f_hat = np.transpose(phi_y_hat)*phi_x
            #count[i,j,y_hat_ind] +=1
            N+=1
        if f_hat > (fyi-1):
            #print 'Taking a gradient step to maximize precision at k'
            # make a gradient step to minimize error function
            gradient_W = np.zeros(W.shape)
            gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.transpose(-V*x)
            gradient_W[:,y_hat_ind] = L(np.floor((Y-1)/N))*np.transpose(V*x)
            W = W - (learning_rate*gradient_W)
            gradient_V = L(int(np.floor((Y-1)/N)))*(W[:,y_hat_ind]*np.transpose(x) - (W[:,j]*np.transpose(x))) 
            V = V - (learning_rate*gradient_V)
        if not(c%100):
            if select_model:
                #print 'evaluating...'
                pre_tr,dist_tr = validate(W, V, Xs_train, Ys_train,k)
                #pre_train = np.append(pre_train,pre_tr)
                pre_train = np.append(pre_train,pre_tr)
                #dist_train = np.append(dist_train,dist_tr)
                dist_train = np.append(dist_train,dist_tr)
                pre_t,dist_t = validate(W, V, Xs_test, Ys_test,k)
                pre_test = np.append(pre_test,pre_t)
                dist_test = np.append(dist_test,dist_t)
                print 'test precision', pre_test[-1]
                print 'test distance', dist_test[-1]
                if best_test_dist == None or dist_test[-1]<best_test_dist:
                    stop_train_pre = pre_train[-1]
                    best_test_dist = dist_test[-1]
                    W_best = W
                    V_best = V
                print 'best test distance:',best_test_dist
            else:
                pre_train,dist_train = validate(W, V, Xs_train, Ys_train,k)
                W_best = W
                V_best = V
            print 'training precision',pre_train[-1]
    return (stim_ind,W_best,V_best,pre_train,pre_test,dist_train, dist_test,stop_train_pre,best_test_dist)

def plot_error_curves(train, test, title='Euc Distance'):
    #fig1 = plt.figure()
    x= range(len(train))
    for i in x:
        plot(x, train[str(i+1)],'b')
        plot(x, test[str(i+1)],'r-')
    legend(('training dist'), ('testing dist'))
    title(title)
    show()

def WARP_opt(Xs, Ys,k,C,D):
    """
    Input:
        Xs - images
        Ys - annotations
        k - for evaluating precision@k
        C - constraint for max norm regularization
        D - dimensionality of joint space
    """
    n = Xs.shape[1]
    Y = Ys.shape[0] # size of dictionary
    d = Xs.shape[0]# dimensionality of images
    #D = 101 # ? choose empirically 
    # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
    W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
    V = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
    pre_new = validate(W,V,Xs,Ys,k)
    #print 'initial precision: ',pre_new
    c=0
    new_rank = np.zeros((Y,n))
    #count=0
    f_new = 1
    rankunchanged=0
    err = np.zeros((n,3)) # 3rd column indicates whether error is still improving or not
    count = np.zeros(n)
    increased = 0
    decreased = 0
    while (c<=1 or not((err[:,2]==1).all())): #pre_new <.90): or count < 100:
    #while True:
        c+=1
        #count+=1
        # pick random labeled example (out of 600 in this case)
        i = random.randrange(0,n,1)
        #print i
        if err[i,2]==1:
            continue
        count[i] +=1
        x = Xs[:,i] # d x 1
        y = Ys[:,i] # Y x 1
        # get index of positive labels 
        positive = [m for m in range(0,Y,1) if y[m]==1]
        # pick a random positive label from y
        j = random.choice(positive)
        phi_x = V*x
        phi_y = W[:,j]
        fyi=np.transpose(phi_y)*phi_x
        N=0
        #print 'choosing negative examples until rank(neg) < rank(pos)-1...'
        while N==0 or not(f_hat > (fyi-1)):
            if N>= (Y-1):
                break;
            # pick a random negative annotation 
            negative = [ne for ne in range(0,Y,1) if y[ne]==0]
            y_hat_ind = random.choice(negative)
            phi_y_hat = W[:,y_hat_ind]
            f_hat = np.transpose(phi_y_hat)*phi_x
            N+=1
        if f_hat > (fyi-1):
            #print 'Taking a gradient step to maximize precision at k'
            # make a gradient step to minimize error function
            gamma = .2 # learning rate
            
            # enforce positive only
            # gradient_W = np.zeros(W.shape)
            # gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(np.transpose(-V*x).shape),np.transpose(-V*x))
            # gradient_W[:,y_hat_ind] = L(np.floor((Y-1)/N))*np.maximum(np.zeros(np.transpose(V*x).shape),np.transpose(V*x))
            # W = W - (gamma*gradient_W)
            # gradient_V = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(V.shape),(W[:,y_hat_ind]*np.transpose(x) - (W[:,j]*np.transpose(x)) ))
            # V = V - (gamma*gradient_V)
            
            gradient_W = np.zeros(W.shape)
            gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.transpose(-V*x)
            gradient_W[:,y_hat_ind] = L(np.floor((Y-1)/N))*np.transpose(V*x)
            W = W - (gamma*gradient_W)
            gradient_V = L(int(np.floor((Y-1)/N)))*(W[:,y_hat_ind]*np.transpose(x) - (W[:,j]*np.transpose(x))) 
            V = V - (gamma*gradient_V)
            
            # max norm regularization
            # Project weigths to enforce constraints ||V_j||_2 <= C (for j=1...d) and ||W_i||_2 <= C (for i=1...Y)
            for dd in range(d):
                b = C/np.linalg.norm(V[:,dd])
                V[:,dd] = V[:,dd]*b
            for YY in range(Y):
                b=C/np.linalg.norm(W[:,YY])
                W[:,YY] = b*W[:,YY]
                
        # Calculate validation error
        # old_rank = new_rank
        # new_rank = np.transpose(W)*V*Xs
        # if (old_rank == new_rank).all():
        #     rankunchanged+=1
            #print 'rank unchanged: ', rankunchanged
        #print new_rank
        
        if count[i]==1:
            err[i, 0] =  L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)))
        elif count[i]==2:
            err[i,1] = L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)))
            if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
                err[i,2] = 1
                print 'error for sample', i, 'stopped improving'
            #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
            if err[i,1] > err[i,0]:
                increased += 1
            elif err[i,1] < err[i,0]:
                decreased +=1
            print 'error increased',increased,'times'
            print 'error decreased', decreased, 'times'
        else:
            err[i,0] = err[i,1]
            err[i,1] =L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)))
            if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
                err[i,2] = 1
                print 'error for sample', i, 'stopped improving'
            #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
            if err[i,1] > err[i,0]:
                increased += 1
            elif err[i,1] < err[i,0]:
                decreased +=1
            print 'error increased',increased,'times'
            print 'error decreased', decreased, 'times'

        #print 'err: ', err_new
        if not(c%100):
            pre_old = pre_new
            pre_new = validate(W, V, Xs, Ys,k)
            print 'precision: ', pre_new
        #print 'f: ', f_new
    return W,V

def validate(W, V, Xs, Ys, k):
    """
    Use learned mapping matrices W and V to the rank of all tags for each image
    Ys = 100x600
    """
    # print 'in validate'
    ranks = np.transpose(W)*V*Xs
    # each column in rank is for a different image
    #prob = np.array([((rank - rank.min())/rank.max())*100 for rank in np.transpose(ranks)])
    # now each row should be for a different image
    ind = np.argsort(ranks,0)[::-1] # descending - highest rank first
    true_prob_ind = np.argsort(Ys, 0)
    n = Ys.shape[1]
    s = Ys.shape[0]
    dist = np.zeros(n)
    p = np.zeros(n)
    pred_prob = np.zeros(s)
    for i,stim in enumerate(np.transpose(Ys)):
        # evaluate precision at k
        stim_pre = stim > 0 # make binary
        # print Ys.shape
        # print stim_pre.shape
        # print ind.shape
        # print i
        truepos = np.sum(stim_pre[0,ind[0:k,i]])
        p[i] = float(truepos)/float(k)
        for j in range(s):
            # assign probabilities present in Ys to tags for Xs based on rank
            pred_prob[ind[j,i]] = stim[0,true_prob_ind[j,i]]
        dist[i] = np.linalg.norm(stim-pred_prob)
        # calculate distance between actual tag vector and model rank-turned-into-probability
        # preserves more than just binary feature info
        #dist[i] = np.linalg.norm(stim-prob[i,:])
    avg_precision = np.mean(p)
    avg_dist = np.mean(dist)
    return avg_precision, avg_dist

def L(k):
    """
    choose alpha = 1/j (instead of alpha = 1/(Y-1)) because Usunier 2009 showed that it yields state-of-the-art results measuring p@k
    """
    loss = 0
    for j in range(1,int(k)):
        alpha = 1/j
        loss = loss + alpha
    return loss

def learn_25_models(subject, k=5,C=1,D=50,learning_rate=.2,feat_expr='_sts+hg_600',tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    t = time.time()
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
    n=brains.shape[1]
    uni_stim_labels = np.unique(stim_labels)
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words[i-1,:]for i in stim_labels ])) # 600 # Y
    Ws = {}
    Vs = {}
    all_dist_test = {}
    all_dist_train = {}
    all_best_test_dists = np.zeros(25)
    queue = pprocess.Queue(limit=100)
    WARP = queue.manage(pprocess.MakeParallel(WARP_opt_weighted3))
    for stim in uni_stim_labels: # 1 based 
        print stim
        trainwords = words[:,stim_labels!=stim]
        trainbrains = brains[:,stim_labels!=stim]
        testwords = words[:,stim_labels==stim]
        testbrains = brains[:,stim_labels==stim]
        WARP(stim, trainbrains,trainwords,k,C,D,learning_rate,Xs_test=testbrains,Ys_test=testwords)
    #time.sleep(20)
    print 'Finishing...'
    for i, W_best, V_best, pre_train ,pre_test, dist_train, dist_test, stop_train_pre, best_test_dist in queue:
        print 'queue', i
        Ws[str(i)] = W_best
        Vs[str(i)] = V_best
        all_dist_test[str(i)] = dist_test
        all_dist_train[str(i)] = dist_train
        all_pre_test[str(i)] = pre_test
        all_pre_train[str(i)] = pre_train
        all_best_test_dists[i-1] = best_test_dist
    total_time = time.time() - t
    print "Time taken:", total_time
    avg_best_dist = np.mean(all_best_test_dists)
    return Ws, Vs,all_dist_test,all_dist_train,all_pre_test,all_pre_train,total_time,avg_best_dist

def evaluate_tag_prediction(subject,k=5,C=1,D=50,tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    # load brain data
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_sts+hg_600.brain')))
    n = brains.shape[1]
    # load tag data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words100 = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    # perform leave-one-out cross validationl
    # 10-fold cross validation
    nfolds = 8
     # initialize array to store precision values
    pak = np.zeros(nfolds)
    test_precision = {}
    train_precision = {}
    cv_train_pre =[]
    nsamples = n/nfolds
    for l in range(nfolds):
        print 'fold:',l
        start = (l*nsamples)
        end = (start + nsamples)-1
        # divide data into training and test sets
        testbrain = brains[:,start:end]
        trainbrains = np.append(brains[:,0:start], brains[:,end+1:],1)
        testwords = words[:,start:end]
        trainwords = np.append(words[:,0:start], words[:,end+1:],1)
        W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,Xs_test=testbrain,Ys_test=testwords)
        #W,V = WARP_opt(trainbrains, trainwords,k,C,D)
        test_precision[str(l)] = pre_test
        test_precision[str(l)] = pre_test
        cv_train_pre = np.append(cv_train_pre,stop_train_pre)
        pak[l] = best_test_pre
        print 'precision on fold ',l,': ',pak[l]
    avg_precision = np.mean(pak)
    to_save = np.append(avg_precision, pak)
    np.savetxt(subject+'_precision_sts+hg_600_run_'+str(k)+'_'+str(D)+'.txt',to_save)
    print 'stopping train precision:',cv_train_pre
    print 'average stopping train precision:', np.mean(cv_train_pre)
    return avg_precision, test_precision, train_precision,cv_train_pre

def evaluate_brain_prediction3(subject='1mar11sj',stop_pre=.88,k=5,C=1,D=75,feat_expr='_sts+hg_600',tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    # leave one out cross validation
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
    n=brains.shape[1]
    if n==75:
        stim_labels = np.unique(stim_labels)
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words[i-1,:]for i in stim_labels ]))
    dist = squareform(pdist(np.transpose(words)))
    WARP_dist = np.zeros(n)
    KNN_dist = np.zeros(n)
    
    #trained = 0
    # train models, one for each of the 25 stimuli
    Ws,Vs,all_dist_test,all_dist_train,all_pre_test,all_pre_train,total_time,avg_best_dist = learn_25_models(subject)

    # synthesize and evaluate predicted brain activity
    # iterate through each data point
    for i in range(n):
        testtags = words[:,i]
        testbrain = brains[:,i]
        cur_stim = stim_labels[i]
        W_best = Ws[str(cur_stim)]
        V_best = Vs[str(cur_stim)]

        # if str(cur_stim) in Ws:
        #     W_best = Ws[str(cur_stim)]
        #     V_best = Vs[str(cur_stim)]
        # else:
        #     trained+=1
        #     print 'training ',trained, 'time'
        #     trainwords = words[:,stim_labels!=stim_labels[i]]
        #     trainbrains = brains[:,stim_labels!=stim_labels[i]]
        #     testwords = words[:,stim_labels==stim_labels[i]]
        #     testbrains = brains[:,stim_labels==stim_labels[i]]
        #     #W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,stop_pre=stop_pre)
        #     W_best,V_best,pre_train,pre_test,dist_train, dist_test,stop_train_pre,best_test_dist= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,Xs_test=testbrains,Ys_test=testwords)
        #     Ws[str(cur_stim)] = W_best
        #     Vs[str(cur_stim)] = V_best
        #     all_dist_test[str(i)] = dist_test
        #     all_dist_train[str(i)] = dist_train
        brainrank = np.array([(validate(W_best, V_best, np.transpose(brain), testtags, k)) for brain in np.transpose(trainbrains)])
        # indices of brains whose distance (after turning rank into prob) to actual tag vec is shortest
        top_brain_ind = np.argsort(brainrank[:,1])[:k]
        pred_brain = np.mean(trainbrains[:,top_brain_ind],1) # brain activity predicted by WARP loss optimization
        # weighted predicted brain - don't use this when using distances to rank brains!
        #pred_brain = np.sum(trainbrains[:,top_brain_ind]*brainrank[top_brain_ind])/np.sum(brainrank[top_brain_ind])

        # KNN 
        print testtags.shape, trainwords.shape
        top_brain_ind_knn = np.argsort([np.linalg.norm(testtags-np.transpose(tagvec)) for tagvec in np.transpose(trainwords)])[-k:]
        pred_brain_knn = np.mean(trainbrains[:,top_brain_ind_knn],1)
        #print pred_brain_knn.shape

        WARP_dist[i]= np.linalg.norm(pred_brain-testbrain)
        print 'WARP_dist:', WARP_dist
        KNN_dist[i]= np.linalg.norm(pred_brain_knn-testbrain)
        print 'KNN_dist:', KNN_dist
    #print 'Model trained only ',trained, 'times'
    #t,prob = ttest_ind(WARP_dist,KNN_dist)
    t,prob = ttest_rel(WARP_dist,KNN_dist)
    print 't:',t,'prob:',prob
    mnWARP = np.mean(WARP_dist)
    mnKNN = np.mean(KNN_dist)
    print 'mean WARP:',mnWARP, 'mean KNN:',mnKNN
    if mnWARP< mnKNN and prob <= .05:
        print 'WARP brain significantly closer to target than KNN brain :D'
    else:
        print ':(' 
    return WARP_dist, KNN_dist, t,prob, all_dist_test, all_dist_train

def calc_baseline(k, tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    """
    Uses random mapping matrices to predict tags
    For comparison to prediction using WARP Loss optimization algorithm
    """
    # load tag data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words[i-1,:]for i in stim_labels ]))
    Y = words.shape[0] # size of dictionary
    # load brain data
    pre = np.ones(15)
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    for i,subject in enumerate(subjects):
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('old_brain_features/'+subject+'_600.brain')))
        n = brains.shape[1]
        d = brains.shape[0]# dimensionality of images
        D = d # ? choose empirically  
        # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
        W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
        V = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
        print i
        pre[i]= validate(W,V,brains,words,k)
    return pre
    
def knn_tags(k=5.0, ntags=5.0, tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    """
    Predict tags using a k-nearest neighbors classifier to use as baseline for comparison.
    Input:
        k - number of neighbors to poll in KNN
        ntags - for evaluating precision@ntags
    Returns:
        precision
    """
    # load tag data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words100 = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    words = (words>0)
    Y = words.shape[0] # size of dictionary
    n=words.shape[1]
    # load brain data
    pre = np.zeros((15,n),float)
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    # leave-one-out cross validation
    for i,subject in enumerate(subjects):
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('old_brain_features/'+subject+'_600.brain')))
        d = brains.shape[0]# dimensionality of images
        dist = squareform(pdist(np.transpose(brains)))
        for j in range(n):
            ind = np.argsort(dist[j,:])[1:] # indeces of nearest neighbors, sorted in ascending order (nearest first), exclude test point from set of neighbors
            ranked = np.array(sum(words[:,ind[:k]],1)) # vote for the best tags
            ind_tags = np.argsort(ranked,0)[::-1] # sorted in descending order, highest rank first
            #print words[ind_tags[:k],j].shape
            #print np.array(words[ind_tags[:k],j]).shape
            ncorrect = sum(sum(np.array(words[ind_tags[:ntags],j]),0))
            #print ncorrect.shape
            pre[i,j] = ncorrect/float(ntags)
    avg_pre = np.mean(pre,1)
    return pre,avg_pre

def learn_params(subject, feat_expr='_sts+hg_600',tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    """
    """
    #subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    #subjects = ['1mar11sj','1mar11yw']
    #Ds = [25,35,50,75,100,200,600]
    Ds=[50,100]
    #learning_rates=[.01,.05, .1, .15 .2, .25 .3, .05, .06 ]
    learning_rates=[.1, .2]

    for D in Ds:
        for lr in learning_rates:
            # load brain data
            stim_labels = np.genfromtxt('stimuli_labels.txt')
            stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
            words100 = np.genfromtxt(tag_feat_file)
            words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
            brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
            Ws, Vs,all_dist_test,all_dist_train,all_pre_test,all_pre_train,total_time,avg_best_dist=learn_25_models(subject)

def main():
    #subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    argv = sys.argv
    subject = argv[1]
    # k = int(argv[2])
    # testtype = argv[3]
    # if len(argv) > 4:
    #     feats_size = int(argv[4])
    # if 'tag' in testtype:
    #     evaluate_tag_prediction(subject,k)
    # elif 'brain' in testtype:
    evaluate_brain_prediction3(subject)


if __name__ == '__main__':
    main()
        