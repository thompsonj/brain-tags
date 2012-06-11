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
import random
import numpy as np
import bregman 
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform



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
            Project weigths to enforce constraints ||V_j||_2 <= C (for j=1...d) and ||W_i||_2 <= C (for i=1...Y)
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
        else:
            err[i,0] = err[i,1]
            err[i,1] =L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)))
            if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
                err[i,2] = 1
                print 'error for sample', i, 'stopped improving'
            #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
            

        #print 'err: ', err_new
        if not(c%100):
            pre_old = pre_new
            pre_new = validate(W, V, Xs, Ys,k)
            print 'precision: ', pre_new
        #print 'f: ', f_new
    return W,V

"""
Use learned mapping matrices W and V to the rank of all tags for each image
Ys = 100x600
"""
def validate(W, V, Xs, Ys, k):
    
    rank = np.transpose(W)*V*Xs
    ind = np.argsort(rank,0)[::-1]
    p = np.zeros((Ys.shape[1]))
    for i,stim in enumerate(np.transpose(Ys)):
        # evaluate precision at k
        truepos = sum(stim[0,ind[0:k,i]])
        p[i] = float(truepos)/float(k)
    #print p
    avg_precision = np.mean(p)
    return avg_precision

def L(k):
    """
    choose alpha = 1/j (instead of alpha = 1/(Y-1)) because Usunier 2009 showed that it yields state-of-the-art results measuring p@k
    """
    loss = 0
    for j in range(1,k):
        alpha = 1/j
        loss = loss + alpha
    return loss

def evaluate_tag_prediction(subject,k=5,C=1,D=25):
    # load brain data
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_600.brain')))
    n = brains.shape[1]
    # load tag data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words100 = np.genfromtxt('top100_tags_features.txt')
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    # perform leave-one-out cross validation
    # 10-fold cross validation
    nfolds = 8
     # initialize array to store precision values
    pak = np.zeros(nfolds)
    nsamples = n/nfolds
    for l in range(nfolds):
        print l
        start = (l*nsamples)
        end = (start + nsamples)-1
        # divide data into training and test sets
        testbrain = brains[:,start:end]
        trainbrains = np.append(brains[:,0:start], brains[:,end+1:],1)
        testwords = words[:,start:end]
        trainwords = np.append(words[:,0:start], words[:,end+1:],1)
        W,V = WARP_opt(trainbrains, trainwords,k,C, D)
        rank = np.transpose(W)*V*testbrain
        ind = np.argsort(rank,0)[::-1]
        p = np.zeros(testwords.shape[1])
        for i, stim in enumerate(np.transpose(testwords)):
            truepos = sum(stim[0,ind[0:k,i]])
            p[i] = truepos/k
        pak[l] = np.mean(p)
        print 'precision on fold ',l,': ',pak[l]
    avg_precision = np.mean(pak)
    to_save = np.append(avg_precision, pak)
    np.savetxt(subject+'_precision_600_run_'+str(k)+'_'+str(D)+'.txt',to_save)
    return avg_precision

def evaluate_brain_prediction(subject='1mar11sj',k=5, feats_size=200):
    """
    Mitchell style evaluation: hold 2 items out, a target and a distractor, evaluate ability to determine target
    """
    # load brain data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    if feats_size ==200:
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_200.brain')))
        s=25
    elif feats_size ==600:
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_600.brain')))
        stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
        s=75
    n = brains.shape[1]
    # load tag data
    words100 = np.genfromtxt('top100_tags_features.txt')
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    ncorrect=0.0
    count=0.0
    for i in range(s):
        for j in range(8):
            count+=1
            samebrains_ind =  [(k*s)+i for k in range(8)] # indeces of other presentations of the same track 
            possibledecoys = np.setdiff1d(range(n),samebrains_ind)
            decoy_ind = random.choice(possibledecoys)
            train_ind = np.setdiff1d(range(n),[(j*s)+i,decoy_ind])
            targetbrain = brains[:,(j*s)+i]
            decoybrain = brains[:,decoy_ind]
            trainbrains = brains[:,train_ind]
            targetwords = words[:,(j*s)+i]
            decoywords = words[:,decoy_ind]
            trainwords = words[:,train_ind]
            W,V = WARP_opt(trainbrains, trainwords,k)
            # predict target and decoy brains 
            ptargetbrain = np.transpose(targetwords)*np.transpose(W)*V
            pdecoybrain = np.transpose(decoywords)*np.transpose(W)*V
            # inspect euclidean distance to determine target
            tdist = bregman.distance.euc2(np.transpose(np.array(targetbrain)), np.array(ptargetbrain))
            ddist = bregman.distance.euc2(np.transpose(np.array(targetbrain)), np.array(pdecoybrain))
            print 'target dist: ', tdist
            print 'decoy dist: ', ddist
            if tdist<ddist:
                ncorrect+=1.0
                print 'correct!'
            cur_acc = ncorrect/float(count)
            print 'current accuracy: ',cur_acc
    accuracy = ncorrect/float(feats_size)
    print 'accuracy: ', accuracy
    np.savetxt(subject+'_mitchelltest_'+str(k)+'_'+str(feats_size)+'.txt',[accuracy])
    return accuracy

def calc_baseline(k):
    """
    Uses random mapping matrices to predict tags
    """
    # load tag data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    words100 = np.genfromtxt('top100_tags_features.txt')
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    Y = words.shape[0] # size of dictionary
    # load brain data
    pre = np.ones(15)
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    for i,subject in enumerate(subjects):
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_600.brain')))
        n = brains.shape[1]
        d = brains.shape[0]# dimensionality of images
        D = d # ? choose empirically  
        # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
        W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
        V = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
        print i
        pre[i]= validate(W,V,brains,words,k)
    return pre
    
def knn(k=5.0, ntags=5.0):
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
    words100 = np.genfromtxt('top100_tags_features.txt')
    words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
    Y = words.shape[0] # size of dictionary
    n=words.shape[1]
    # load brain data
    pre = np.zeros((15,n),float)
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    # leave-one-out cross validation
    for i,subject in enumerate(subjects):
        brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+'_600.brain')))
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

def main():
    #subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    argv = sys.argv
    subject = argv[1]
    k = int(argv[2])
    testtype = argv[3]
    if len(argv) > 4:
        feats_size = int(argv[4])
    if 'tag' in testtype:
        evaluate_tag_prediction(subject,k)
    elif 'brain' in testtype:
        evaluate_brain_prediction(subject,k,feats_size)

if __name__ == '__main__':
    main()
        