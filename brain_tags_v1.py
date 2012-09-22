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
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

def WARP_opt_weighted3(Xs_train,Ys_train,k,C,D,gamma=.2,Xs_test=None,Ys_test=None,stop_pre=None):
    """
    Input:
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
    print 'learning rate:', gamma
    n = Xs_train.shape[1]
    Y = Ys_train.shape[0] # size of dictionary
    d = Xs_train.shape[0]# dimensionality of images
    if Xs_test!=None and Ys_test !=None:
        select_model=True
    else:
        select_model=False
        pre_test = 0
    stop_train_pre=0
    #D = 101 # ? choose empirically 
    # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
    # W = np.array(np.zeros((D,Y,1)))
    # V = np.array(np.zeros((D,d,1)))
    # W[:,:,0] = np.array([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
    # V[:,:,0] = np.array([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
    W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
    W_best = W
    V= np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
    V_best = V
    if select_model:
        pre_init,dist_init = validate(W,V,Xs_test,Ys_test,k)
        pre_test = [pre_init]
        dist_test = [dist_init]
        print 'initial test precision: ',pre_test
        stop_pre = .95 # almost 100%, won't increase much more past this value, best test error is surely before we get this
    pre_tr, dist_tr=validate(W,V,Xs_train,Ys_train,k)
    pre_train = [pre_tr]
    dist_train = [dist_tr]
    best_test_dist = None
    print 'initial train precision:',pre_train[-1]
    c=0
    #err = np.zeros((n,Y,Y,3)) # 3rd column indicates whether error is still improving or not
    #count = np.zeros((n,Y,Y))
    #increased = 0
    #decreased = 0
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
            # for dd in range(d):
            #     b = C/np.linalg.norm(V[:,dd])
            #     V[:,dd] = V[:,dd]*b
            # for YY in range(Y):
            #     b=C/np.linalg.norm(W[:,YY])
            #     W[:,YY] = b*W[:,YY]

        # Calculate error
        # if count[i,j,y_hat_ind]==1:
        #     if (1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)) < 0:
        #         err[i,j,y_hat_ind,0] = 0
        #     else:
        #         err[i,j,y_hat_ind, 0] =  L(np.floor((Y-1)/N))*(1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x))
        # elif count[i,j,y_hat_ind]==2:
        #     if (1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)) < 0:
        #         err[i,j,y_hat_ind,1] = 0
        #     else:
        #         err[i,j,y_hat_ind,1] = L(np.floor((Y-1)/N))*(1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x))
        #     if err[i,j,y_hat_ind,1] == 0 or err[i,j,y_hat_ind,1]==err[i,j,y_hat_ind,0]:# or err[i,1] > err[i,0]:
        #         err[i,j,y_hat_ind,2] = 1
        #         print 'error for sample', i, 'stopped improving'
        #     #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
        #     if err[i,j,y_hat_ind,1] > err[i,j,y_hat_ind,0]:
        #         increased += 1
        #     elif err[i,j,y_hat_ind,1] < err[i,j,y_hat_ind,0]:
        #         decreased +=1
        #     # print 'error increased',increased,'times'
        #     # print 'error decreased', decreased, 'times'
        # else:
        #     err[i,j,y_hat_ind,0] = err[i,j,y_hat_ind,1]
        #     if (1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x)) < 0:
        #         err[i,j,y_hat_ind,1] = 0
        #     else:
        #         err[i,j,y_hat_ind,1] =L(np.floor((Y-1)/N))*(1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,y_hat_ind])*V*x))
        #     if err[i,j,y_hat_ind,1] == 0 or err[i,j,y_hat_ind,1]==err[i,j,y_hat_ind,0]:# or err[i,1] > err[i,0]:
        #         err[i,j,y_hat_ind,2] = 1
        #         print 'error for sample', i, 'stopped improving'
        #     #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
        #     if err[i,j,y_hat_ind,1] > err[i,j,y_hat_ind,0]:
        #         increased += 1
        #     elif err[i,j,y_hat_ind,1] < err[i,j,y_hat_ind,0]:
        #         decreased +=1
            # print 'error increased',increased,'times'
            # print 'error decreased', decreased, 'times'
        #print 'err: ', err_new
        if not(c%100):
            if select_model:
                print 'evaluating...'
                pre_tr,dist_tr = validate(W, V, Xs_train, Ys_train,k)
                #pre_train = np.append(pre_train,pre_tr)
                pre_train = np.append(pre_train,pre_tr)
                #dist_train = np.append(dist_train,dist_tr)
                dist_train = np.append(dist_train,dist_tr)
                pre_t,dist_t = validate(W, V, Xs_test, Ys_test,k)
                pre_test = np.append(pre_test,pre_t)
                dist_test = np.append(dist_test,dist_t)
                print 'test precision', pre_test
                print 'test distance', dist_test
                if best_test_dist == None or dist_test<best_test_dist:
                    stop_train_pre = pre_train[-1]
                    best_test_dist = dist_test[-1]
                    W_best = W
                    V_best = V
                print 'best test distance:',best_test_dist
            else:
                pre_train,dist_train = validate(W, V, Xs_train, Ys_train,k)
                W_best = W
                V_best = V
            print 'training precision',pre_train

                #improving = False
    return W_best,V_best,pre_train,pre_test,dist_train, dist_test,stop_train_pre,best_test_dist

# def WARP_opt_weighted2(Xs,Ys,k,C,D):
#     """
#     Input:
#         Xs - images
#         Ys - annotations
#         k - for evaluating precision@k
#         C - constraint for max norm regularization
#         D - dimensionality of joint space
#     """
#     n = Xs.shape[1] # number of images: 
#     Y = Ys.shape[0] # size of dictionary: 213
#     d = Xs.shape[0] # dimensionality of images: varies with subject
#     #D = 101 # ? choose empirically 
#     # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
#     W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
#     V = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
#     pre_new = validate(W,V,Xs,Ys,k)
#     #print 'initial precision: ',pre_new
#     c=0
#     new_rank = np.zeros((Y,n))
#     #count=0
#     f_new = 1
#     rankunchanged=0
#     err = np.zeros((n,3)) # 3rd column indicates whether error is still improving or not
#     count = np.zeros(n)
#     increased = 0
#     decreased = 0
#     while (c<=1 or not((err[:,2]==1).all())): #pre_new <.90): or count < 100:
#         c+=1
#         # pick random labeled example (out of 600 in this case)
#         i = random.randrange(0,n,1)
#         if err[i,2]==1:
#             continue
#         count[i] +=1
#         x = Xs[:,i] # d x 1
#         y = Ys[:,i] # Y x 1

#         # choose random label
#         j = random.randrange(0,Y,1)
#         N=0
#         while N==0 or not(f_n > (f_p-1)):
#             if N>= (Y-1):
#                 break; 
#             # find a label s.t. the difference in rank between the two labels is at least 1 and the lower ranked label has a higher probability
#             # i.e. the ranking is incorrect
#             # then proceed as in previous version, update mapping matrices
#             m=random.randrange(0,Y,1)
#             if Ys[m, i]==Ys[j,i]: # if the labels have the same probability, we can't learn from them, skip
#                 continue
#             else:
#                 N+=1
#                 if Ys[m, i] >Ys[j,i]: # which ever label has the highest prob becomes the positive example
#                     pos = m
#                     neg = j
#                 else:
#                     pos = j
#                     neg = m
#                 print 'positive prob:',Ys[pos,i]
#                 print 'negative prob:',Ys[neg,i]

#                 phi_x = V*x
#                 phi_p = W[:,pos]
#                 phi_n = W[:,neg]
#                 f_p=np.transpose(phi_p)*phi_x # rank of positive (more probable) label
#                 f_n = np.transpose(phi_n)*phi_x   # rank of negative (less probable) label
#                 if f_n > (f_p -1): # ranking is incorrect, so adjust mapping
#                     break # appropriate pair of labels has been found, leave loop and continue with gradient descent step
                    
#         if f_n > (f_p -1):
#             #print 'Taking a gradient step to maximize precision at k'
#             # make a gradient step to minimize error function
#             gamma = .2 # learning rate
        
#             # enforce positive only
#             # gradient_W = np.zeros(W.shape)
#             # gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(np.transpose(-V*x).shape),np.transpose(-V*x))
#             # gradient_W[:,y_hat_ind] = L(np.floor((Y-1)/N))*np.maximum(np.zeros(np.transpose(V*x).shape),np.transpose(V*x))
#             # W = W - (gamma*gradient_W)
#             # gradient_V = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(V.shape),(W[:,y_hat_ind]*np.transpose(x) - (W[:,j]*np.transpose(x)) ))
#             # V = V - (gamma*gradient_V)
        
#             gradient_W = np.zeros(W.shape)
#             gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.transpose(-V*x)
#             gradient_W[:,m] = L(np.floor((Y-1)/N))*np.transpose(V*x)
#             W = W - (gamma*gradient_W)
#             gradient_V = L(int(np.floor((Y-1)/N)))*(W[:,m]*np.transpose(x) - (W[:,j]*np.transpose(x))) 
#             V = V - (gamma*gradient_V)
        
#             # max norm regularization
#             # Project weigths to enforce constraints ||V_j||_2 <= C (for j=1...d) and ||W_i||_2 <= C (for i=1...Y)
#             for dd in range(d):
#                 b = C/np.linalg.norm(V[:,dd])
#                 V[:,dd] = V[:,dd]*b
#             for YY in range(Y):
#                 b=C/np.linalg.norm(W[:,YY])
#                 W[:,YY] = b*W[:,YY]
                    
#         if count[i]==1:
#             err[i, 0] =  L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,m])*V*x)))
#         elif count[i]==2:
#             err[i,1] = L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,m])*V*x)))
#             if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
#                 err[i,2] = 1
#                 print 'error for sample', i, 'stopped improving'
#             #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
#             if err[i,1] > err[i,0]:
#                 increased += 1
#             elif err[i,1] < err[i,0]:
#                 decreased +=1
#             print 'error increased',increased,'times'
#             print 'error decreased', decreased, 'times'
#         else:
#             # maybe need new error function
#             err[i,0] = err[i,1]
#             err[i,1] =L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,m])*V*x)))
#             if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
#                 err[i,2] = 1
#                 print 'error for sample', i, 'stopped improving'
#             #print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
#             if err[i,1] > err[i,0]:
#                 increased += 1
#             elif err[i,1] < err[i,0]:
#                 decreased +=1
#             print 'error increased',increased,'times'
#             print 'error decreased', decreased, 'times'
            

#         if not(c%100):
#             pre_old = pre_new
#             pre_new = validate(W, V, Xs, Ys,k)
#             print '*** precision: ', pre_new 
#     return W,V

# def WARP_opt_weighted(Xs,Ys,k,C,D):
#     """
#     Input:
#         Xs - images
#         Ys - annotations
#         k - for evaluating precision@k
#         C - constraint for max norm regularization
#         D - dimensionality of joint space
#     """
#     n = Xs.shape[1] # number of images: 
#     Y = Ys.shape[0] # size of dictionary: 213
#     d = Xs.shape[0] # dimensionality of images: varies with subject
#     #D = 101 # ? choose empirically 
#     # initialize matrices V and W at random with zero mean and std of 1/sqrt(d)
#     W = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(Y)] for col in range(D)])
#     V = np.matrix([[random.gauss(0, 1/np.sqrt(d)) for row in range(d)] for col in range(D)])
#     pre_new = validate(W,V,Xs,Ys,k)
#     #print 'initial precision: ',pre_new
#     c=0
#     new_rank = np.zeros((Y,n))
#     #count=0
#     f_new = 1
#     rankunchanged=0
#     err = np.zeros((n,3)) # 3rd column indicates whether error is still improving or not
#     count = np.zeros(n)
#     brain_ind = range(n)
#     while (c<=1 or not((err[:,2]==1).all())): #pre_new <.90): or count < 100:
#         random.shuffle(brain_ind)
#         # pick random labeled example (out of 600 in this case)
#         # iterate through 600 brain volumes in a new order each time, i.e. random selection without replacement, iteratively
#         for i in brain_ind:
#             labels = range(Y)
#             #print i
#             if err[i,2]==1:
#                 continue
#             count[i] +=1
#             x = Xs[:,i] # d x 1
#             y = Ys[:,i] # Y x 1
#                             # get index of positive labels 
#                             #positive = [m for m in range(0,Y,1) if y[m]==1]

#                             # pick a random positive label from y
#             # iterate through all 213 possible labels
#                             #j = random.choice(positive)
#             random.shuffle(labels)
#             for j in labels:
#                 N=0
#                             #print 'choosing negative label until rank(neg) > rank(pos)-1...' same as rank(neg) >= rank(pos)
#                 # find a label s.t. the difference in rank between the two labels is at least 1 and the lower ranked label has a higher probability
#                 # i.e. the ranking is incorrect
#                 # then proceed as in previous version, update mapping matrices
#                 labels2 = list(labels)
#                 random.shuffle(labels2)
#                 for m in labels2:
#                     c+=1
#                     if N>= (Y-1):
#                         break;
#                     N+=1
#                     if Ys[m, i]==Ys[j,i]: # if the labels have the same probability, we can't learn from them, skip
#                         continue
#                     else:
#                         [neg,pos] = np.sort([k, j]) 
#                         phi_x = V*x
#                         phi_y = W[:,pos]
#                         phi_y_hat = W[:,neg]
#                         fyi=np.transpose(phi_y)*phi_x # rank of positive (more probable) label
#                         f_hat = np.transpose(phi_y_hat)*phi_x   # rank of negative (less probable) label
#                         if f_hat >= fyi: # ranking is incorrect, so adjust mapping
#                             break # appropriate pair of labels has been found, leave loop and continue with gradient descent step
                    
#                 if f_hat >= fyi:
#                     #print 'Taking a gradient step to maximize precision at k'
#                     # make a gradient step to minimize error function
#                     gamma = .2 # learning rate
                
#                     # enforce positive only
#                     # gradient_W = np.zeros(W.shape)
#                     # gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(np.transpose(-V*x).shape),np.transpose(-V*x))
#                     # gradient_W[:,y_hat_ind] = L(np.floor((Y-1)/N))*np.maximum(np.zeros(np.transpose(V*x).shape),np.transpose(V*x))
#                     # W = W - (gamma*gradient_W)
#                     # gradient_V = L(int(np.floor((Y-1)/N)))*np.maximum(np.zeros(V.shape),(W[:,y_hat_ind]*np.transpose(x) - (W[:,j]*np.transpose(x)) ))
#                     # V = V - (gamma*gradient_V)
                
#                     gradient_W = np.zeros(W.shape)
#                     gradient_W[:,j] = L(int(np.floor((Y-1)/N)))*np.transpose(-V*x)
#                     gradient_W[:,k] = L(np.floor((Y-1)/N))*np.transpose(V*x)
#                     W = W - (gamma*gradient_W)
#                     gradient_V = L(int(np.floor((Y-1)/N)))*(W[:,k]*np.transpose(x) - (W[:,j]*np.transpose(x))) 
#                     V = V - (gamma*gradient_V)
                
#                     # max norm regularization
#                     # Project weigths to enforce constraints ||V_j||_2 <= C (for j=1...d) and ||W_i||_2 <= C (for i=1...Y)
#                     for dd in range(d):
#                         b = C/np.linalg.norm(V[:,dd])
#                         V[:,dd] = V[:,dd]*b
#                     for YY in range(Y):
#                         b=C/np.linalg.norm(W[:,YY])
#                         W[:,YY] = b*W[:,YY]
#                     if not(c%100):
#                         pre_old = pre_new
#                         pre_new = validate(W, V, Xs, Ys,k)
#                         print 'precision: ', pre_new 
#         if count[i]==1:
#             err[i, 0] =  L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,k])*V*x)))
#         elif count[i]==2:
#             err[i,1] = L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,k])*V*x)))
#             if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
#                 err[i,2] = 1
#                 print 'error for sample', i, 'stopped improving'
#             print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
#         else:
#             err[i,0] = err[i,1]
#             err[i,1] =L(np.floor((Y-1)/N))*abs((1 - (np.transpose(W[:,j])*V*x) + (np.transpose(W[:,k])*V*x)))
#             if err[i,1] == 0 or err[i,1]==err[i,0]:# or err[i,1] > err[i,0]:
#                 err[i,2] = 1
#                 print 'error for sample', i, 'stopped improving'
#             print 'error for sample',i,'went from ', err[i,0], 'to', err[i,1]
            

#         #print 'err: ', err_new
#         #if not(c%100):
#         # if True:
#         #     pre_old = pre_new
#         #     pre_new = validate(W, V, Xs, Ys,k)
#         #     print 'precision: ', pre_new
#         #print 'f: ', f_new
#     return W,V

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


"""
Use learned mapping matrices W and V to the rank of all tags for each image
Ys = 100x600
"""
def validate(W, V, Xs, Ys, k):
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
        # print testbrain.shape
        # print trainbrains.shape
        # print testwords.shape
        # print trainwords.shape
        W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,Xs_test=testbrain,Ys_test=testwords)
        #W,V = WARP_opt(trainbrains, trainwords,k,C,D)
        test_precision[str(l)] = pre_test
        test_precision[str(l)] = pre_test
        cv_train_pre = np.append(cv_train_pre,stop_train_pre)
        #pak[l]=validate(W_best, V_best, testbrain, testwords,k)
        pak[l] = best_test_pre
        # rank = np.transpose(W_best)*V_best*testbrain
        # ind = np.argsort(rank,0)[::-1]
        # p = np.zeros(testwords.shape[1])
        # testwords_bin = testwords > 0
        # for i, stim in enumerate(np.transpose(testwords_bin)):
        #     truepos = sum(stim[0,ind[0:k,i]])
        #     p[i] = truepos/k
        # pak[l] = np.mean(p)
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
    Ws = {}
    Vs = {}
    trained = 0
    # iterate through each 
    for i in range(n):
        testtags = words[:,i]
        print testtags.shape
        testbrain = brains[:,i]
        cur_stim = stim_labels[i]
        if str(cur_stim) in Ws:
            W_best = Ws[str(cur_stim)]
            V_best = Vs[str(cur_stim)]
        else:
            trained+=1
            print 'training ',trained, 'time'
            trainwords = words[:,stim_labels!=stim_labels[i]]
            trainbrains = brains[:,stim_labels!=stim_labels[i]]
            testwords = words[:,stim_labels==stim_labels[i]]
            testbrains = brains[:,stim_labels==stim_labels[i]]
            #W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,stop_pre=stop_pre)
            W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,Xs_test=testbrains,Ys_test=testwords)
            Ws[str(cur_stim)] = W_best
            Vs[str(cur_stim)] = V_best
        #brainrank = np.sum(np.transpose(testwords)*np.transpose(W_best)*V_best*trainbrains,0)
        #brainrank = np.array([(validate(W_best, V_best, np.transpose(brain), testwords, k)) for brain in np.transpose(trainbrains)])
        brainrank = np.array([(validate(W_best, V_best, np.transpose(brain), testtags, k)) for brain in np.transpose(trainbrains)])
        # indices of brains whose distance (after turning rank into prob) to actual tag vec is shortest
        top_brain_ind = np.argsort(brainrank[:,1])[:k]
        pred_brain = np.mean(trainbrains[:,top_brain_ind],1) # brain activity predicted by WARP loss optimization
        # weighted predicted brain - don't use this when using distances to rank brains!
        #pred_brain = np.sum(trainbrains[:,top_brain_ind]*brainrank[top_brain_ind])/np.sum(brainrank[top_brain_ind])
        #print pred_brain.shape

        # KNN 
        print testtags.shape, trainwords.shape
        top_brain_ind_knn = np.argsort([np.linalg.norm(testtags-np.transpose(tagvec)) for tagvec in np.transpose(trainwords)])[-k:]
        pred_brain_knn = np.mean(trainbrains[:,top_brain_ind_knn],1)
        #print pred_brain_knn.shape

        WARP_dist[i]= np.linalg.norm(pred_brain-testbrain)
        print 'WARP_dist:', WARP_dist
        KNN_dist[i]= np.linalg.norm(pred_brain_knn-testbrain)
        print 'KNN_dist:', KNN_dist
    print 'Model trained only ',trained, 'times'
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
    return WARP_dist, KNN_dist, t,prob

def evaluate_brain_prediction2(subject='1mar11sj',k=5,C=1,D=75,feat_expr='_sts+hg_avgrun_75x72',tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    """

    """
    # load brain data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
    n=brains.shape[1]
    if n==75:
        print '75'
        # stimuli_ids = np.array([line.strip() for line in open('stimuli_labels.txt')])
        # stimuli_ids_tr = list(itertools.chain(*[[sid+'a', sid+'b', sid+'c'] for sid in stimuli_ids]))
        stim_labels = np.unique(stim_labels)
        stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
        #print stim_labels.shape
        s=75
    elif n==600:
        stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
        s=75
    elif n==200:
        s=25
    dims = brains.shape[0]
    # load tag data
    words = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words[i-1,:]for i in stim_labels ]))
    ncorrect=0.0
    count=0.0
    for i in range(s):
        if n==75:
            # simple leave one out cross validation
            count+=1
            possibledecoys = np.setdiff1d(range(n),[i])
            possibledecoys = range(n)
            possibledecoys.remove(i)
            for decoy_ind in possibledecoys:
            #decoy_ind = random.choice(possibledecoys)
            #print decoy_ind
                train_ind = np.setdiff1d(range(n),[i,decoy_ind])
                targetbrain = brains[:,i]
                decoybrain = brains[:,decoy_ind]
                trainbrains = brains[:,train_ind]
                targetwords = words[:,i]
                decoywords = words[:,decoy_ind]
                trainwords = words[:,train_ind]
                W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,stop_pre=.92)
                # predict target and decoy brains 
                # Method 1: direct application of mapping matrices (unllikely to work)
                ptargetbrain = np.transpose(targetwords)*np.transpose(W_best)*V_best
                pdecoybrain = np.transpose(decoywords)*np.transpose(W_best)*V_best

                # Method 2
                rank
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

        else:
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
                W_best,V_best,pre_train,pre_test,stop_train_pre,best_test_pre= WARP_opt_weighted3(trainbrains,trainwords,k,C,D,stop_pre=.92)
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
    print 'accuracy: ', cur_acc
    np.savetxt(subject+'_hg+sts_mitchelltest_'+str(k)+'_'+str(n)+'.txt',[accuracy])
    return cur_acc

def evaluate_brain_prediction(subject='1mar11sj',k=5, feat_expr='_sts+hg_avgrun_75x72',tag_feat_file='weighted_top100_lastfm+pandora_feats.txt'):
    """
    Mitchell style evaluation: hold 2 items out, a target and a distractor, evaluate ability to determine target
    """
    # load brain data
    stim_labels = np.genfromtxt('stimuli_labels.txt')
    brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
    n=brains.shape[1]
    if s==75:
        stim_labels = np.unique(stim_labels)
    elif s==600:
        stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
    dims = brains.shape[0]
    # load tag data
    words = np.genfromtxt(tag_feat_file)
    words = np.transpose(np.matrix([words[i-1,:]for i in stim_labels ]))
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
            W,V,pre_train,pre_test = WARP_opt(trainbrains, trainwords,k)
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

    Ds = [25,35,50,75,100,200,600]
    gammas[.01,.05, .1, .15 .2, .25 .3, .05, .06 ]
    for D in Ds:
        for gamma in gammas:
            # load brain data
            stim_labels = np.genfromtxt('stimuli_labels.txt')
            stim_labels = np.ravel([np.tile(i,(1,3)) for i in stim_labels])
            words100 = np.genfromtxt(tag_feat_file)
            words = np.transpose(np.matrix([words100[i-1,:]for i in stim_labels ]))
            brains = np.matrix(np.transpose(bregman.audiodb.adb.read('brain_features/'+subject+feat_expr+'.brain')))
            W_best,V_best,pre_train,pre_test,dist_train, dist_test,stop_train_pre,best_test_dist=WARP_opt_weighted3(Xs_train,Ys_train,k,C,D,gamma=gamma,Xs_test=None,Ys_test=None,stop_pre=None)


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
        