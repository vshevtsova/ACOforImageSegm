# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:40:12 2016

@author: ВШевцова
"""
import numpy as np
import random as rnd
import cv2

num_ants = 5000
num_iter = 5000
writeEveryStep = 100
isPrint = False
ksize = 3
halfOfKsize = ksize/2
beta = 1.5 #отвечает за степень случайности, с которой каждый муравей может пойти по градиенту феромона
delta = 0.1; #1/q показатель восприимчивости, который указывает на уменьшение восприимчивости феромона в больших концентрациях муравьями.
#beta = 3.5 #отвечает за степень случайности, с которой каждый муравей может пойти по градиенту феромона
#delta = 2; #1/q показатель восприимчивости, который указывает на уменьшение восприимчивости феромона в больших концентрациях муравьями.
eta = 0; #постоянное количество феромона для update
ro = 1.2; #постоянное количество феромона для update
V = 0; #величина испарения феромона
#V = 0.015; #величина испарения феромона
#ro = 10; #постоянное количество феромона для update
#V = 0.0015; #величина испарения феромона



def delta_gl(img,prev,cur):
    return abs(img[prev].astype('int16') - img[cur].astype('int16'))

w_former = np.array([[[1., 1./2, 1./4, 1./2, 0, 1./12, 1./4, 1./12, 1./20],\
                      [1./2, 1., 1./2, 1./4, 0, 1./4, 1./12, 1./20, 1./12],\
                      [1./4, 1./2, 1., 1./12, 0, 1./2, 1./20, 1./12, 1./4]],
                      [[1./2, 1./4, 1./12, 1., 0, 1./20, 1./2, 1./4, 1./12],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [1./12, 1./4, 1./2, 1./20, 0, 1., 1./12, 1./4, 1./2]],\
                      [[1./4, 1./12, 1./20, 1./2, 0, 1./12, 1., 1./2, 1./4],\
                      [1./12, 1./20, 1./12, 1./4, 0, 1./4, 1./2, 1., 1./2],\
                      [1./20, 1./12, 1./4, 1./12, 0, 1./2, 1./4, 1./2, 1.]]])



if __name__ == '__main__':
#--------------------------------------seed
    rnd.seed(1)
    np.random.seed(1)
#--------------------------------------imread
#    img = cv2.imread('ex.png',0)
#    img = cv2.imread('5x5.png',0)
#    img = cv2.imread('blox.jpg',0)
    img = cv2.imread('blox1.png',0)
#--------------------------------------pheromone

    pheromone = np.zeros_like(img).astype('float64')

#--------------------------------------расчет характеристик изображения
    kernel_mean = np.ones((ksize,ksize),np.float32)/(ksize**2)
    img_mean = cv2.filter2D(img.astype('float64'),-1,kernel_mean)
#--------------------------------------расставляем муравьев
    antsPosCur0 = np.random.random_integers(1, img.shape[0] - 2,(num_ants,1)).astype('int16')
    antsPosCur1 = np.random.random_integers(1, img.shape[1] - 2,(num_ants,1)).astype('int16')

    img_draw = np.zeros((img.shape[0],img.shape[1]),dtype = np.uint8)
    img_draw[antsPosCur0, antsPosCur1] = 255
    if isPrint:
        print img_draw
        print pheromone
    cv2.imwrite('draw0.png',img_draw)
#--------------------------------------задаем направления муравьев
    antsPosPrev0 = np.zeros_like(antsPosCur0)
    antsPosPrev1 = np.zeros_like(antsPosCur1)

    antsAngle = np.random.random_integers(-1,1,(num_ants,2)).astype('int8')
    antsAngle[(antsAngle == 0).all(axis=1)] = [-1, 0]
    antsAngle0, antsAngle1 = np.hsplit(antsAngle, 2)
#--------------------------------------основной цикл
    for i in range(num_iter):
        if isPrint:
            print i
#--------------------------------------step
        inds0 = antsPosCur0 == 0
        inde0 = antsPosCur0 == (img.shape[1] - 1)
        inds1 = antsPosCur1 == 0
        inde1 = antsPosCur1 == (img.shape[0] - 1)
        ind = np.bitwise_or(np.bitwise_or(inds0, inds1).flatten(),np.bitwise_or(inde0, inde1).flatten())
        antsAngle0[ind] = -antsAngle0[ind]
        antsAngle1[ind] = -antsAngle1[ind]

        w = w_former[antsAngle0 +1, antsAngle1 + 1]
        w = w.reshape((w.shape[0],w.shape[2]))

        W_full = pheromone.copy()
        W_full = (1+pheromone/(1+delta*pheromone))**beta
        W_full = cv2.copyMakeBorder(W_full,halfOfKsize,halfOfKsize,halfOfKsize,halfOfKsize,\
                    cv2.BORDER_CONSTANT,value=(0))

        w0 = W_full[antsPosCur0, antsPosCur1]
        w1 = W_full[antsPosCur0, antsPosCur1 + 1]
        w2 = W_full[antsPosCur0, antsPosCur1 + 2]
        w3 = W_full[antsPosCur0 + 1, antsPosCur1]
        w4 = W_full[antsPosCur0 + 1, antsPosCur1 + 1]
        w5 = W_full[antsPosCur0 + 1, antsPosCur1 + 2]
        w6 = W_full[antsPosCur0 + 2, antsPosCur1]
        w7 = W_full[antsPosCur0 + 2, antsPosCur1 + 1]
        w8 = W_full[antsPosCur0 + 2, antsPosCur1 + 2]
        W = np.hstack((w0,w1,w2,w3,w4,w5,w6,w7,w8))
        p_ik = w * W
        p_ik_sum = p_ik.sum(axis = 1).reshape(num_ants,1)
        P_ik = (p_ik/p_ik_sum).cumsum(axis=1)
        for i in range(num_ants):
            if p_ik_sum[i] == 0:
                antsAngle0[i] = [-1]
                antsAngle1[i] = [0]
            else:
                r = rnd.random()
                if r < P_ik[i, 0]:
                    antsAngle0[i] = [-1]
                    antsAngle1[i] = [-1]
                elif r < P_ik[i, 1]:
                    antsAngle0[i] = [-1]
                    antsAngle1[i] = [0]
                elif r < P_ik[i, 2]:
                    antsAngle0[i] = [-1]
                    antsAngle1[i] = [1]
                elif r < P_ik[i, 3]:
                    antsAngle0[i] = [0]
                    antsAngle1[i] = [-1]
                elif r < P_ik[i, 5]:
                    antsAngle0[i] = [0]
                    antsAngle1[i] = [1]
                elif r < P_ik[i, 6]:
                    antsAngle0[i] = [1]
                    antsAngle1[i] = [-1]
                elif r < P_ik[i, 7]:
                    antsAngle0[i] = [1]
                    antsAngle1[i] = [0]
                else:
                    antsAngle0[i] = [1]
                    antsAngle1[i] = [1]
        antsPosPrev0 = antsPosCur0.copy()
        antsPosPrev1 = antsPosCur1.copy()
        antsPosCur0 = antsPosCur0 + antsAngle0
        antsPosCur1 = antsPosCur1 + antsAngle1

#--------------------------------------update_pheromone
        pheromone[antsPosCur0, antsPosCur1] = pheromone[antsPosCur0, antsPosCur1] + \
                                              eta + ro * \
        abs(img_mean[antsPosPrev0, antsPosPrev1].astype('int16') - img_mean[antsPosCur0, antsPosCur1].astype('int16'))/255
#--------------------------------------evaporate_pheromone
        pheromone = pheromone - V
        pheromone[pheromone < 0] = 0
 #-----------------------------------------------------------------------------------
        if isPrint:
            if i % writeEveryStep == 0:
                img_draw[:,:] = 0
                img_draw[antsPosCur0, antsPosCur1] = 255
                print img_draw
                print pheromone
#--------------------------------------normalize pheromone
    pheromone = (pheromone/pheromone.max())*255
    pheromone = pheromone.astype('uint8')
    cv2.imwrite('reswcd'+str(num_ants)+' '+str(num_iter)+' '+str(ro)+' '+str(V)+'.png',pheromone)
#--------------------------------------imshow
    cv2.imshow('img',img)

    cv2.imshow('res',pheromone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()