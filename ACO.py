# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:22:48 2016

@author: ВШевцова
"""
import numpy as np
import random as rnd
import cv2
num_ants = 100
num_iter = 200
ksize = 3
halfOfKsize = ksize/2
beta = 3.5 #отвечает за степень случайности, с которой каждый муравей может пойти по градиенту феромона
delta = 20; #1/q показатель восприимчивости, который указывает на уменьшение восприимчивости феромона в больших концентрациях муравьями.
eta = 0; #постоянное количество феромона для update
ro = 1.2; #постоянное количество феромона для update
V = 0.0015; #величина испарения феромона
#ro = 20; #постоянное количество феромона для update
#V = 0.015; #величина испарения феромона



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

#[-1,-1]  ->  [0,0]
#[1., 1./2, 1./4, 1./2, 0, 1./12, 1./4, 1./12, 1./20]
#[-1,0]  ->  [0,1]
#[1./2, 1., 1./2, 1./4, 0, 1./4, 1./12, 1./20, 1./12]
#[-1,1]  ->  [0,2]
#[1./4, 1./2, 1., 1./12, 0, 1./2, 1./20, 1./12, 1./4]
#
#[0,-1]  ->  [1,0]
#[1./2, 1./4, 1./12, 1., 0, 1./20, 1./2, 1./4, 1./12]
#[0,0]  ->  [1,1]
#[0, 0, 0, 0, 0, 0, 0, 0, 0]
#[0,1]  ->  [1,2]
#[1./12, 1./4, 1./2, 1./20, 0, 1., 1./12, 1./4, 1./2]
#
#[1,-1]  ->  [2,0]
#[1./4, 1./12, 1./20, 1./2, 0, 1./12, 1., 1./2, 1./4]
#[1,0]  ->  [2,1]
#[1./12, 1./20, 1./12, 1./4, 0, 1./4, 1./2, 1., 1./2]
#[1,1]  ->  [2,2]
#[1./20, 1./12, 1./4, 1./12, 0, 1./2, 1./4, 1./2, 1.]



def calc_w(a):
        return w_former[a[0] + 1, a[1] + 1]

def calc_W(pheromone):
    W_full = pheromone.copy()
    W_full = (1+pheromone/(1+delta*pheromone))**beta
    W_full = cv2.copyMakeBorder(W_full,halfOfKsize,halfOfKsize,halfOfKsize,halfOfKsize,\
                cv2.BORDER_CONSTANT,value=(0))
    return W_full

def evaporate_pheromone(pheromone):
    pheromone = pheromone - V
    pheromone[pheromone < 0] = 0

class Ant:
    def __init__(self,img):
        self._img = img
# генерируем случайную позицию муравья так, чтобы он не попал на границу
        self.posCur = np.array([rnd.randint(1, self._img.shape[1] - 2), \
                                rnd.randint(1, self._img.shape[0] - 2)])
        self.angle = np.array([rnd.randint(-1, 1), rnd.randint(-1, 1)])
        self.posPrev = self.posCur - self.angle
#
##0   1   2  |  i-1,j-1   i-1,j     i-1,j+1
##3  i,j  2  |    i,j-1     i,j       i,j+1
##5   4   3  |  i+1,j-1   i+1,j     i+1,j+1

    def step(self,pheromone):
        if self.posCur[0] == 0 or self.posCur[1] == 0 or \
           self.posCur[0] == (self._img.shape[1] - 1) or \
           self.posCur[1] == (self._img.shape[0] - 1):
               self.angle = - self.angle
        w = calc_w(self.angle)
        W_full = calc_W(pheromone)
        W = W_full[self.posCur[0]:(self.posCur[0] + ksize),\
                   self.posCur[1]:(self.posCur[1] + ksize)]
        p_ik = w * W.flatten()
        p_ik_sum = p_ik.sum()
        if p_ik_sum == 0:
            self.angle = np.array([rnd.randint(-1, 1), rnd.randint(-1, 1)])
        else:
            P_ik = p_ik/p_ik.sum()
            P_ik = P_ik.cumsum()
    #[-1,-1] [-1,0] [-1,1]             [0,0] [0,1] [0,2]
    #[0,-1]  [0,0]  [0,1]    +  1 =    [1,0] [1,1] [1,2]
    #[1,-1]  [1,0]  [1,1]              [2,0] [2,1] [2,2]
            r = rnd.random()
            if r < P_ik[0]:
                self.angle = np.array([-1,-1])
            elif r < P_ik[1]:
                self.angle = np.array([-1,0])
            elif r < P_ik[2]:
                self.angle = np.array([-1,1])
            elif r < P_ik[3]:
                self.angle = np.array([0,-1])
            elif r < P_ik[5]:
                self.angle = np.array([0,1])
            elif r < P_ik[6]:
                self.angle = np.array([1,-1])
            elif r < P_ik[7]:
                self.angle = np.array([1,0])
            else:
                self.angle = np.array([1,1])
        self.posPrev = self.posCur
        self.posCur = self.posCur + self.angle

    def update_pheromone(self,pheromone):
        pheromone[self.posCur] = pheromone[self.posCur] + eta + \
                ro * delta_gl(self._img,self.posPrev,self.posCur)/255;

    def __str__(self):
        return "(position = (%i, %i), angle = (%i, %i)" % \
               (self.posCur[0],self.posCur[1],self.angle[0],self.angle[1])
def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
if __name__ == '__main__':
    rnd.seed(1)
#    img = cv2.imread('ex.png',0)
#    img = cv2.imread('blox.jpg',0)
    img = cv2.imread('baboon.jpg',0)
    kernel_mean = np.ones((ksize,ksize),np.float32)/(ksize**2)
    img_mean = cv2.filter2D(img.astype('float64'),-1,kernel_mean)
    ants = []
    for i in range(num_ants):
        ants.append(Ant(img))
    filters = build_filters()
    res1 = process(img, filters)
    ret,thresh1 = cv2.threshold(res1,200,20,cv2.THRESH_BINARY)

    pheromone = thresh1.copy().astype('float64')

    pheromone = np.zeros_like(img).astype('float64')
    for i in range(num_iter):
        for ant in ants:
            ant.step(pheromone)
            ant.update_pheromone(pheromone)
        evaporate_pheromone(pheromone)

    pheromone = (pheromone/pheromone.max())*255
    pheromone = pheromone.astype('uint8')
    cv2.imwrite('res'+str(num_ants)+' '+str(num_iter)+' '+str(ro)+' '+str(V)+'.png',pheromone)
    cv2.imshow('img',img)
    laplacian1 = cv2.Laplacian(img,cv2.CV_64F)
    laplacian = np.zeros_like(laplacian1)
    laplacian = laplacian1 - laplacian1.min()
    laplacian = cv2.normalize(laplacian,laplacian,norm_type = cv2.NORM_INF)
    laplacian = laplacian * 255

    laplacian = laplacian.astype('uint8')
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('res',pheromone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

