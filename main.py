import numpy
import numpy as np
import time
import torch
import scipy.io as scio

def Pad(result,pix,autoscale):
    x = np.arange(-1 + 1 / pix, 1, 2 / pix)
    y = np.arange(-1 + 1 / pix, 1, 2 / pix)
    k = 0
    z=0
    data = np.empty((64*64, 3),dtype=float)
    for i in range(pix):
        for j in range(pix):
            if (x[i] ** 2 + y[j] ** 2) <= 1:
                data[k, 0] = x[i]
                data[k, 1] = x[j]
                data[k, 2] = result[z]
                z+=1
            else:
                data[k, 2]=0
            k += 1
    if autoscale:
        data[:,2] = data[:,2]/max(result)

    img = np.reshape(data[:,2],(64,64))
    return img

def Onestep(S, Input, mu, pix):
    x = np.arange(-1 + 1 / pix, 1, 2 / pix)
    y = np.arange(-1 + 1 / pix, 1, 2 / pix)
    co = np.empty((pix ** 2, 2))
    coxy = np.empty((pix ** 2, 2))
    k = 0
    for i in range(pix):
        for j in range(pix):
            if (x[i] ** 2 + y[j] ** 2) <= 1:
                co[k, 0] = x[i]
                co[k, 1] = x[j]
                coxy[k, 0] = i
                coxy[k, 1] = j
                k += 1
    totalPixel = k
    L = np.zeros((totalPixel, totalPixel))
    co = co[:k]
    def findAdj(i, co, n):
        pIndex = (((np.reshape(co[:, 0], (-1, 1)) == co[i, 0]) & abs(np.reshape(co[:, 1], (-1, 1)) - np.dot(co[i, 1], np.ones((n, 1))) == 2 / pix)) |
                  ((np.reshape(co[:, 1], (-1, 1)) == co[i, 1]) & (abs(np.reshape(co[:, 0], (-1, 1)) - np.dot(co[i, 0], np.ones((n, 1)))) == 2 / pix)))
        pIndex = np.flatnonzero(pIndex)
        return pIndex

    for i in range(totalPixel):
        pIndex = findAdj(i, co, totalPixel)
        L[i, i] = len(pIndex)
        L[i, pIndex] = -1
    LAOP = L.T * L
    I = LAOP
    result = np.dot(np.dot(np.linalg.inv(np.dot(S.T, S) + mu * I), S.T), Input)
    J_matrix = np.dot(np.linalg.inv(np.dot(S.T, S) + np.dot(mu,I)), S.T)
    np.save('J_matrix.npy',J_matrix)
    return result
