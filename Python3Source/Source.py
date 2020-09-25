#Vladimir Gudkov, Ilia Moiseev
#South Ural State University, Chelyabinsk, Russia, 2020
#Image smoothing Algorithm Based on Gradient Analysis

import numpy as np
from math import sqrt, atan2, cos, isnan
from random import random, randint

def euclid_norm(vect): 
    return sqrt( vect[0]*vect[0] + vect[1]*vect[1] )

def angle_rad(vect):
    return atan2(vect[1],vect[0])

def grad(x, y, image):
    gradx = image[y][x-1] - image[y][x+1]
    grady = image[y+1][x] - image[y-1][x] 
    return np.array([gradx, grady])

def compute_grads(image):
    grads = np.zeros((image.shape[0], image.shape[1], 2))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if( x > 0 and y > 0 and x < image.shape[1] - 1 and y < image.shape[0] - 1):
                grads[y,x] = grad(x,y,image)
    return grads

def compute_modules(image, grads=None):
    newimage = np.zeros(image.shape)
    if(grads is None):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if( x > 0 and y > 0 and x < image.shape[1] - 1 and y < image.shape[0] - 1):
                    newimage[y][x] = euclid_norm(grad(x,y,image))
    else:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if( x > 0 and y > 0 and x < image.shape[1] - 1 and y < image.shape[0] - 1):
                    newimage[y][x] = euclid_norm(grads[y][x])
    return newimage

def compute_angles(image, grads=None):
    newimage = np.zeros(image.shape)
    if(grads is None):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if( x > 0 and y > 0 and x < image.shape[1] - 1 and y < image.shape[0] - 1):
                    angle = angle_rad(grad(x,y,image))
                    if(not isnan(angle)):
                        newimage[y][x] = angle
                    else:
                        newimage[y][x] = 0
    else:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if( x > 0 and y > 0 and x < image.shape[1] - 1 and y < image.shape[0] - 1):
                    angle = angle_rad(grad(x,y,image))
                    if(not isnan(angle)):
                        newimage[y][x] = angle
                    else:
                        newimage[y][x] = 0
    return newimage

def filter_channel(src, dst, modules, angles, ksize):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            up = i - ksize // 2
            left = j - ksize // 2
            down = i + ksize // 2 + 1
            right = j + ksize // 2 + 1
            sumWeights = 0
            result = 0
            for k in range(up, down):
                if (k < 0 or k >= src.shape[0]):
                    continue
                for l in range(left, right):
                    if (l < 0 or l >= src.shape[1]):
                        continue
                    if (modules[k][l] == .0):
                        continue
                    weight = .0
                    if (k != i and l != j):
                        angle = 2.* (angles[k][l] - angles[i][j])
                        weight = (cos(angle) + 1) / modules[k][l]
                    else:
                        #weight of central pixel
                        weight = 1.
                    result += weight * src[k][l]
                    sumWeights += weight
            if (sumWeights != 0):
                dst[i][j] = round(result / sumWeights)
            else:
                #pixel remains without changes if sum of weights = 0
                dst[i][j] = src[i][j]

def filter_(src, ksize, grads=None, modules=None, angles=None):
    if(len(src.shape) == 3):
        red, green, blue = src[:,:,0], src[:,:,1], src[:,:,2]
        red   = filter_(red,   ksize)
        green = filter_(green, ksize)
        blue  = filter_(blue,  ksize)

        result = np.dstack((red, green, blue))
        return result

    newimage = np.zeros(src.shape)
    if(grads is None):
        grads = compute_grads(src)
    if(modules is None):
        modules =  compute_modules(src, grads=grads)
    if(angles is None):
        angles = compute_angles(src, grads=grads)

    filter_channel(src, newimage, modules, angles, ksize)
    return newimage