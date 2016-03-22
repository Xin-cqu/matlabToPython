__author__ = 'xinwen'
import numpy as np


def gradientDescent(x,y,theta,alpha,num_iters):
    m=len(y)
    J_history=np.zeros((num_iters,1))
    for i in range(num_iters):
        theta=theta-(alpha/m)*(np.dot(x.transpose(),(np.dot(x,theta)-y)))
        J_history[i]=computeCost(x,y,theta)
    return theta,J_history



def computeCost(x,y,theta):
    m=len(y)
    j=np.mean(np.array(np.dot(x,theta)-y)**2)/2
    return j

