__author__ = 'xinwen'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
data=np.loadtxt('ex1data1.txt',delimiter=',').astype('float')
x_input=data[:,0]
y_input=data[:,1]
m=len(x_input)
#test=np.zeros((2,1),dtype='float')
#print test[1]
m_ones=np.ones((m,1))
x_add=np.matrix([np.reshape(m_ones,m),np.reshape(x_input,len(x_input))]).transpose()
y_add=np.matrix(y_input).transpose()
theta=np.zeros((2,1)).astype('float')
iterators=1500;
alpha=0.01
