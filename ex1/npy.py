import function_xin

__author__ = 'xinwen'
import numpy as np
theta0_vals=np.arange(-10,10,0.2)
theta1_vals=np.arange(-4,1,0.05)
j_vals=np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(theta1_vals):
        t=np.matrix[theta0_vals(i),theta1_vals(j)]
        j_vals[i,j]=function_xin.computeCost(X,y,t)

print len(j_vals)