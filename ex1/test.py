__author__ = 'xinwen'
import numpy as np
import matplotlib.pyplot as plt
import function_xin
data = np.loadtxt('ex1data1.txt', delimiter=',').astype('float')
x_input = data[:, 0]
y_input=data[:, 1]
m=len( x_input )
#test=np.zeros((2,1),dtype='float')
#print test[1]
m_ones=np.ones((m,1))
x_add=np.matrix([np.reshape(m_ones,m),np.reshape(x_input,len(x_input))]).transpose()
y_add=np.matrix(y_input).transpose()
theta=np.zeros((2,1)).astype('float')
iterators = 1500
alpha=0.01
#print y_add.shape
#print (np.dot(x_add,theta)-y_add).shape

j=function_xin.computeCost(x_add,y_add,theta)
print('here should be 32.073: ',j)
theta_after,J_history=function_xin.gradientDescent(x_add,y_add,theta,alpha,iterators)
#plt.figure()

plt.plot(x_input,y_input,'rx')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot(x_input,np.dot(x_add,theta_after),'-')
plt.legend('training data')
plt.show()

print 'Theta found by gradient descent:',theta_after[0],theta_after[1]

print 'For population = 70,000, we predict a profit of $%f\n' %(np.dot(np.array([1,7]),theta_after)*1000)

theta0_vals=np.arange(-10,10,0.2)
theta1_vals=np.arange(-4,1,0.05)
j_vals=np.zeros((len(theta0_vals), len(theta1_vals)))
#print np.matrix([theta0_vals[1],theta1_vals[1]])
#print theta1_vals[1]

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t=np.matrix([theta0_vals[i],theta1_vals[j]]).transpose()
        j_vals[i,j]=function_xin.computeCost(x_add,y_add,t)
j_vals=np.transpose(j_vals)

