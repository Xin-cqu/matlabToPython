__author__ = 'xinwen'
import matplotlib.pyplot as plt
import numpy as np
import time
plt.ion()
x = np.linspace(0, 50, 1000)
plt.figure(1)
plt.plot(x, np.sin(x))
plt.draw()
time.sleep(5)
plt.close(1)
plt.figure(2)
plt.plot(x, np.cos(x))
plt.draw()
time.sleep(5)
print 'it is ok'