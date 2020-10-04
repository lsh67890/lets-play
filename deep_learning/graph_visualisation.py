# Graph visualisation with matplotlib!
# comment out plt.show() for the graph you want to view.

import numpy as np
import matplotlib.pyplot as plt

# draw a graph
data = np.random.randn(50).cumsum()
plt.plot(data)
plt.title('My Graph')
plt.xlabel('my x')
plt.ylabel('my y')
#plt.show()
plt.close()

# draw multiple graphs
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 2)
#plt.show()
plt.close()

# draw a multi-graph
hist_data = np.random.randn(100)
scat_data = np.arange(30)
plt.subplot(2, 2, 1)
plt.plot(data)
plt.subplot(2, 2, 2)
plt.hist(hist_data, bins=20)
plt.subplot(2, 2, 3)
plt.scatter(scat_data, np.arange(30) + 3 * np.random.randn(30))
#plt.show()
plt.close()

# change graph colour and marker
# colours: b - blue, g - green, r - red, y - yellow, k - black, w - white
# marker: o - circle, v - reverse triangle, ^ - triangle, s - square, + - plus, . - dot
colmar = np.random.randn(50).cumsum()
plt.plot(colmar, 'yv')
#plt.show()
plt.close()

# change graph size
plt.figure(figsize=(10, 5)) # in inches. figure must be at the top of the graph to be applied
plt.plot(data, 'k+')
#plt.show()
plt.close()

# overlap graphs with label and legends
oldata = np.random.randn(30).cumsum()
plt.plot(oldata, 'k--', label='Default')
plt.plot(oldata, 'k-', drawstyle='steps-post', label='steps-post')
plt.legend()
#plt.show()
plt.close()
