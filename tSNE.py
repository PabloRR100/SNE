
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

colors = ['red', 'blue', 'green']

# High Dimensional Space

p1 = np.array([1,6,0]).T
p2 = np.array([3,5,1]).T
p3 = np.array([7,1,5]).T


# def plot_3D(matrix):

def adjusts_axis(points):
    return np.array()


points = np.array([p1,p2,p3])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*points, c=colors, s=100)
ax.invert_yaxis()
plt.show()


for (i,j,k) in zip(*points.T):
    print(i,j,k)


sigmas = np.array([0.2,0,3,0.5])

