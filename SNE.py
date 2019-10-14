
import numpy as np 
from scipy.stats import multivariate_normal

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g = 30 # granularity
colors = ['red', 'blue', 'green']

# High Dimensional Space

p1 = np.array([1,14,3]).T # red
p2 = np.array([3,5,1]).T # blue
p3 = np.array([7,1,5]).T # green

q1 = np.array([1,5])
q2 = np.array([3,2])
q3 = np.array([4,4])

s1 = np.array([0.2,0,3,0.5]).T
s2 = np.array([0.2,0,3,0.5]).T
s3 = np.array([0.2,0,3,0.5]).T

points = np.array([p1,p2,p3])
low_dim_points = np.array([q1,q2,q3])

sigmas = np.array([s1,s2,s3])
low_dim_sigmas = np.array([np.array([0.5,0.5]) for _ in range(len(low_dim_points))])

print(*points)
print(*points.T) # This is how plt likes it [Xs, Ys, Zs]


def plot_3D(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*matrix.T, c=colors, s=100)
    ax.set_ybound(ax.get_ybound()[::-1])
    plt.show()


def plot_2D(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(*matrix.T, c=colors, s=100)
    plt.show()

plot_3D(points)
plot_2D(low_dim_points)


## ELLIPSES - 2D GAUSSIANS



## ELLIPSOIDS - 3D GAUSSIANS


