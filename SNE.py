
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

g = 30 # granularity
colors = ['red', 'blue', 'green']

# High Dimensional Space

p1 = np.array([1,6,6]).T # red
p2 = np.array([3,5,1]).T # blue
p3 = np.array([7,1,5]).T # green

s1 = np.array([0.2,0,3,0.5]).T
s2 = np.array([0.2,0,3,0.5]).T
s3 = np.array([0.2,0,3,0.5]).T

points = np.array([p1,p2,p3])
sigmas = np.array([s1,s2,s3])


def plot_3D(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*matrix, c=colors, s=100)
    ax.invert_yaxis()
    plt.show()

plot_3D(points)



x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
xy = np.column_stack([x.flat, y.flat])
mu = np.array([0.0, 0.0])
sigma = np.array([.5, .5])
covariance = np.diag(sigma**2)
z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z)
#ax.plot_wireframe(x,y,z)
plt.show()


fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.contour(z.reshape(30,30))
plt.show()