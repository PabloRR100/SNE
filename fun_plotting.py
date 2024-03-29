
import numpy as np 
from scipy.stats import multivariate_normal

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.linspace(0,10,10)
x2 = np.linspace(0,10,10)
x3 = np.linspace(0,10,10)
colors = ['red', 'blue', 'green']

'''

Instead of having the samples, we are already given the Gaussian Distributions.
The domain of each dimension is [0 -> 10]

1D
==
mu = 6
si = 0.3

2D
==
mu = [6, 3]
sigma = [[0.7, 0.2], [0.2, 0.8]] --> [0.7, 0.8]


3D
==
mu = [6, 3, 8]
sigma = [[0.7, 0.2, 0.1], [0.2, 0.8, 0.4]] --> [0.1, 0.4, 0.5] --> [0.7, 0.8, 0.5]

'''


# 1D GAUSSIAN
# -----------
# -----------

# 1 DIST
# ------

# 0 - Calculate the Gaussian
mu = 6
si = [0.03]
z = multivariate_normal(mean=mu, cov=si)

# 1 - SURFACE 
plt.figure(figsize=(10,10))
plt.plot(x1, z.pdf(x1))
plt.show()

# 2 - CONTOURS DOESNT MAKE SENSE IN 1D !!


# 2 DISTS
# -------

n = 2
mus = [6, 3]
sis = [[0.03], [0.6]]
zs = [multivariate_normal(mu, si) for mu,si in zip(mus, sis)]

plt.figure(figsize=(10,10))
for z,col in zip(zs, colors):
    plt.plot(x1, z.pdf(x1), c=col)
plt.show()






# BIVARIATE GAUSSIAN
# ------------------
# ------------------

xx1, xx2 = np.meshgrid(x1,x2)
grid = np.array([xx1.flatten(), xx2.flatten()]).T 

# 1 DIST
# ------

# 0 - Calcualte the Gaussian

mu = np.array([6, 3])
sigma = [[2, 0.5], [0.5, 1.8]]
si = np.diag(sigma)
z = multivariate_normal(mean=mu, cov=sigma) 

z_ = z.pdf(grid)

# 1 - SURFACE

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx1, xx2, z.pdf(grid).reshape(xx1.shape), 
    cmap=cm.coolwarm,linewidth=0, antialiased=True)
plt.show()

# 2 - CONTOURS
plt.figure(figsize=(10,10))
plt.contour(z.pdf(grid).reshape(xx1.shape))
plt.contourf(z.pdf(grid).reshape(xx1.shape))
plt.show()



# 2 DISTS
# -------

n = 2
mu1 = np.array([6, 3])
mu2 = np.array([3, 7])
mus = [mu1,mu2]

sigma1 = [[2,0.5], [0.5,1.8]]
sigma2 = [[1,0.13], [0.4,3]]
sigmas = [sigma1,sigma2]
sis = [np.diag(sigma) for sigma in sigmas]
zs = [multivariate_normal(mu, si) for mu,si in zip(mus, sis)]
z1_ = z.pdf(grid).reshape(xx1.shape)

# 1 - SURFACE

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for z in zs:
    ax.plot_surface(xx1, xx2, z.pdf(grid).reshape(xx1.shape), 
        cmap=cm.coolwarm,linewidth=0, antialiased=True)
plt.show()

# 2 - CONTOURS
plt.figure(figsize=(10,10))
for z,c in zip(zs,colors):
    plt.contour(z.pdf(grid).reshape(xx1.shape), colors=c)
plt.show()



# 3D MULTIVARIATE GAUSSIAN
# -------------------------

xx11, xx22, xx33 = np.meshgrid(x1,x2,x3)
grid = np.array([xx11.flatten(), xx22.flatten(), xx33.flatten()]).T 

# 1 DIST
# ------

# 0 - Calcualte the Gaussian

mu = np.array([6, 3, 8])
sigma = [[2, 0.5, 0.2], [0.5, 1.8,-0.9], [0.2, -0.9, 3]]
si = np.diag(sigma)
z = multivariate_normal(mean=mu, cov=sigma) 

# 1 - SURFACE --> DOESN'T MAKE SENSE - IT GOES TO 4D

# 2 - CONTOURS

z_ = z.pdf(grid).reshape(xx11.shape)
z2_ = z_[0]

# We have to divide the dimensions ???
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(xx11,xx22,, 50, cmap='binary')
ax.plot_wireframe(*z_, color='black')
plt.show()




