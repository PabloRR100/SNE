

import numpy as np 
from scipy.stats import multivariate_normal

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.linspace(0,10,100)
x2 = np.linspace(0,10,100)
x3 = np.linspace(0,10,100)


# 1D GAUSSIAN
# -----------

# 1 - CONTOURS DOESNT MAKE SENSE IN 1D

# 2 - SURFACE PLOT
mu = 2.5
si = [0.5]
z = multivariate_normal.pdf(x1, mean=2.5, cov=si)
plt.figure(figsize=(10,10))
plt.plot(x1, z)
plt.show()

# BIVARIATE GAUSSIAN
# ------------------

# 1 - CONTOURS

xx1, xx2 = np.meshgrid(x1, x2)



from scipy.stats import multivariate_normal
x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
plt.plot(x, y)



def np_bivariate_normal_pdf(domain, mean, variance):
  X = np.arange(-domain+mean, domain+mean, variance)
  Y = np.arange(-domain+mean, domain+mean, variance)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt(X**2 + Y**2)
  Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
  return X+mean, Y+mean, Z

def plt_plot_bivariate_normal_pdf(x, y, z, name=""):
  fig = plt.figure(figsize=(12, 6))
  ax = fig.gca(projection='3d')
  ax.plot_surface(x, y, z, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=True)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z');
  plt.show()


# def bivariate_pdf(mean, variance):

mu = np.array([6,4])
si = np.array([[1, 3], [3,4]])

x1 = np.linspace(0,10,10)
x2 = np.linspace(0,10,10)


xx1, xx2 = np.meshgrid(x1, x2)
z = multivariate_normal.pdf(np.meshgrid(x1, x2), mu, np.diag(si))

plt.figure(figsize=(20,20))
plt.contour(x)
plt.show()

plt.figure(figsize=(20,20))
plt.contourf(z)
plt.show()




def np_bivariate_normal_pdf(domain, mean, variance):
  X = np.arange(-domain+mean, domain+mean, variance)
  Y = np.arange(-domain+mean, domain+mean, variance)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt(X**2 + Y**2)
  Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
  return X+mean, Y+mean, Z


def plt_plot_bivariate_normal_pdf(x, y, z, name=""):
  fig = plt.figure(figsize=(12, 6))
  ax = fig.gca(projection='3d')
  ax.plot_surface(x, y, z, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=True)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z');
  plt.show()

plt_plot_bivariate_normal_pdf(*np_bivariate_normal_pdf(6, 4, .25))
