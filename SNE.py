
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

colors = ['red', 'blue', 'green']

# High Dimensional Space

p1 = np.array([1,6,6]).T # red
p2 = np.array([3,5,1]).T # blue
p3 = np.array([7,1,5]).T # green


def adjusts_axis(points):
    return np.array()


points = np.array([p1,p2,p3])

def plot_3D(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*matrix, c=colors, s=100)
    ax.invert_yaxis()
    plt.show()

plot_3D(points)

sigmas = np.array([0.2,0,3,0.5])

