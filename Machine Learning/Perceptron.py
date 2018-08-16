
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

inPut = sys.argv[1].lower()
outPut = sys.argv[2].lower()


# In[7]:


def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()


def visualize_3d(df, lin_reg_weights=[1,1,1], feat1=0, feat2=1, labels=2,
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):

    # Setup 3D figure
    ax = plt.figure().gca(projection='3d')
    plt.hold(True)

    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array( [lin_reg_weights[0] +
                       lin_reg_weights[1]*f1 +
                       lin_reg_weights[2]*f2  # height
                       for f1, f2 in zip(axes1, axes2)] )
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()


# In[3]:


#data = pd.read_csv('input1.csv', header=None)
data = pd.read_csv(inPut, header=None)


# In[5]:


df = pd.DataFrame(data)
weight_1 = 0
weight_2 = 0
b = 0
conv = False

w1 = []
w2 = []
bs = []

while not conv:
    conv = True
    w1.append(weight_1)
    w2.append(weight_2)
    bs.append(b)
    
    for i in range (len(data)):
        
        if data[2][i]*(weight_1*data[0][i]+weight_2*data[1][i]+b) <= 0:
            weight_1 = weight_1 + data[2][i] * data[0][i]
            weight_2 = weight_2 + data[2][i] * data[1][i]
            b = b + data[2][i] * 1
            conv = False
            
w1.append(weight_1)
w2.append(weight_2)
bs.append(b)            
#text_file = open("Output_1.csv", "w") 
text_file = open(outPut, "w")
for i in range (len(w1)):
    text_file.write(str(int(w1[i])) + ",")
    text_file.write(str(int(w2[i])) + ",")
    text_file.write(str(int(bs[i])) + "\n")
text_file.close()
            

#visualize_scatter(data, weights=[weight_1, weight_2, b])

