
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

inPut = sys.argv[1].lower()
outPut = sys.argv[2].lower()


# In[2]:


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


#data = pd.read_csv('input2.csv', header=None)
data = pd.read_csv(inPut, header=None)
df = pd.DataFrame(data)


# In[4]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 2].values


# In[5]:


#data preprocessing
X_temp = np.copy(X)
meanValue = np.mean(X_temp, axis=0)
stdValue = np.std(X_temp, axis=0)
X_scaled = (X_temp - meanValue) / stdValue
X_inte = np.ones((X.shape[0], X.shape[1]+1))
X_inte[:, 1:X.shape[1]+1] = X_scaled


# In[6]:


def riskFunction(B0, B1, B2, X, y):
    R = 0
    for i in range (len(X)):
        R += (y[i] - (B0 + B1 * X[i][1] + B2 * X[i][2]))**2
        R = R / (2*len(X))
    return R


# In[7]:


def gradientDescent(X, y, B0, B1, B2, alpha):
    b0 = 0
    b1 = 0
    b2 = 0
    n = float(len(X))
    
    for i in range (len(X)):
        b0 += (1/n)*(B0 + B1*X[i][1] + B2*X[i][2] - y[i])
        b1 += (1/n)*(B0 + B1*X[i][1] + B2*X[i][2] - y[i])*X[i][1]
        b2 += (1/n)*(B0 + B1*X[i][1] + B2*X[i][2] - y[i])*X[i][2]
        
    B0 = B0 - alpha*b0
    B1 = B1 - alpha*b1
    B2 = B2 - alpha*b2
        
    return(B0, B1, B2)


# In[8]:


iteration = 100
pick_alpha = 0.8
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, pick_alpha]
betas = []
risks = []

for alpha in alphas:
    B0 = 0
    B1 = 0
    B2 = 0
    for i in range (iteration):
        B0, B1, B2 = gradientDescent(X_inte, y, B0, B1, B2, alpha)
        #risk = riskFunction(B0, B1, B2, X_inte, y)
        
    betas.append([B0, B1, B2])
    #risks.append(risk)
        


# In[9]:


#text_file = open("Output_2.txt", "w") 
text_file = open(outPut, "w") 
for i in range(len(alphas)):
    text_file.write(str(alphas[i]) + ",")
    text_file.write(str(100) + ",")
    text_file.write(str(betas[i][0]) + ",")
    text_file.write(str(betas[i][1]) + ",")
    text_file.write(str(betas[i][2]) + "\n")
text_file.close()

