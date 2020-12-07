import numpy as np
import math
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(19970922)

#### Things to implement

## Generate data

classA = np.concatenate((
    np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    ))
classB=np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs=np.concatenate((classA, classB))
targets=np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]

permute = list(range(N))
random.shuffle(permute)
x = inputs[permute,:]
t = targets[permute]


## Kernel functions

# Linear kernel

def linearKernel(x,y):
    return np.dot(x,y)

def polynomialKernel(x,y,p):
    return math.pow(np.dot(x, y) + 1, p)

def rbfKernel(x, y,sigma):
    return math.exp(-math.pow(np.linalg.norm(np.subtract(x, y)), 2)/(2 * math.pow(sigma,2)))


# Objective function

P = np.zeros((N,N))
for i in range(N):
    for j in range(N):
                                #Change kernel for tests
        P[i][j] = t[i] * t[j] * rbfKernel(x[i],x[j],4)


def objective(alpha):
    suma = 0.0
    for i in range(N):
        for j in range(N):
            suma += alpha[i] * alpha[j] * P[i][j]
    return (1/2) * suma - np.sum(alpha)


# Zerofun function

def zerofun(alpha):
    suma = 0.0
    for i in range(N):
        suma += alpha[i] * t[i]
    return suma

## Call minimize

# Constants for calling minimize

C = 100 #importance of avoiding slack, noisy data = lower C
start = np.zeros(N) # initial alpha guesses
bound = [(0,C) for b in range(N)] #upperbound
lowerbound = [(0,None) for b in range(N)] #lowerbound only if wanted
contraints = {'type':'eq', 'fun':zerofun} #given constraints

ret = minimize(objective,start,bounds=bound,constraints=contraints)

#Extract non zero alphas, x and t values
alphas = ret['x']
nzm = [(alphas[i], x[i], t[i]) for i in range(N) if abs(alphas[i]) > 10e-5]

## Calculate b values using the no zeros

def bfun():
    b_val = 0.0
    for i in nzm:
        #nzm[0] = alphas, nzm[1] = x, nzm[2] = t
                                #Change kernel here
        b_val += i[0] * i[2] * rbfKernel( i[1], nzm[0][1],4)
        #print(nzm[0][1])
    return b_val - nzm[0][2]


##Indicator function

#Takes x,y, new points and classifies em
def indicator(x,y,b):
    ind_val = 0.0
    for i in nzm:
        #nzm[0] = alphas, nzm[1] = x, nzm[2] = t
                                #Change kernel here
        ind_val += i[0] * i[2] * rbfKernel( i[1],[x,y],4)
    return ind_val - b

## Plot data with decision boundry using linear kernel

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
xgrid = np.linspace(-5,5)
ygrid = np.linspace(-4,4)

grid = np.array([[indicator(x, y, bfun()) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))

plt.axis('equal')
plt.show()

## Begin exploration and repporting