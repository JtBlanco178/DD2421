import numpy , random , math
import numpy as np
from scipy . optimize import minimize
import matplotlib . pyplot as plt
import scipy


# Define x and target as global variables
# x = numpy.random.normal(size=(, ))
# t =
p = []
P = np.array(p)
#targets = np.array(targets)
C = 9
# Kernal Functions depending on type Equations from 3.3
def kernel(x, y, type):
    if(type == 'Linear'):
        return numpy.dot(x.T, y)
    elif(type == 'Poly'):
        tmp = numpy.dot(x.T, y)
        tmp += 1
        return tmp ** 2 # p == 2
    elif(type == 'RBF'):
        num = abs(x - y) ** 2
        den = 2 * numpy.var(x)
        return numpy.exp(-(num/den))

# Objective Function Equation #4
# CHANGE: Only take in x
def objective(x, alpha):
    alpha_sum = numpy.sum(alpha)
    tmp = 0
    tmp = np.sum(alpha * alpha * kernel(x, x, 'Linear'))
    # for i in range(len(x)):
    #     for j in range(len(x)):
    #        tmp += alpha[i] *  alpha[j] * kernel(x[i], x[j])
    return (tmp / 2) - alpha_sum

# Zeros Function for Constraints
def zerofun(alpha):
    # Find sum of all dots with alpha and targets equal to zero
    func = scipy.optimize.LinearConstraint(np.dot(alpha, targets), 0, 0)
    return func


# Extract Non Zeroes
def nonzero_alpha(alpha):
    a = []
    a = np.array(a)
    for i in alpha:
        if alpha[i] <= 10 ** -5:  # Scalar for removing all float approximate zero
            a.append(alpha[i])
    return a

def indicator(x, targets, alpha, bias, s):
    total, tmp = 0, 0
    # Get b value
    for i in range(len(s)):
        tmp += alpha[i] * targets[i] * kernel(s, x[i], 'Poly')
    b = tmp - targets
    # Get result from indicator function
    for i in range(len(s)):
        tmp = alpha[i] * targets[i] * kernel(s, x[i], 'Poly')
        total += tmp
    total -= b


# Data generation (Source: Lab 2 instructions)

classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
inputs = numpy.concatenate((classA , classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N = inputs.shape [0] # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# Plotting (Source: Lab 2 instructions)

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal') # Force same scale on both axes
#plt.savefig('lab2/plots/svmplot.pdf') # Save a copy in a file
plt.show() # Show the plot on the screen

# Decision Boundary


alpha = np.zeros(N)
#alpha = [5+x for x in targets]
print(alpha)
alpha = np.array(alpha)
P = np.dot(targets, kernel(inputs, inputs, 'Linear'))
bounds = [(0, C) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun(alpha)}
scipy.optimize.minimize(objective(inputs, alpha), alpha, bounds=bounds, constraints=XC)

xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)
grid=numpy.array([[indicator(x,y)
                    for x in xgrid ]
                    for y in ygrid ] )
plt.contour(xgrid, ygrid , grid ,(-1.0, 0.0, 1.0),
        colors=('red' , 'black', 'blue'),
        linewidths = (1, 3, 1))