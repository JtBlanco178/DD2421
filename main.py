import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy

# Define x and target as global variables
# x = np.random.normal(size=(, ))
# t 3
C = 100 # C parameter
kernel_ = 'Poly'


# Kernel Functions depending on type Equations from 3.3
def kernel(x, y):
    if (kernel_ == 'Linear'):
        return np.dot(x.T, y)
    elif (kernel_ == 'Poly'):
        tmp = np.dot(x.T, y)
        tmp += 1
        return tmp ** 5  # p == 2
    elif (kernel_ == 'RBF'):
        #print(x, y)
        num = (x[0] - y[0]) ** 2
        num += (x[1] - y[1]) ** 2
        #num = np.sqrt(num)
        den = 2 * 5  # scale to a parameter
        #print(np.exp(-(num / den)))
        return np.exp(-(num / den))


# Objective Function Equation #4
def objective(alpha):
    alpha_sum = np.sum(alpha)
    tmp = 0
    # alpha_dot = np.dot(alpha, alpha)
    # tmp = np.dot(alpha_dot, P)
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            tmp += alpha[i] * alpha[j] * P[i][j]
    return (tmp / 2) - alpha_sum


# Equation 10
def zerofun(alpha):
    # REQUIRES: alpha vector
    # EFFECTS: Returns constraints
    return sum([alpha[i] * targets[i] for i in range(N)])


def nonzero_alpha(inputs, alpha, targets):
    inp = []
    a = []
    t = []
    for i in range(len(alpha)):
        if alpha[i] >= 10 ** -5:  # Scalar for removing all float approximate zero
            a.append(alpha[i])
            inp.append(inputs[i])
            t.append(targets[i])
    return inp, a, t


def b():
    return sum(support_vector['alpha'][i] * support_vector['targets'][i] * kernel(support_vector['inputs'][0],
                                                                                  support_vector['inputs'][i]) for i in
               range(len(support_vector['alpha']))) - support_vector['targets'][0]


# Equation 6
def indicator(x, y):  # TODO: Debug this
    total = 0
    for i in range(len(support_vector['alpha'])):
        total += support_vector['alpha'][i] * support_vector['targets'][i] * kernel(np.array([x, y]),
                                                                                    support_vector['inputs'][i])
    total -= b()
    return total


# Data generation (Source: Lab 2 instructions)
def generate_data():
    classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [0.0, -0.1], np.random.randn(10, 2) * 0.2 + [-1.5, 0.0]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0]  # Number of rows (samples)
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB


# Plotting (Source: Lab 2 instructions)

def plot(classA, classB):
    # Data
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

    # Decision Boundary
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
    print("x: ", xgrid)
    print("y: ", ygrid)
    print("grid: ", grid)
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Axis + Save Plot
    plt.axis('equal')  # Force same scale on both axes
    plt.savefig('./plots/svmplot_3.pdf')  # Save a copy in a file
    #plt.title('RBF', 'center')
    plt.show()  # Show the plot on the screen


inputs, targets, classA, classB = generate_data()
N = inputs.shape[0]  # Number of inputs
start = np.zeros(N)  # First alpha vector
alpha = start  # Initial alpha vector N list of 0s

P = np.zeros((N, N))  # Pre-computed matrix

for i in range(N):
    for j in range(N):
        P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])

# Calling minimize function
bounds = [(0, C) for _ in range(N)]
XC = {'type': 'eq', 'fun': zerofun}
ret = minimize(objective, start, bounds=bounds, constraints=XC)
alpha = ret['x']

# Extract Non-zero values
inputs, alpha, targets = nonzero_alpha(inputs, alpha, targets)
print(alpha)
# Global directories alpha and targets
# Indicator function passing x and y
# Therefore can calculate sv values by modifying support vector director

# support_vector = {'alpha': alpha, 'inputs': inputs, 'targets': targets}
support_vector = {'alpha': [], 'inputs': [], 'targets': []}
for i in range(len(alpha)):
    support_vector['alpha'].append(alpha[i])
    support_vector['inputs'].append(inputs[i])
    support_vector['targets'].append(targets[i])

plot(classA, classB)