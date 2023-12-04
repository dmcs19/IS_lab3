import matplotlib.pyplot as plt
import numpy as np
from random import random

# Preparing data
x = np.arange(0, 1, 1/22)  # Input
# Given formula for desired output
d = ((1 + 0.6*np.sin(2*np.pi*x/0.7)) + (0.3*np.sin(2*np.pi*x)))/2

# Initiate parameters
w1 = random()  # Peso para RBF 1
w2 = random()  # Peso para RBF 2
b = random()   # Bias
c1 = 0.2       # Centro para RBF 1
r1 = 0.2       # Largura para RBF 1
c2 = 0.9        # Centro para RBF 2
r2 = 0.2      # Largura para RBF 2
eta = 0.01      # Learning rate

# Training
test = np.zeros(len(x))
for k in range(1000):
    for i in range(len(x)):
        # Ativações RBF (Gaussian)
        v1 = np.exp(-((x[i]-c1)**2)/(2*r1**2))
        v2 = np.exp(-((x[i]-c2)**2)/(2*r2**2))

        # Output da rede (Linear)
        y = w1 * v1 + w2 * v2 + b
        test[i] = y
        
        # Calculo do erro
        e = d[i] - y

        # Regra de aprendizagem para atualização do peso e bias
        w1 = w1 + eta * e * v1
        w2 = w2 + eta * e * v2
        b = b + eta * e

# Testing
X = np.arange(0, 1, 1/220)
Y = np.zeros(len(X))
for i in range(len(X)):
    # Ativações RBF para teste
    v1 = np.exp(-((X[i]-c1)**2)/(2*r1**2))
    v2 = np.exp(-((X[i]-c2)**2)/(2*r2**2))

    # Output da rede para teste
    Y[i] = w1 * v1 + w2 * v2 + b

plt.plot(x, d, label='Desired Output')
plt.plot(X, Y, 'g', label='Output')
plt.legend()
plt.show()
