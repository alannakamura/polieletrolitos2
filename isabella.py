import numpy as np
import matplotlib.pyplot as plt

N = 1001 # N = numero de pontos

c_salt =  6.02e-8
lb =  7.2
k = 8 * np.pi * lb * c_salt
# k = 1
print('k', k)

c = -1 #contorno
d = 0
l = 4
h = l/(N-1)
a = 2+k**2*h**2
A = np.zeros((N,N))
B = np.zeros((N,1))

# dirichlet = [True, True]
dirichlet = [False, False]

if dirichlet[0]:
    A[0, 0] = 1
    B[0] = c
    B[-1] = d
else:
    A[0,0] = -a
    A[0,1] = 1
    B[0] = c * h
    B[-1] = d * h

if dirichlet[1]:
    A[N - 1, N - 1] = 1
else:
    A[N - 1, N - 1] = -a
    A[N - 1, N - 2] = 1

for i in range(1,N-1,1):
    A[i,i] = -a
    A[i,i-1] = 1
    A[i,i+1] = 1

Y = np.linalg.solve(A,B)

print(A, end='\n\n')
print(B, end='\n\n')
print(Y, end='\n\n')

plt.plot(Y)
plt.show()