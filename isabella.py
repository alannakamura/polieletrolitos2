import numpy as np
import matplotlib.pyplot as plt

N = 101 # N = numero de pontos

c_salt =  6.02e-8
lb =  7.2
k = 8 * np.pi * lb * c_salt
# k = 1
print('k', k)

c = -1 #contorno
# d = 2
d = 0
l = 4
h = l/(N-1)
a = 2+k*h**2
A = np.zeros((N,N))
B = np.zeros(N)

# dirichlet = [True, True]
dirichlet = [False, False]

if dirichlet[0]:
    A[0, 0] = 1
    B[0] = c
else:
    A[0,0] = -a
    A[0,1] = 2
    B[0] = 2 * c * h

if dirichlet[1]:
    A[N - 1, N - 1] = 1
    B[-1] = d
else:
    A[N - 1, N - 1] = -a
    A[N - 1, N - 2] = 2
    B[-1] = -2 * d * h

for i in range(1,N-1,1):
    A[i,i] = -a
    A[i,i-1] = 1
    A[i,i+1] = 1

Y = np.linalg.solve(A,B)

def analitica_neumann(l, N, c, d, k):
    x = np.linspace(0, l, N)

    y = d*np.cosh(np.sqrt(k)*(x-x[0])) - c*np.cosh(np.sqrt(k)*(x[-1]-x))
    y /= np.sqrt(k)*np.sinh(np.sqrt(k)*(l))
    return x, y

def analitica_dirichlet(l, N, c, d, k):
    x = np.linspace(0, l, N)
    k2 = np.sqrt(k)
    y =  c * np.sinh(k2 * (x[-1] - x))
    y += d * np.sinh(k2 * (x - x[0]))
    y /= np.sinh(k2*(x[-1] - x[0]))

    return x, y

print(A, end='\n\n')
print(B, end='\n\n')
print(Y, end='\n\n')

if dirichlet[0] == False and dirichlet[1] == False:
    x, Y2 = analitica_neumann(l, N, c, d, k)
elif dirichlet[0] == True and dirichlet[1] == True:
    x, Y2 = analitica_dirichlet(l, N, c, d, k)

# dY = Y-Y2
# dY = np.abs(dY)
# v_max =  np.max(dY)
# i_max = np.where(dY == v_max)
# print(i_max, Y[i_max], Y2[i_max])
# print(Y2-Y, end='\n\n')
print(np.linalg.norm(Y2-Y, ord=2),
      np.linalg.norm(Y2-Y, ord=np.inf), end='\n\n')

plt.plot(x, Y, 'b', x, Y2, 'r')
plt.show()