import numpy as np
import matplotlib.pyplot as plt

# acrescentei
def analitica(t, k, L, q0, qL):

    sqrt_k = np.sqrt(k)
    A = qL - q0 * np.cosh(sqrt_k * L)
    A /=  sqrt_k * np.sinh(sqrt_k * L)
    B = q0 / sqrt_k

    return A * np.cosh(sqrt_k * t) + B * np.sinh(sqrt_k * t)

N = 101 # N = numero de pontos
k= 1e-6
# k=1
c=-1 #contorno
# acrescentei
d = -1
l= 4
h= l/(N-1)
# a= 2+k**2*h**2
a= 2+k*h**2
A= np.zeros((N,N))

# A[0,0] = -a
# A[0,1] = 1
# A[N-1, N-1] = -1
# A[N-1, N-2] = 1
A[0,0] = -1
A[0,1] = 1
A[N-1, N-1] = 1
A[N-1, N-2] = -1

B= np.zeros((N,1))
B[0] = c * h
# acrescentei
B[-1] = d * h

for i in range(1,N-1,1):
    A[i,i] = -a
    A[i,i-1] = 1
    A[i,i+1] = 1

Y = np.linalg.solve(A,B)

# print(A, end='\n\n')
# print(B, end='\n\n')
# print(Y, end='\n\n')

# acrescentei
t = np.linspace(0, l, len(Y))
Y2 = analitica(t, k, l, c, d)

Y.shape = N,1
Y2.shape = N,1
print(np.linalg.norm(Y-Y2))
plt.plot(t, Y, 'r', t, Y2, 'b')
plt.legend(['numerico','analitico'])
plt.show()