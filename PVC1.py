import numpy as np
import matplotlib.pyplot as plt

n = 5
a = 3
b = 4
c = 3
x0 = -1
xn = 1
x = np.linspace(x0,xn,n)
dx = x[1]-x[0]
d = np.ones((n-2,1))*dx**2 * c
d[0] -= a
d[-1] -= b
A1 = np.diag(np.ones(n-2)*-2)
A2 = np.diag(np.ones(n-3), 1)
A3 = np.diag(np.ones(n-3), -1)
A = A1+A2+A3
y = np.linalg.solve(A, d)
v = np.linspace(a,b,n)
v .shape = (n, 1)
v [1:-1] = y
print(A)
print(d)
print(v)
print(dx, dx**2*3)

plt.title('n='+str(n))
plt.plot(x,v, 'r')
# x2 = np.linspace(x0,xn,10000)
# plt.plot(x2,1.5*x2**2+0.5*x2+2,'r')
plt.show()