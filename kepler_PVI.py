import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

G = 4*np.pi**2
m = 3.003e-6
M = 1

def fx(x,y,vx,vy):
  #print(px/m)
  return vx
def fy(x,y,vx,vy):
  #print(py/m)
  return vy
def fvx(x,y,vx,vy):
  #print(-x*G*np.sqrt(x**2+y**2)**3)
  return G*M*(-x)/(np.sqrt(x**2+y**2))**3
def fvy(x,y,vx,vy):
  return G*M*(-y)/(np.sqrt(x**2+y**2))**3

x0 = 0.9833
y0 = 0
vx0 = 0
vy0 = 6.389

t0 = 0
tf = 1
dt = 1e-2
t = np.arange(0,tf+dt,dt)
x = np.zeros(len(t))
y = np.zeros(len(t))
vx = np.zeros(len(t))
vy = np.zeros(len(t))
x[0] = x0
y[0] = y0
vx[0] = vx0
vy[0] = vy0

#verlet
for i in range(0, len(t)-1):
  vx[i+1] = vx[i] + dt/2*fvx(x[i],y[i],vx[i],vy[i])
  vy[i+1] = vy[i] + dt/2*fvy(x[i],y[i],vx[i],vy[i])
  x[i+1] = x[i] + dt* fx(x[i],y[i],vx[i+1],vy[i+1])
  y[i+1] = y[i] + dt* fy(x[i],y[i],vx[i+1],vy[i+1])
  vx[i + 1] = vx[i+1] + dt/2 * fvx(x[i+1], y[i+1], vx[i], vy[i])
  vy[i + 1] = vy[i+1] + dt/2 * fvy(x[i+1], y[i+1], vx[i], vy[i])

plt.plot(x,y)
plt.show()