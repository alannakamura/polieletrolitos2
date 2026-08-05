import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

gamma = 98.04e-4
delta = 8.14e-2
omega = 0.0108
zeta = 216

def fy(y,h,y1,h1):
  return y1
def fh(y,h,y1,h1):
  return h1
def fy1(y,h,y1,h1):
  return gamma * np.sinh(y) + delta * (np.exp(y) - h**2)
def fh1(y,h,y1,h1):
  return omega * (h**3 - h) + zeta * y * h

fx = fy
fy = fh
fvx = fy1
fvy = fh1

ys=-1
x0 = ys
y0 = 0
xn = 0
yn = 1

t0 = 0
tf = 1
nT = 3000
# dt = 1e-4
dt = (tf-t0)/(nT-1)
V = np.random.rand(2,1).ravel()
# V[0] = 0
# V[1] = 6
epsilon = 1e-6
maxError = 1e-3
error = 2*maxError

while abs(error)>maxError:
    t = np.arange(0,tf+dt,dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    # vx[0] = V[0].item()
    # vy[0] = V[1].item()
    vx[0] = V[0]
    vy[0] = V[1]

    for i in range(0, len(t)-1):
        vx[i + 1] = vx[i] + dt / 2 * fvx(x[i], y[i], vx[i], vy[i])
        vy[i + 1] = vy[i] + dt / 2 * fvy(x[i], y[i], vx[i], vy[i])
        x[i + 1] = x[i] + dt * fx(x[i], y[i], vx[i + 1], vy[i + 1])
        y[i + 1] = y[i] + dt * fy(x[i], y[i], vx[i + 1], vy[i + 1])
        vx[i + 1] = vx[i + 1] + dt / 2 * fvx(x[i + 1], y[i + 1], vx[i], vy[i])
        vy[i + 1] = vy[i + 1] + dt / 2 * fvy(x[i + 1], y[i + 1], vx[i], vy[i])
    F = np.array([x[-1]-xn, y[-1]-yn])


    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = V[0].item()+epsilon/2
    vy[0] = V[1].item()

    for i in range(0, len(t)-1):
        vx[i + 1] = vx[i] + dt / 2 * fvx(x[i], y[i], vx[i], vy[i])
        vy[i + 1] = vy[i] + dt / 2 * fvy(x[i], y[i], vx[i], vy[i])
        x[i + 1] = x[i] + dt * fx(x[i], y[i], vx[i + 1], vy[i + 1])
        y[i + 1] = y[i] + dt * fy(x[i], y[i], vx[i + 1], vy[i + 1])
        vx[i + 1] = vx[i + 1] + dt / 2 * fvx(x[i + 1], y[i + 1], vx[i], vy[i])
        vy[i + 1] = vy[i + 1] + dt / 2 * fvy(x[i + 1], y[i + 1], vx[i], vy[i])
    F3 = np.array([x[-1]-xn, y[-1]-yn])

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = V[0].item() - epsilon / 2
    vy[0] = V[1].item()

    for i in range(0, len(t) - 1):
        vx[i + 1] = vx[i] + dt / 2 * fvx(x[i], y[i], vx[i], vy[i])
        vy[i + 1] = vy[i] + dt / 2 * fvy(x[i], y[i], vx[i], vy[i])
        x[i + 1] = x[i] + dt * fx(x[i], y[i], vx[i + 1], vy[i + 1])
        y[i + 1] = y[i] + dt * fy(x[i], y[i], vx[i + 1], vy[i + 1])
        vx[i + 1] = vx[i + 1] + dt / 2 * fvx(x[i + 1], y[i + 1], vx[i], vy[i])
        vy[i + 1] = vy[i + 1] + dt / 2 * fvy(x[i + 1], y[i + 1], vx[i], vy[i])
    F2 = np.array([x[-1] - xn, y[-1] - yn])

    J = np.zeros((2,2))
    J[:,0] = (F3 - F2)/epsilon
    # print(F2,F,J)

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = V[0].item()
    vy[0] = V[1].item()+epsilon/2

    for i in range(0, len(t)-1):
        vx[i + 1] = vx[i] + dt / 2 * fvx(x[i], y[i], vx[i], vy[i])
        vy[i + 1] = vy[i] + dt / 2 * fvy(x[i], y[i], vx[i], vy[i])
        x[i + 1] = x[i] + dt * fx(x[i], y[i], vx[i + 1], vy[i + 1])
        y[i + 1] = y[i] + dt * fy(x[i], y[i], vx[i + 1], vy[i + 1])
        vx[i + 1] = vx[i + 1] + dt / 2 * fvx(x[i + 1], y[i + 1], vx[i], vy[i])
        vy[i + 1] = vy[i + 1] + dt / 2 * fvy(x[i + 1], y[i + 1], vx[i], vy[i])
    F3 = np.array([x[-1]-xn, y[-1]-yn])

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    vx[0] = V[0].item()
    vy[0] = V[1].item() - epsilon/2

    for i in range(0, len(t) - 1):
        vx[i + 1] = vx[i] + dt / 2 * fvx(x[i], y[i], vx[i], vy[i])
        vy[i + 1] = vy[i] + dt / 2 * fvy(x[i], y[i], vx[i], vy[i])
        x[i + 1] = x[i] + dt * fx(x[i], y[i], vx[i + 1], vy[i + 1])
        y[i + 1] = y[i] + dt * fy(x[i], y[i], vx[i + 1], vy[i + 1])
        vx[i + 1] = vx[i + 1] + dt / 2 * fvx(x[i + 1], y[i + 1], vx[i], vy[i])
        vy[i + 1] = vy[i + 1] + dt / 2 * fvy(x[i + 1], y[i + 1], vx[i], vy[i])
    F2 = np.array([x[-1] - xn, y[-1] - yn])



    J[:,1] = (F3 - F2)/epsilon
    # print(F2,F,J)

    dV = solve(J, -F)
    # print(dV)
    V = V + 1.0*dV
    error = norm(dV)
    print(error)

print(min(x), max(x), x)
print(min(y), max(y), y)
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x')
plt.show()
plt.plot(t,y)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

