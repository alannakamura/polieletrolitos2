import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

G = 4*np.pi**2
M = 1

def fx(x,y,vx,vy):
  return vx
def fy(x,y,vx,vy):
  return vy
def fvx(x,y,vx,vy):
  return G*M*(-x)/(np.sqrt(x**2+y**2))**3
def fvy(x,y,vx,vy):
  return G*M*(-y)/(np.sqrt(x**2+y**2))**3

ecc = 0.0167086
x0 = (1-ecc)
y0 = 0
xn = -1*(1+ecc)
yn = 0

t0 = 0
tf = 0.5
dt = 1e-4
V = np.random.rand(2,1).ravel()
V[0] = 0
V[1] = 6
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
    print(error, np.linalg.cond(J))

print(min(x), max(x), x)
print(min(y), max(y), y)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# plt.plot(t,x)
# plt.xlabel('t')
# plt.ylabel('x')
# plt.show()
# plt.plot(t,y)
# plt.xlabel('t')
# plt.ylabel('y')
# plt.show()

