import numpy as np
import matplotlib.pyplot as plt

"""Parâmetros"""

params = {
    # Tipo de condição de contorno: 'potential' (y(0)=y_s) ou 'charge' (y'(0)=yprime0)
    'bc_type': 'potential',

    # Se 'potential', fixe o potencial superficial (adimensional) y(0)=y_s
    'y_s': -1.0,  # típico: -0.5 a -2 (em unidades e*psi/kBT)

    # Se 'charge', fixe a derivada do potencial na parede
    'yprime0': 1.0,

    # Parâmetros adimensionais do modelo (valores exemplo do artigo)
    'gamma': 98.04e-4,   # ~  (kappa.D)^2
    'delta': 8.14e-2,    # ~ (kappa_M.D)^2
    'omega': 0.01080,   # (6/a^2).D^2.(v.phi_b)^2
    'zeta': 216.0, # (6/a^2).D^2.f

    # Grade numérica
    'Xmax': 2.0,   # extensão do domínio (em unidades de 1/kappa)
    'nx': 101
}

"""Sistema de EDOs"""

# def odes(X, Y, gamma, delta, omega, zeta):
#     y, y1, h, h1 = Y
#     dy_dx = y1
#     dy1_dx = params['gamma'] * np.cosh(y) + delta * (np.exp(y) - h**2)
#     dh_dx = h1
#     dh1_dx = omega * (h**3 - h) + zeta * y * h
#     return np.vstack((dy_dx, dy1_dx, dh_dx, dh1_dx))

N = params['nx']
R = np.zeros(2*N)
Y = np.zeros(2*N)
J = np.zeros((2*N, 2*N))
X = np.linspace(0, params['Xmax'], N)
dx = X[1]-X[0]
dy = np.ones(N)
iter = 0

J[0,0] = 1
J[1,1] = 1
J[2*N-1, 2*N-1] = 1
J[2*N-2, 2*N-2] = 1

# A = 3.0
# alpha = 0.6406
# beta  = 2.4453
# def eta(A2, alpha2, beta2, Y2):
#     return A2*Y2**alpha2*np.exp(-beta2*Y2)
# Y[3:-2:2] = eta(A, alpha, beta, X[1:-1])
#
# A3 = 0.69
# beta3 = 10.0
# def y(A4, beta4, Y2):
#     return -1+A4*(1-np.exp(-beta4*Y2))
# Y[2:-3:2] = y(A3, beta3, X[1:-1])
#
# Y[0] = -1
# Y[-1] = 1
#
# plt.subplot(1, 2, 1)
# plt.plot(X, Y[0::2], 'ro-')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.subplot(1, 2, 2)
# plt.plot(X, Y[1::2], 'ro-')
# plt.xlabel('x')
# plt.ylabel('$\\eta(x)$')
# plt.show()

# exit(0)
while np.linalg.norm(dy)>1e-6:
    print('iter', iter, 'error', np.linalg.norm(dy))
    R[0] = Y[0] - params['y_s']
    R[1] = Y[1]
    R[-2] = Y[-2]
    R[-1] = Y[-1] -1

    for i in range(2, 2*N-2, 2):

        J[i, i - 2] = 1
        J[i, i] = -2 - dx**2 * params['gamma'] * np.cosh(Y[i])
        J[i, i] -= dx**2 * params['delta'] * np.exp(Y[i])
        J[i, i + 1] = - dx**2*params['delta']*(-2*Y[i+1])
        J[i, i + 2] = 1

        J[i + 1, i - 1] = 1
        J[i + 1, i] = -dx**2 * params['zeta'] * Y[i+1]
        J[i + 1, i + 1] = -2 - 3 * dx**2*params['omega']*Y[i+1]**2
        J[i + 1, i + 1] += dx**2*params['omega']-dx**2*params['zeta']*Y[i+1]
        # J[i + 1, i + 1] += dx ** 2 * params['omega'] - dx ** 2 * params['zeta'] * Y[i]
        J[i + 1, i + 3] = 1

        R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * params['gamma'] * np.sinh(Y[i])
        R[i] -= dx**2 * params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
        R[i+1] = Y[i-1] -2*Y[i+1] + Y[i+3] - dx**2*params['omega']*(Y[i+1]**3-Y[i+1])
        R[i+1] -= dx**2*params['zeta']*Y[i]*Y[i+1]

    dy = np.linalg.solve(J, -R)
    Y = Y +dy
    iter+=1
    # plt.subplot(1, 2, 1)
    # plt.plot(X, Y[0::2], 'ro-')
    # plt.xlabel('x')
    # plt.ylabel('y(x)')
    # plt.subplot(1, 2, 2)
    # plt.plot(X, Y[1::2], 'ro-')
    # plt.xlabel('x')
    # plt.ylabel('$\\eta(x)$')
    # plt.show()

# plt.subplot(1,3,1)
# plt.plot(X,Y[0::2],'ro-')
# plt.xlabel('x')
# plt.ylabel('y(x)')
# plt.subplot(1,3,2)
# plt.plot(X,Y[1::2]**2,'ro-', X, np.ones(len(X)),'b')
# plt.xlabel('x')
# plt.ylabel('$\\eta^{2}$')
# plt.legend(['eta**2','eta**2 = 1'])
# plt.subplot(1,3,3)
# plt.plot(X,Y[1::2]**2/max(Y[1::2]**2),'ro-',X, np.zeros(len(X)),'b')
# plt.xlabel('x')
# plt.ylabel('$\\eta^{2}$')
# plt.legend(['eta**2','eta**2 = 0'])
# plt.show()
# print(J)
# print(R)

plt.subplot(1,2,1)
plt.plot(X,Y[0::2],'ro-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.subplot(1,2,2)
plt.plot(X,Y[1::2]**2,'ro-', X, np.ones(len(X)),'b')
plt.xlabel('x')
plt.ylabel('$\\eta^{2}$')
plt.legend(['eta**2','eta**2 = 1'])
plt.show()