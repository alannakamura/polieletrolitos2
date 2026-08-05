import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

"""Parâmetros"""

params = {
    # Tipo de condição de contorno: 'potential' (y(0)=y_s) ou 'charge' (y'(0)=yprime0)
    'bc_type': 'potential',

    # Se 'potential', fixe o potencial superficial (adimensional) y(0)=y_s

    'lb':0,

    'y_s': -1,  # típico: -0.5 a -2 (em unidades e*psi/kBT)1
    # Se 'charge', fixe a derivada do potencial na parede
    'yprime0': 1.0,

    'a': 5,
    'c_s': 6.02e-8,
    'f':1,
    'phib2':1e-6,
    'v':50,
    'e':80,
    'T':300,
    'lb':7.2,


    # Parâmetros adimensionais do modelo (valores exemplo do artigo)
    # 'gamma': 98.04e-4,   # ~  (kappa.D)^2
    # 'delta': 8.14e-2,    # ~ (kappa_M.D)^2
    # 'omega': 0.01080,   # (6/a^2).D^2.(v.phi_b)^2
    # 'zeta': 216.0, # (6/a^2).D^2.f

    # Grade numérica
    # 'Xmax': 2.0,   # extensão do domínio (em unidades de 1/kappa)
    'nx': 101
}

# params['D'] = params['lb']

normalization = 0

if normalization == 0:
    params['D'] = 30
elif normalization == 1:
    params['D'] = params['lb']
elif normalization == 2:
    params['D'] = 1

exemplo = 1

if exemplo == 2:
    params['y_s'] = -0.5
elif exemplo == 3:
    params['y_s'] = -0.5
    params['omega']/= 4
    params['zeta'] /= 4
elif exemplo == 4:
    params['y_s'] = -0.5
    params['zeta'] /= 10
elif exemplo == 5:
    params['y_s'] = -0.5
    params['omega'] /= 1
    params['zeta'] /= 1

params['k'] = np.sqrt(8 * np.pi * params['lb'] * params['c_s'])
params['km'] = np.sqrt(4 * np.pi * params['lb'] * params['phib2'] * params['f'])
params['gamma'] = (params['D']*params['k'])**2
params['delta'] = (params['D']*params['km'])**2
params['omega'] = 6*(params['D']/params['a'])**2*params['v']*params['phib2']
params['zeta'] = 6*(params['D']/params['a'])**2*params['f']
params['Xmax'] = 60/params['D']


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
        # // nao teria q ser Y[i](y) em vez de Y[i+1](eta)?
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

arquivo = 'res.pkl'
if os.path.exists(arquivo):
    f = open(arquivo,"rb")
    l = pickle.load(f)
    l.append([X, Y])
    f.close()
    f = open(arquivo,'wb')
    pickle.dump(l, f)
    f.close()
else:
    f = open(arquivo, "wb")
    pickle.dump([[X, Y]], f)


plt.subplot(1,2,1)
plt.plot(X,Y[0::2],'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
# plt.show()
plt.subplot(1,2,2)
plt.plot(X,Y[1::2]**2,'r-', X, np.ones(len(X)),'b')
plt.xlabel('x')
plt.ylabel('$\\eta^{2}$')
plt.legend([r'$\eta^{2}(x)$',r'$\eta^{2}(x)$ = 1'])
plt.show()