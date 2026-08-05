import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

"""Parâmetros"""

params = {
    # Tipo de condição de contorno: 'potential' (y(0)=y_s) ou 'charge' (y'(0)=yprime0)
    'bc_type': 'potential',

    # Se 'potential', fixe o potencial superficial (adimensional) y(0)=y_s
    'a': 5,
    'c_salt': 6.02e-8,
    'f':1,
    'y_s': -1,  # típico: -0.5 a -2 (em unidades e*psi/kBT)1
    'phib2':1e-6,
    'v':50,
    'e':80,
    'T':300,
    'lb':7.2,

    # Se 'charge', fixe a derivada do potencial na parede
    'yprime0': 1.0,


    # Grade numérica
    'Xmax': 60.0,
    'nx': 101
}

# params['D'] = params['lb']

normalization = 2

if normalization == 0:
    params['D'] = 30
elif normalization == 1:
    params['D'] = params['lb']
elif normalization == 2:
    params['D'] = 1

exemplo = 1

if exemplo == 2:
    params['a'] = 5
    params['f'] = 1
    params['y_s'] = -0.5
elif exemplo == 3:
    params['a'] = 10
    params['f'] = 1
    params['y_s'] = -0.5
elif exemplo == 4:
    params['a'] = 5
    params['f'] = 0.1
    params['y_s'] = -0.5
elif exemplo == 5:
    params['y_s'] = -0.5
    params['c_salt'] *= 700
    params['f'] = 0.12

params['k2'] = 8 * np.pi * params['lb'] * params['c_salt']
params['km2'] = 4 * np.pi * params['lb'] * params['phib2'] * params['f']
params['gamma'] = params['k2']*params['D']**2
params['delta'] = params['km2']*params['D']**2
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

    delta = 1e-6
    for i in range(2, 2*N-2, 2):
        R[i] = Y[i - 2] - 2 * Y[i] + Y[i + 2] - dx ** 2 * params['gamma'] * np.sinh(Y[i])
        R[i] -= dx ** 2 * params['delta'] * (np.exp(Y[i]) - Y[i + 1] ** 2)
        R[i + 1] = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
        R[i + 1] -= dx ** 2 * params['zeta'] * Y[i] * Y[i + 1]

        J[i, i - 2] = 1

        r1 = Y[i - 2] - 2 * (Y[i]) + Y[i + 2] - dx ** 2 * params['gamma'] * np.sinh(Y[i])
        r1 -= dx ** 2 * params['delta'] * (np.exp(Y[i]) - (Y[i + 1] + delta) ** 2)
        r2 = Y[i - 2] - 2 * (Y[i]) + Y[i + 2] - dx ** 2 * params['gamma'] * np.sinh(Y[i])
        r2 -= dx ** 2 * params['delta'] * (np.exp(Y[i]) - (Y[i + 1]- delta) ** 2)
        J[i, i+1] = (r1 - r2) / (2 * delta)

        r1 = Y[i - 2] - 2 * (Y[i]+delta) + Y[i + 2] - dx ** 2 * params['gamma'] * np.sinh(Y[i]+delta)
        r1 -= dx ** 2 * params['delta'] * (np.exp(Y[i]+delta) - Y[i + 1] ** 2)
        r2 = Y[i - 2] - 2 * (Y[i] - delta) + Y[i + 2] - dx ** 2 * params['gamma'] * np.sinh(Y[i] - delta)
        r2 -= dx ** 2 * params['delta'] * (np.exp(Y[i] - delta) - Y[i + 1] ** 2)
        J[i, i] = (r1 - r2) / (2 * delta)

        J[i, i + 2] = 1

        J[i + 1, i - 1] = 1

        # J[i + 1, i] = -dx**2 * params['zeta'] * Y[i+1]
        r1 = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
        r1 -= dx ** 2 * params['zeta'] * (Y[i] + delta) * Y[i + 1]
        r2 = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
        r2 -= dx ** 2 * params['zeta'] * (Y[i] - delta) * Y[i + 1]
        J[i + 1, i] = (r1 - r2) / (2 * delta)

        J[i + 1, i + 1] = -2 - 3 * dx**2*params['omega']*Y[i+1]**2
        J[i + 1, i + 1] += dx**2*params['omega']-dx**2*params['zeta']*Y[i+1]
        # Y[i + 1] += delta
        # r1 = Y[i - 1] - 2 * (Y[i + 1]) + Y[i + 3] - dx ** 2 * params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
        # r1 -= dx ** 2 * params['zeta'] * (Y[i]) * (Y[i + 1])
        # Y[i+1] -= 2*delta
        # r2 = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
        # r2 -= dx ** 2 * params['zeta'] * (Y[i]) * (Y[i + 1])
        # Y[i+1] += delta
        # J[i + 1, i + 1] = (r1 - r2) / (2 * delta)

        J[i + 1, i + 3] = 1

    dy = np.linalg.solve(J, -R)
    Y = Y +dy
    iter+=1

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