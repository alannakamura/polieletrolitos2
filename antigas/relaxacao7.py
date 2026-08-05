import matplotlib.pyplot as plt
import pickle
import os

gpu  = False

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
    'nx': 100,
    'relaxation_tax': 0.007,
}

# params['D'] = params['lb']

if gpu:
    import cupy as np
else:
    import numpy as np

normalization = 2

if normalization == 0:
    params['D'] = 30
elif normalization == 1:
    params['D'] = params['lb']
elif normalization == 2:
    params['D'] = 1

exemplo = 12

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
    params['Xmax'] = 150
    params['y_s'] = -0.5
    params['c_salt'] *= 700
    params['f'] = 0.12
elif exemplo == 6:
    params['Xmax'] = 150
    params['y_s'] = -0.5
    params['c_salt'] *= 700
    params['f'] = 0.1
elif exemplo == 7:
    params['Xmax'] = 150
    params['y_s'] = -0.5
    params['c_salt'] *= 700
    params['f'] = 0.09
elif exemplo == 8:
    params['Xmax'] = 150
    params['y_s'] = -0.5
    params['c_salt'] *= 700
    params['f'] = 0.08
elif exemplo == 9:
    params['c_salt'] *= 10
elif exemplo == 10:
    params['c_salt'] *= 100
elif exemplo == 11:
    params['c_salt'] *= 1e-1
elif exemplo == 12:
    params['c_salt'] *= 1e-2

params['k2'] = 8 * np.pi * params['lb'] * params['c_salt']
params['km2'] = 4 * np.pi * params['lb'] * params['phib2'] * params['f']
params['gamma'] = params['k2']*params['D']**2
params['delta'] = params['km2']*params['D']**2
params['omega'] = 6*(params['D']/params['a'])**2*params['v']*params['phib2']
params['zeta'] = 6*(params['D']/params['a'])**2*params['f']
params['Xmax'] /= params['D']

# params['omega'] = 0
# params['zeta'] = 0
# params['delta'] = 0
# params['gamma'] = 0
# params['gamma']*=-1
# params['delta']*=-1

# params['zeta']*=-1

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
    # R[0] = Y[0] - (-np.sinh(0))
    R[1] = Y[1]
    # R[1] = Y[1] - 0
    R[-2] = Y[-2]
    # R[-2] = Y[-2] - (-np.sinh(2))
    R[-1] = Y[-1] -1
    # R[-1] = Y[-1]

    for i in range(2, 2*N-2, 2):

        J[i, i - 2] = 1
        J[i, i] = -2 - dx**2 * params['gamma'] * np.cosh(Y[i])
        J[i, i] -= dx**2 * params['delta'] * np.exp(Y[i])
        J[i, i + 1] = - dx**2*params['delta']*(-2*Y[i+1])
        J[i, i + 2] = 1

        J[i + 1, i - 1] = 1
        J[i + 1, i] = -dx**2 * params['zeta'] * Y[i+1]
        J[i + 1, i + 1] = -2 - 3 * dx**2*params['omega']*Y[i+1]**2
        J[i + 1, i + 1] += dx**2*params['omega']-dx**2*params['zeta']*Y[i]

        J[i + 1, i + 3] = 1

        R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * params['gamma'] * np.sinh(Y[i])
        R[i] -= dx**2 * params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
        R[i+1] = Y[i-1] -2*Y[i+1] + Y[i+3] - dx**2*params['omega']*(Y[i+1]**3-Y[i+1])
        R[i+1] -= dx**2*params['zeta']*Y[i]*Y[i+1]

    dy = np.linalg.solve(J, -R)
    Y = Y +params['relaxation_tax']*dy
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


if gpu:
    X = X.get()
    Y = Y.get()
    Y2 = np.ones(len(X)).get()
else:
    Y2 = np.ones(len(X))

plt.subplot(1,2,1)
plt.plot(X,Y[0::2],'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
# plt.show()
plt.subplot(1,2,2)
plt.plot(X,Y[1::2]**2,'r-', X, Y2,'b')
plt.xlabel('x')
plt.ylabel('$\\eta^{2}$')
plt.legend([r'$\eta^{2}(x)$',r'$\eta^{2}(x)$ = 1'])
plt.show()