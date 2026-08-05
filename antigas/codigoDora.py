import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

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
    'Xmax': 10.0,   # extensão do domínio (em unidades de 1/kappa)
    'nx': 300
}

"""Sistema de EDOs"""

def odes(X, Y, gamma, delta, omega, zeta):
    # Y = [y, y', h, h']
    y, y1, h, h1 = Y
    dy_dx = y1
    dy1_dx = gamma * np.sinh(y) + delta * (np.exp(y) - h**2)
    dh_dx = h1
    dh1_dx = omega * (h**3 - h) + zeta * y * h
    return np.vstack((dy_dx, dy1_dx, dh_dx, dh1_dx))

"""Condições de contorno"""

def bc(Y0, Yf, bc_type='potential', y_s=None, yprime0=None):
    if bc_type == 'potential':
        # y(0)=y_s, h(0)=0 ; y(Xmax)=0, h(Xmax)=1
        return np.array([Y0[0] - y_s, Y0[2] - 0.0, Yf[0] - 0.0, Yf[2] - 1.0])
    else:
        # y'(0)=yprime0, h(0)=0 ; y(Xmax)=0, h(Xmax)=1
        return np.array([Y0[1] - yprime0, Y0[2] - 0.0, Yf[0] - 0.0, Yf[2] - 1.0])

""" chute inicial"""

def initial_guess(X, params):
    y_s = params['y_s']
    # y0 = y_s * np.exp(-0.5 * X)
    y0 = y_s * (np.exp(-0.5 * X) - np.exp(-0.5 * X[-1])) / (1 - np.exp(-0.5 * X[-1]))# decai exponencialmente
    # h0 = 1.0 - np.exp(-0.3 * X)   # cresce para 1
    a = -0.3
    h0 = (1 - np.exp(-a * X)) / (1 - np.exp(-a * X[-1]))
    y1_0 = np.gradient(y0, X)
    h1_0 = np.gradient(h0, X)
    return np.vstack((y0, y1_0, h0, h1_0))

"""Resolver - Mexer aqui!"""

X = np.linspace(0, params['Xmax'], params['nx'])
Y_guess = initial_guess(X, params)

# plt.subplot(2,2,1)
# plt.plot(Y_guess[0])
# plt.subplot(2,2,2)
# plt.plot(Y_guess[2])
# plt.subplot(2,2,3)
# plt.plot(Y_guess[1])
# plt.subplot(2,2,4)
# plt.plot(Y_guess[3])
# plt.show()

sol = solve_bvp(
    lambda x, y: odes(x, y, params['gamma'], params['delta'],
                      params['omega'], params['zeta']),
    lambda y0, yf: bc(y0, yf, params['bc_type'],
                      y_s=params['y_s'], yprime0=params['yprime0']),
    X, Y_guess, verbose=2, tol=1e-6, max_nodes=1000000
)

if sol.status != 0:
    print("Aviso: solver não convergiu perfeitamente (status={}).".format(sol.status))

X_plot = sol.x
y_plot = sol.y[0]
h_plot = sol.y[2]

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(X_plot, y_plot, label='y(X) (potencial)', color='tab:blue')
ax1.set_xlabel('X = x/D (distância adimensional)')
ax1.set_ylabel('y (potencial adimensional)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(X_plot, h_plot, '--', label='h(X) (polímero)', color='tab:orange')
ax2.set_ylabel('h (amplitude do polímero)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
#plt.title('Perfis de potencial e polímero (parâmetros exemplo)')
plt.show()
# plt.savefig("main.pdf")