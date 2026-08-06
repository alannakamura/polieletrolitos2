import matplotlib.pyplot as plt
import pickle
import os
import time
import numpy as np
from numpy.linalg import norm
import scipy as sp
from scipy.sparse.linalg import eigs

class Polieletrolito:
    def __init__(self, gpu = False, normalisation = 2, exemplo = 13):
        self.params = {
            # Tipo de condição de contorno: 'potential' (y(0)=y_s) ou 'charge' (y'(0)=yprime0)
            # 'bc_type': 'potential',

            # Se 'potential', fixe o potencial superficial (adimensional) y(0)=y_s
            'a': 5,
            'c_salt': 6.02e-8,  # A^{-3} 0,1 mM #fig 1a
            'f': 1,
            # 'y_s': -1,  # típico: -0.5 a -2 (em unidades e*psi/kBT)1
            'phib2': 1e-6,
            'v': 50,
            'e': 80,
            'T': 300,
            'lb': 7.2,  # angstrom
            'w2': 0.0,

            # Se 'charge', fixe a derivada do potencial na parede
            # 'yprime0': 1.0,

            # Grade numérica
            'x0': 0.,
            'xn': 60.,
            'nx': 11,
            # 'h' : 0.1,
            'relaxation_tax': 0.007,

            'y0': -1,
            'h0': 0,
            'yn': 0,
            'hn': 1,

            'dirichlet_boundary': [True, True, True, True],
            'boundary': 'dirichlet',
            'error': 1e-6,
            'D' : 30,
            'adaptative' : False,
            'iterative_plot' : False,

            'convergence_error':100,
            'lambda0': 0,
            'lambdaf':-1,
            'n_lambda':-1
        }

        self.gpu = gpu

        self.filename = 'file.pkl'
        # self.calculate_constants()

    def run4(self):

        if self.gpu:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            from cupyx.scipy.sparse.linalg import spsolve
        else:
            import numpy as np
            import scipy as sp

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        # J = np.zeros((2 * N, 2 * N))
        J = sp.sparse.lil_matrix((2 * N, 2 * N))
        X = np.linspace(0, self.params['xn'], N)
        dx = X[1] - X[0]
        dy = np.ones(N)
        iter = 0

        data = [1,0,0,1,0,0,2,2,2,2,2,3,3,3,3,3,0,0,1,0,0,1]
        columns = [0,1,2,1,2,3,0,1,2,3,4,1,2,3,4,5,2,3,4,3,4,5]
        rows = [0,3,6,11,16,19, 22]

        if self.params['dirichlet_boundary'][0]:
            J[0, 0] = 1
        else:
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        if self.params['dirichlet_boundary'][1]:
            J[1, 1] = 1
        else:
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        if self.params['dirichlet_boundary'][2]:
            J[2 * N - 2, 2 * N - 2] = 1
        else:
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        if self.params['dirichlet_boundary'][3]:
            J[2 * N - 1, 2 * N - 1] = 1
        else:
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        while np.linalg.norm(dy) > self.params['error']:
            print('iter', iter, 'error', np.linalg.norm(dy))

            if self.params['dirichlet_boundary'][0]:
                R[0] = Y[0] -self.params['y0']
            else:
                R[0] = (Y[2] - Y[0])/dx -self.params['y0']

            if self.params['dirichlet_boundary'][1]:
                R[1] = Y[1] -self.params['h0']
            else:
                R[1] = (Y[3] - Y[1]) / dx -self.params['h0']

            if self.params['dirichlet_boundary'][2]:
                R[-2] = Y[-2] - self.params['yn']
            else:
                R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

            if self.params['dirichlet_boundary'][3]:
                R[-1] = Y[-1] - self.params['hn']
            else:
                R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

            for i in range(2, 2*N-2, 2):

                J[i, i - 2] = 1
                J[i, i] = -2 - dx**2 * self.params['gamma'] * np.cosh(Y[i])
                J[i, i] -= dx**2 * self.params['delta'] * np.exp(Y[i])
                J[i, i + 1] = - dx**2*self.params['delta']*(-2*Y[i+1])
                J[i, i + 2] = 1

                J[i + 1, i - 1] = 1
                J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

                J[i + 1, i + 1] = -2
                J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                J[i + 1, i + 1] += dx ** 2 * self.params['omega']
                J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

                J[i + 1, i + 3] = 1

                R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                R[i + 1] = Y[i-1] -2*Y[i+1] + Y[i+3]
                R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i+1] ** 3-Y[i+1])
                R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])
                R[i + 1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

            inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            fim = time.perf_counter()
            tempo_total = fim - inicio1
            print(f"tempo total: {tempo_total:.6f} s")

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

    def run3(self):

        if self.gpu:
            import cupy as np
        else:
            import numpy as np
            import scipy as sp

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        # J = np.zeros((2 * N, 2 * N))
        J = sp.sparse.lil_matrix((2*N, 2*N))
        X = np.linspace(0, self.params['xn'], N)
        dx = X[1] - X[0]
        dy = np.ones(N)
        iter = 0

        if self.params['dirichlet_boundary'][0]:
            J[0, 0] = 1
        else:
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        if self.params['dirichlet_boundary'][1]:
            J[1, 1] = 1
        else:
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        if self.params['dirichlet_boundary'][2]:
            J[2 * N - 2, 2 * N - 2] = 1
        else:
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        if self.params['dirichlet_boundary'][3]:
            J[2 * N - 1, 2 * N - 1] = 1
        else:
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        for i in range(2, 2 * N - 2, 2):
            J[i, i - 2] = 1
            J[i, i] = 1
            J[i, i + 1] = 1
            J[i, i + 2] = 1

            J[i + 1, i - 1] = 1
            J[i + 1, i] = 1
            J[i + 1, i + 1] = 1
            J[i + 1, i + 3] = 1

        J_csr = J.tocsr()

        while np.linalg.norm(dy) > self.params['error']:
            print('iter', iter, 'error', np.linalg.norm(dy))

            if self.params['dirichlet_boundary'][0]:
                R[0] = Y[0] -self.params['y0']
            else:
                R[0] = (Y[2] - Y[0])/dx -self.params['y0']

            if self.params['dirichlet_boundary'][1]:
                R[1] = Y[1] -self.params['h0']
            else:
                R[1] = (Y[3] - Y[1]) / dx -self.params['h0']

            if self.params['dirichlet_boundary'][2]:
                R[-2] = Y[-2] - self.params['yn']
            else:
                R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

            if self.params['dirichlet_boundary'][3]:
                R[-1] = Y[-1] - self.params['hn']
            else:
                R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

            for i in range(2, 2*N-2, 2):

                J[i, i - 2] = 1
                J[i, i] = -2 - dx**2 * self.params['gamma'] * np.cosh(Y[i])
                J[i, i] -= dx**2 * self.params['delta'] * np.exp(Y[i])
                J[i, i + 1] = - dx**2*self.params['delta']*(-2*Y[i+1])
                J[i, i + 2] = 1

                J[i + 1, i - 1] = 1
                J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

                J[i + 1, i + 1] = -2
                J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                J[i + 1, i + 1] += dx ** 2 * self.params['omega']
                J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

                J[i + 1, i + 3] = 1

                R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                R[i + 1] = Y[i-1] -2*Y[i+1] + Y[i+3]
                R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i+1] ** 3-Y[i+1])
                R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])
                R[i + 1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

            inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            fim = time.perf_counter()
            tempo_total = fim - inicio1
            print(f"tempo total: {tempo_total:.6f} s")

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

    def ci2(self, x, ys, D):

        Ly = D
        Leta = D
        L2 = 2 * D
        L3 = 2*D
        Ay = abs(ys)*10
        Aeta = 20

        y0 = ys * np.exp(-x / Ly) + Ay * (np.exp(-x / L2)- np.exp(-x / Ly))
        eta0 =  1.0 - np.exp(-x / Leta) + Aeta* (np.exp(-x / L3)- np.exp(-x / Leta))

        plt.plot(x, y0, 'r', x, eta0, 'b')
        plt.show()
        exit(0)
        return y0, eta0

    def ci3(self, x):
        f = open('phib2_1e-06_w2_0.0_xn_60_nx_1025_lambda_0.007_f_1_v_50_c_6.02e-08_y0_-1_t_tttt_salt_6.02e-08_D_30.pkl', 'rb')
        Y = pickle.load(f)
        xr = Y[0]
        yr = Y[1][0::2]
        etar = Y[1][1::2]

        n1 = len(xr)
        n2 = len(x)
        y0 = np.zeros(n2)
        eta0 = np.ones(n2)

        for i in range(n1):
            y0[i] = yr[i]
            eta0[i] = etar[i]
        for i in range(n1, n2):
            y0[i] = 1/x[i]
            eta0[i] = 1+1/x[i]



        return y0, eta0

    def ci(self, x):
        f = open('phib2_1e-06_w2_0.0_xn_60_nx_1025_lambda_0.007_f_1_v_50_c_6.02e-08_y0_-1_t_tttt_salt_6.02e-08_D_30.pkl', 'rb')
        Y = pickle.load(f)
        xr = Y[0]
        yr = Y[1][0::2]
        etar = Y[1][1::2]
        # plt.plot(x, y, 'r', x, eta, 'b')
        # plt.show()
        n1 = len(xr)-1
        n2 = len(x)-1
        fator = int(n2/n1)
        print(n1, n2, fator)
        y0 = np.zeros(len(x))
        y0[0::fator] = yr
        # print(y0[:9], len(y0))
        # print(yr[:9], len(yr))
        eta0 = np.zeros(len(x))
        eta0[0::fator] = etar
        # print(eta0[:10], eta0[-10:], len(eta0))
        # print(etar[:10], etar[-10:], len(etar))

        xr *= x[-1]/xr[-1]
        for i in range(len(yr)-1):
            # print(yr[i], yr[i+1])
            base = fator*i
            for j in range(1, fator):
                x1 = xr[i]
                x2 = xr[i+1]
                y1 = yr[i]
                y2 = yr[i+1]
                eta1 = etar[i]
                eta2 = etar[i + 1]
                y0[base + j] = ((x[base + j] - x2) / (x1 - x2) * y1 +
                                (x[base + j] - x1) / (x2 - x1) * y2)
                eta0[base + j] = ((x[base + j] - x2) / (x1 - x2) * eta1 +
                                (x[base + j] - x1) / (x2 - x1) * eta2)
        # print(y0[:17], len(y0))
        # print(yr[:17], len(yr))
        # plt.subplot(1,2,1)
        # plt.plot(x, y0, 'ro')
        # plt.subplot(1,2,2)
        # plt.plot(x, eta0, 'ro')
        # plt.show()
        # exit(0)
        return y0, eta0

    def calcular_jacobiana1(self, J, dx, N):

        if self.params['dirichlet_boundary'][0]:
            J[0, 0] = 1
        else:
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        if self.params['dirichlet_boundary'][1]:
            J[1, 1] = 1
        else:
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        if self.params['dirichlet_boundary'][2]:
            J[2 * N - 2, 2 * N - 2] = 1
        else:
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        if self.params['dirichlet_boundary'][3]:
            J[2 * N - 1, 2 * N - 1] = 1
        else:
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        # if self.params['dirichlet_boundary'][0]:
        #     # J[0, 0] = 1
        #     J[0, 0] = dx
        # else:
        #     # J[0, 0] = -1 / dx
        #     # J[0, 2] = 1 / dx
        #     J[0, 0] = -1
        #     J[0, 2] = 1
        #
        # if self.params['dirichlet_boundary'][1]:
        #     # J[1, 1] = 1
        #     J[1, 1] = dx
        # else:
        #     # J[1, 1] = -1 / dx
        #     # J[1, 3] = 1 / dx
        #     J[1, 1] = -1
        #     J[1, 3] = 1
        #
        # if self.params['dirichlet_boundary'][2]:
        #     # J[2 * N - 2, 2 * N - 2] = 1
        #     J[2 * N - 2, 2 * N - 2] = dx
        # else:
        #     # J[2 * N - 2, 2 * N - 4] = -1 / dx
        #     # J[2 * N - 2, 2 * N - 2] = 1 / dx
        #     J[2 * N - 2, 2 * N - 4] = -1
        #     J[2 * N - 2, 2 * N - 2] = 1
        #
        # if self.params['dirichlet_boundary'][3]:
        #     # J[2 * N - 1, 2 * N - 1] = 1
        #     J[2 * N - 1, 2 * N - 1] = dx
        # else:
        #     # J[2 * N - 1, 2 * N - 3] = -1 / dx
        #     # J[2 * N - 1, 2 * N - 1] = 1 / dx
        #     J[2 * N - 1, 2 * N - 3] = -1
        #     J[2 * N - 1, 2 * N - 1] = 1

    def calcular_jacobiana2(self, J, dx, N, Y):

        for i in range(2, 2 * N - 2, 2):
            J[i, i - 2] = 1
            J[i, i] = -2 - dx ** 2 * self.params['gamma'] * np.cosh(Y[i])
            J[i, i] -= dx ** 2 * self.params['delta'] * np.exp(Y[i])
            J[i, i + 1] = - dx ** 2 * self.params['delta'] * (-2 * Y[i + 1])
            J[i, i + 2] = 1

            J[i + 1, i - 1] = 1
            J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

            J[i + 1, i + 1] = -2
            J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
            J[i + 1, i + 1] += dx ** 2 * self.params['omega']
            J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
            J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
            J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

            J[i + 1, i + 3] = 1

        # for i in range(2, 2 * N - 2, 2):
        #     J[i, i - 2] = dx
        #     J[i, i] = -2 - dx ** 3 * self.params['gamma'] * np.cosh(Y[i])
        #     J[i, i] -= dx ** 3 * self.params['delta'] * np.exp(Y[i])
        #     J[i, i + 1] = - dx ** 3 * self.params['delta'] * (-2 * Y[i + 1])
        #     J[i, i + 2] = dx
        #
        #     J[i + 1, i - 1] = dx
        #     J[i + 1, i] = -dx ** 3 * self.params['zeta'] * Y[i + 1]
        #
        #     J[i + 1, i + 1] = -2*dx
        #     J[i + 1, i + 1] += -3 * dx ** 3 * self.params['omega'] * Y[i + 1] ** 2
        #     J[i + 1, i + 1] += dx ** 3 * self.params['omega']
        #     J[i + 1, i + 1] += -5 * dx ** 3 * self.params['alpha'] * Y[i + 1] ** 4
        #     J[i + 1, i + 1] += dx ** 3 * self.params['alpha']
        #     J[i + 1, i + 1] += -dx ** 3 * self.params['zeta'] * Y[i]
        #
        #     J[i + 1, i + 3] = dx

    def calcular_residuo(self, R, Y, N, dx):

        if self.params['dirichlet_boundary'][0]:
            R[0] = Y[0] - self.params['y0']
        else:
            R[0] = (Y[2] - Y[0]) / dx - self.params['y0']

        if self.params['dirichlet_boundary'][1]:
            R[1] = Y[1] - self.params['h0']
        else:
            R[1] = (Y[3] - Y[1]) / dx - self.params['h0']

        if self.params['dirichlet_boundary'][2]:
            R[-2] = Y[-2] - self.params['yn']
        else:
            R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

        if self.params['dirichlet_boundary'][3]:
            R[-1] = Y[-1] - self.params['hn']
        else:
            R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

        for i in range(2, 2 * N - 2, 2):
            R[i] = Y[i - 2] - 2 * Y[i] + Y[i + 2] - dx ** 2 * self.params['gamma'] * np.sinh(Y[i])
            R[i] -= dx ** 2 * self.params['delta'] * (np.exp(Y[i]) - Y[i + 1] ** 2)
            R[i + 1] = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3]
            R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])
            R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])
            R[i + 1] -= dx ** 2 * self.params['zeta'] * Y[i] * Y[i + 1]

        # R *= dx

    def run(self, p = True):

        self.calculate_constants()
        N = self.params['nx']
        J = sp.sparse.lil_matrix((2 * N, 2 * N))
        R = np.zeros(2 * N)
        X = np.linspace(0, self.params['xn'], N)
        # Y2 = self.ci3(X)
        Y = np.zeros(2 * N)
        # Y[0::2] = Y2[0]
        # Y[1::2] = Y2[1]
        dx = X[1] - X[0]
        iter = 0

        self.calcular_jacobiana1(J, dx, N)

        # plot iterativo
        if self.params['iterative_plot']:
            plt.ion()
            fig, (ax_y, ax_eta) = plt.subplots(1, 2, sharex=True)
            linha_y, = ax_y.plot(X, Y[::2], 'r')
            linha_eta, = ax_eta.plot(X, Y[1::2] ** 2, 'r')
            ax_y.set_ylabel("y")
            ax_eta.set_ylabel(r"$\eta^2$")
            ax_eta.set_xlabel("x")
            plt.show(block=False)
            plt.pause(0.1)

        error = np.inf
        while error > self.params['error']:

            if p:
                print('iter', iter,
                      'error', error,
                      'tax', self.params['relaxation_tax'])

            self.calcular_residuo(R, Y, N, dx)
            self.calcular_jacobiana2(J, dx, N, Y)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)

            if self.params['adaptative']:
                Yp = Y + self.params['relaxation_tax']*dy
                Rp = np.zeros(2*N)
                self.calcular_residuo(Rp, Yp, N, dx)
                if norm(Rp) < norm(R):
                    Y = Yp
                    iter += 1
                    error = norm(Rp)
                    if  self.params['relaxation_tax'] < 0.5:
                        self.params['relaxation_tax'] *= 2
                else:
                    if self.params['relaxation_tax'] > 1e-16:
                        self.params['relaxation_tax'] /= 2
                    else:
                        raise Exception

            else:
                Y = Y + self.params['relaxation_tax']*dy
                error = norm(self.params['relaxation_tax']*dy)
                # error = norm(R)
                iter += 1

            # plot iterativo
            if self.params['iterative_plot']:
                y = Y[::2]
                eta2 = Y[1::2] ** 2
                linha_y.set_ydata(y)
                linha_eta.set_ydata(eta2)
                ax_y.relim()
                ax_y.autoscale_view()
                ax_eta.relim()
                ax_eta.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.01)

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        return X, Y, self.params

    def run5(self, p = True):
        N = self.params['nx']
        Y = np.zeros(2 * N)

        if self.gpu:
            J = sp.sparse.lil_matrix((2 * N, 2 * N))
        else:
            J = sp.sparse.lil_matrix((2*N, 2*N))
        X = np.linspace(0, self.params['xn'], N)
        dx = X[1] - X[0]
        iter = 0

        if self.params['dirichlet_boundary'][0]:
            J[0, 0] = 1
        else:
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        if self.params['dirichlet_boundary'][1]:
            J[1, 1] = 1
        else:
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        if self.params['dirichlet_boundary'][2]:
            J[2 * N - 2, 2 * N - 2] = 1
        else:
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        if self.params['dirichlet_boundary'][3]:
            J[2 * N - 1, 2 * N - 1] = 1
        else:
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        error = 1e100
        while error > self.params['error']:

            R = self.calcular_residuo(Y, dx, N)
            error = np.linalg.norm(R)

            if p:
                print('iter', iter,
                      'error', error,
                      'tax', self.params['relaxation_tax'])

            # if self.params['dirichlet_boundary'][0]:
            #     R[0] = Y[0] -self.params['y0']
            # else:
            #     R[0] = (Y[2] - Y[0])/dx -self.params['y0']
            #
            # if self.params['dirichlet_boundary'][1]:
            #     R[1] = Y[1] -self.params['h0']
            # else:
            #     R[1] = (Y[3] - Y[1]) / dx -self.params['h0']
            #
            # if self.params['dirichlet_boundary'][2]:
            #     R[-2] = Y[-2] - self.params['yn']
            # else:
            #     R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']
            #
            # if self.params['dirichlet_boundary'][3]:
            #     R[-1] = Y[-1] - self.params['hn']
            # else:
            #     R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

            for i in range(2, 2*N-2, 2):
                J[i, i - 2] = 1
                J[i, i] = -2 - dx**2 * self.params['gamma'] * np.cosh(Y[i])
                J[i, i] -= dx**2 * self.params['delta'] * np.exp(Y[i])
                J[i, i + 1] = - dx**2*self.params['delta']*(-2*Y[i+1])
                J[i, i + 2] = 1

                J[i + 1, i - 1] = 1
                J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

                J[i + 1, i + 1] = -2
                J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                J[i + 1, i + 1] += dx ** 2 * self.params['omega']
                J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

                J[i + 1, i + 3] = 1

                # R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                # R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                # R[i + 1] = Y[i-1] -2*Y[i+1] + Y[i+3]
                # R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i+1] ** 3-Y[i+1])
                # R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])
                # R[i + 1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

            # inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            # error0 = np.linalg.norm(dy) * self.params['relaxation_tax']
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            # fim = time.perf_counter()
            # tempo_total = fim - inicio1
            # print(f"tempo total: {tempo_total:.6f} s")

            R2 = self.calcular_residuo(Y+dy*self.params['relaxation_tax'], dx, N)
            R2 = np.linalg.norm(R2)

            while R2 > error:
                if self.params['relaxation_tax']/2 > 1e-16:
                    self.params['relaxation_tax'] /= 2
                else:
                    print('taxa muito peoquena')
                    exit(1)
                R2 = self.calcular_residuo(Y + dy * self.params['relaxation_tax'], dx, N)
                R2 = np.linalg.norm(R2)
            Y = Y + self.params['relaxation_tax'] * dy
            error = R2
            iter += 1
            # self.params["relaxation_tax"] = 1.0

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

        return X, Y, self.params

    def set_filename(self, name):
        self.filename = name

    def calculate_constants(self):

        self.params['k2'] = 8 * np.pi * self.params['lb'] * self.params['c_salt']
        self.params['km2'] = 4 * np.pi * self.params['lb'] * self.params['phib2'] * self.params['f']
        self.params['gamma'] = self.params['k2'] * self.params['D'] ** 2
        self.params['delta'] = self.params['km2'] * self.params['D'] ** 2
        self.params['omega'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['v'] * self.params['phib2']
        self.params['zeta'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['f']
        self.params['alpha'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['w2'] / 2 * self.params[
            'phib2'] ** 2
        self.params['xn'] /= self.params['D']

    def run2(self):

        if self.gpu:
            import cupy as np
        else:
            import numpy as np

        if self.params['lambdaf']!= -1:
            lambda_l = np.linspace(self.params['lambda0'],
                                   self.params['lambdaf'],
                                    self.params['n_lambda'])
        else:
            lambda_l = [self.params['lambda0']]

        f = open(self.filename, "wb")
        data = []
        for l in lambda_l:
            self.params['relaxation_tax'] = l
            print('lambda', self.params['relaxation_tax'])
            N = self.params['nx']
            R = np.zeros(2 * N)
            Y = np.zeros(2 * N)
            J = np.zeros((2 * N, 2 * N))
            X = np.linspace(0, self.params['xn'], N)
            dx = X[1] - X[0]
            dy = np.ones(N)
            iter = 0

            if self.params['dirichlet_boundary'][0]:
                J[0, 0] = 1
            else:
                J[0, 0] = -1 / dx
                J[0, 2] = 1 / dx

            if self.params['dirichlet_boundary'][1]:
                J[1, 1] = 1
            else:
                J[1, 1] = -1 / dx
                J[1, 3] = 1 / dx

            if self.params['dirichlet_boundary'][2]:
                J[2 * N - 2, 2 * N - 2] = 1
            else:
                J[2 * N - 2, 2 * N - 4] = -1 / dx
                J[2 * N - 2, 2 * N - 2] = 1 / dx

            if self.params['dirichlet_boundary'][3]:
                J[2 * N - 1, 2 * N - 1] = 1
            else:
                J[2 * N - 2, 2 * N - 4] = -1 / dx
                J[2 * N - 2, 2 * N - 2] = 1 / dx
                J[2 * N - 1, 2 * N - 3] = -1 / dx
                J[2 * N - 1, 2 * N - 1] = 1 / dx

            while np.linalg.norm(dy) > self.params['error']:
                # print('iter', iter, 'error', np.linalg.norm(dy))

                if self.params['dirichlet_boundary'][0]:
                    R[0] = Y[0] -self.params['y0']
                else:
                    R[0] = (Y[2] - Y[0])/dx -self.params['y0']

                if self.params['dirichlet_boundary'][1]:
                    R[1] = Y[1] -self.params['h0']
                else:
                    R[1] = (Y[3] - Y[1]) / dx -self.params['h0']

                if self.params['dirichlet_boundary'][2]:
                    R[-2] = Y[-2] - self.params['yn']
                else:
                    R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

                if self.params['dirichlet_boundary'][3]:
                    R[-1] = Y[-1] - self.params['hn']
                else:
                    R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

                for i in range(2, 2*N-2, 2):

                    J[i, i - 2] = 1
                    J[i, i] = -2 - dx**2 * self.params['gamma'] * np.cosh(Y[i])
                    J[i, i] -= dx**2 * self.params['delta'] * np.exp(Y[i])
                    J[i, i + 1] = - dx**2*self.params['delta']*(-2*Y[i+1])
                    J[i, i + 2] = 1

                    J[i + 1, i - 1] = 1
                    J[i + 1, i] = -dx**2 * self.params['zeta'] * Y[i+1]
                    J[i + 1, i + 1] = -2
                    J[i + 1, i + 1] += -3 * dx**2*self.params['omega']*Y[i+1]**2
                    J[i + 1, i + 1] +=  dx**2*self.params['omega']
                    J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                    J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                    J[i + 1, i + 1] -= -dx**2*self.params['zeta']*Y[i]

                    J[i + 1, i + 3] = 1

                    R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                    R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                    R[i+1] = Y[i-1] -2*Y[i+1] + Y[i+3] - dx**2*self.params['omega']*(Y[i+1]**3-Y[i+1])
                    R[i+1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

                dy = np.linalg.solve(J, -R)
                Y = Y + self.params['relaxation_tax']*dy
                iter+=1

            X1 = X.copy()
            Y1 = Y.copy()

            error = 1
            error_l = []
            N_l = []
            print('N = ', N, 'antes convergencia', 'iter = ', iter)
            while error> self.params['convergence_error']:
                N -= 1
                N *= 2
                N += 1

                # N = self.params['nx']
                R = np.zeros(2 * N)
                Y = np.zeros(2 * N)
                J = np.zeros((2 * N, 2 * N))
                X = np.linspace(0, self.params['xn'], N)
                dx = X[1] - X[0]
                dy = np.ones(N)
                iter = 0

                if self.params['dirichlet_boundary'][0]:
                    J[0, 0] = 1
                else:
                    J[0, 0] = -1 / dx
                    J[0, 2] = 1 / dx

                if self.params['dirichlet_boundary'][1]:
                    J[1, 1] = 1
                else:
                    J[1, 1] = -1 / dx
                    J[1, 3] = 1 / dx

                if self.params['dirichlet_boundary'][2]:
                    J[2 * N - 2, 2 * N - 2] = 1
                else:
                    J[2 * N - 2, 2 * N - 4] = -1 / dx
                    J[2 * N - 2, 2 * N - 2] = 1 / dx

                if self.params['dirichlet_boundary'][3]:
                    J[2 * N - 1, 2 * N - 1] = 1
                else:
                    J[2 * N - 2, 2 * N - 4] = -1 / dx
                    J[2 * N - 2, 2 * N - 2] = 1 / dx
                    J[2 * N - 1, 2 * N - 3] = -1 / dx
                    J[2 * N - 1, 2 * N - 1] = 1 / dx

                while np.linalg.norm(dy) > self.params['error']:
                    # print('iter', iter, 'error', np.linalg.norm(dy))

                    if self.params['dirichlet_boundary'][0]:
                        R[0] = Y[0] - self.params['y0']
                    else:
                        R[0] = (Y[2] - Y[0]) / dx - self.params['y0']

                    if self.params['dirichlet_boundary'][1]:
                        R[1] = Y[1] - self.params['h0']
                    else:
                        R[1] = (Y[3] - Y[1]) / dx - self.params['h0']

                    if self.params['dirichlet_boundary'][2]:
                        R[-2] = Y[-2] - self.params['yn']
                    else:
                        R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

                    if self.params['dirichlet_boundary'][3]:
                        R[-1] = Y[-1] - self.params['hn']
                    else:
                        R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

                    for i in range(2, 2 * N - 2, 2):
                        J[i, i - 2] = 1
                        J[i, i] = -2 - dx ** 2 * self.params['gamma'] * np.cosh(Y[i])
                        J[i, i] -= dx ** 2 * self.params['delta'] * np.exp(Y[i])
                        J[i, i + 1] = - dx ** 2 * self.params['delta'] * (-2 * Y[i + 1])
                        J[i, i + 2] = 1

                        J[i + 1, i - 1] = 1
                        J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]
                        J[i + 1, i + 1] = -2 - 3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                        J[i + 1, i + 1] += dx ** 2 * self.params['omega'] - dx ** 2 * self.params['zeta'] * Y[i]

                        J[i + 1, i + 3] = 1

                        R[i] = Y[i - 2] - 2 * Y[i] + Y[i + 2] - dx ** 2 * self.params['gamma'] * np.sinh(Y[i])
                        R[i] -= dx ** 2 * self.params['delta'] * (np.exp(Y[i]) - Y[i + 1] ** 2)
                        R[i + 1] = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * self.params['omega'] * (
                                    Y[i + 1] ** 3 - Y[i + 1])
                        R[i + 1] -= dx ** 2 * self.params['zeta'] * Y[i] * Y[i + 1]

                    dy = np.linalg.solve(J, -R)
                    Y = Y + self.params['relaxation_tax'] * dy
                    iter += 1

                X2 = X.copy()
                Y2 = Y.copy()

                y1 = Y1[0::2]
                eta1 = Y1[1::2]
                y2 = Y2[0::2]
                eta2 = Y2[1::2]
                y3 = y2[0::2]
                eta3 = eta2[0::2]
                deltay = y1 - y3
                deltae = eta1 - eta3
                v = np.concatenate((deltay, deltae), axis=0)
                error = np.linalg.norm(v)
                # print('erro',erro)
                X1 = X2.copy()
                Y1 = Y2.copy()
                print('N = ', N, 'error = ', error, 'iter = ', iter, ' dx =  ' , dx)
                error_l.append(error)
                # if len(error_l) > 2 and error_l[-1]> error_l[-2]> error_l[-3]:
                #     break
                N_l.append([((N-1)/2)+1, N])
            data.append([N_l[-1][0],  l, X2, Y2, self.params, N_l, error_l])
        pickle.dump(data, f)
        f.close()

    def set_filename(self, name):
        self.filename = name

    def examples(self, exemplo = 2):
        # exemplos do artigo

        if exemplo == 2:
            self.params['a'] = 5
            self.params['f'] = 1
            self.params['y0'] = -0.5
        elif exemplo == 3:
            self.params['a'] = 10
            self.params['f'] = 1
            self.params['y0'] = -0.5
        elif exemplo == 4:
            self.params['a'] = 5
            self.params['f'] = 0.1
            self.params['y0'] = -0.5
        elif exemplo == 5:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.12
        elif exemplo == 6:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.1
        elif exemplo == 7:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.09
        elif exemplo == 8:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.08
        elif exemplo == 9:
            self.params['c_salt'] *= 10
        elif exemplo == 10:
            self.params['c_salt'] *= 100
        elif exemplo == 11:
            self.params['c_salt'] *= 1e-1
        elif exemplo == 12:
            self.params['c_salt'] *= 1e-2
        elif exemplo == 13:
            self.params['y0'] = -1