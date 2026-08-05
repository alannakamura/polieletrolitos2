import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURAÇÃO DO PROBLEMA
# -----------------------------

gamma = 98.04e-4
delta = 8.14e-2
omega = 0.0108
zeta = 216

def g(x, y):
    """Função do sistema de 3 EDOs: y' = g(x, y)."""
    y1, y2, y3, y4 = y
    return np.array([
        y1,
        h1,
        gamma * np.sinh(y) + delta * (np.exp(y) - h ** 2),
        omega * (h ** 3 - h) + zeta * y * h
    ])

# def jacobian_g(x, y):
#     """Jacobiana 3x3 de g(x, y)."""
#     return np.array([
#         [ 0.0,  1.0,  0.1],
#         [-1.0,  0.0,  1.0],
#         [ 0.0, -0.5,  0.0]
#     ])

def jacobian_g(x, y):
    """Jacobiana 3x3 de g(x, y)."""
    return np.array([
        [ 0.0,  1.0],
        [ -np.pi**2,  0.0]
    ])

# # Condições de contorno
# a1, a2 = 0.0, 1.0   # y1(0), y2(0)
# b3     = 2.0        # y3(1)
# Condições de contorno
a1     = 0.0   # y1(0), y2(0)
b2     = -np.pi        # y3(1)

# Malha
nn = 100
x = np.linspace(0, 1, nn)
h = x[1] - x[0]
# nv = 3
nv = 2
NV = nv * nn

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def unpack(Y):  # transforma vetor longo em lista de vetores 3D
    return [Y[i*nv:(i+1)*nv] for i in range(nn)]

def pack(nodes):
    return np.concatenate(nodes)

# -----------------------------
# CONSTRUÇÃO DE A E R
# -----------------------------
def build_AR(Y):
    """Constroi matriz A (Jac) e vetor R (resíduos) para o passo de Newton."""
    Y_nodes = unpack(Y)
    A = np.zeros((NV, NV))
    R = np.zeros(NV)
    I = np.eye(nv)
    row = 0

    # --- BC esquerda (y1(0)=a1, y2(0)=a2)
    # A[row, 0:3] = np.array([1, 0, 0]); R[row] = Y_nodes[0][0] - a1; row += 1
    # A[row, 0:3] = np.array([0, 1, 0]); R[row] = Y_nodes[0][1] - a2; row += 1
    A[row, 0:2] = np.array([1, 0])
    R[row] = Y_nodes[0][0] - a1
    row += 1

    # --- Blocos internos (4 intervalos)
    for k in range(nn - 1):
        yk = Y_nodes[k]
        ykp1 = Y_nodes[k+1]
        mid = 0.5*(yk + ykp1)
        J = jacobian_g((x[k]+x[k+1])/2, mid)
        gmid = g((x[k]+x[k+1])/2, mid)

        Rk = ykp1 - yk - h*gmid
        R[row:row+nv] = Rk

        Bk = -I - 0.5*h*J
        Ck =  I - 0.5*h*J

        A[row:row+nv, k*nv:(k+1)*nv] = Bk
        A[row:row+nv, (k+1)*nv:(k+2)*nv] = Ck
        row += nv

    # --- BC direita (y3(1)=b3)
    # A[row, -3:] = np.array([0, 0, 1]); R[row] = Y_nodes[-1][2] - b3
    A[row, -2:] = np.array([0, 1])
    R[row] = Y_nodes[-1][1] - b2
    return A, R

# -----------------------------
# LOOP DE NEWTON (RELAXAÇÃO)
# -----------------------------
Y = np.zeros(NV)  # chute inicial
tol = 1e-8
maxiter = 15

for it in range(maxiter):
    A, R = build_AR(Y)
    deltaY = np.linalg.solve(A, -R)
    Y += deltaY
    err = np.linalg.norm(R, np.inf)
    print(f"Iter {it+1:2d}: ||R||_inf = {err:.3e}")
    if err < tol:
        break

# -----------------------------
# RESULTADOS
# -----------------------------
Y_nodes = unpack(Y)
y1 = []
y2 = []

print("\nSolução nos nós:")
for i, xi in enumerate(x):
    # print(f"x = {xi:.2f} -> y1={Y_nodes[i][0]:.6f}, y2={Y_nodes[i][1]:.6f}, y3={Y_nodes[i][2]:.6f}")
    print(f"x = {xi:.2f} -> y1={Y_nodes[i][0]:.6f}, y2={Y_nodes[i][1]:.6f}")
    y1.append(Y_nodes[i][0])
    y2.append(Y_nodes[i][1])

plt.subplot(2,2,1)
plt.plot(x, y1,'r')
plt.subplot(2,2,2)
plt.plot(x, np.sin(np.pi*x),'r')
plt.subplot(2,2,3)
plt.plot(x, y2,'r')
plt.subplot(2,2,4)
plt.plot(np.pi*np.cos(np.pi*x),'r')
plt.show()