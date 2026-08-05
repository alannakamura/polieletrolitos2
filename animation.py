import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --------------------------------------------------
# Carrega os arquivos .pkl (um por frame)
# --------------------------------------------------
# files = sorted(glob.glob("animation/res_*.pkl"))
list_dir = os.listdir("animation")
# list_dir.sort(reverse=True)

y_list = []
for i in list_dir:
    temp = i.split('_')
    y_list.append(float(temp[1].split('.pkl')[0]))

y0_min = -1.0
y0_max = 1.0
y_list.sort()
y_list = np.array(y_list)
y_list = y_list[(y_list >= y0_min) & (y_list <= y0_max)]

results = []
xmin = -np.inf
xmax = np.inf
ymin = -np.inf
ymax = np.inf

for fname in y_list:
    fname = 'animation/res_'+f"{fname:.10f}"+'.pkl'
    with open(fname, "rb") as f:
        temp = pickle.load(f)
        results.append([temp[0], temp[1], temp[2]])
        x = temp[0]
        y = temp[1][1::2]**2
        if np.min(x) < xmin          or xmin == -np.inf:
            xmin = np.min(x)
        if np.max(x) > xmax          or xmax ==  np.inf:
            xmax = np.max(x)
        if np.min(y) < ymin or ymin == -np.inf:
            ymin = np.min(y)
        if np.max(y) > ymax or ymax ==  np.inf:
            ymax = np.max(y)


# plt.figure(figsize=(largura_polegadas, altura_polegadas), dpi=100)
fig, ax = plt.subplots()
fig.set_size_inches(20, 7)
line, = ax.plot([], [], lw=2)

tol = 1.01
ax.set_xlim(tol*xmin, tol*xmax)
ax.set_ylim(tol*ymin, tol*ymax)

# --------------------------------------------------
# Inicialização da animação
# --------------------------------------------------
def init():
    line.set_data([], [])
    return line,

# --------------------------------------------------
# Atualização de cada frame
# --------------------------------------------------
def update(frame):
    data = results[frame]      # array N×2
    x = data[0]
    y = data[1]

    line.set_xdata(x)
    # line.set_ydata(y[1::2]**2)
    line.set_ydata(y[1::2] ** 2)

    # ax.set_title('y0 = '+str(data[2]['y0']))
    ax.set_title('y0 = ' + f"{data[2]['y0']:.2f}")
    ax.set_xlabel(r'$x(\AA)$', fontsize=30)
    ax.set_ylabel(r'$\frac{c(x)}{c_b}$', rotation=0, fontsize=40, labelpad=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    line.set_color('#c00')
    return line,

# --------------------------------------------------
# Criação da animação
# --------------------------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=len(results),
    init_func=init,
    # blit=True,
    interval=1000
)

# --------------------------------------------------
# Mostrar ou salvar
# --------------------------------------------------
plt.show()
# ani.save("animacao.mp4", dpi=150, fps=5)
ani.save("animacao.gif", writer="pillow", fps=1)
