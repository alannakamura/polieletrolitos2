import matplotlib.pyplot as plt
import numpy as np
import pickle

f = open('convergencia_teste_dirichlet2.pkl','rb')
res = pickle.load(f)

# plt.subplot(2,2,1)
# plt.plot(res[0], res[1][::2],'ro')
# plt.subplot(2,2,2)
# plt.plot(res[0], res[1][1::2]**2,'ro')
# plt.subplot(2,2,3)
# plt.plot(np.array(res[-2])[:,1], res[-1], 'ro')
# plt.subplot(2, 2, 4)
# plt.plot(np.array(res[-2])[:, 1], np.log10(res[-1]), 'ro')
# plt.show()
# print(res)

N = []
l = []
N_l = []
for i in res:
    N.append(i[0])
    l.append(i[1])
    N_l.append((i[0], i[1]))
print(N)
print(l)
print(N_l)
N_l.sort()
print(N_l)
pass