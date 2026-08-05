import pickle
import matplotlib.pyplot as plt
import numpy as np

arquivo = 'res_For.pkl'
f = open(arquivo,'rb')

l = pickle.load(f)

f.close()

X = l[0][0]

l2 = []
# l = l[-10:]
for i in range(len(l)):
    Y = l[i][1]
    a = i / (len(l) - 1)
    plt.subplot(1,2,1)
    plt.plot(X,Y[0::2], color=(a,0,1-a))
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.subplot(1,2,2)
    plt.plot(X,Y[1::2]**2, color=(a,0,1-a))
    plt.xlabel('x')
    plt.ylabel('$\\eta^{2}$')
    # plt.title(' y0 = '+str(l[i][2]))
    l2.append('y0 = '+ str(l[i][2]))
    # plt.legend(l2)
plt.show()