import pickle
import matplotlib.pyplot as plt

arquivo = 'res.pkl'
f = open(arquivo,'rb')

l = pickle.load(f)

f.close()
pass

X = l[0][0]
Y1 = l[0][1]
Y2 = l[1][1]
Y3 = l[2][1]
Y4 = l[3][1]
Y5 = l[4][1]

plt.plot(X,Y1[0::2],'r', X,Y2[0::2],'b',  X,Y3[0::2],'g', X,Y3[0::2],'m', X,Y3[0::2],'#0ff')
# plt.legend(['a=5, f=1, $y_s$=-1','a=5, f=1, $y_s$=-0,5','a=10, f=1, $y_s$=-0,5','a=5, f=0,1, $y_s$=-0,5'])
plt.xlabel('x(A)')
plt.ylabel('y(x)')
plt.show()
plt.plot(X,Y1[1::2]**2,'r', X,Y2[1::2]**2,'b', X,Y3[1::2]**2,'g', X,Y3[1::2]**2,'m', X,Y3[1::2]**2,'#0ff')
# plt.legend(['a=5, f=1, $y_s$=-1','a=5, f=1, $y_s$=-0,5','a=10, f=1, $y_s$=-0,5','a=5, f=0,1, $y_s$=-0,5'])
plt.xlabel('x(A)')
plt.ylabel('$\\eta(x)^{2}$')
plt.show()