import pickle
import matplotlib.pyplot as plt
import numpy as np

# arquivo = 'animation/res_-0.9000000000.pkl'
# arquivo = 'teste_mixed.pkl'
# arquivo = 'teste_artigo.pkl'
# w2 = 50.0*1e4
# arquivo = 'teste_artigo1_w2_'
# arquivo += str(w2)
# arquivo += '.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_60_nx_2049_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_120_nx_2049_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_120_nx_8193_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_240_nx_2049_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_240_nx_4097_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_240_nx_8193_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_360_nx_2049_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_360_nx_4097_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_360_nx_8193_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_60_nx_2049_lambda_0.007.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_60_nx_2049_lambda_0.007_f_0.2.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_60_nx_2049_lambda_0.007_f_0.8.pkl'
# arquivo = 'teste_artigo1_phib2_7e-06_w2_0.0_xn_120_nx_2049_lambda_0.007_f_0.2.pkl'
# arquivo = 'teste_artigo1_phib2_1e-06_w2_0.0_xn_180_nx_4097_lambda_0.007_f_0.1.pkl'
arquivo = 'teste_artigo1_phib2_7e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.2'

f = open(arquivo,'rb')
l = pickle.load(f)

f.close()
pass

X = l[0]
Y = l[1]
for i in l[2]:
    print(i, l[2][i])

plt.subplot(1,2,1)
plt.plot(X,Y[0::2], color='#c00')
plt.xlabel(r'$x(\AA)$', fontsize=20)
plt.ylabel('y(x)',fontsize=20)
plt.subplot(1,2,2)
plt.plot(X,Y[1::2]**2, color='#c00')
plt.xlabel(r'$x(\AA)$', fontsize=20)
plt.ylabel('$\eta^2(x)$',fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.yticks(np.linspace(-0.9, 0.1, 11))
plt.show()