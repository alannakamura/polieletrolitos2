import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# diretorio = "080426/testesPhib2/"
diretorio = "290426/exp4/"
# diretorio = "210426/testesw2_2/"
list_dir = os.listdir(diretorio)

for i in range(len(list_dir)):
    list_dir[i] = float(list_dir[i].split('_')[4]), list_dir[i]
list_dir.sort()

inicio = 0
fim = 3
# list_dir = [list_dir[1]]
list_dir = list_dir[inicio:fim+1]
# list_dir = list_dir[-10]

# list_dir = 'teste_artigo1_phib2_4e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.1.pkl'
# list_dir = 'teste_artigo1_phib2_5e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.08.pkl'
# list_dir = 'teste_artigo1_phib2_7e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.3.pkl'
# list_dir = 'teste_artigo1_phib2_0.001_w2_0.0_xn_120_nx_4097_lambda_0.007_error_0.0001_a_5_f_0.16_v_1.pkl'
# list_dir = 'teste_phib2_0.001_w2_1500_xn_120_nx_4097_lambda_0.007_error_0.0001_a_5_f_0.14_v_-1.5.pkl'
# list_dir = [[0, list_dir]]

print(list_dir)

y_list = []
e_list = []
x_list = []
t_list = []
for i in list_dir:
    f = open(diretorio+i[1], 'rb')
    l = pickle.load(f)
    x_list.append(l[0])
    y_list.append(l[1][0::2])
    e_list.append(l[1][1::2])
    # t_list.append(i[1].split('_')[17])
    t_list.append(i[1].split('_')[-1][:-4])
    f.close()

l = list_dir[0][1]
l = l.split('_')
# l2 = l[0:2]+l[4:]
l2 = l[:-2]
l = str.join('_', l2)

red = np.linspace(1.,0., len(list_dir))
green = np.linspace(0,0., len(list_dir))
blue = np.linspace(0.,1., len(list_dir))

# colours = []
# for i in range(len(x_list)):
#     colours.append('#'+str(hex(15-i)[-1])+'0'+str(hex(i)[-1]))

plt.title(l)
# plt.subplot(1,2,1)
for i in range(len(x_list)):
    plt.plot(x_list[i], y_list[i], color = (red[i], green[i], blue[i]))
plt.legend(t_list)
plt.show()

plt.title(l)
for i in range(len(x_list)):
    plt.plot(x_list[i], e_list[i]**2, color = (red[i], green[i], blue[i]))
# plt.plot( x_list[0], np.ones((len(x_list[0]), 1)), color = (0, 0.5, 0))
plt.legend(t_list)
plt.show()

# plt.title(l)
# for i in range(len(x_list)):
#     if i == 0:
#         plt.plot(x_list[i], e_list[i]**2-1, color = (red[i], green[i], blue[i]))
#     elif i == 1:
#         plt.plot(x_list[i], (e_list[i] ** 2-1) *150, color=(red[i], green[i], blue[i]))
#     # plt.plot(x_list[i], np.log(e_list[i] ** 2), color=(red[i], green[i], blue[i]))
# plt.legend(t_list)
# plt.show()