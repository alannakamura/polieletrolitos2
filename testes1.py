from antigas.relaxacao9 import *

# y_list = np.linspace(-0.944,-0.943,101)
y_list = np.linspace(-1.0,1.0,11)
p = Polieletrolito()

inicio1 = time.perf_counter()
for i in y_list:
    inicio2 = time.perf_counter()
    p.params['y0'] = i
    filename = 'res_' + f"{p.params['y0']:.10f}" + '.pkl'
    p.set_filename(filename)
    p.params['relaxation_tax'] = 0.005
    p.params['nx'] = 100
    p.run()

    fim = time.perf_counter()
    tempo_total1 = fim - inicio1
    tempo_total2 = fim - inicio2
    print('y0', i, f"O processo levou {tempo_total2:.4f} segundos.", 'total', tempo_total1)