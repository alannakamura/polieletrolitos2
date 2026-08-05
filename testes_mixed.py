from polieletrolito import *

p = Polieletrolito()

inicio1 = time.perf_counter()

p.params['dirichlet_boundary'][0] = False
p.params['y0'] = -9.048e-3
p.params['dirichlet_boundary'][2] = False
p.params['yn'] = 0
filename = 'teste_mixed.pkl'
p.set_filename(filename)
p.params['relaxation_tax'] = 0.005
p.params['nx'] = 5
p.run()

fim = time.perf_counter()
tempo_total = fim - inicio1
print(f"tempo total: {tempo_total:.3f} s")