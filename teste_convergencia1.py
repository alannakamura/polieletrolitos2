from polieletrolito import *

p = Polieletrolito()

inicio1 = time.perf_counter()

# p.params['dirichlet_boundary'][0] = False
# p.params['y0'] = -9.048e-3
# p.params['dirichlet_boundary'][2] = False
# p.params['yn'] = 0
# p.params['relaxation_tax'] = 0.005
# p.params['relaxation_tax'] = 0.01
# p.params['x0'] = 0.
# p.params['xn'] = 1.

# nx inicial de cada lambda
p.params['nx'] = 3

p.params['lambda0'] = 0.01
# p.params['lambdaf'] = 1.0
# p.params['n_lambda'] = 10
p.params['convergence_error'] = 1e-2
# p.params['nx'] = 100
# filename = 'convergencia_teste_mixed'
filename = 'convergencia_teste_dirichlet2'
# filename += '_'+'p.'
filename +='.pkl'
p.set_filename(filename)
p.run2()

fim = time.perf_counter()
tempo_total1 = fim - inicio1
print('tempo total', tempo_total1)