from polieletrolito import *

p = Polieletrolito(gpu=False)

inicio1 = time.perf_counter()

# p.params['relaxation_tax'] = 0.007
p.params['nx'] = int(2**12)
p.params['nx'] += 1
p.params['xn'] = 30*4
p.params['f'] = 0.14
p.params['v'] = -1.6
p.params['w2'] = 1500
p.params['phib2']=1e-3
p.params['lambda'] = 7e-3
p.params['error'] = 1e-4
# p.params['w2'] = 1.6
print('v', p.params['v'], 'w', p.params['w2'])

# filename = 'teste_artigo1_w2_'
# filename += str(p.params['w2'])
# filename += '.pkl'

filename = "../simulacoesAntigas/290426/exp4/"
# filename = '080426/testesPhib2/'
filename += 'teste_phib2_'
filename += str(p.params['phib2'])
filename += '_w2_'
filename += str(p.params['w2'])
filename += '_xn_'
filename += str(p.params['xn'])
filename += '_nx_'
filename += str(p.params['nx'])
filename += '_lambda_'
filename += str(p.params['lambda'])
filename += '_error_'
filename += str(p.params['error'])
filename += '_a_'
filename += str(p.params['a'])
filename += '_f_'
filename += str(p.params['f'])
filename += '_v_'
filename += str(p.params['v'])
filename += '.pkl'

p.set_filename(filename)

p.calculate_constants()
p.run()

fim = time.perf_counter()
tempo_total = fim - inicio1
print(f"tempo total: {tempo_total:.3f} s")