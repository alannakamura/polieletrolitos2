from polieletrolito import *

p = Polieletrolito()

inicio1 = time.perf_counter()

filename = 'teste_artigo.pkl'
p.set_filename(filename)
# p.params['relaxation_tax'] = 0.007
p.params['relaxation_tax'] = 0.1
p.params['nx'] = 129
p.run()

fim = time.perf_counter()
tempo_total = fim - inicio1
print(f"tempo total: {tempo_total:.3f} s")