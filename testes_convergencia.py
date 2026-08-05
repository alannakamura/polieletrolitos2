import gc

from polieletrolito import *
import numpy as np

error = 1e20

n = 2
while error>1e-3 :

    p = Polieletrolito(gpu=False)
    inicio1 = time.perf_counter()

    p.params['nx'] = n
    p.params['nx'] += 1
    p.params['xn'] = 30*15
    p.params['f'] = 0.05
    p.params['v'] = 0.1
    p.params['w2'] = 1500
    p.params['phib2']=1e-3
    p.params['relaxation_tax'] = 1e-4
    p.params['error'] = 1e-4
    p.params['dirichlet_boundary'] = [False, True, False, True]
    # p.params['y0'] = -4*np.pi*p.params['lb']*1e-4*30
    p.params['y0'] = -0.01
    p.params['yn'] = 0.0

    p.calculate_constants()
    res1 = p.run(p=False)
    del p
    gc.collect()

    n *= 2
    p = Polieletrolito(gpu=False)
    p.params['nx'] = n
    p.params['nx'] += 1
    p.params['xn'] = 30 * 15
    p.params['f'] = 0.05
    p.params['v'] = 0.1
    p.params['w2'] = 1500
    p.params['phib2'] = 1e-3
    p.params['relaxation_tax'] = 1e-3
    p.params['error'] = 1e-4
    p.params['dirichlet_boundary'] = [False, True, False, True]
    # p.params['y0'] = -4*np.pi*p.params['lb']*1e-4*30
    p.params['y0'] = -0.01
    p.params['yn'] = 0.0

    p.calculate_constants()
    res2 = p.run(p=False)
    del p
    gc.collect()

    error1 = res1[1][0::2] - res2[1][0::4]
    error2 = res1[1][1::2] - res2[1][1::4]
    error = np.linalg.norm(error1) + np.linalg.norm(error2)
    print(n, error)

    # print(res1[0], res2[0])
