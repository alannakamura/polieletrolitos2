from numpy import *

cm_A = 1e-8 # centimetros por angstrom

kb = 1.3806e-16 #erg/K
T = 300 #K
beta = 1/(kb*T)
# print(beta)

e = 4.803e-10 # CGS
epsilon = 80 # agua
lb = beta*e**2/epsilon # cm
print(lb, lb/cm_A)

M = 0.1e-3 # M =  molar = 1 mol/L
cm3_l = 1e3 # cm3 por litro
Na = 6.02e23 # numerod e avogrado

# l
M/=cm3_l # mol/cm^3
M*=Na # ions/cm3
# print(M)

kappa = sqrt(8*pi*lb*M)
kappa_i = 1/kappa
print(kappa_i, kappa_i)

kappa = sqrt(8*pi*lb*M)*cm_A
kappa_i = 1/kappa
print(kappa_i, kappa_i)