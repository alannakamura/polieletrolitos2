from numpy import *
from matplotlib.pyplot import *

F=12
m=3
mi=0.4

def f(theta):
    return F/m*cos(theta)-mi*(9.8-F/m*sin(theta))


t = 21.8*pi/180
t2 = t-0.01
t3 = t+0.01
print(f(t2)-f(t), f(t3)-f(t))

t = linspace(0,pi/2,100)
plot(t,f(t),'r',t,ones(len(t))*f(22.8*pi/180),'b')
show()

