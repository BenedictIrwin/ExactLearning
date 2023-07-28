import scipy.integrate as integrate
from functools import partial
import numpy as np
from scipy.special import gamma, jn
from matplotlib import pyplot as plt

def integrand(x,noise): return jn(0,(x+ np.random.normal(loc = 0, scale = noise))) 
def real_integrand(x,s,k,noise): return np.real(x**(s-1)*integrand(x,noise)*np.log(x)**k)
def imag_integrand(x,s,k,noise): return np.imag(x**(s-1)*integrand(x,noise)*np.log(x)**k)
def special_int(s,order, noise):  
    r = integrate.quad(real_integrand, 0, np.inf, args=(s,order,noise), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)
    i = integrate.quad(imag_integrand, 0, np.inf, args=(s,order,noise), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)  
    return r[0]+ 1j*i[0], r[1], i[1]


s = np.linspace(-3,8,30)
n = np.array([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])
noises = []
for noise in n:
    partial_function = partial(special_int, order = 0, noise = noise)
    vectorised_integration_function = np.vectorize(partial_function)
    data = [vectorised_integration_function(s)[0] for _ in range(20)]
    moments = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    #plt.plot(s,np.clip(moments,-100,100), label = f'n={noise}')
    plt.plot(s,np.log(std), label = f'n={noise}')
    noises.append(np.log(std))


noises = [ np.corrcoef(x = nn, y = n)[0][1] for nn in np.array(noises).T ]
plt.plot(s, noises,'ko', label = f'corr')

plt.legend()
plt.show()

