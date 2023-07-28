import numpy as np

from matplotlib import pyplot as plt


def spin_from_num(i, N):
    """
    From binary, get by bitshifting?
    """
    return [(-1)**((i>>k)&1) for k in range(N)]

# J, essentially a banded matrix, or similar.

class Hamiltonian():
    """
    Ising Model Hamiltonian
    """
    def __init__(self, J, h, N, mu):
        self.J = J
        self.h = h
        self.N = N
        self.mu = mu
        self.all_spins = None
        self.Z = None

        self.compute_all_spins()
        self.compute_partition_function()
    
    def compute_all_spins(self):
        """
        Compute all unique 2**N spin states and hold them locally
        """
        self.all_spins = np.array([spin_from_num(i,self.N) for i in range(2**self.N)])

    def compute_partition_function(self):
        """
        Compute the partition function exactly, as a partial function of beta
        """
        def pre_Z(beta):
            return np.sum( [np.exp(- beta * self.evaluate(sigma)) for sigma in self.all_spins] )
        self.Z = np.vectorize(pre_Z)

    def expectation(self, f, beta):
        """
        Returns general expectation of a function f that takes spin takes as an argument
        """
        return np.sum( [f(sigma)*self.P(sigma, beta) for sigma in self.all_spins] )

    def evaluate(self, sigma):
        """
        Evalulate the hamiltonian for one configuration
        """
        sigma_outer_product = np.einsum('i,j->ij',sigma,sigma)
        J_term = - np.einsum("ij,ij->", self.J, sigma_outer_product)
        h_term = - self.mu * np.einsum("j,j->",self.h, sigma)
        return  J_term + h_term
    
    def P(self, sigma, beta):
        """
        Returns the propbability of a single state, for a given beta
        """
        return np.exp( - beta * self.evaluate(sigma))/self.Z(beta)



# spin state example
ss = [-1,1,1,-1,1]

# TODO: Just make a class called 'hamiltonian' and ave it have methods to compute stuff,
# i.e. compute partiion function, (once, and hold it as self.Z)

N = 5
J = np.random.random(size = [N,N])
h = np.random.random(size = [N])
mu = 1
H = Hamiltonian(J=J, h=h, N=N, mu=mu)

beta = np.linspace(0,10,10000)



# TODO: Now H.P is an example of an 'integrable' function that could be 'analytically' integrated to arbitrary precision?
# Obviously we have a closed form expression above anyway for simple aspects, but getting an approximation is interesting

# TODO: Free energy is lim L-> inf, log(Z)/beta
plt.plot(beta, H.P(ss,beta), label = 'P_ss(beta)' )
#plt.plot(beta, np.log(H.Z(beta))/beta, label = 'f = (log Z)/beta' )
plt.legend()
plt.grid()
plt.show()

from moments import MomentsBundle
#from functools import partial
mb = MomentsBundle("Ising_5", upper_integration_bound=np.inf)

def partial_fun(x): return H.P(ss, x)
#partial_fun = lambda x : partial(H.P, sigma=ss)(beta = x)
mb.ingest(beta, partial_fun, y_is_function = True)
