from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import numpy as np

@dataclass
class MomentsBundle():
    """
    Store the internalised exact learning data, ready for fitting
    Generates an interpolating function
    Generates an integration technique that can yeild moments
    """
    def __init__(self, name):

        # Definable variables
        self.name = name
        self.num_s_samples = 40
        self.s_domain = {'Re':[1,5],'Im':[-2*np.pi,2*np.pi]}
        self.max_moment_log_derivative_order = 3

        # TODO: Generalise this
        self.real_errors_exist = True
        self.imag_errors_exist = True

        # Placeholder variables
        self.num_dims = None
        self.interpolating_function = None

        # These concepts will be expanded to kth order
        self.vectorised_integration_function = [None for _ in range(self.max_moment_log_derivative_order)]
        self.moments = [None for _ in range(self.max_moment_log_derivative_order)]
        self.real_error_in_moments = [None for _ in range(self.max_moment_log_derivative_order)]
        self.imaginary_error_in_moments = [None for _ in range(self.max_moment_log_derivative_order)]

    def ingest(self,x,y):

        #TODO: Generalise this
        self.num_dims = 1

        dimension_1 = True
        if(dimension_1):
            self.x_max = np.amax(x)
            self.x_min = np.amin(x)

        # TODO: idiot_proofing etc.

        # TODO: Determine how likely it is a probability distribution

        # TODO: Understand if y!=0 at x_max and asymptotics

        # Fit interpolating form
        self.interpolating_function = interp1d(x,y, kind='cubic')
        #psi_inp_lin = interp1d(x,y)

        for k in range(self.max_moment_log_derivative_order):
            # Generate integral representation
            def real_integrand(x,s): return np.real(x**(s-1)*self.interpolating_function(x)*np.log(x)**k)
            def imag_integrand(x,s): return np.imag(x**(s-1)*self.interpolating_function(x)*np.log(x)**k)
            def special_int(s):  
                # N.B. Can use np.inf for exact equations if overriding
                r = integrate.quad(real_integrand, self.x_min, self.x_max, args=(s))
                i = integrate.quad(imag_integrand, self.x_min, self.x_max, args=(s))  
                return r[0]+ 1j*i[0], r[1], i[1]
            self.vectorised_integration_function[k] = np.vectorize(special_int)

        # TODO: add timing as optional outputs

        # Generate s-domain [once, remains fixed]
        s1 = np.random.uniform(
            low = self.s_domain['Re'][0], 
            high = self.s_domain['Re'][1], 
            size = self.num_s_samples)
        s2 = np.random.uniform(
            low = self.s_domain['Im'][0], 
            high = self.s_domain['Im'][1], 
            size = self.num_s_samples)
        
        # Array of complex s-values
        self.complex_s_samples = np.array([ t1 + t2*1j for t1,t2 in zip(s1,s2) ])

        # TODO: determine domain of validity

        # Perform the numerical integration for moments m(s)
        # Calculate m'(s), m''(s),..., m^{(k)}(s) etc. as  E[log(X)^k X^{s-1} f(x)](s)
        for k in range(self.max_moment_log_derivative_order):
            self.moments[k], self.real_error_in_moments[k], self.imaginary_error_in_moments[k] = self.vectorised_integration_function[k](self.complex_s_samples)
        


@dataclass
class ExactLearningResult():
    """
    Store an exact learning result, sort of represents a fact
    """
    def __init__(self, result_dict):
        self.equation = result_dict["equation"]
        self.num_dims = result_dict["num_dims"]
        self.complex_moment = result_dict["complex_moments"]
        self.loss = result_dict['losses']

    #TODO: Might have a method to yeild TeXForm etc.
    def __repr__(self) -> str:
        return self.equation





def ingest(x,y,name) -> MomentsBundle:
    """
    Take x and y as plt.plot would
    Interpolate, generate this function as part of moments bundle
    Determine settings (distribution etc.?)
    Convert to a dataset, via complex integration for distributions?
    """

    # determine dimensionality of data

    # dig out an example of interpolation (i.e. Schrodinger)

    # dig out an example of integration (i.e. moment fitting)

    # name is used as the folder name for this problem,
    # e.g. "Beta_Distribution"
    return MomentsBundle(name).ingest(x,y)
