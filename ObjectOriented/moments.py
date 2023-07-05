from copyreg import constructor
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import gamma, polygamma
from matplotlib import pyplot as plt
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

        # TODO: Make this a range?
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

        # TODO: Consider additional properties within the scope of this method.
        # Define None to be not True and not False
        self.pole_at_0  = None
        self.pole_at_m1 = None
        self.pole_at_m2 = None
        self.normalised = None
        self.asymptotic_to_zero = None
        self.unit_intercept = None
        self.scale_gamma_pole = None

        # TODO: Make this a single object?
        self.x_max = None
        self.x_min = None

    def ingest(self,x,y):

        #TODO: Generalise this
        self.num_dims = 1

        dimension_1 = True
        if(dimension_1):
            self.x_max = np.amax(x)
            self.x_min = np.amin(x)

        # TODO: idiot_proofing etc.

        # TODO: Determine how likely it is a probability distribution

        # TODO: Determine if it is a cumulative type function
        # Consider taking derivative

        # TODO: Understand if y!=0 at x_max and asymptotics

        # TODO: Consider order k = -1

        # TODO: Consider the case where we have a histogram rather than curve data.

        # Fit interpolating form
        self.interpolating_function = interp1d(x,y, kind='cubic')
        x = np.linspace(self.x_min,self.x_max,100)
        plt.plot(x,self.interpolating_function(x))
        plt.show()
        #psi_inp_lin = interp1d(x,y)

        # TODO: Consider fractional log order? This would surely be the fractional derivative of the mellin transform?
        # Or if it is not, then it is something else interesting?
        # For simple fractions, Split log(x)^(1/k) into interval [0,1] (Im) and [1,Infty) (Re)
        # Expressions exist for real x (WolframAlpha -> Im[Log[x]^(1/3)], alternate exprsn.)
        # Can always fit interpolating function

        from functools import partial
        def real_integrand(x,s,k): return np.real(x**(s-1)*self.interpolating_function(x)*np.log(x)**k)
        def imag_integrand(x,s,k): return np.imag(x**(s-1)*self.interpolating_function(x)*np.log(x)**k)
        def special_int(s,order):  
            # N.B. Can use np.inf for exact equations if overriding
            # TODO: Catch cases with negative x_min or something else weird
            r = integrate.quad(real_integrand, self.x_min, self.x_max, args=(s,order))
            i = integrate.quad(imag_integrand, self.x_min, self.x_max, args=(s,order))  
            return r[0]+ 1j*i[0], r[1], i[1]
        
        for k in range(-3,self.max_moment_log_derivative_order):
            partial_dict = { k : partial(special_int, order = k)}
            self.vectorised_integration_function[k] = np.vectorize(partial_dict[k])

        # Get domain of validity


        def approx_log_gamma_1(s):
            return (s-0.5)*np.log(s)-s+ np.log(np.pi * 2)/2 + 1/s/12
        
        # https://dlmf.nist.gov/5.11
        # gamma(a z + b) ~ 
        def approx_log_gamma_large(z,a,b):
            power = a*z + b - 0.5
            x = np.sqrt(2*np.pi) * np.exp(- a * z) * (a * z)**power
            return np.log(x)
        
        # |gamma(x + i y)|, y->infty, ~ 
        def abs_gamma_complex(x,y):
            return np.sqrt(2 * np.pi) * np.abs(y)**(x-0.5) * np.exp(- np.pi * np.abs(y)/2)

        # exp( delta s m) = Gamma(a s1 + b)/Gamma(a s0 + b)

        # Point estimate of large line
        s = np.array([50.0, 50.1])
        moments, re, im = self.vectorised_integration_function[0](s)
        y = np.log(moments)
        # dy/dx
        m = (y[1]-y[0])/(s[1]-s[0])
        c = y[0] - m*s[0]
        print(m,c)
        print(np.log(s) - 0.5/s)
        def line(x,m,c):
            return m*x + c


        # TODO: Consider taking the numerical derivative (like in holonomic script)
        # This might mean we don't need to integrate
        # Also a divergence between the derivate and the sampled moments would be a tell-tale
        # sign of a domain boundary.



        # Get the function max and min?
        # TODO: Get a list of points of interest
        zeros_dict = {}
        poles_dict = {}


        # TODO: Removing 'noise' which comes as spikes
        # Apply some form of smoothing (k=-3)
        s = np.linspace(1,3,70)

        # Test the accuracy
        if(False):
            for k in range(0,self.max_moment_log_derivative_order):
                moments, re, im = self.vectorised_integration_function[k](s)
                if(k == 0):
                    plt.plot(s,moments, label=f'{k}')
                    plt.plot(s,gamma(s),'k:')
                if(k == 1):
                    plt.plot(s,moments, label=f'{k}')
                    plt.plot(s,gamma(s)*polygamma(0,s),'k:')
                if(k == 2):
                    plt.plot(s,moments, label=f'{k}')
                    plt.plot(s,gamma(s)*(polygamma(0,s)**2 + polygamma(1,s)),'k:')
                plt.show()

        # TODO: Remember we are then interested in dq/q and ddq/q
        for k in range(0,self.max_moment_log_derivative_order):
            moments, re, im = self.vectorised_integration_function[k](s)
            plt.plot(s,np.log(moments),label=f'k = {k}')
            # TODO: Also get arg min (abs(f(x))
            mn, am = np.amin(moments), s[np.argmin(moments)]
            print(f"Min {k}:    {mn}")
            print(f"Argmin {k}: {am}")
            zeros_dict[k] = [mn, am]

            mx, ax = np.amax(moments), s[np.argmax(moments)]
            print(f"Max {k}:    {mx}")
            print(f"Argmax {k}: {ax}")
            poles_dict[k] = [mx, ax]


        def template_to_pole(template, result, real_error):
            # Is it a pole
            a,b,c,d,e = result
            # Get the 'sign' of the pole
            # Todo, [-ve, +ve], [+ve, -ve], [-ve, -ve], [+ve, +ve]
            left, right = 0,0
            if a<0 and b<0 : left = -1
            if a>0 and b>0 : left = 1
            if d<0 and e<0 : right = -1
            if d>0 and e>0 : right = 1
            
            error_signal = False
            # If there is a large error on the middle
            if real_error[2] > 1e-1:
                if np.sum(real_error) - real_error[2] < 1e-1:
                    error_signal = True

            increasing_test = False
            # Sharply increasing test
            if( np.abs(b)>np.abs(a) and np.abs(d) > np.abs(e)):
                increasing_test = True

            # Construct Result
            # TODO: Get derivative using the template
            if( increasing_test and error_signal):
                return True, left, right
            else:
                return False, left, right

        # Is the integration divergent at s = 0 ?
        # Template
        # Can see error is high in middle region
        template = np.array([-0.002, -0.001, 0, 0.001, 0.002])
        div0, div0_r_e, div0_im_e = self.vectorised_integration_function[0](0 + template)
        print(div0)
        print(div0_r_e)
        print(div0_im_e)

        b,l,r = template_to_pole(template,div0,div0_r_e)
        print(b,l,r)
        if(b): self.pole_at_0 = True


        if(False):
            # Contour integral template about point
            ss = np.linspace(0, 2* np.pi, 100)
            sr = 0.01*np.real(np.exp(1j*ss))
            si = 0.01*np.imag(np.exp(1j*ss))
            sq = 0.001*np.exp(1j*ss)
            print(ss)
            print(sr)
            print(si)
            q0, q0_r_e, q0_im_e = self.vectorised_integration_function[0](-2 + sq)
            print(q0)
            print(q0_r_e)
            print(q0_im_e)
            plt.plot(ss,np.real(q0),label='real')
            plt.plot(ss,np.imag(q0),label='imag')
            plt.legend()
            plt.show()

        # Is the integration divergent at s = -1 ?
        # Template
        # Can see error is high in middle region
        divm1, divm1_r_e, divm1_im_e = self.vectorised_integration_function[0](template-1)
        print(divm1)
        print(divm1_r_e)
        print(divm1_im_e)

        b,l,r = template_to_pole(template,divm1,divm1_r_e)
        print(b,l,r)
        if(b): self.pole_at_m1 = True

        divm1, divm1_r_e, divm1_im_e = self.vectorised_integration_function[0](template-2)
        print(divm1)
        print(divm1_r_e)
        print(divm1_im_e)

        b,l,r = template_to_pole(template,divm1,divm1_r_e)
        print(b,l,r)
        if(b): self.pole_at_m2 = True



        # TODO: Contour integration if needed...
        # TODO: Consider idea of Kapetyn Series
        breakpoint()

        # If k=0 at s=1, is close to 1, then likely normalised
        # TODO: Look at errors as well
        norm, _, _ = self.vectorised_integration_function[0]([1])
        print(f'norm {norm}')
        if(np.isclose(norm, 1, rtol=1e-04, atol=1e-4)):
            print("Likely normalised!")
            self.normalised = True

        # If asymptotic to zero
        y_max  = self.interpolating_function([self.x_max])
        print(f'y_max {y_max}')
        if(np.isclose(y_max, 0, rtol=1e-02, atol=1)):
            print("Likely asymptotic to zero")
            self.asymptotic_to_zero = True

        # If y_min is exactly 1 (implies pure hypergeometric)
        y_min = self.interpolating_function([self.x_min])
        print(f'y_min {y_min}')
        if(np.isclose(y_min, 1, rtol=1e-04, atol=1e-4)):
            print("Intercept is 1")
            self.unit_intercept = True

        # TODO: Case by case
        # Use FCA if needed to arrange the tests?

        # Order k = 0, argmax -> does it equal zero?
        # Implies -> gamma(a s)
        a, b = poles_dict[0]
        if( np.isclose(b,0, rtol=1e-02, atol=1) ):
            print("Scale gamma pole, gamma(a s)")

            # TODO: Could also have poles at negative integers
            # TODO: We should also have a minimum
            # TODO: Point minimisations?
            self.scale_gamma_pole = True


        # argmax -> At fraction, implies gamma(s +/- m/n)

        



        #plt.plot(s,approx_log_gamma_1(s),':k',label='line')
        #plt.plot(s,approx_log_gamma_large(s,1,0),':b',label='large')
        #plt.plot(s,line(s,m,c),':r',label='line 1')
        #plt.plot(s,line(s,m,c - np.log(2 * np.pi)/2),':b',label='line 2')
        #plt.plot(s,moments+re,':r')
        #plt.plot(s,moments-re,':r')
        #plt.plot(s, (s-1)*np.log(self.x_max), label = 'x_max')

        # TODO: of the -1 curve Zeros number is --> ?
        #zero_a = >1 < 1.46, about 1.164

        # TODO: Use these points to identify gamma(a s + b) term?
        zero_0 = 1.46163214496836 # Where logGamma'[x] = 0, x > 0
        zero_1 = -0.504083008264455 # x < 0
        zero_2 = -1.57349847316239



        # You could just fit the ansatz asymptotics ??? I.e. Sum_i loggamma(a_i + b_i *s)
        plt.legend()
        plt.show()


        # Minimise
        raise NotImplementedError
        moments, re, im = self.vectorised_integration_function[2](s)
        plt.plot(s,np.log(moments) - line(s,m,c),label=f'residual')
        plt.show()



        # For exp(-x) -> Gamma(s), we clearly have zeros for k = 1 at s = 1/2 (??? 1.477), -1/2, -3/2
        # For exp(-x) -> Gamma(s), we clearly have poles for s = 0, for k = 0, 1, 2

        # TODO: Learn a function that gives the zeros / poles of "a PolyGamma[0, a x + b]"
        # Can we use this to invert
        # Well the infinities are at 0, -1, -2, -3, etc.

        print("innovation, technical excellence, feasibility and business impact")
        # TODO: Grab some stuff from the literature review?
        raise NotImplementedError


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
