from dataclasses import dataclass
from types import NoneType
from typing import Iterable, Tuple, Callable
#from Holonomic.pytorch_fit_holonomic import training_loop
#TODO: Work on the holonomic routine as above

#from ObjectOriented.AdvancedFunctions import trigamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d, splrep, splev, Akima1DInterpolator, BarycentricInterpolator, KroghInterpolator, CubicHermiteSpline, PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.special import gamma, polygamma, digamma
from sklearn.metrics import r2_score
from scipy.optimize import root, minimize
from scipy.stats import f
from AdvancedFunctions import trigamma_vec as trigamma 
from AdvancedFunctions import tetragamma_vec as tetragamma 
from matplotlib import pyplot as plt
#from scipy.interpolate import approximate_taylor_polynomial
import numpy as np



def get_derivatives(root_polynomial : str, order : int) -> dict:
    """
    Populate a dictionary input with strings representing the derivatives
    root_polynomial : string to take analytic derivatives of
    order : 
    """
    from sympy import symbols
    from sympy.parsing.sympy_parser import parse_expr
    s, p, sym_p = symbols('s'), {}, parse_expr(root_polynomial)
    for i in range(order+1):
        p[i] = str(sym_p.diff(s,i))
    return p

@dataclass
class MomentsBundle():
    """
    Store the internalised exact learning data, ready for fitting
    Generates an interpolating function
    Generates an integration technique that can yeild moments
    """
    def __init__(self, name, upper_integration_bound = np.inf):

        # Definable variables
        self.name = name.replace('/','-')
        self.num_s_samples = 100
        self.s_domain = {'Re':[1,5],'Im':[-2*np.pi,2*np.pi]}

        # TODO: Make this a range?
        self.max_moment_log_derivative_order = 4

        # Sample range for plotting moments as function of s
        self.moments_sample_s_min = -10
        self.moments_sample_s_max = 10
        self.moments_sample_n_samples = 2000

        self.max_integration = upper_integration_bound

        # TODO: Generalise this
        self.real_errors_exist = True
        self.imag_errors_exist = True

        # Placeholder variables
        self.num_dims = None
        self.interpolating_function = None

        # Mostly for pictorial and checking
        self.moments_interpolating_function = [None for _ in range(self.max_moment_log_derivative_order)]

        # These concepts will be expanded to kth order
        self.vectorised_integration_function = [None for _ in range(self.max_moment_log_derivative_order)]
        self.vectorised_integration_function_an = [None for _ in range(self.max_moment_log_derivative_order)]
        self.vectorised_trap = [None for _ in range(self.max_moment_log_derivative_order)]
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
        self.root_polynomial = None
        self.poly_derivatives = []

        # TODO: Try to scan how many times the moments cross 0 and where they do
        self.moments_have_roots = None # Implies moments have a polynomial or other non-meromorphic term of order "num roots"

        # TODO: Make this a single object?
        self.x_max = None
        self.x_min = None

    def ingest(self,x,y, y_is_function = False):

        # Truncated the range
        #print(x.shape)
        #x = x[:10000]
        #y = y[:10000]

        #TODO: Generalise this
        self.num_dims = 1

        dimension_1 = True
        if(dimension_1):
            self.x_max = np.amax(x)
            self.x_min = np.amin(x)

        # TODO: idiot_proofing etc. Crashes like out of bounds etc.

        # TODO: Determine how likely it is a probability distribution

        # TODO: Determine if it is a cumulative type function
        # Consider taking derivative

        # TODO: Understand if y!=0 at x_max and asymptotics

        # TODO: Consider order k = -1
        # Need to know what expression to write to cancel out terms

        # TODO: Consider the case where we have a histogram rather than curve data.

        # Fit interpolating form -------------------
        # TODO: We have many options for the tail density where it is cut short
        # We can try a range of values from 'zero everywhere' to, linear tail, to exponential
        # or power law tail, pade tail etc. But the main thing is to get an ensemble of values
        # and consider the variance in the 'Mellin transform' as a result.

        def test_function(x): return np.exp(-1/(2*x))/2/x**2
        def test_function(x): return np.sqrt(x) * np.exp(-x/2) / np.sqrt(2 * np.pi)

        #self.interpolating_function = interp1d(x, y, kind='cubic', bounds_error = False, fill_value = 0)
        if(y_is_function == False):
            self.spl = splrep(x, y, k=3)
            self.interpolating_function = lambda x : splev(x, self.spl)

        else:
            self.interpolating_function = y
        #self.interpolating_function = Akima1DInterpolator(x,y)
        #self.interpolating_function = BarycentricInterpolator(x,y)
        #self.interpolating_function = KroghInterpolator(x,y) # Did not finish
        #self.interpolating_function = CubicHermiteSpline(x,y) # Do not have dydx
        #self.interpolating_function=PchipInterpolator(x,y)
        #from sklearn.preprocessing import SplineTransformer
        #from sklearn.linear_model import Ridge
        #from sklearn.pipeline import make_pipeline

        #self.interpolating_function = SplineTransformer(knots = 'uniform', degree = 5).fit(x[:,None],y)
        #self.interpolating_function = make_pipeline(SplineTransformer(n_knots=300, degree=5), Ridge(alpha=1e-3))
        #self.interpolating_function.fit(x[:,None],y)



        # TODO: What we want is a way to rule out the moments datapoints that shift drastically under the change of 
        # interpolation technique. These are not trustworthy/pure, as we don't want to rely on them to fit our final
        # 'exact' moments representation. 
        # We are never going to 'prefectly' reconstruct the moments space, whatever interpolation we use, only with an
        # exact equation, could we probably evalulate the integrals to that level of precision.

       
        if(False):
            # TODO: Try Fitting '100' smaller - processes, with a small amount of overlap
            # Split the interpolating function into a bunch of regions and lookup the relevant process?
            def x_y_to_sections(x,y, n_sections=10, overlap = 2):

                # Get sizes
                N = len(x)   # if 100, 10 chunks of 

                # n sections, each with k data points
                # Each section also has 2 on the left and 2 on right
                # Total data = N
                # Left: All datapoints and repeats
                # N + 2o * (n-2) + 2o/2 = (n-2) * ( k + 2o) + 2*(k+o)
                # Find expression for k
                # k =

                chunk_size = int((N - overlap)/n_sections)
                x_list = []
                y_list = []

                x_list.append(x[:chunk_size + overlap])
                y_list.append(y[:chunk_size + overlap])
                for i in range(1,n_sections-1):
                    x_list.append(x[i*chunk_size - overlap:(i+1)*chunk_size + overlap])
                    y_list.append(y[i*chunk_size - overlap:(i+1)*chunk_size + overlap])
                x_list.append(x[chunk_size*(n_sections-1)-overlap:])
                y_list.append(y[chunk_size*(n_sections-1)-overlap:])


                return [np.array(xx) for xx in x_list], [np.array(yy) for yy in y_list]


        #x_list, y_list = x_y_to_sections(x,y, 20, 10)
        #GP_list = []
        #from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        #print("Fitting GPs")
        #for xx,yy in zip(x_list,y_list):
        #    print("Fitting!")
        #    #TODO: https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr
        #    # TODO: Set legnth scale to be 1/'delta x'
        #    gp = GaussianProcessRegressor(kernel= ConstantKernel(1, (0.0001, 1.0)) * RBF(length_scale=1, length_scale_bounds=(1e-7, 1e-3)), alpha=1e-6).fit(xx[:,None], yy)
        #    GP_list.append(gp)
        #    print("Done Fitting")

        #x_start = self.interpolating_function.x[0]
        x_end = x[-1]
        #y_start = self.interpolating_function.y[0]
        if(y_is_function == False):
            y_end = y[-1]
        else:
            breakpoint()
            y_end = y(x_end)

        #plt.plot(x,y,'ro')
        # TODO: Do not use variables called x and y
        x = np.linspace(self.x_min,self.x_max,10000)
        x_dash = np.linspace(0.0,2*self.x_max,10000)
        # TODO: Design a function that represents a tail locally.
        # Like 1/log(x)
        # When x = E, this is 1
        # y_end / log(E + x - x_end)

        # We can fit it to the last few data points ? Try to extrapolate knowing it will asymptote to 0
        # Could fit with power and log power term, and exponential, and even a sum of terms.
        # I.e. a local 'ML' but windowed Pade or whatever
        # Long term just train a network to smooth tails correctly given masked data.
        def logtail(x, a): 
            return y_end / np.log(np.exp(1) + x - x_end)**a
        def powertail(x, a): 
            return y_end / (1 + x - x_end)**a
        def exptail(x, s): raise NotImplementedError

        if(False):
            x_dd = np.linspace(x_end, 2*self.x_max, 1000)
            
            #print(self.interpolating_function([self.x_min,self.x_max]))
            #idx =0
            #for xx,yy,gp in zip(x_list,y_list,GP_list):
            #    gp_pred, gp_var = gp.predict(xx.reshape([-1,1]), return_std=True)
            #    plt.plot(xx,gp_pred)
            #    plt.plot(xx,gp_pred + gp_var, 'k:')
            #    plt.plot(xx,gp_pred - gp_var, 'k:')
            #    idx += 1
            plt.plot(x,self.interpolating_function(x),label='interp.')
            #plt.plot(x_dash,self.interpolating_function.predict(x_dash.reshape([-1,1])),'--',label='interp. ext')
            plt.plot(x_dash, test_function(x_dash), 'r--', label='exact')
            #for a in [0.5,1,1.5]:
            #    plt.plot(x_dd, logtail(x_dd, a), label = f'Logtail guess a = {a}')
            #for a in [0.3,0.4,0.5]:
            #    plt.plot(x_dd, powertail(x_dd, a), label = f'Powertail guess a = {a}')

            plt.legend()
            plt.show()

        # TODO: Develop an ensemble of tails approach

        # TODO: Develop a good tail extrapolation routine
        #train_x
        #val_x
        #test_x


        # TODO: Examine the spread of moments under different tail approximations
        def tail_extrapolate_one(x_i, tail_parameter):
            ret = []
            #for x_i in x:
            # TODO: Make it a linear, or even a left tail right tail type thing using ther same smoothing model as above.
            if(x_i < self.x_min): ret = 0
            if(x_i > self.x_max): ret = logtail(x_i, tail_parameter)
            #if(x_i > self.x_max): ret = np.sqrt(x_i) * np.exp(-x_i/2) / np.sqrt(2 * np.pi)
            else:
                ret = self.interpolating_function(x_i) 
                #for xx,gp in zip(x_list, GP_list):
                
                # Would need to make this much quicker for GP
                #   if(x_i < np.amax(xx)):
                #        ret = gp.predict([[x_i]])
                #        break
            return ret

        if(y_is_function == False):
            self.adjusted_interpolating_function = tail_extrapolate_one
        else:
            # TODO: Clean up this
            self.adjusted_interpolating_function = self.interpolating_function


        #psi_inp_lin = interp1d(x,y)

        # TODO: Consider fractional log order? This would surely be the fractional derivative of the mellin transform?
        # Or if it is not, then it is something else interesting?
        # For simple fractions, Split log(x)^(1/k) into interval [0,1] (Im) and [1,Infty) (Re)
        # Expressions exist for real x (WolframAlpha -> Im[Log[x]^(1/3)], alternate exprsn.)
        # Can always fit interpolating function

        # TODO: Add a variation of tail parameter 'a'
        from functools import partial
        def real_integrand(x,s,k,a): return np.real(x**(s-1)*self.adjusted_interpolating_function(x,a)*np.log(x)**k)
        def imag_integrand(x,s,k,a): return np.imag(x**(s-1)*self.adjusted_interpolating_function(x,a)*np.log(x)**k)
        def special_int(s,order,a):  
            # N.B. Can use np.inf for exact equations if overriding
            # TODO: Catch cases with negative x_min or something else weird
            r = integrate.quad(real_integrand, 0, self.max_integration, args=(s,order,a), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)
            i = integrate.quad(imag_integrand, 0, self.max_integration, args=(s,order,a), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)  
            return r[0]+ 1j*i[0], r[1], i[1]
        
        for k in range(-3,self.max_moment_log_derivative_order):
            partial_dict = { k : partial(special_int, order = k)}
            self.vectorised_integration_function[k] = np.vectorize(partial_dict[k])

        if(False):
            # This did not really work
            from functools import partial
            # Define 'integration points'
            def trap_int(s,order):  
                # N.B. Can use np.inf for exact equations if overriding
                # TODO: Catch cases with negative x_min or something else weird
                X = [xx for xx in x]
                Y = [xx**(s-1)*yy*np.log(xx)**order for xx,yy in zip(x,y)]
                Yr = np.real(Y)
                Yi = np.imag(Y)
                r = integrate.simpson(Yr, X)
                i = integrate.simpson(Yi, X)  
                return r+ 1j*i#, r[1], i[1]
            
            for k in range(-3,self.max_moment_log_derivative_order):
                partial_dict = { k : partial(trap_int, order = k)}
                self.vectorised_trap[k] = np.vectorize(partial_dict[k])



        def c(x): return np.clip(np.real(x),-10,10)
        def d(x): return np.clip(np.imag(x),-10,10)
        test_function_dict = {
            "inv_chi_sq" : "np.exp(-1/(2*x))/2/x**2",
            "exp" : "np.exp(-x)",
        }
        test_moments_dict = {
            "inv_chi_sq" : "2**(1-s) * gamma(2-s)",
            "exp" : "gamma(s)",
        }

        def bound(a,s1,b,s2,c):
            if isinstance(b,Iterable):
                if(s1 == 'le' and s2 == 'le'):
                    return np.array([ a <= xx <= c for xx in b])
                if(s1 == 'l' and s2 == 'le'):
                    return np.array([ a < xx <= c for xx in b])
                else:
                    raise Exception
            else:
                if(s1 == 'le' and s2 == 'le'):
                    return a <= b <= c
                if(s1 == 'l' and s2 == 'le'):
                    return a < b <= c 
        
        def square_line_picking(x):
            b1 = bound(0,'le',x,'le',1)
            b2 = bound(1,'l',x,'le',np.sqrt(2))
            t1 = (2*x *(x**2 - 4 * x + np.pi))
            t2 = np.nan_to_num(  (2*x*(4*np.sqrt(x**2 - 1)-(x**2 + 2 - np.pi)-4*np.arctan(np.sqrt(x**2 - 1)))) )
            return 0 + t1*b1 + t2*b2


        

        if(True):
            # An analytic comparison for testing the integrals over interpolating functions
            # Perhaps they are breaking down...
            # def test_function(x): return np.exp(-1/(2*x))/2/x**2  #inv chi sq
            #def test_function(x): return np.sqrt(x) * np.exp(-x/2) / np.sqrt(2 * np.pi)  # chi-sq k=3
            def test_function(x): return square_line_picking(x)  # chi-sq k=3
            #def test_function(x): return 12* (1-x)**2 * np.heaviside(1-x,0.5) # Beta(2,3) , might work
            #def test_function(x): return 1.0/np.sqrt(x)/(2 + x)**(3/2) # F-ratio
            #def test_function(x): return np.exp(-x) # Exponential
            def real_integrand_an(x,s,k): return np.real(x**(s-1)*test_function(x)*np.log(x)**k)
            def imag_integrand_an(x,s,k): return np.imag(x**(s-1)*test_function(x)*np.log(x)**k)
            def special_int_an(s,order):  
                # N.B. Can use np.inf for exact equations if overriding
                # TODO: Catch cases with negative x_min or something else weird
                r = integrate.quad(real_integrand_an, 0, self.max_integration, args=(s,order), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)
                i = integrate.quad(imag_integrand_an, 0, self.max_integration, args=(s,order), complex_func=True, limit=100, epsabs = 1e-16, epsrel=1e-16)  
                return r[0]+ 1j*i[0], r[1], i[1]
            
            for k in range(-3,self.max_moment_log_derivative_order):
                partial_dict = { k : partial(special_int_an, order = k)}
                self.vectorised_integration_function_an[k] = np.vectorize(partial_dict[k])


        #s_analytic = np.linspace(-10,2,100)
        #m_analytic = self.vectorised_integration_function_an[0](s_analytic)[0]
        # Look for poles?
        #plt.plot(s_analytic,np.clip(m_analytic,-10,10))
        #plt.grid()
        #plt.show()
        #exit()

        # TODO: Consider 'Integrating functions, given fixed samples' in scipy, i.e. do not use any interpolating method
        if(False):
            # Experiment to see if varying the tail varies the moments
            # Test
            s_old = np.linspace(-1,5,100)
            s_2 = np.linspace(-1,5,200)
            s_5 = np.linspace(-1,5,500)
            for a in [10]:
                q , _, _ = self.vectorised_integration_function[0](s_5, a=a)
                print(q)
                plt.plot(s_5,c(np.real(q)),label=f'{a}')
                #np.save('pchip_10m.npy',q)

            q10_orig = np.load('q10_orig.npy')
            q10_k5_60k = np.load('q10_k5_60k.npy')
            q10_k5_600k = np.load('q10_k5_600k.npy')
            q10_k5_6m = np.load('q10_k5_6m.npy')
            pchip10m = np.load('pchip_10m.npy')
            plt.plot(s_old,c(q10_orig), 'b:', label = 'q10_orig')
            plt.plot(s_old,c(q10_k5_60k), 'g:', label = 'q10_k5_60k')
            plt.plot(s_2,c(q10_k5_600k), 'g--', label = 'q10_k5_600k')
            plt.plot(s_2,c(q10_k5_6m), 'b--', label = 'q10_k5_6m')
            plt.plot(s_2,c(pchip10m), '--', label = 'pchip_10m')
            q_an, _, _ = self.vectorised_integration_function_an[0](s_5)
            plt.plot(s_5,c(np.real(q_an)),'k--',label=f'Analytic Num. int.')
            plt.plot(s_5,c(2**(s_5-1) * gamma(1/2+s_5)/gamma(3/2)), 'r:', label = 'correct')
            plt.plot([min(s_5),max(s_5)],[0,0],'k:')
            plt.xlim([-1,1])
            plt.legend()
            plt.show()


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

        # Generate a space for investigation
        s = np.linspace(self.moments_sample_s_min,
                        self.moments_sample_s_max,
                        40)
        
        # Wick Rotation type effect to view complex 
        if(False):
            s = 1 + 1j*s

        print("If one derivative is a sum of step functions")
        print("The next derivative is a sum of delta functions")

        print("Need to cap the range, in case it affects the integration")
        print("Need to compare against numericised versions of standard sums of decaying exponentials etc.")
        print("Consider making an adaptive integration scheme when most of the function is 0")

        
        # Test the accuracy
        if(True):
            q, re1, im1 = self.vectorised_integration_function[0](s,a=1)
            dq, re2, im2 = self.vectorised_integration_function[1](s,a=1)
            ddq, re3, im3 = self.vectorised_integration_function[2](s,a=1)
            dddq, re4, im4 = self.vectorised_integration_function[3](s,a=1)

            # TODO: We definately notice a discrepancy in the integration of certain functions.
            # This indicates the sampling has problems, probably for very sharp or divergent functions
            # Potentially the tail not long enough, or not exactly zero at the origin etc.
            
            q_an, re1a, im1a = self.vectorised_integration_function_an[0](s)
            dq_an, re2a, im2a = self.vectorised_integration_function_an[1](s)
            ddq_an, re3a, im3a = self.vectorised_integration_function_an[2](s)
            dddq_an, re4a, im4a = self.vectorised_integration_function_an[3](s)



            # Define local s for quick plotting changes
            #s_loc = np.imag(s)
            s_loc = s

            if(True):
                plt.title("Errors in Integration - For Input Curve")
                plt.plot(s_loc, np.log(re1), label = 're1')
                plt.plot(s_loc, np.log(re2), label = 're2')
                plt.plot(s_loc, np.log(re3), label = 're3')
                plt.plot(s_loc, np.log(re4), label = 're4')
                plt.plot(s_loc, np.log(im1), label = 'im1')
                plt.plot(s_loc, np.log(im2), label = 'im2')
                plt.plot(s_loc, np.log(im3), label = 'im3')
                plt.plot(s_loc, np.log(im4), label = 'im4')
                plt.legend()
                plt.show()

            # TODO: Establish a weight function from the errors profile
            # I.e. flat bottom region is very good

            try:
                s_loc_filter = s_loc[np.log(re1) < -15]
            except:
                print("The integration domain is not defined due to poor convergence of numerical integration under current assumptions.")
                raise NotImplementedError
            try:
                low_s = np.amin(s_loc_filter)
                high_s = np.amax(s_loc_filter)
            except:
                print("Skipping!")
                exit()
                return

            # Resample these values
            s = np.linspace(low_s, high_s, 200)
            s_loc = s

            # Set this as the actual sampling domain
            self.s_domain['Re'][0] = low_s
            self.s_domain['Re'][1] = high_s

            q, re1, im1 = self.vectorised_integration_function[0](s,a=1)
            dq, re2, im2 = self.vectorised_integration_function[1](s,a=1)
            ddq, re3, im3 = self.vectorised_integration_function[2](s,a=1)
            dddq, re4, im4 = self.vectorised_integration_function[3](s,a=1)

            test_data = True

            if(test_data):
                # TODO: We definately notice a discrepancy in the integration of certain functions.
                # This indicates the sampling has problems, probably for very sharp or divergent functions
                # Potentially the tail not long enough, or not exactly zero at the origin etc.
                q_an, re1a, im1a = self.vectorised_integration_function_an[0](s)
                dq_an, re2a, im2a = self.vectorised_integration_function_an[1](s)
                ddq_an, re3a, im3a = self.vectorised_integration_function_an[2](s)
                dddq_an, re4a, im4a = self.vectorised_integration_function_an[3](s)

                # TODO: Do create a sliding method that detects sudden flips in polarity
                # I.e. pole detector.

                def local_moment(s):
                    #return 2**(1-s) * gamma(2-s) # Inv-chi_sq k=2
                    return 2**(s-1) * gamma(0.5+s)/gamma(3/2) # chi_sq k=3
                



            #plt.plot(s, c(np.real(q)), 'b:',label = 'numeric')
            #plt.show()
            # Need to solve why they are not the same...
            if(True):
                plt.title("Real part of moments")
                plt.plot(s, c(np.real(q)),label = 'numeric')
                plt.plot([min(s),max(s)],[0,0],'k:')
                if(test_data):
                    plt.plot(s, c(re1), label = 'real error numeric')
                    plt.plot(s, c(q_an), 'r-', label = 'anyl. int.')
                    plt.plot(s, c(re1a), label = 'real error analytic')
                    plt.plot(s, c(local_moment(s)),'k:', label = 'theoretical')
            #plt.plot(mma_s, mma_res, 'ro', label= 'MMA n. int.')
            plt.legend()
            plt.show()

            #exit()

            def root_wrapper(s):
                res, _, _ = self.vectorised_integration_function[0](s,a=1)
                return np.real(res)
            def pole_wrapper(s):
                res, _, _ = self.vectorised_integration_function[0](s,a=1)
                return 1.0/np.real(res)

            # Find roots of the moments function
            # These can be directly forwarded to the algorithm
            # (or even divided out directly for simplicity)
            # Root Finding
            potential_root_list = []
            evaluated_root_list = []
            potential_pole_list = []
            evaluated_pole_list = []
            for i in range(len(q)-1):
                m1, m2 = np.real(q)[i], np.real(q)[i+1]
                if(np.sign(m1) != np.sign(m2)):
                    #TODO: Rule out 'poles' but add them to 'potential pole list'
                    # Criterion could be ??? 
                    critereon = np.abs(m1) + np.abs(m2) < 10*(s[i+1]-s[i])
                    if(not critereon): 
                        potential_pole_list.append(i)
                        plt.plot(s[i],0,'bo')
                        res = root(pole_wrapper, x0=s[i])
                        print(res)
                        if(res.success):
                            plt.plot(res.x[0],0,'bo')
                            plt.arrow(s[i],0,res.x[0]-s[i],0)
                            evaluated_pole_list.append([res.x[0],res.fun[0]])
                    else:
                        potential_root_list.append(i)
                        plt.plot(s[i],m1,'ro')
                        res = root(root_wrapper, x0=s[i])
                        print(res)
                        if(res.success):
                            plt.plot(res.x[0],res.fun[0],'ro')
                            plt.arrow(s[i],m1,res.x[0]-s[i],res.fun[0]-m1)
                            evaluated_root_list.append([res.x[0],res.fun[0]])

            print("Roots: -------")
            for i in potential_root_list:
                print(s[i],q[i])
            print(evaluated_root_list)

            print("Poles: -------")
            for i in potential_pole_list:
                print(s[i],q[i])
            print(evaluated_pole_list)

            self.num_poly_terms = len(evaluated_root_list)
            self.num_evaluated_poles = len(evaluated_pole_list)

            # Logic to remove a polynomial cofactor from 'q'
            if(self.num_poly_terms > 0):
                # Divide through by polynomial
                root_polynomial = ""
                for i, rt in enumerate(evaluated_root_list):
                    root_polynomial += f"(s-{rt[0]})"
                    if(i<len(evaluated_root_list)-1): root_polynomial+="*"

                self.root_polynomial = root_polynomial
                poly = eval("lambda s: "+root_polynomial)
                poly_data = np.array([poly(ss) for ss in s])
                plt.plot(s,c(poly_data),label='Root Polynomial')
                plt.plot(s,c(np.real(q)/poly_data),label='Factorized Moments')


                if(True):
                    print(f"Renormalising, removing order-{len(evaluated_root_list)} polynomial!")

                    # chi_0 = phi_0/p(s)
                    #q = q/poly_data

                    print("Careful for analysis of dq, etc. which are not normalised?")
                    self.poly_derivative_expressions = get_derivatives(self.root_polynomial, order = 3)
            
                    p = {}
                    poly_order = 3
                    for i in range(poly_order + 1):
                        poly = eval("lambda s: "+self.poly_derivative_expressions[i])
                        p[i] = np.array([poly(ss) for ss in s])
                    
                    self.poly_samples = p
                    
                    # TODO: Check the polynomial in the complex plane makes sense?
                    # TODO: Extend to any dimension up to say max_root_order=10
                    # TODO: Error estimates will be a bit wrong!
                    chi = {}

                    # Simply divide through chi_0 = phi_0/p_0
                    chi[0] = q/p[0]

                    # chi_1 from chi_1/chi_0 = phi_1/phi_0 - p_1/p_0
                    chi[1] = (dq/q - p[1]/p[0]) * chi[0]

                    # Rearrange phi_2/phi = D[p(s)chi(s),{s,2}]/p/chi
                    chi[2] = (ddq/q - 2.0 * chi[1]/chi[0] * p[1]/p[0] - p[2]/p[0]) * chi[0]

                    # Rearrage phi_3/phi_0 = D[p(s)chi(s),{s,2}]/(p_0chi_0)
                    chi[3] = (dddq/q - 3.0 * chi[2]/chi[0] * p[1]/p[0] - 3.0 * chi[1]/chi[0] * p[2]/p[0] - p[3]/p[0]) * chi[0]

                    q = chi[0]
                    dq = chi[1]
                    ddq = chi[2]
                    dddq = chi[3]

                    # phi_1 / phi_0  

            plt.legend()
            plt.show()

            # Take the factorised expression.
            # Plot phi(s+1)/phi(s) , which may be a polynomial or rational expression?
            # The order of the polynomial on top and bottom is important, because it is derived from
            # a in Gamma (as + b) for integer a, and the total number of gamma on top and bottom.
            
            # We can fit the moments for easy shifting
            #OMG, still using interp1d here... 
            #phi_0 = self.normalised_moments_interpolation = interp1d(s, q, kind='cubic',  bounds_error = False, fill_value = 0)
            #phi_1 = self.normalised_moments_interpolation = interp1d(s, dq, kind='cubic',  bounds_error = False, fill_value = 0)

            if(False):
                ## Define a secret function 'moments'
                #def phi(s, v):
                #    return v[0]*v[1]**s*gamma(v[2] + v[3]*s)

                #def logdivphi(s,v):
                #    return log(v[1]) + v[3]*digamma(v[2] + v[3]*s)

                ## EulerGamma
                g = 0.57721566490153286060651209008240243104216
                c1,c2,c3,c4 = np.random.uniform(0,1,size=[4])
                #v = np.random.uniform(0,1,size=[4])
                print(c1,c2,c3,c4)

                p0 = phi_0(0)
                #p1 = phi(1, v)

                for i in range(1000):
                    ## Solve
                    #p0 = phi(0, v)
                    #dp0 = logdivphi(0, v)

                    t1 = phi_0((1-c3)/c4)
                    t2 = phi_0((2-c3)/c4)
                    ##log_term = np.log(t2/t1) / np.log(c2) 

                    th1 = phi_1((1-c3)/c4)/t1
                    th2 = phi_1((2-c3)/c4)/t2

                    c3_new = t1*phi_0(1/c4)/p0/t2


                    #c2_new = t2**c4/t1**c4
                    #c2_new = p1/c1/gamma(c3+c4)
                    #c2_new = exp(dp0 - c4*digamma(c3))
                    c1_new = p0/gamma(c3)
                    #c4_new = 1/log_term
                    #c4_new = (dp0 - log(c2))/digamma(c3)
                    #c4_new = (dp0 - log(c2_new))/digamma(c3)# + 0.05*(th2-th1)/(1-2*g)
                    c4_new = th2 - th1
                    c2_new = np.exp((th2 + th1 - c4_new*(1 - 2*g))/2)

                    c1 = c1_new
                    c2 = c2_new
                    c3 = c3_new
                    c4 = c4_new
                    
                    print(c1,c2,c3,c4)

                plt.plot(s,phi_0(s),label = 'true')
                plt.plot(s,c1 * c2**s * gamma(c3 + c4 * s),label = 'fit')
                plt.legend()
                plt.show()


            # Inverted Function as a line
            #cc_inv = phi_0(0)/phi_0(1)
            #mm_inv = phi_0(1)/phi_0(2)


            #cc = phi_0(1)/phi_0(0)
            #mm = phi_0(2)/phi_0(1)-cc

            #print(mm,cc)
            #print(mm_inv, cc_inv)

            """
            if  Gamma+ Gamma-
                fit rational 
                if |a| = 1 , C (a + bs )/(c + ds )
                C is C^s

            if( a is 1):
                if(line_correlation_high):
                    cc = power_constant_base * b
                    mm = power_constant_base
                    C * mm**s * gamma(s + cc/mm)
            if( a is -1):
            
            if( a is 2):
                If second order polynomial

            if( a is -2):

            if( a is integer)

            # TODO: CONSIDER polynomial terms as well

            # TODO: Consider rational optimisation overpolygamma ratios as well. At constant s
            # I.e. solve for constants
            """

            # TODO: Make a poly fit routine
            # TODO: Make a rational fit routine
            # TODO: factor out any zeros or poles.
            if(True):
                from scipy.optimize import curve_fit
                
                def line(x,a,b): return a * x + b

                # TODO: Check the coefficients order
                def poly2(x, a, b, c): return c* x**2 + b * x + a

                def gamma_ratio(s, a, b, c): 
                    """
                    Ratio of gamma functions is well behaved and stronger generalisation of a line
                    a = 1 is a line
                    a = 2 is a poly_2
                    a = fractional -> Not representable by simple poly
                    """
                    return  np.real(c * gamma(a * (s+1) + b)/gamma(a * s + b))
                
                def gamma_ratio_2(s, a, b, c, d, e): 
                    """
                    Ratio of gamma functions is well behaved and stronger generalisation of a line
                    a = 1 is a line
                    a = 2 is a poly_2
                    a = fractional -> Not representable by simple poly
                    """
                    return  np.real(c * (gamma(a * (s+1) + b)/gamma(a * s + b)) * (gamma(d * (s+1) + e)/gamma(d * s + e)))

                # This domain
                s_loc = np.linspace(min(s),max(s)-1,200)
                top, rett, iett = self.vectorised_integration_function[0](s_loc+1,a=1)
                bottom, rebb, iebb = self.vectorised_integration_function[0](s_loc,a=1)


                if(self.num_poly_terms > 0):
                    p = {}
                    p_shift = {}
                    poly_order = 3
                    for i in range(poly_order + 1):
                        poly = eval("lambda s: "+self.poly_derivative_expressions[i])
                        p[i] = np.array([poly(ss) for ss in s_loc])
                        p_shift[i] = np.array([poly(ss + 1) for ss in s_loc])

                    # Try cancelling out the polynomial
                    y_true = top*p[0]/bottom/p_shift[0]

                else: y_true = top/bottom

                # Get the extended analytic integral
                #s_ext = np.linspace(min(s)-4,max(s)+6,400)
                #y_true_ext = self.vectorised_integration_function_an[0](s_ext+1)[0]/self.vectorised_integration_function_an[0](s_ext)[0]

                #plt.plot(s_ext, c(y_true_ext), label = 'extended')
                #plt.legend()
                #plt.show()

                # TODO: IF we know there is a pole, we should be careful when fitting.
                # TODO: Calculate the errorin the ratio...

                #well_behaved_region = [a,b] and [c,d] and [e,f]


                if(True):
                    popt = (1,1,1)
                    popt_2 = (1,1,1,1,1)
                    try:
                        popt, pcov = curve_fit(gamma_ratio, s_loc, y_true, p0=popt)
                    except:
                        print("Warning! -> Gamma Ratio Exception")
                    try:
                        popt_2, _ = curve_fit(gamma_ratio_2, s_loc, y_true, p0=popt_2)
                    except:
                        print("Warning! -> Gamma Ext. Exception")
                    print("gamma rat:", popt)
                    print("gamma rat_2:", popt_2)
                    

                    plt.plot(s_loc, c(y_true), label = 'correct')
                    #plt.plot(s_ext, y_true_ext, label = 'extended')
                    plt.plot(s_loc, gamma_ratio(s_loc, *popt), label = 'gamma ratio')
                    plt.plot(s_loc, gamma_ratio_2(s_loc, *popt_2), label = 'gamma ratio 2')
                    plt.legend()

                    plt.grid()
                    plt.show()

                self.gamma_ratio_popt = popt
                self.y_true = y_true
                self.s_loc = s_loc
                return



                # TODO: Fit in the complex plane!
                if(True):

                    fit_dict = {}
                    init_params = {}
                    popt = {}
                    pcov = {}
                    r2 = {}
                    ssq = {}
                    df = {}
                    # Create an automated rational fitter.
                    for a1 in range(4):
                        for a2 in range(4):
                            for b1 in range(4):
                                for b2 in range(4):
                                    total_num_params = 1+a1+a2+b1+b2
                                    if(total_num_params == 1): 
                                        # skip the trival case? m(s) = c1 c2^s -> f(x) = Delta
                                        continue
                                    fit_string = f"{a1}-{a2}-{b1}-{b2}"
                                    # TODO: Add a single scale parameter
                                    # Use up the parameters in the order a1,a2,b1,b2
                                    total_num_params = 1+a1+a2+b1+b2
                                    init_params = [1 for _ in range(total_num_params)]

                                    #TODO: Write a function file
                                    # load this_function

                                    function_list = [
                                    "def fit_function(s,{}):".format(",".join([f"p{i}" for i in range(total_num_params)])),
                                    " y = p0", " eps = 1e-8"]
                                    p_idx = 1
                                    for _ in range(a1):
                                        function_list.append(f" y *= (p{p_idx} + s)")
                                        p_idx+=1
                                    for _ in range(a2):
                                        function_list.append(f" y *= (p{p_idx} - s)")
                                        p_idx+=1
                                    for _ in range(b1):
                                        function_list.append(f" y /= (p{p_idx} + s + eps)")
                                        p_idx+=1
                                    for _ in range(b2):
                                        function_list.append(f" y /= (p{p_idx} - s + eps)")
                                        p_idx+=1
                                    function_list.append(" return y")
                                    function_string = "\n".join(function_list)
                                    # Make this function
                                    exec(function_string,globals())
                                    #s_loc = np.linspace(min(s),max(s)-1,200)
                                    try:
                                        y_true = top/bottom
                                        popt[fit_string], pcov[fit_string] = curve_fit(fit_function, s_loc, y_true, p0=init_params)
                                        fit_dict[fit_string] = fit_function
                                        y_pred = fit_function(s_loc, *popt[fit_string])
                                        r2[fit_string] = r2_score(np.real(y_true), np.real(y_pred))
                                        ssq[fit_string] = np.real(np.sum((y_pred - y_true)**2))
                                        df[fit_string] = len(s_loc) - total_num_params
                                        print(fit_string, r2[fit_string])

                                        if(r2[fit_string] > 0.999):
                                            # Plot on the wider domain
                                            y_pred = fit_function(s, *popt[fit_string])
                                            plt.plot(s, y_pred,label=fit_string + " " + str(r2[fit_string]))
                                    except:
                                        continue
                    
                    # Get a matrix of F-tests between the models?

                    # If p = 1, model j is better than i
                    # If p = 0, model i is better than j
                    keys_list = [key for key in df.keys()]
                    p = np.array([[1-f.cdf((ssq[j]-ssq[i])/(ssq[i]/df[i]), 1, df[i]) for j in keys_list] for i in keys_list])
                    sample = np.random.binomial(1,p)
                    print(sample)

                    header = [""] + [str(key) for key in keys_list]
                    with open("test.csv",'w') as fi:
                        fi.write(",".join(header)+"\n")
                        for i, row in enumerate(sample):
                            fi.write(",".join(['a'+keys_list[i]] + [ "X" if elem==1 else "" for elem in row])+"\n")

                    if(False):
                        import concepts as con

                        cccc = con.load_csv("test.csv")

                        print(cccc.objects)

                        l=cccc.lattice
                        dot = l.graphviz()

                        objects = cccc.objects 
                        print(objects)
                        properties = cccc.properties
                        print(properties)

                        for i in objects:
                            node_list = cccc.neighbors([i])
                            print(i)
                            for j in node_list: 
                                print("  ",end="")
                                print(j) 

                        dot.render('test/test.gv', view=True)
                        # We could sample form the above...



                    #for i in df.keys():
                    #  for j in df.keys():
                    #    print(i, " vs ", j)
                    #    f_ratio=(ssq[j]-ssq[i])/(ssq[i]/df[i])
                    #    p=1-f.cdf(f_ratio, 1, df[i])
                    #    print("df, r2, i:",df[i],r2[i])
                    #    print("df, r2, i:",df[j],r2[j])
                    #    print(f_ratio,p)
                    #    breakpoint()
                    
                    if(False):
                        # Print out all the curves??
                        plt.plot(s,phi_0(s+1)/phi_0(s),'k:',label='Data')
                        plt.legend()
                        plt.show()



                # Can also imagine top/bottom is like (-1)^(k) k in [0,1], but, imagine complex with a1,b1,c1, or higher.


                line_start = (1,1)
                poly2_start = (1,1,1)

                # y = rational(x, [-0.2, 0.3, 0.5], [-1.0, 2.0])
                #ynoise = y * (1.0 + np.random.normal(scale=0.1, size=x.shape))
                #s_loc = np.linspace(min(s),max(s)-1,50)
                #popt, pcov = curve_fit(rational4_4, s_loc, phi_0(s_loc+1)/phi_0(s_loc), p0=(0.0, 0.0, 0.9, 0.0, -1.0, 1, 1))
                popt_line, pcov = curve_fit(line, s_loc, top/bottom, p0=line_start)
                popt_poly2, pcov = curve_fit(poly2, s_loc, top/bottom, p0=poly2_start)

                np.save(self.name+"_ratio_phi", [s, top/bottom, bottom])

                print("line",popt_line)
                print("poly2",popt_poly2)

                plt.plot(s_loc, top/bottom, label='original')
                #plt.plot(x, ynoise, '.', label='data')
                plt.plot(s_loc, line(s, *popt_line), label='fit line')
                plt.plot(s_loc, poly2(s, *popt_poly2), label='fit poly2')
                plt.legend()
                plt.show()

                # SO for arbitrarily high polynomials shift only
                # We can have {scale_constant}*MeijerG[{{}, {}}, {{-r1, -r2, -r3},{}}, x/base_constant]

                # Clearly once we factorize Gamma[a+s], Gamma[a-s], and 1/Gamma[b+s], 1/Gamma[b-s]
                # We can populate the other fields as well, so any rational type expression

                # Problem is, how to identify if a poly factor on the top or the bottom is +/- s ?
                # Seems when fitting the curve, we just choose a form... 

                # Fit  ->  C * (a+s)(b-s)/(c+s)(d-s) etc.
                # TODO: How to fit all gammas with a scaled 's'-> 'As'
                # For this we just allow -> C * (a+As)(b-As)/(c+As)(d-As) etc.
                # TODO: Use mpmath to evalulate MeijerG functions if needed.





                # Assuming only a line
                base_constant_line = popt_line[0]
                gamma_shift_line = popt_line[1]/popt_line[0]
                trial_form_line = (popt_line[0]**s * gamma(s + gamma_shift_line))

                # TODO: Upgrade this to some kind of mode of a distribution of scale values
                scale_constant_line = np.mean(bottom/trial_form_line)
                trial_form_line = scale_constant_line * trial_form_line
                print(f"{scale_constant_line}*{base_constant_line}^s Gamma[s+{gamma_shift_line}]")

                def gamma_from_params_poly2(popt) -> Tuple[Tuple[float], Callable[[Tuple[float], float], float]]:
                    # Assuming a poly2 -> Construct from roots and get constant = base
                    from numpy.polynomial.polynomial import polyroots, polyfromroots, Polynomial
                    roots_2 = polyroots(popt_poly2)
                    print("roots_2",roots_2)
                    test_points = np.array([1,2,3])
                    p_coeffs = Polynomial(popt_poly2)(test_points)
                    p_roots = Polynomial(polyfromroots(roots_2))(test_points)
                    base_constant = p_coeffs/p_roots

                    print("Base constant 2", base_constant)
                    #TODO: Assert that the constant is the same all over
                    base_constant = base_constant[0]
                    gamma_shift_1 =  -roots_2[0]
                    gamma_shift_2 =  -roots_2[1]

                    # Return p, form
                    return (base_constant, gamma_shift_1, gamma_shift_2), lambda p, s : p[0]**s * gamma(s + p[1]) * gamma(s + p[2])

                
                p_gamma_poly2, form_gamma_poly2 = gamma_from_params_poly2(popt_poly2)
                base_constant_poly2, gamma_shift_1, gamma_shift_2 = p_gamma_poly2

                #trial_form = base_constant_poly2**s * gamma(s + gamma_shift_1) * gamma(s + gamma_shift_2)
                trial_form = form_gamma_poly2(p_gamma_poly2, s)

                # TODO: Upgrade this to some kind of mode of a distribution of scale values
                scale_constant_poly2 = np.mean(phi_0(s)/trial_form)
                trial_form = scale_constant_poly2 * trial_form


                plt.plot(s, phi_0(s), label = 'true')
                plt.plot(s,scale_constant_line*(popt_line[0]**s * gamma(s + popt_line[1]/popt_line[0])), label = 'line')

                plt.plot(s, trial_form, label = 'poly_2')
                # Print MMA string for manual exploration
                print(f"{scale_constant_poly2}*{base_constant_poly2}^s Gamma[s + {gamma_shift_1}]Gamma[s + {gamma_shift_2}]")
                plt.legend()
                plt.show()

                #cc = power_constant_base * b
                #mm = power_constant_base
                #C * mm**s * gamma(s + cc/mm)



                # Further optimise. Can I wiggle my parameters, so get the ratio as flat as possible in as many as possible section
                # Loss ->
                # If no flat sections at all, fail
                # The more flat sections the better.
                # Has to be very flat, i.e. derivative is zero
                # Bonus:
                # Able to disregard up to one half of the curve without large penalty.
                # Bin the curve into sections
                # Get the maximum derivative across sections (i.e. completely flat is good)
                # Take the lowest best half, as the score, i.e. rank and filter.

                def bin_deriv_rank_filter_loss(x : Iterable, y : Iterable, n_sections : int) -> Tuple[float]:
                    """
                    Split a list into n_sections, and get the max-gradient in each section
                    Sort the chunks and ignore the worst few
                    """
                    N = len(x)
                    section_width = int(N/n_sections)
                    max_abs_gradient = []
                    ys = []
                    for i in range(n_sections):
                        top = min((i+1)*section_width, N)
                        x_i = x[i*section_width:top]
                        y_i = y[i*section_width:top]
                        #TODO: verify this is a good function call to make
                        dy_dx_i = np.gradient(y_i, x_i)
                        max_dy_dx_i = np.amax(dy_dx_i)
                        max_abs_gradient.append(np.abs(np.real(max_dy_dx_i)))
                        ys.append(np.mean(y_i))
                    
                    flattest_i = np.argmin(max_abs_gradient)
                    norm = ys[flattest_i]

                    # Keep a few chunks
                    keep = int(n_sections/2)+1
                    loss = sorted(max_abs_gradient)[:keep]
                    loss = np.mean(loss)

                    # norm is the average value of the function in the flattest section
                    return loss, norm
                
                # Minimise: from scipy.optimize import minimize... 
                loss, norm = bin_deriv_rank_filter_loss(s, phi_0(s)/trial_form_line, 10)
                print(loss)

                # TODO: The ratio is quite unstable... So we want a 'masked' difference.
                def bin_diff_rank_filter_loss(y : Iterable, n_sections : int, ridge = False, alpha = 1e-3, p = None):
                    """
                    Split a list into n_sections, and get the max-difference in each section
                    Sort the chunks and ignore the worst few
                    """
                    N = len(y)
                    section_width = int(N/n_sections)
                    max_abs_diff = []
                    for i in range(n_sections):
                        top = min((i+1)*section_width, N)
                        delta_y_i = np.abs(np.real(y[i*section_width:top]))
                        max_abs_diff.append(np.amax(delta_y_i))
                    
                    # Keep a few chunks
                    keep = int(n_sections/2)+1
                    loss = sorted(max_abs_diff)[:keep]
                    loss = np.mean(loss)
                    if(ridge):
                        loss += alpha * np.sum(np.abs(p))
                    return loss



                p_gamma_poly2, form_gamma_poly2 = gamma_from_params_poly2(popt_poly2)
                trial_form = form_gamma_poly2(p_gamma_poly2, s)

                #def form_to_loss(p,s,form : Callable):
                #    #trial_form = (p[0]**s * gamma(s + p[1]/p[0]))
                #    return bin_deriv_rank_filter_loss(s, phi_0(s)/form(p,s), 10)[0]
                
                # TODO: This might be unstable because of gamma parameters....
                def form_to_loss(p,s,form : Callable):
                    return bin_diff_rank_filter_loss(phi_0(s)-form(p,s), 10)
                
                def form_to_loss(p,s,form):
                    # TODO, consider over a reduced section of s
                    # As this is trying to fit line to line, or poly to poly
                    return bin_diff_rank_filter_loss(phi_0(s+1)/phi_0(s) - form(p,s+1)/form(p,s), 10)

                def form_to_norm(p,s,form : Callable):
                    """
                    Wrapper of more stable? difference
                    """
                    #trial_form = (p[0]**s * gamma(s + p[1]/p[0]))
                    return bin_deriv_rank_filter_loss(s, phi_0(s)/form(p,s), 10)[1]

                # Do for a line
                # TODO: Do for as many shapes as needed
                plt.plot(s,phi_0(s)/form_gamma_poly2(popt_poly2,s),label = 'phi/fit assumption poly2')
                res = minimize(form_to_loss, x0=popt_poly2, args = (s, form_gamma_poly2), method = "BFGS", tol = 1e-8)
                solution = res.x
                constant = form_to_norm(solution, s, form_gamma_poly2)

                # TODO: Can get the loss for each ratio.

                # TODO: Can we vary the parameters of our fit to flatten, this
                # I.e. minimise the 'curvature'? Or similar.
                print(self.root_polynomial)
                plt.title("Is it constant?")
                exact_form = 2**(s-1)*gamma(1/2 + s)
                plt.plot(s,phi_0(s)/trial_form_line,label = 'phi/fit assumption line')

                plt.plot(s,phi_0(s)/exact_form,label = 'phi/fit exact best case')
                plt.plot(s,phi_0(s)/form_gamma_poly2(solution,s)/constant,label = 'phi/fit solved-flatten')
                #plt.plot(s,phi_0(s), label = 'true')
                plt.legend()
                plt.show()

                exit()

                if(False):
                    # TODO: Automatic sympy evalulation of function
                    def sympy_inverse_mellin(moments_string):
                        from sympy import inverse_mellin_transform, oo, gamma, sqrt
                        from sympy.abc import x, s
                        return inverse_mellin_transform(moments_string, s, x, (0, oo))

                    moments_string = self.root_polynomial + f"*({scale_constant})*({base_constant})^s * Gamma[s + {gamma_shift}]".replace('--','+')
                    print(moments_string)
                    #MMA_function = input("Get the MMA function as input, and parse into sympy etc.")

                    string = """0.005573457742959165*((-15.925025447742174*x^1.4332110586835183)/E^(3.2464714279171454*x) +(33.259591472696314*(1 - 2.265173303155506*x)*x^(53270321/37168511))/E^(3.2464714279171454*x) +(3.91390619491436*^-15*x^(53270321/37168511)*(2837727099443041 - 1.7340881800943048*^16*x +1.4560406389353844*^16*x^2))/E^(3.2464714279171454*x))"""
                    #result = sympy_inverse_mellin(moments_string)

                    def get_vals(string):
                        from sympy.parsing.mathematica import parse_mathematica
                        from sympy.abc import x
                        expression = parse_mathematica(string)
                        values = [expression.evalf(subs={x:xx}) for xx in [0,1,2,3]]
                        print(values)
                        breakpoint()
                        return values
                    
                    values = get_vals(string)
                    #func = lambdify(x, expression, 'numpy') # returns a numpy-ready function

                from scipy.special import kn
                from mpmath import meijerg

                

                # TODO: Simplify and figure out how to plot the result
                plt.title("Comparison")
                x = np.linspace(self.x_min,self.x_max,2000)

                #MeijerG[{{}, {0}}, {{1, 1.4741178655552896, 4.254005239427203}, {}}, 12.695630543690696*x]
                def temp_f(x): return complex(meijerg([[],[0]], [[1, 1.4741178655552896, 4.254005239427203],[]], 12.695630543690696*x))
                curve51_mg = np.vectorize(temp_f)

                def temp_f(x): 
                    ret = -0.5202487172672269*1.74787866*meijerg([[], []], [[3.21408191], []], 23.604485135429552*x, 1.74787866)
                    ret += meijerg([[],[0]], [[1,3.21408191],[]], 23.604485135429552*x, 1.74787866)/1.74787866**2
                    return complex(ret)
                curve51_mg_2 = np.vectorize(temp_f)

                plt.plot(x, self.interpolating_function(x), label = 'Interpolant')

                #plt.plot(x,(-1.438492776325737*x**1.2369673295454544)* np.exp(-3.5511363636363633*x),label = 'line approx') # line curve ??
                plt.plot(x,(0.23609829490609555*x**0.8218310997655742)*np.exp(-1.3003278181026625*x),label = 'line approx') # line curve 50
                plt.plot(x,77.98273638661385*x**4.1178080078394705*kn(5.662842095677695, 7.679414303164028*np.sqrt(x)),label = 'poly2 approx') # pol2 curve 50
                plt.plot(x,-38.425663041946635*x**2.864061552491246*kn(2.7798873738719134, 7.1261856679967845*np.sqrt(x)) + 0.02549450467547057*curve51_mg(x), label = 'mg') # pol2 curve 51
                plt.plot(x,curve51_mg_2(x), label = 'mg_2') # pol2 curve 51
                #mma_x = [0,0.1,0.2,0.05,0.15,0.2,0.25, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
                #mma_y = np.array([0., 0.0195802, 0.00540991,0.012437, 0.0167003, 0.00540991, -0.0115681,-0.0315771, -0.0723896, -0.105614, -0.126737,-0.135614, -0.134308, -0.12565, -0.112417, -0.0969391, -0.0809731, -0.0657189, -0.0519062, -0.0399023, -0.0298153, -0.0215822, -0.0150381, -0.00996711, -0.00613756])*0.1
                #plt.plot([0,1,2,3], values, 'r-', label = 'Curve_0 - Approximation')
                plt.legend()
                plt.show()

                plt.plot(x, self.interpolating_function(x)/((0.23609829490609555*x**0.8218310997655742)*np.exp(-1.3003278181026625*x)), label = 'line' )
                plt.plot(x, self.interpolating_function(x)/(77.98273638661385*x**4.1178080078394705*kn(5.662842095677695, 7.679414303164028*np.sqrt(x))), label = 'poly2' )
                plt.legend()
                plt.show()
                exit()



            if(False):
                for i in [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]:
                    plt.plot(s, phi_0(s+i)/phi_0(s))
                plt.plot(s, phi_0(s+1)/phi_0(s), label = 'phi0(s+1)/phi0(s)')
                plt.plot(s, phi_0(s)/phi_0(s+1), label = 'phi0(s)/phi0(s+1)')
                plt.plot(s, phi_0(s+2)/phi_0(s), label = 'phi0(s+2)/phi0(s)')
                plt.plot(s, mm*s + cc, label = f'line {mm}s+{cc}')
                plt.plot(s, mm_inv*s + cc_inv, label = f'line {mm_inv}s+{cc_inv}')
                #plt.plot(s, 3 * (4+s)/5/(3- 4*s), label = 'F rat')
                plt.plot(s, 1 + 2*s)
                plt.plot(s, 3 + 8 * s + 4 * s**2)
                #plt.plot(s, phi_1(s+1)/phi_1(s), label = 'phi1(s+1)/phi1(s)')
                #plt.plot(s, (mm*s+ cc)/(s - 2.401076))

                # TODO: Fit a polynomial/line etc.
                # TODO: Find the poles, or roots of the reciprocal function?

                plt.legend()
                plt.show()

            # Print

            print("Also look at shifted dq/q over non-shifted dq/q which evaluates to a rational constant? As cancellation of digamma etc.")



            plt.show()
            #exit()

            
            from mpl_toolkits import mplot3d
            



            xdata = []
            ydata = []
            zdata = []
            dzdata = []
            an_zdata = []

            for i in range(len(s)):
              for j in range(len(s)):
                  #if(i==j): continue
                  xdata.append(s[i])
                  ydata.append(s[j])
                  if(True):
                    zdata.append(np.log(q[i]/q[j]))
                    dzdata.append(np.log(q[i]/q[j]))
                  if(True):
                    local_moment = lambda s: gamma(s)**2
                    an_zdata.append(np.log(local_moment(s[i])/local_moment(s[j])))

            xdata=np.array(xdata)
            ydata=np.array(ydata)
            zdata=np.array(zdata)
            dzdata = np.array(dzdata)
            an_zdata=np.array(an_zdata)
            idx = np.abs(zdata)<0.1
            anidx = np.abs(an_zdata)<0.1
            didx = np.abs(dzdata)<0.1

            fig = plt.figure()

            # TODO: Use these points to identify gamma(a s + b) term?
            zero_0 = 1.46163214496836 # Where logGamma'[x] = 0, x > 0
            zero_1 = -0.504083008264455 # x < 0
            zero_2 = -1.57349847316239
            zero_3 = -2.61072086844414
            
            ax = plt.axes(projection='3d')
            ax.set_title("Ratio of q(s)/q(t)")
            #ax.scatter3D(xdata[~idx], ydata[~idx], zdata[~idx])
            ax.scatter3D(xdata[idx], ydata[idx], zdata[idx], cmap='reds')
            if(True):
                ax.scatter3D(xdata[anidx], ydata[anidx], an_zdata[anidx], cmap='reds')
            # ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
            ax.set_xlabel('s')
            ax.set_ylabel('t')
            ax.set_zlabel('q(s)/q(t)')
            plt.show()

            #TODO: Somehow solve for continuous curves
            
            plt.title("q/q Projection to s-t plane")

            plt.plot(xdata[idx], ydata[idx], 'r.',label='data')
            if(True):
                plt.plot(xdata[anidx], ydata[anidx], 'k.',label ='local moment')
            plt.grid(True,'both','both')
            plt.legend()
            plt.plot(zero_0,zero_0,'ro')
            plt.plot(zero_1,zero_1,'ro')
            plt.plot(zero_2,zero_2,'ro')
            plt.plot(zero_3,zero_3,'ro')
            plt.show()

            plt.title("dq/dq Projection to s-t plane")

            plt.plot(xdata[didx], ydata[didx], 'r.',label='data')
            plt.grid(True,'both','both')
            plt.legend()
            plt.plot(zero_0,zero_0,'ro')
            plt.plot(zero_1,zero_1,'ro')
            plt.plot(zero_2,zero_2,'ro')
            plt.show()

            from scipy.spatial import ConvexHull, KDTree


            plt.title("Projection to s-dq/dq plane")
            plt.plot(xdata[idx], zdata[idx], 'r.',label='data')
            points = np.array([xdata[idx],zdata[idx]]).T
            data_hull = ConvexHull(points)

            # Identify points which are symmetric about 0
            degenerate_points = np.array([xdata[idx],np.abs(np.real(zdata[idx]))]).T
            lower_points_filter = zdata[idx]<0
            lower_points = np.real(points[lower_points_filter])

            tree = KDTree(np.real(degenerate_points))

            upper_points = np.real(points[~lower_points_filter])

            # Sort the upper points by 's' from lowest to highest
            # Plot max(y) as a function of sorted x
            sorted_upper_points = np.array(sorted(upper_points, key= lambda x: x[0]))
            window_length = 6
            # Smoothing
            pos = [ np.mean(sorted_upper_points[i:i+window_length+1][:,0]) for i in range(len(sorted_upper_points) - window_length)]
            vals = [ np.amax(sorted_upper_points[i:i+window_length+1][:,1]) for i in range(len(sorted_upper_points) - window_length)]

            xx = np.linspace(min(xdata),max(xdata),200)
            plt.plot(xx, [min(xxx, 0.1) for xxx in (xx-2.39)**2 /6] )
            plt.plot(pos,vals)
            plt.show()
            upper_tree = KDTree(upper_points)
            results = tree.query(degenerate_points, k=2)
            
            reflected_lower_points = np.array([1,-1])*lower_points
            asym_results = upper_tree.query(reflected_lower_points, k=1)

            # Consider larger than average asym distances
            asym_distances = asym_results[0]
            filt_point = np.percentile(asym_distances, q=0.99)
            filt_idx = asym_distances > filt_point
            #plt.hist(asym_results, bins=100, density=True)
            #plt.show()


            # If a point has two exact neighbours, it is likely degenerate
            print(results)
            #plt.show()

            distances_sum = np.sum(results[0],axis=1)
            filter = distances_sum < 0.001
            #plt.hist(distances_sum, density=True, bins=100)
            #plt.show()
            plt.plot(points[filter,0],points[filter,1],'bx')
            #plt.plot(lower_points[~filt_idx,0],lower_points[~filt_idx,1],'gx')

            # Or could look for unusually large neighbour distances in the reflected lower points fitting to the upper points. 
            # I.e. to get the 'void'



            plt.plot(points[data_hull.vertices,0], points[data_hull.vertices,1], 'r--', lw=2)
            if(False):
                plt.plot(xdata[anidx], an_zdata[anidx], 'k.',label ='local moment')
            plt.grid(True,'both','both')
            plt.plot(zero_0,0,'ro')
            plt.plot(zero_1,0,'ro')
            plt.plot(zero_2,0,'ro')
            plt.legend()
            plt.show()

            
            def local_line(x,y):
                # Should be x=s and y=log_errors
                # For each pair of neighbouring points get the 'grad' and 'intercept'
                N = len(x)
                m = [(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(N-1)]
                c = [y[i] - m[i] * x[i] for i in range(N-1)]
                return m, c

            m1, c1 = local_line(s_loc, np.log(re1))
            print(m1)
            print(c1)
            #breakpoint()

            # TODO: Fit lines to the log(error) in integration
            # Establish the domain as the region when this linear relationship breaks down
            # Do the coefficients of the line reveal the internals of the Gamma?

            if(test_data):
                plt.title("Errors in Integration - For Analytic Curve")
                plt.plot(s_loc, np.log(re1a), label = 're1_an')
                plt.plot(s_loc, np.log(re2a), label = 're2_an')
                plt.plot(s_loc, np.log(re3a), label = 're3_an')
                plt.plot(s_loc, np.log(re4a), label = 're4_an')
                plt.plot(s_loc, np.log(im1a), label = 'im1_an')
                plt.plot(s_loc, np.log(im2a), label = 'im2_an')
                plt.plot(s_loc, np.log(im3a), label = 'im3_an')
                plt.plot(s_loc, np.log(im4a), label = 'im4_an')
                plt.legend()
                plt.show()

            # [DONE] Revise s to a domain with acceptable errors
            # TODO: Where this is not possible, produce an error! I.e. 'curve appears unstable'
            # Numerical integration failed to produced acceptable error for learning.

            # TODO: Consider adding covariance to error propagation as might be a factor

            # TODO: Remember to only fit in the domain where errors are tolerable.
            # This will improve the performance significantly
            # Retest all the BFGS type fitting after that.
            # TODO: Remember to restrict the coefficient domains to [-1,1,-1/2,1/2] etc.

            # TODO: IF WE ARE DIVIDING THROUGH BY POLYNOMIAL, THEN WE NEED TO ISOLATE THE LOG DERIVATIVE OF POLYNOMIAL FROM dQ, ddQ etc. ??

            # TODO: Try to remove unstable regions from sampling s, i.e. poles
            # But do note them down?
            if(False):
                plt.title("q, dq, ddq, dddq")
                plt.plot(s_loc,c(np.real(q)), label=f'q')
                plt.plot(s_loc,c(np.real(dq)), label=f'dq')
                plt.plot(s_loc,c(np.real(ddq)), label=f'ddq')
                plt.plot(s_loc,c(np.real(dddq)), label=f'dddq')
                plt.legend()
                plt.show()



            if(False):
                plt.title("dq/q")
                plt.plot(s_loc,c(np.real(dq/q)), label=f'{k} real')
                plt.plot([min(s),max(s)],[0,0],'k:')
            #plt.plot(s_loc,np.real(dq/q) * (re1/np.real(q) + re2/np.real(dq)), label=f'{k} real error via prop')
            #plt.plot(s_loc,np.real(dq_an/q_an) * (re1a/np.real(q_an) + re2a/np.real(dq_an)), label=f'{k} real error_an via prop')
            if(test_data):
                plt.plot(s_loc,c(np.real(dq_an/q_an)), label=f'{k} real analytic')
                #plt.plot(s_loc, -1/(0.86-s_loc),'k:' ,label='fitted curve')
                #plt.plot(s_loc,np.imag(dq/q), label=f'{k} imag')
                #plt.plot(s_loc,c(digamma(s)), label = "p(0,s)")
                #plt.plot(s_loc,c(np.log(2) - digamma(0.5 - s) + digamma(1+s)), label = "r log 2 - p(0,0.5-s) + p(0,1+s)")
                #plt.plot(s_loc,d(np.log(2) - digamma(0.5 - s) + digamma(1+s)), label = "i log 2 - p(0,0.5-s) + p(0,1+s)")
                plt.plot(s_loc,c(-np.log(2) - digamma(2 - s)) , label = "r -log 2 - p(0,2-s)")
                #plt.plot(s_loc,c(-np.log(2) - digamma(1 - 1j * s_loc)) , label = "r* -log 2 - p(0,2-s)")
                #plt.plot(s_loc,d(-np.log(2) - digamma(2 - s)) , label = "i -log 2 - p(0,2-s)")
                plt.legend()
                plt.show()

            if(test_data):
                plt.title("ddq/q for numeric and analytic comparison")
                plt.plot(s_loc,c(np.real(ddq/q)), label=f'{k} real')
                #plt.plot(s_loc,np.imag(ddq/q), label=f'{k} imag')
                plt.legend()
                plt.show()

                plt.title("ddq/q - (dq/q)^2 for numeric and analytic comparison")
                plt.plot(s_loc,c(np.real(ddq/q - (dq/q)**2)), label=f'{k} real')
            if(test_data):
                plt.plot(s_loc,c(np.real(ddq_an/q_an - (dq_an/q_an)**2)), label=f'{k} real analytic')
                #np.save("curve_3_2ndorder",np.real(ddq/q - (dq/q)**2))
                #exit()
                #plt.plot(s_loc,np.imag(ddq/q - (dq/q)**2), label=f'{k} imag')
                #plt.plot(s_loc,c(trigamma(0.5 - s) + trigamma(1+s)), label = "p(1,0.5-s) + p(1,1+s)")
                plt.plot(s_loc,c(trigamma(2 - s)), label = "p(1,2-s)")
                plt.plot([min(s),max(s)],[0,0],'k:')
                plt.legend()
                plt.show()

            if(False):
                def ff(ss):
                    eps = 0.00001
                    ret = []
                    for sss in ss:
                        if(sss<-1) : ret.append(1)
                        else: ret.append(np.sin(np.pi*sss)**2 + eps)
                    return np.array(ret)
                
                plt.plot(s_loc, c(np.real(ddq_an/q_an - (dq_an/q_an)**2)/trigamma(2 - s)/ff(s)))
                plt.plot(s_loc, 1/s_loc)
                plt.plot(s_loc, np.exp(- s_loc))
                plt.show()

            if(False):
                plt.title("dddq/q - 3(ddq/q)(dq/q) + 2 (dq/q)^3 for numeric and analytic comparison")
                plt.plot(s_loc, c(np.real(2 *(dq/q)**3 - 3 *(dq/q) * (ddq/q) + (dddq/q))), label='3rd order, real')
            if(test_data):
                #plt.plot(s_loc, np.imag(2 *(dq/q)**3 - 3 *(dq/q) * (ddq/q) + (dddq/q)), label='3rd order, imag')
                #plt.plot(s,c(-polygamma(2, 0.5 - s) + polygamma(2, 1+s)), label = "-p(2,0.5-s) + p(2,1+s)")
                #plt.plot(s,c(-polygamma(2, 2 - s)), label = "-p(2,2-s)")
                plt.plot(s_loc,c(-tetragamma(2 - s)), label = "p(2,2-s)")
                plt.plot([min(s),max(s)],[0,0],'k:')
                plt.legend()
                plt.show()

            #plt.plot(s_loc, c((np.real(2 *(dq/q)**3 - 3 *(dq/q) * (ddq/q) + (dddq/q)))*ff(s)/tetragamma(2 - s)), label='3rd order, real')
            #plt.show()

            if(False):
                for k in range(0,self.max_moment_log_derivative_order):
                    moments, re, im = self.vectorised_integration_function[k](s)
                    if(k == 0):
                        plt.title("Moments / gamma(s)")
                        plt.plot(s,moments/gamma(s), label=f'{k}')
                        plt.plot([min(s),max(s)],[0,0],'k:')
                        #plt.plot(s,gamma(s),'k:')
                    if(k == 1):
                        plt.title("log Moments")
                        plt.plot(s,np.log(moments), label=f'{k}')
                        plt.plot([min(s),max(s)],[0,0],'k:')
                        #plt.plot(s,gamma(s)*polygamma(0,s),'k:')
                    if(k == 2):
                        plt.plot(s,np.log(moments), label=f'{k}')
                        #plt.plot(s,gamma(s)*(polygamma(0,s)**2 + polygamma(1,s)),'k:')
                    plt.show()

        for k in range(0,self.max_moment_log_derivative_order):
            moments, re, im = self.vectorised_integration_function[k](s)

            # Store this guy for analysis
            self.moments_interpolating_function[k] = interp1d(s, moments, kind='cubic')

            #plt.plot(s,np.log(moments),label=f'k = {k}')
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


        # Root Finding
        def wrapper(s):
            res, _, _ = self.vectorised_integration_function[1](s)
            return np.real(res)

        res = root(wrapper, x0=5)
        print(res)

    
        # TODO: Recall 
        # M^{-1}[\Gamma(s) M[f(-t)](1-s)](t) -> MGF or GF 

        # TODO: Check the alignment of spaces
        from lookup import inverse_polygamma as tbl
        x_tbl = np.linspace(0,10,1001)
        print(len(x_tbl))
        print(len(tbl))
        inv_poly = interp1d(x_tbl, tbl, kind='cubic')

        def inverse_polygamma(x):
          # TODO: all negative values approximation
          #if(x < 0): raise ValueError("x must be >= 0")
          #if( x == 0): return -inf * x
          if( x < 0): return -1/x
          if( x > 9):
            # For large x, i.e. x > 3 is accurate
            # Series Reversion Coefficients
            c = [-1.0000000000000000000, 
                0.57721566490153286061, 
                1.3117561430405077622,
                -1.4540727133641656986, 
                -3.9273504590258703791, 
                4.5506845688740145645, 
                15.345703059377175597,
                -15.388198880666277106, 
                -67.809754536905425774, 
                51.932247003817346795, 
                319.74330918808715017,
                -159.30257094433569794]
            res = 0
            for i, cc in enumerate(c):
                res += cc/x**(i+1)
            return res
          else:
              # Use a lookup table for x in [0,3]
              return inv_poly(x)
        vec_inv_poly = np.vectorize(inverse_polygamma)

    
        # Test
        #test_x = np.linspace(-20,20,200)
        #plt.plot(test_x, polygamma(0, vec_inv_poly(test_x)) - test_x,label='phi(invphi(x))')
        #plt.plot(test_x, polygamma(0, inv_poly(test_x)),label='cubic spline')
        #plt.plot(test_x,0*test_x)
        #plt.legend()
        #plt.show()
         

        # Looking for a root of the moment derivative
        # These reveal information about the parameters
        # If there is no root, it might be a ratio Gamma(s)/Gamma(s)
        self.large_k1_root = None
        if res.success == True:
            self.large_k1_root = res.x

        # TODO: Get second root as well... 
        #breakpoint()


        # TODO: Contour integration if needed...
        # TODO: Consider idea of Kapetyn Series

        # If k=0 at s=1, is close to 1, then likely normalised
        # TODO: Look at errors as well
        norm, _, _ = self.vectorised_integration_function[0]([1])
        print(f'norm {norm}')
        if(np.isclose(norm, 1, rtol=1e-04, atol=1e-4)):
            print("Likely normalised!")
            self.normalised = True

        # If asymptotic to zero
        y_max  = self.interpolating_function([self.x_max])
        print(f'y(x_max) {y_max}')
        if(np.isclose(y_max, 0, rtol=1e-02, atol=1)):
            print("Likely asymptotic to zero")
            self.asymptotic_to_zero = True

        # If y_min is exactly 1 (implies pure hypergeometric)
        y_min = self.interpolating_function([self.x_min])
        print(f'y(x_min) {y_min}')
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





        # You could just fit the ansatz asymptotics ??? I.e. Sum_i loggamma(a_i + b_i *s)
        #plt.legend()
        #plt.show()



        # For exp(-x) -> Gamma(s), we clearly have zeros for k = 1 at s = 1/2 (??? 1.477), -1/2, -3/2
        # For exp(-x) -> Gamma(s), we clearly have poles for s = 0, for k = 0, 1, 2

        # TODO: Learn a function that gives the zeros / poles of "a PolyGamma[0, a x + b]"
        # Can we use this to invert
        # Well the infinities are at 0, -1, -2, -3, etc.

        print("innovation, technical excellence, feasibility and business impact")
        # TODO: Grab some stuff from the literature review?

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
        # TODO: Test this for complex numbers
        self.complex_s_samples = np.array([ t1 + t2*1j for t1,t2 in zip(s1,s2) ])

        # TODO: determine domain of validity

        print("We must control logic to remove the polynomial correctly before proceeding! Moments are resampled above and not correct.")

        # Perform the numerical integration for moments m(s)
        # Calculate m(s), m'(s), m''(s),..., m^{(k)}(s) etc. as  E[log(X)^k X^{s-1} f(x)](s)
        for k in range(self.max_moment_log_derivative_order):
            self.moments[k], self.real_error_in_moments[k], self.imaginary_error_in_moments[k] = self.vectorised_integration_function[k](self.complex_s_samples)

        # Renormalise by factoring out a polynomial
        if(self.root_polynomial is not None):
            
            self.poly_derivative_expressions = get_derivatives(self.root_polynomial, order = 3)
            
            p = {}
            poly_order = 3
            for i in range(poly_order + 1):
                poly = eval("lambda s: "+self.poly_derivative_expressions[i])
                p[i] = np.array([poly(ss) for ss in self.complex_s_samples])
            
            self.poly_samples = p
            
            # TODO: Check the polynomial in the complex plane makes sense?
            # TODO: Extend to any dimension up to say max_root_order=10
            # TODO: Error estimates will be a bit wrong!
            chi = {}

            # Simply divide through chi_0 = phi_0/p_0
            chi[0] = self.moments[0]/p[0]

            # chi_1 from chi_1/chi_0 = phi_1/phi_0 - p_1/p_0
            chi[1] = (self.moments[1]/self.moments[0] - p[1]/p[0]) * chi[0]

            # Rearrange phi_2/phi = D[p(s)chi(s),{s,2}]/p/chi
            chi[2] = (self.moments[2]/self.moments[0] - 2.0 * chi[1]/chi[0] * p[1]/p[0] - p[2]/p[0]) * chi[0]

            # Rearrage phi_3/phi_0 = D[p(s)chi(s),{s,2}]/(p_0chi_0)
            chi[3] = (self.moments[3]/self.moments[0] - 3.0 * chi[2]/chi[0] * p[1]/p[0] - 3.0 * chi[1]/chi[0] * p[2]/p[0] - p[3]/p[0]) * chi[0]

            # Renormalise moments that will be passed on to the gamma finding code...
            self.has_polynomial = True
            for k in range(self.max_moment_log_derivative_order):
                self.moments[k] = chi[k]


        


@dataclass
class ExactLearningResult():
    """
    Store an exact learning result
    """
    def __init__(self, result_dict):
        self.equation = result_dict["equation"]
        self.factored_polynomial = result_dict["factored_polynomial"]
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

    # TODO: determine dimensionality of data

    # TODO: dig out an example of interpolation (i.e. Schrodinger)

    # TODO: dig out an example of integration (i.e. moment fitting)

    # name is used as the folder name for this problem,
    # e.g. "Beta_Distribution"
    return MomentsBundle(name).ingest(x,y)
