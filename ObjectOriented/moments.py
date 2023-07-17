from copyreg import constructor
from dataclasses import dataclass
from math import inf
from typing import Iterable
#from ObjectOriented.AdvancedFunctions import trigamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import gamma, polygamma, digamma
from scipy.optimize import root
from AdvancedFunctions import trigamma_vec as trigamma 
from AdvancedFunctions import tetragamma_vec as tetragamma 
from matplotlib import pyplot as plt
#from scipy.interpolate import approximate_taylor_polynomial
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
        self.num_s_samples = 100
        self.s_domain = {'Re':[1,5],'Im':[-2*np.pi,2*np.pi]}

        # TODO: Make this a range?
        self.max_moment_log_derivative_order = 4

        # Sample range for plotting moments as function of s
        self.moments_sample_s_min = -10
        self.moments_sample_s_max = 10
        self.moments_sample_n_samples = 2000

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

        # TODO: Try to scan how many times the moments cross 0 and where they do
        self.moments_have_roots = None # Implies moments have a polynomial or other non-meromorphic term of order "num roots"

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
        self.interpolating_function = interp1d(x, y, kind='cubic', bounds_error = False, fill_value = 0)
        #breakpoint()

        x_start = self.interpolating_function.x[0]
        x_end = self.interpolating_function.x[-1]
        y_start = self.interpolating_function.y[0]
        y_end = self.interpolating_function.y[-1]

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
            print(self.interpolating_function([self.x_min,self.x_max]))
            plt.plot(x,self.interpolating_function(x),label = 'interp' )
            plt.plot(x_dash,self.interpolating_function(x_dash),'r-',label='interp. ext')
            plt.plot(x_dash, test_function(x_dash),label='exact')
            for a in [0.5,1,1.5]:
                plt.plot(x_dd, logtail(x_dd, a), label = f'Logtail guess a = {a}')
            for a in [0.3,0.4,0.5]:
                plt.plot(x_dd, powertail(x_dd, a), label = f'Powertail guess a = {a}')

            plt.legend()
            plt.show()

        #breakpoint()
        # TODO: Examine the spread of moments under different tail approximations
        def tail_extrapolate_one(x_i):
            ret = []
            #for x_i in x:
            # TODO: Make it a linear, or even a left tail right tail type thing using ther same smoothing model as above.
            if(x_i < self.x_min): ret = 0
            if(x_i > self.x_max): ret = logtail(x_i, 1)
            else: 
                ret = self.interpolating_function(x_i)
            return ret

        self.adjusted_interpolating_function = tail_extrapolate_one


        #psi_inp_lin = interp1d(x,y)

        # TODO: Consider fractional log order? This would surely be the fractional derivative of the mellin transform?
        # Or if it is not, then it is something else interesting?
        # For simple fractions, Split log(x)^(1/k) into interval [0,1] (Im) and [1,Infty) (Re)
        # Expressions exist for real x (WolframAlpha -> Im[Log[x]^(1/3)], alternate exprsn.)
        # Can always fit interpolating function

        from functools import partial
        def real_integrand(x,s,k): return np.real(x**(s-1)*self.adjusted_interpolating_function(x)*np.log(x)**k)
        def imag_integrand(x,s,k): return np.imag(x**(s-1)*self.adjusted_interpolating_function(x)*np.log(x)**k)
        def special_int(s,order):  
            # N.B. Can use np.inf for exact equations if overriding
            # TODO: Catch cases with negative x_min or something else weird
            r = integrate.quad(real_integrand, 0, np.inf, args=(s,order))
            i = integrate.quad(imag_integrand, 0, np.inf, args=(s,order))  
            return r[0]+ 1j*i[0], r[1], i[1]
        
        for k in range(-3,self.max_moment_log_derivative_order):
            partial_dict = { k : partial(special_int, order = k)}
            self.vectorised_integration_function[k] = np.vectorize(partial_dict[k])

        test_function_dict = {
            "inv_chi_sq" : "np.exp(-1/(2*x))/2/x**2",
            "exp" : "np.exp(-x)",
        }
        test_moments_dict = {
            "inv_chi_sq" : "2**(1-s) * gamma(2-s)",
            "exp" : "gamma(s)",
        }

        if(True):
            # An analytic comparison for testing the integrals over interpolating functions
            # Perhaps they are breaking down...
            # def test_function(x): return np.exp(-1/(2*x))/2/x**2  #inv chi sq
            def test_function(x): return np.sqrt(x) * np.exp(-x/2) / np.sqrt(2 * np.pi)  # chi-sq k=3
            #def test_function(x): return 12* (1-x)**2 * np.heaviside(1-x,0.5) # Beta(2,3) , might work
            #def test_function(x): return 1.0/np.sqrt(x)/(2 + x)**(3/2) # F-ratio
            #def test_function(x): return np.exp(-x) # Exponential
            def real_integrand_an(x,s,k): return np.real(x**(s-1)*test_function(x)*np.log(x)**k)
            def imag_integrand_an(x,s,k): return np.imag(x**(s-1)*test_function(x)*np.log(x)**k)
            def special_int_an(s,order):  
                # N.B. Can use np.inf for exact equations if overriding
                # TODO: Catch cases with negative x_min or something else weird
                r = integrate.quad(real_integrand_an, 0, np.inf, args=(s,order),complex_func=True, limit=100)
                i = integrate.quad(imag_integrand_an, 0, np.inf, args=(s,order),complex_func=True, limit=100)  
                return r[0]+ 1j*i[0], r[1], i[1]
            
            for k in range(-3,self.max_moment_log_derivative_order):
                partial_dict = { k : partial(special_int_an, order = k)}
                self.vectorised_integration_function_an[k] = np.vectorize(partial_dict[k])


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

        if(False):
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
            q, re1, im1 = self.vectorised_integration_function[0](s)
            dq, re2, im2 = self.vectorised_integration_function[1](s)
            ddq, re3, im3 = self.vectorised_integration_function[2](s)
            dddq, re4, im4 = self.vectorised_integration_function[3](s)

            # TODO: We definately notice a discrepancy in the integration of certain functions.
            # This indicates the sampling has problems, probably for very sharp or divergent functions
            # Potentially the tail not long enough, or not exactly zero at the origin etc.
            q_an, re1a, im1a = self.vectorised_integration_function_an[0](s)
            dq_an, re2a, im2a = self.vectorised_integration_function_an[1](s)
            ddq_an, re3a, im3a = self.vectorised_integration_function_an[2](s)
            dddq_an, re4a, im4a = self.vectorised_integration_function_an[3](s)
            def c(x): return np.clip(np.real(x),-10,10)
            def d(x): return np.clip(np.imag(x),-10,10)


            # Define local s for quick plotting changes
            #s_loc = np.imag(s)
            s_loc = s

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
                s_loc_filter = s_loc[np.log(re1) < -5]
            except:
                print("The integration domain is not defined due to poor convergence of numerical integration under current assumptions.")
                raise NotImplementedError
            low_s = np.amin(s_loc_filter)
            high_s = np.amax(s_loc_filter)

            # Resample these values
            s = np.linspace(low_s, high_s, 200)
            s_loc = s

            # Set this as the actual sampling domain
            self.s_domain['Re'][0] = low_s
            self.s_domain['Re'][1] = high_s

            q, re1, im1 = self.vectorised_integration_function[0](s)
            dq, re2, im2 = self.vectorised_integration_function[1](s)
            ddq, re3, im3 = self.vectorised_integration_function[2](s)
            dddq, re4, im4 = self.vectorised_integration_function[3](s)

            test_data = False

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
                
                mma_s = [1.0,2,3,1.5,2.5,0.5,0.1,1.75,1.85,1.25,0.75,0.2,0.3,0.4]
                mma_res = [1.0,2.0,8.0,1.25331,3.75994,1.25331,5.09816,1.54567,1.70447,1.0779,1.03045,2.63675,1.84153,1.46344]
                #plt.plot(s, c(np.real(q)), label = 'numeric')
                #plt.show()


            #plt.plot(s, c(np.real(q)), 'b:',label = 'numeric')
            #plt.show()
            # Need to solve why they are not the same...
            plt.title("Real part of moments")
            plt.plot(s, c(np.real(q)),label = 'numeric')
            plt.plot([min(s),max(s)],[0,0],'k:')
            if(test_data):
                plt.plot(s, c(re1), label = 'real error numeric')
                plt.plot(s, c(q_an), 'r-', label = 'anyl. int.')
                plt.plot(s, c(re1a), label = 'real error analytic')
                plt.plot(s, c(local_moment(s)),'k:', label = 'theoretical')
            #plt.plot(mma_s, mma_res, 'ro', label= 'MMA n. int.')

            #exit()

            def root_wrapper(s):
                res, _, _ = self.vectorised_integration_function[0](s)
                return np.real(res)
            def pole_wrapper(s):
                res, _, _ = self.vectorised_integration_function[0](s)
                return np.real(1.0/res)

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
                            plt.arrow(s[i],m1,res.x[0]-s[i],res.fun[0]-m1)
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

                poly = eval("lambda s: "+root_polynomial)
                poly_data = np.array([poly(ss) for ss in s])
                plt.plot(s,c(poly_data),label='Root Polynomial')
                plt.plot(s,c(np.real(q)/poly_data),label='Factorized Moments')


                if(True):
                    print(f"Renormalising, removing order-{len(evaluated_root_list)} polynomial!")
                    q = q/poly_data

            plt.legend()
            plt.show()

            
            





            
            




            from mpl_toolkits import mplot3d
            



            xdata = []
            ydata = []
            zdata = []
            an_zdata = []

            for i in range(len(s)):
              for j in range(len(s)):
                  #if(i==j): continue
                  xdata.append(s[i])
                  ydata.append(s[j])
                  if(True):
                    zdata.append(np.log(q[i]/q[j]))
                  if(True):
                    local_moment = lambda s: gamma(1/2+s)
                    an_zdata.append(np.log(local_moment(s[i])/local_moment(s[j])))

            xdata=np.array(xdata)
            ydata=np.array(ydata)
            zdata=np.array(zdata)
            an_zdata=np.array(an_zdata)
            idx = np.abs(zdata)<0.1
            anidx = np.abs(an_zdata)<0.1

            fig = plt.figure()

            # TODO: Use these points to identify gamma(a s + b) term?
            zero_0 = 1.46163214496836 # Where logGamma'[x] = 0, x > 0
            zero_1 = -0.504083008264455 # x < 0
            zero_2 = -1.57349847316239
            
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
            
            plt.title("Projection to s-t plane")

            plt.plot(xdata[idx], ydata[idx], 'r.',label='data')
            if(True):
                plt.plot(xdata[anidx], ydata[anidx], 'k.',label ='local moment')
            plt.grid(True,'both','both')
            plt.legend()
            plt.plot(zero_0,zero_0,'ro')
            plt.plot(zero_1,zero_1,'ro')
            plt.plot(zero_2,zero_2,'ro')
            plt.show()

            plt.title("Projection to s-dq/dq plane")
            plt.plot(xdata[idx], zdata[idx], 'r.',label='data')
            if(True):
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
            plt.title("q, dq, ddq, dddq")
            plt.plot(s_loc,c(np.real(q)), label=f'q')
            plt.plot(s_loc,c(np.real(dq)), label=f'dq')
            plt.plot(s_loc,c(np.real(ddq)), label=f'ddq')
            plt.plot(s_loc,c(np.real(dddq)), label=f'dddq')
            plt.legend()
            plt.show()



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
        self.complex_s_samples = np.array([ t1 + t2*1j for t1,t2 in zip(s1,s2) ])

        # TODO: determine domain of validity

        print("We must control logic to remove the polynomial correctly before proceeding! Moments are resampled above and not correct.")
        breakpoint()

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
