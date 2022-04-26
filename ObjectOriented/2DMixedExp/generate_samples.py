import numpy as np
from matplotlib import pyplot as plt 
from scipy.special import gamma, digamma

def prob_mass(x,y): return np.exp(- x/y)*np.exp(- x*y**2)
## Sample points for x1,x2
samples = np.random.uniform(low = [0,0], high = [5,100], size = [1000000,2])

values = prob_mass(samples[:,0],samples[:,1])

print(values)

chances = np.random.uniform(low = 0, high = 1, size = values.shape)
bools = chances < values

valid_samples = samples[bools]

print(valid_samples)
print(valid_samples.shape)

print(np.amax(valid_samples, axis = 0))


## Get ||x||_2 max
X = np.linalg.norm(valid_samples,axis = 1)
print(X)
argmax_x = np.argmax(X)
argmin_x = np.argmin(X)
print("||x||_2 [min,max]",X[argmin_x],X[argmax_x])
print("x [min,max]",valid_samples[argmin_x],valid_samples[argmax_x])


yyy= np.array([valid_samples[argmin_x],valid_samples[argmax_x]]).T

plt.plot(valid_samples[:,0],valid_samples[:,1],'ok')
plt.plot(yyy[0],yyy[1],'ob')
plt.show()

## Sample at s points
#s = np.array([[1,1],[1,1.9],[2,1],[1.5,1.5]])
#s = np.random.uniform(low=1, high = 2, size = [100,2])
s1 = np.random.uniform(low=0, high = 5, size = [1000])
s2 = np.random.uniform(low=-7, high = 7, size = [1000])

s = np.array([s1,s2]).T



#test = np.array([[1/2,1/3],[1,1]])

res = np.array([np.power(t,s-1) for t in valid_samples])
print(res.shape)
print(res)


E = np.mean(np.product(res,axis = 2), axis = 0)
print(E.shape)
print(E)


key = np.power(valid_samples[argmax_x],s-1)
print(key)
print(key.shape)


E_max = np.product(np.power(valid_samples[argmax_x],s-1),axis = 1)
E_min = np.product(np.power(valid_samples[argmin_x],s-1),axis = 1)


def moments(s1,s2): return np.sqrt(3) * gamma( (2*s1 - s2)/3) * gamma( (s1+s2)/3) / (2 * np.pi)

mom = moments(s[:,0],s[:,1])
print("Theoretical",mom)
print("Expectation",E)


### A plot of log E on the s-surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(s1, s2, np.log(E))
x=np.linspace(0,4,10)
ax.plot(x,2*x,':k')
plt.plot(x,2*x+3,':k')
plt.plot(x,2*x+6,':k')
plt.plot(x,-x,':k')
plt.plot(x,-x-3,':k')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('log E')

##### Calculate the planes

points_1 = []
ids_1 = []
for ss_i in range(len(s)):
  ss = s[ss_i]
  if(ss[1]>=2*ss[0]): 
    points_1.append(ss)
    ids_1.append(ss_i)

points_1 = np.array(points_1)
ids_1 = np.array(ids_1)

## We want to end up looping over this somehow...
points_2 = []
ids_2 = []
for ss_i in range(len(s)):
  ss = s[ss_i]
  if(ss[1]<=-ss[0]): 
    points_2.append(ss)
    ids_2.append(ss_i)

points_2 = np.array(points_2)
ids_2 = np.array(ids_2)

ax.scatter(points_1[:,0],points_1[:,1],np.log(E[ids_1]),c = 'red')

ax.scatter(points_2[:,0],points_2[:,1],np.log(E[ids_2]),c = 'green')

from sklearn.linear_model import LinearRegression


reg = LinearRegression(fit_intercept = False)
reg.fit(points_1, np.log(E[ids_1]))
print("R_sq",reg.score(points_1, np.log(E[ids_1])))
print("coeff ",reg.coef_)
print("intercept ",reg.intercept_)

print("Est. x1_max ", np.exp(reg.coef_[0]))
print("Est. x2_max ", np.exp(reg.coef_[1]))

print("Est. N: ",np.exp(-(reg.intercept_ + np.sum(reg.coef_))))

### Generate a bunch of points in the plane
p1_s1 = np.random.uniform(low = 0, high = 4, size = 1000)
p1_s2 = np.random.uniform(low = 0, high = 10, size = 1000)
P = np.array([p1_s1,p1_s2]).T
ax.scatter(p1_s1,p1_s2,reg.predict(P), c = 'blue', marker = '.')

s1s1_1 = np.linspace(0,4,100)
s2s2_1 = (-reg.intercept_ - reg.coef_[0]*s1s1_1)/reg.coef_[1]
ax.plot(s1s1_1,s2s2_1,'r-')

reg = LinearRegression(fit_intercept = False)
reg.fit(points_2, np.log(E[ids_2]))
print("R_sq",reg.score(points_2, np.log(E[ids_2])))
print("coeff ",reg.coef_)
print("intercept ",reg.intercept_)

print("Est. x1_max ", np.exp(reg.coef_[0]))
print("Est. x2_max ", np.exp(reg.coef_[1]))

print("Est. N: ",np.exp(-(reg.intercept_ + np.sum(reg.coef_))))

### Generate a bunch of points in the plane
p2_s1 = np.random.uniform(low = 0, high = 4, size = 1000)
p2_s2 = np.random.uniform(low = 0, high = -10, size = 1000)
P = np.array([p2_s1,p2_s2]).T
ax.scatter(p2_s1,p2_s2,reg.predict(P), c = 'green', marker = '.')


s1s1_2 = np.linspace(0,4,100)
s2s2_2 = (-reg.intercept_ - reg.coef_[0]*s1s1_2)/reg.coef_[1]
ax.plot(s1s1_2,s2s2_2,'r-')

plt.show()

plt.plot(s1s1_1,s2s2_1,'r-')
plt.plot(s1s1_2,s2s2_2,'r-')



plt.plot(s[:,0],s[:,1],'ok')
plt.title("Moment Samples")
plt.xlabel("s1")
plt.ylabel("s2")
x=np.linspace(0,4,10)
plt.plot(x,2*x,':k')
plt.plot(x,2*x+3,':k')
plt.plot(x,2*x+6,':k')
plt.plot(x,-x,':k')
plt.plot(x,-x-3,':k')
plt.show()

plt.plot(E,mom,'or')
plt.xlabel('Expectation')
plt.ylabel('Theoretical')
low = np.amin([E,mom])
high = np.amax([E,mom])
print("low,high",low,high)

plt.plot([low,high],[low,high],':k')
plt.show()


plt.plot(np.sum(s,axis=1),mom,'or')
plt.plot(np.sum(s,axis=1),E,'ok')
plt.plot(np.sum(s,axis=1),E_max,'ob')
plt.plot(np.sum(s,axis=1),E_min,'ob')
plt.show()



diff = abs(E-mom)
problem = (2*s[:,0] - s[:,1])/3
for d,p in zip(diff,problem):
  print(d,p)

am = np.argmax(diff)
print(s[am])

###### Do something to filter them here
print("WARNING: Missing parts")
###
###
###### Save out the moments as required by Exact Estimator

if(True):
  print("WARNING! Generation currently bypassed, outputting exact moments for testing purposes!")
  
  s1 = np.random.uniform(low=0, high = 5, size = [100])
  s2 = np.random.uniform(low=-s1, high = 2*s1, size = [100])
  
  q  = moments(s1,s2)
  #print(mom)
  print(mom.shape)
 
  s = np.array([s1,s2]).T

  def moments(s1,s2): 
    return np.sqrt(3) * gamma( (2*s1 - s2)/3) * gamma( (s1+s2)/3) / (2 * np.pi)
  def d_1_moments(s1,s2): 
    return (gamma((2*s1 - s2)/3)*gamma((s1 + s2)/3)*(2*digamma((2*s1 - s2)/3) + digamma((s1 + s2)/3)))/(2*np.sqrt(3)*np.pi) 
  def d_2_moments(s1,s2): 
    return (gamma((2*s1 - s2)/3)*gamma((s1 + s2)/3)*(-digamma((2*s1 - s2)/3) + digamma((s1 + s2)/3)))/(2*np.sqrt(3)*np.pi)
  ##np.save("")
 
  #def d_11_moments(s1,s2): return (Gamma[(2*s1 - s2)/3]*Gamma[(s1 + s2)/3]*((2*PolyGamma[0, (2*s1 - s2)/3] + PolyGamma[0, (s1 + s2)/3])^2 + 4*PolyGamma[1, (2*s1 - s2)/3] + PolyGamma[1, (s1 + s2)/3]))/(6*Sqrt[3]*Pi) 
  #def d_12_moments(s1,s2): return (Gamma[(2*s1 - s2)/3]*Gamma[(s1 + s2)/3]*(-2*PolyGamma[0, (2*s1 - s2)/3]^2 + PolyGamma[0, (2*s1 - s2)/3]*PolyGamma[0, (s1 + s2)/3] + PolyGamma[0, (s1 + s2)/3]^2 - 2*PolyGamma[1, (2*s1 - s2)/3] + PolyGamma[1, (s1 + s2)/3]))/(6*Sqrt[3]*Pi)
  #def d_22_moments(s1,s2): return (Gamma[(2*s1 - s2)/3]*Gamma[(s1 + s2)/3]*((PolyGamma[0, (2*s1 - s2)/3] - PolyGamma[0, (s1 + s2)/3])^2 + PolyGamma[1, (2*s1 - s2)/3] + PolyGamma[1, (s1 + s2)/3]))/(6*Sqrt[3]*Pi)

## Try this
dq_1 = d_1_moments(s1,s2)
dq_2 = d_2_moments(s1,s2)

####### SAVING OUTPUTS ######
keyword = "2DMixedExp"
## Save out exact learning data (complex moments)
np.save("s_values_{}".format(keyword),s)
np.save("moments_{}".format(keyword),q)
#np.save("logmoments_{}".format(keyword),np.log(q))
#np.save("derivative_{}".format(keyword),dq)

### Save a few derivatives?
np.save("logderivative_1_{}".format(keyword),dq_1/q)
np.save("logderivative_2_{}".format(keyword),dq_2/q)

## Save a few?
print("Warning: missing parts!")
#np.save("logderivative2_11_{}".format(keyword),ddq_11/q)
#np.save("logderivative2_12_{}".format(keyword),ddq_12/q)
#np.save("logderivative2_22_{}".format(keyword),ddq_22/q)



##### Also sampling? Derivatives?

### log(x1)
### log(x2)
### log(x1)log(x2) --> actually second order... 

### This gets unweildly for many dimensions


### Also want to pass along information w.r.t the approximate bounds and stuff
### I.e. we have detected there are ... (say 3) lines that define a convex hull of convergent sampling points.
### Return domain, and also the fact there are likely to be 3 gammas (on top), and approximate constants.
print("WARNING: Missing parts")



