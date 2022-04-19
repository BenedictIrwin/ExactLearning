import numpy as np
from matplotlib import pyplot as plt 
from scipy.special import gamma

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

plt.plot(valid_samples[:,0],valid_samples[:,1],'ok')

## Sample at s points
#s = np.array([[1,1],[1,1.9],[2,1],[1.5,1.5]])
s = np.random.uniform(low=1, high = 2, size = [100,2])

#test = np.array([[1/2,1/3],[1,1]])

res = np.array([np.power(t,s-1) for t in valid_samples])
print(res.shape)
print(res)


E = np.mean(np.product(res,axis = 2), axis = 0)
print(E.shape)
print(E)

def moments(s1,s2): return np.sqrt(3) * gamma( (2*s1 - s2)/3) * gamma( (s1+s2)/3) / (2 * np.pi)

mom = moments(s[:,0],s[:,1])
print("Theoretical",mom)
print("Expectation",E)

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
plt.show()



