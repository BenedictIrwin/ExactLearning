import numpy as np

A = np.random.uniform(low = 1, high = 5, size = 100)
B = np.random.uniform(low = -np.pi, high = np.pi, size = 100)
S = [1 + a + b*1j for a,b in zip(A,B)]

D = np.load("3D_particles.npy")

print(D)
res = np.array([ np.mean(np.power(D,s-1)) for s in S])
order= "pgame3"

## Save the results
np.save("moments_{}".format(order),res)
np.save("s_values_{}".format(order),np.array(S))
np.save("real_error_{}".format(order),np.array([1e-7 for s in S]))
np.save("imag_error_{}".format(order),np.array([1e-7 for s in S]))

