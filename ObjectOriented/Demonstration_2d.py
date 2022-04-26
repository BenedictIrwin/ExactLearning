from ObjectOriented import *

#### Working Example... ####
EE = ExactEstimator("2DMixedExp", folder = "2DMixedExp")

print("Managed to output moments, s (vector), derivatives (sort of)...")




print("Define how 2D/3D/ND fingerprints are constructed!")
print("Define how 2D/3D/ND function_files are constructed!")


EE.set_fingerprint( gen_fpdict(['c','shift-gamma','neg-shift-gamma']))


## c --> constant (in ND)
## c^s --> c^(a.s) for vector a of length n_s_dims

## c^s dict for n dimension in general... 
## {"D": {
##  "1" : "",
##  "2" : "",}}

## For a vector derivative n = [n1,n2,n3] giving
## d_(a1)^(n1) d_(a2)^(n2) d_(a3)^(n3) ... c^s = vec{s}^vec{n} log(c)^(|n|) c^(a.s) 

## 



fp = gen_fpdict()
EE.set_ND_fingerprint()



n_bfgs = 10
for i in range(n_bfgs): 
  EE.BFGS(order=2)
  print("{}%".format(100*(i+1)/n_bfgs),flush=True)
EE.speculate(k = 4)
############################

EE.cascade_search()

exit()

