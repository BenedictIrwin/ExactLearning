from ObjectOriented import *

#### Working Example... ####
EE = ExactEstimator("2DMixedExp", folder = "2DMixedExp")


print("Define how 2D/3D/ND fingerprints are constructed!")
print("Define how 2D/3D/ND function_files are constructed!")
EE.set_fingerprint( gen_fpdict(['c','shift-gamma','neg-shift-gamma']))


n_bfgs = 10
for i in range(n_bfgs): 
  EE.BFGS(order=2)
  print("{}%".format(100*(i+1)/n_bfgs),flush=True)
EE.speculate(k = 4)
############################

EE.cascade_search()

exit()

