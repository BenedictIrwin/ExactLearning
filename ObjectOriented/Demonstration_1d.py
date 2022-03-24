from ObjectOriented import *



##
EE = ExactEstimator("Disk_Line_Picking", folder = "Disk_Line_Picking")
#EE = ExactEstimator("Chi_Distribution", folder = "Chi_Distribution")
fps = [gen_fpdict(['c','c^s','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma'])]
fps = [gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','neg-P1'])]
fps = [gen_fpdict(['c','c^s','alt-linear-gamma','alt-neg-linear-gamma','neg-P1'])]
#fps = [gen_fpdict(['c','c^s','linear-gamma'])]
for k in fps:
  print("Setting Fingerprint: ",k)
  EE.set_fingerprint(k)

  ## Do a bit of a random search
  ##EE.preseed(1000, logd = True)

  ##exit()
  n_bfgs = 10
  for i in range(n_bfgs):
    EE.BFGS(order = 2)
    print("{}%".format(100*(i+1)/n_bfgs),flush=True)

print("Completed!")
EE.speculate(k = 4)

EE.cascade_search()

exit()

#### Working Example... ####
EE = ExactEstimator("Simple_Exponential", folder = "Simple_Exponential")
EE.set_fingerprint( gen_fpdict(['linear-gamma']))
EE.set_fingerprint( gen_fpdict(['scale-gamma']))
n_bfgs = 1
for i in range(n_bfgs): 
  EE.BFGS(order=2)
  print("{}%".format(100*(i+1)/n_bfgs),flush=True)
EE.speculate(k = 4)
############################

EE.cascade_search()
exit()



#### Working Example... ####
EE = ExactEstimator("Beta_Distribution", folder = "Beta_Distribution")
EE.set_fingerprint( gen_fpdict(['c','shift-gamma','neg-shift-gamma']))
n_bfgs = 10
for i in range(n_bfgs): 
  EE.BFGS(order=2)
  print("{}%".format(100*(i+1)/n_bfgs),flush=True)
EE.speculate(k = 4)
############################

EE.cascade_search()

exit()


#### Working Example... ####
EE = ExactEstimator("ChiSquare_Distribution", folder = "ChiSquare_Distribution")
EE.set_fingerprint( gen_fpdict(['c','c^s','shift-gamma']))
n_bfgs = 1
for i in range(n_bfgs): 
  EE.BFGS(order=2)
  print("{}%".format(100*(i+1)/n_bfgs),flush=True)
EE.speculate(k = 4)
############################

EE.cascade_search()






##
EE = ExactEstimator("Disk_Line_Picking", folder = "Disk_Line_Picking")
#EE = ExactEstimator("Chi_Distribution", folder = "Chi_Distribution")
fps = [gen_fpdict(['c','c^s','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma'])]
#fps = [gen_fpdict(['c','c^s','linear-gamma'])]
for k in fps:
  print("Setting Fingerprint: ",k)
  EE.set_fingerprint(k)

  ## Do a bit of a random search
  ##EE.preseed(1000, logd = True)

  ##exit()
  n_bfgs = 50
  for i in range(n_bfgs):
    EE.BFGS(order = 1)
    print("{}%".format(100*(i+1)/n_bfgs),flush=True)

print("Completed!")
EE.speculate(samples = 1, k = 4)

optimal_points = [2/np.pi**(1/4),np.sqrt(2),1,1/2,1,1,2,1,5/2,1/2]
#optimal_points = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
print(optimal_points)
print(np.array(optimal_points).shape)


print("Theoretical Best!")
loss = EE.point_evaluation(optimal_points, order = 0)
print("log phi(s)",loss,optimal_points)
loss = EE.point_evaluation(optimal_points, order = 1)
print("D_s log phi(s)",loss,optimal_points)
loss = EE.point_evaluation(optimal_points, order = 2)
print("D_s D_s log phi(s)",loss,optimal_points)

plots(EE.s_values,EE.logmoments,fingerprint(*optimal_points))
plots(EE.s_values,EE.ratio,logderivative(*optimal_points))
plots(EE.s_values,EE.ratio2-EE.ratio**2,logderivative2(*optimal_points))

print("\n~~~ Numerical Best! ~~~")
loss = EE.point_evaluation(EE.best_params, order = 0)
print("log phi(s)",loss,EE.best_params)
loss = EE.point_evaluation(EE.best_params, order = 1)
print("D_s log phi(s)",loss,EE.best_params)
loss = EE.point_evaluation(EE.best_params, order = 2)
print("D_s D_s log phi(s)",loss,EE.best_params)

pp = EE.best_params
plots(EE.s_values,EE.logmoments,fingerprint(*pp))
plots(EE.s_values,EE.ratio,logderivative(*pp))
plots(EE.s_values,EE.ratio2-EE.ratio**2,logderivative2(*pp))

exit()


#########################################
## Start the code here pointing to a file
EE = ExactEstimator("Chi_Distribution", folder = "Chi_Distribution")

fps=[
#gen_fpdict(['c','linear-gamma']),
#gen_fpdict(['c^s','linear-gamma']),
gen_fpdict(['c','c^s','linear-gamma']),
#gen_fpdict(['c','c^s','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','linear-gamma','linear-gamma','linear-gamma','neg-linear-gamma','neg-linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','P1']),
#gen_fpdict(['c','c^s','linear-gamma','neg-linear-gamma','P2']),
#gen_fpdict(['c','c^s','2F1'],N=4),
#gen_fpdict(['c','c^s','2F1','linear-gamma'],N=4),
#gen_fpdict(['c','c^s','2F1','linear-gamma','neg-linear-gamma'],N=4)
]
#fp_x = gen_fpdict(['c','c^s','G-1-0-1-1'])   ### Look up how to do kwargs properly...

## Looping over fingerprints to see which is best
for k in fps:
  print("Setting Fingerprint: ",k)
  EE.set_fingerprint(k)
  n_bfgs = 300
  for i in range(n_bfgs):
    EE.BFGS()
    print("{}%".format(100*(i+1)/n_bfgs),flush=True)

print("Completed!")

## Get a few samples
samples_to_try = EE.speculate(samples = 100, k =4)

## Constrains that normalisation is true (s=1) --> Integral = 1
for q in samples_to_try:
  loss = EE.point_evaluation(q)
  print(q, loss)


EE.set_fingerprint(EE.best_fingerprint)

print(EE.best_params)
#plots(EE.s_values,EE.logmoments,fingerprint(*optimal_params))
plots(EE.s_values,EE.logmoments,fingerprint(*EE.best_params))

##print(EE.__dict__)
print(EE.results)

best_loss = EE.point_evaluation(EE.best_params)
print("Best Param Loss:", best_loss)
optimal_params = [np.sqrt(2**(-7/2)/3),np.sqrt(2**(1/2)),9/2,1/2]
best_loss = EE.point_evaluation(optimal_params)
print("Theoretical Best Loss:", best_loss)

### Consider a routine to enforce certain parameter values...




exit()






exit()
EE.set_fingerprint(fp_2)
for i in range(10): EE.BFGS()
EE.set_fingerprint(fp_3)
for i in range(10): EE.BFGS()

res = EE.results
print(res)

EE.summarise()

EE.set_mode('log')



