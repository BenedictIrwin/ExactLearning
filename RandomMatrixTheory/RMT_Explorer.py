import numpy as np
import sys
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

import scipy.stats as scs

import os
import subprocess
import sklearn.metrics

fingerprint_length = 2048

## Grabbing a fingerprint using RDkit
def get_fingerprint(smiles): return np.array(list(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smiles),minPath=1,maxPath=10,fpSize=fingerprint_length,bitsPerHash=2,useHs=False,tgtDensity=0.3,minSize=fingerprint_length)))

def mp_dist(gamma,x): return np.sqrt(((1+np.sqrt(gamma))**2-x)*(x-(1-np.sqrt(gamma))**2))/(2.0*math.pi*gamma*x)
def svd_dist(gamma,x): return 2.0*x*mp_dist(gamma,x**2)

## Sum of the square differences
def ssd(A,B): return np.sum((A-B)**2)

## Takes a np.array as input
class RMTclassifier:
  def __init__(self, matrix):
    self.data_matrix = matrix

    self.predrop_num_desc = self.data_matrix.shape[1]

    self.drop_args = np.array([])
    self.drop_redundant_columns()
    
    ## Define key variables
    self.num_obs = self.data_matrix.shape[0] 
    self.num_desc = self.data_matrix.shape[1]
    self.gamma = self.num_desc/self.num_obs
    self.gamma_dashed = self.predrop_num_desc/self.num_obs
    self.MP_threshold = (1 + np.sqrt(self.gamma))**2
    self.MP_threshold_dashed = (1 + np.sqrt(self.gamma_dashed))**2
    self.eigenvalues = np.array([])
    self.plotting_eigenvalues = np.array([])
    self.eigenvectors = np.array([])
    self.threshold = 0.0
    self.mu_list = np.array([])
    self.sigma_list = np.array([])
    self.min_list = np.array([])
    self.max_list = np.array([])
    self.eigenbins = []
    self.eigencounts = []
    self.singularbins = []
    self.singularcounts = []

    self.plotting_singularvalues = np.array([])
    self.singularuvectors = np.array([])
    self.singularvhvectors = np.array([])

    ## Just a list of things to do/prepare
    self.Z_transform()
    self.min_max_transform()

    ## Show the realtionship between svd and eig
    #plt.clf()
    #plt.plot(self.singularvalues,self.plotting_eigenvalues[:self.singularvalues.shape[0]])
    #plt.show()
    #plt.clf()
    stack_count = 10
    eigenvalue_stack = []
    for i in range(stack_count):
      eigenvalue_stack.append(self.generate_randomized_samples())
    self.eigenbins, self.eigencounts = self.plot(np.array(eigenvalue_stack).flatten(),"Randomised Eigenvalue Stack")
    
    singularvalue_stack = []
    for i in range(stack_count):
      singularvalue_stack.append(self.generate_randomized_svd())
    self.singularbins, self.singularcounts = self.plot(np.array(singularvalue_stack).flatten(),"Randomised Singular Value Stack")
   
    self.largest_random_singular_value = np.max(np.array(singularvalue_stack).flatten())
    self.perc_singular_value = np.percentile(np.array(singularvalue_stack).flatten(),90)

    self.get_eigen()
    self.get_svd()

    plt.plot(self.eigenbins,self.eigencounts,"-",color="black")
    self.plot(self.plotting_eigenvalues,"Matrix Eigenvalues") ## Comparing to MP distribution

    plt.plot(self.singularbins,self.singularcounts,"-",color="black")
    self.plot(self.singularvalues,"Matrix Singular Values") ## Comparing to MP

  ## Run the sigma tau routine for an input data matrix and activity column
  def Generate_Variational_Method(self, input_matrix, activity_column):
    ## Find the range of tau and sigma parameters to sample over
    self.tau_sigma_list = []
    
    num_rows = input_matrix.shape[0]
    print(num_rows)
    minimum_sigma = 0 ## The smallest offset allowed
    minimum_tau = 15  ## The smallest subsample can only contain this number of rows
    maximum_sigma = num_rows - minimum_tau
    tau_threshold = num_rows
    for s in range(minimum_sigma,maximum_sigma+1):
      #maximum_tau = np.amin( [num_rows - s, tau_threshold])
      #for t in range(minimum_tau,maximum_tau+1):
      t = minimum_tau
      self.tau_sigma_list.append((t,s))
    #self.tau_sigma_list = random.sample(self.tau_sigma_list, 3000)
    print("Performing {} eigendecompositions!".format(len(self.tau_sigma_list)))

    ## Clear these lists
    self.st_eigenvalue_list = []
    self.st_eigenvector_list = []
    self.st_mean_activity_list = []
    self.st_std_activity_list = []
    self.st_min_activity_list = []
    self.st_max_activity_list = []
    self.st_mu_list = []  ## For the Z-transforms of inputs
    self.st_std_list = [] ## For the Z-transforms of inputs
    self.st_drop_args_list = [] ## For the dropping of zero variance columns
    self.st_activity_list = []
    ## Get the eigenvalue, vectors, mean, std, min, max activity for these parameters
    ## Store these parameters for easy access (i.e. the lists will become object properties)
    index = 1
    for tau_value, sigma_value in self.tau_sigma_list:
      results = self.sigma_tau_matrix(sigma_value,tau_value,input_matrix,activity_column)
      self.st_eigenvalue_list.append(results[0])
      self.st_eigenvector_list.append(results[1])
      self.st_mean_activity_list.append(results[2])
      self.st_std_activity_list.append(results[3])
      self.st_min_activity_list.append(results[4])
      self.st_max_activity_list.append(results[5])
      self.st_mu_list.append(results[6])
      self.st_std_list.append(results[7])
      self.st_drop_args_list.append(results[8])
      self.st_activity_list.append(results[9])
      print("{}/{} with K = {}".format(index,len(self.tau_sigma_list),len(self.st_eigenvalue_list[-1])))
      index += 1
    return       

  ## A routine to take an input and produce a probability distribution for the activity
  ## This is made according to the set of eigensystems made from subsampling the dataset
  def Variational_Prediction(self, input_row):
    
    ## These will hold critical values
    projected_vectors = []
    distances = []
    smallest_dist = 1e90
    smallest_index = np.nan
    
    ## Define the kernel function to be used for this assesment (as a function of the activity information)
    kernel_functions = [] ## Store a list of lambdas?
    num_pairs = len(self.st_mean_activity_list)
    
    ## Make a function to make lambdas from inputs
    def create_kernel_lambda(mu,sig):
      narrowing_constant = 1.0
      return lambda x: np.exp(-narrowing_constant*(x - mu)**2/2.0/(sig**2))/np.sqrt(2.0*np.pi)/sig 

    for kf in range(num_pairs):
      kf_sig = self.st_std_activity_list[kf] 
      kf_mu = self.st_mean_activity_list[kf]
      act_list = self.st_activity_list[kf]
      ## Add a gaussian kernel function about the mean and std of this eigensystem
      #kernel_functions.append( lambda x : np.sum( [np.exp(-narrowing_constant*(x - jj)**2/2.0/(kf_sig**2))/np.sqrt(2.0*np.pi)/kf_sig for jj in act_list] ) )
      kernel_functions.append( create_kernel_lambda(kf_mu,kf_sig) )

    ## For each of the eigensystems(sigma,tau) in the self.lists
    for es in range(num_pairs):
      ## Drop the args of the row to fit with this system
      drop_row = np.delete(input_row, self.st_drop_args_list[es])

      ## Z-transform the input row to meet this predictor
      z_trans_row = np.divide( drop_row - self.st_mu_list[es], self.st_std_list[es] )

      ## Project the vector into subspace = span(v_1,..,v_K) for the K RMT significant eigenvectors
      temp = np.array([ np.dot(eigv,z_trans_row)*eigv for eigv in self.st_eigenvector_list[es]])
      projected_val = np.sum(temp,axis=0)

      ## Add projected vector to list
      projected_vectors.append(projected_val)

      ## Measure the distance from the original input ( ||u-u_p||_2 )
      ## DO WE NEED TO COMPENSATE FOR THE AVERAGE LENGTH OF VECTORS i.e. RANDOM VECTORS?
      diff = np.sqrt( ssd(z_trans_row,projected_val) )
      distances.append(diff)


      ## Check if this is the smallest distance so far (d_min)
      ## Note down index
      if(diff < smallest_dist):
        smallest_dist = diff
        smallest_index = es

    ## Calculate weights relative to the smallest distance?
    #weights = [ 1.0/(distances[ii]+0.001*smallest_dist)/self.tau_sigma_list[ii][0] for ii in range(len(distances))]
    #weights = [ 1.0/(distances[ii]+0.00001*smallest_dist) for ii in range(len(distances))]
    weights = [ np.exp((smallest_dist - distances[ii])/2.0)/self.tau_sigma_list[ii][0] for ii in range(len(distances))]
    normalisation_constant = np.sum(weights)
    #print("Norm = {}".format(normalisation_constant)) 
    #plt.title("Distances")
    #plt.hist(distances,bins=100)
    #plt.show()
    #plt.title("Weights")
    #plt.hist(weights,bins=100)
    #plt.show()
    #plt.title("Means")
    #plt.hist(self.st_mean_activity_list,bins=100)
    #plt.show()
    #plt.title("Sigmas")
    #plt.hist(self.st_std_activity_list,bins=100)
    #plt.show()

    ## Calculate the activities between the largest and smallest seen
    
    #plt.title("RMT Derived Activity Density Estimation")
    #x = np.linspace(np.min(self.st_min_activity_list),np.max(self.st_max_activity_list),200)
    ### For each of the x sum up the lambdas with the weights to make y
    #def weighted_sum(v,w): return np.sum(np.dot(v,w))
    #y = [ weighted_sum([f(xx) for f in kernel_functions], weights)/normalisation_constant for xx in x]
    #plt.plot(x,y,'-',color='black')
    #plt.show()

    ## For this sample, plot the activity distributions as a function of the distances
    #plt.title("RMT Derived Activity Estimation")
    #plt.xlabel("Distances [RMT Projection]")
    #plt.ylabel("Activities")
    #for xe, ye in zip(distances, self.st_activity_list):
    #  plt.plot([xe] * len(ye), ye,'o',color='black',alpha=0.01)
    #plt.show()
    ## Draw a distribution of the 'predicted' activity
    #print("Smallest Diff = {}".format(smallest_dist))
   
    ## Return an interator for the plotting points
    return zip(distances, self.st_activity_list)

  ## Get the K significant eigenvectors and values for a matrix which is a cut of the current data matrix for varaitional methods
  ## Assumes them to be sorted in the same order
  def sigma_tau_matrix(self,sigma,tau,input_matrix,activity_column): 
    ## Get activity column slice
    activity_slice = activity_column[sigma:tau+sigma]

    ## Get data slice
    ## if sigma=0 and tau=N then the whole matrix
    data_slice = input_matrix[sigma:tau+sigma,:]

    ## Get normalisation for this data
    st_mu_list =    np.array([ np.mean(i) for i in data_slice.T])
    st_sigma_list = np.array([ np.std(i) for i in data_slice.T])

    ## Get the indices of sigma=0.0 columns (i.e. drop_args)
    st_drop_args = []
    for i in range(len(st_sigma_list)):
      if(st_sigma_list[i]==0.0): st_drop_args.append(i)
     
    ## Drop these parts
    st_mu_list = np.delete(st_mu_list,st_drop_args)
    st_sigma_list = np.delete(st_sigma_list,st_drop_args)
    data_slice = np.delete(data_slice,st_drop_args,axis=1) ## axis = ???
    
    ## Get dimensions of final data section
    num_rows = data_slice.shape[0]
    num_cols = data_slice.shape[1]
    ## Make normalised matrix
    norm_slice = np.array(np.divide(data_slice-st_mu_list,st_sigma_list))

    ## Make correlation matrix
    correlation_matrix = np.matmul(norm_slice.T,norm_slice)/(num_rows)

    values, vectors = np.linalg.eig(correlation_matrix)
    values = np.real(values) 

    ## Get MP threshold
    st_gamma = num_cols/num_rows
    st_MP_threshold = (1 + np.sqrt(st_gamma))**2

    ## Calculate eigenvalue demposition
    sig_eig_index = []
    sig_eig = []
    sig_eigv = []
    for i in range(num_cols):
      if(values[i] > st_MP_threshold): sig_eig_index.append(i)
    for i in sig_eig_index:
      u = vectors[:,i]
      sig_eigv.append(np.real(u))
      sig_eig.append(values[i])

    ## Get the eigenvectors/values
    ## The final numpy arrays here
    sig_eig  = np.array(sig_eig)
    sig_eigv = np.array(sig_eigv)

    ## Get activity range for this data (to link to this set of vectors)
    max_activity  = np.amax(activity_slice)
    min_activity  = np.amin(activity_slice)
    mean_activity = np.mean(activity_slice)
    activity_std  = np.std(activity_slice)

    ## Output these quantities and the additional information required
    return sig_eig, sig_eigv, mean_activity, activity_std, min_activity, max_activity, st_mu_list, st_sigma_list, st_drop_args, activity_slice
    
  def reconstruct_svd_data(self):
    sd = np.zeros(shape=[self.singularuvectors.shape[0],self.singularvhvectors.shape[0]])
    count_sig_sv = np.sum([1 if sv > self.perc_singular_value else 0 for sv in self.singularvalues])
    for i in range(count_sig_sv): sd[i,i]=self.singularvalues[i]
    print("Reconstructing using:")
    print(sd)
    reconstructed_data_matrix = np.matmul(np.matmul(self.singularuvectors,sd),self.singularvhvectors) 
    reconstructed_data_matrix = np.array([ np.multiply(self.sigma_list,row)+self.mu_list for row in reconstructed_data_matrix ]) 
    print("Not un-normalised")
    return reconstructed_data_matrix

  def generate_randomized_svd(self):
    ## From the data matrix get the columns and shuffle the values up so they are not related
    shuffled_matrix = []
    for i in self.data_matrix.copy().T:
      random.shuffle(i)
      shuffled_matrix.append(i)
    shuffled_matrix = np.array(shuffled_matrix).T
    matrix_norm = np.array([ np.divide(shuffled_matrix[i]-self.mu_list,self.sigma_list) for i in range(self.num_obs) ])
    
    #matrix_norm = self.data_matrix
    #correlation_matrix = np.matmul(matrix_norm.T,matrix_norm)/(self.num_obs) 
    temp_u, temp_s, temp_vh = np.linalg.svd(matrix_norm)
    #temp_eigenvalues = np.real(temp_eigenvalues) 
    #self.plot(temp_s)
    return temp_s
  
  def generate_randomized_samples(self):
    ## From the data matrix get the columns and shuffle the values up so they are not related
    shuffled_matrix = []
    for i in self.data_matrix.copy().T:
      random.shuffle(i)
      shuffled_matrix.append(i)
    shuffled_matrix = np.array(shuffled_matrix).T
    matrix_norm = np.array([ np.divide(shuffled_matrix[i]-self.mu_list,self.sigma_list) for i in range(self.num_obs) ])
    correlation_matrix = np.matmul(matrix_norm.T,matrix_norm)/(self.num_obs) 
    temp_eigenvalues, temp_eigenvectors = np.linalg.eig(correlation_matrix)
    temp_eigenvalues = np.real(temp_eigenvalues) 
    ##self.plot(temp_eigenvalues)
    return temp_eigenvalues

  def drop_redundant_columns(self):
    ## Drop any columns in the data matrix with zero variance
    flip = self.data_matrix.T
    flip_var = [np.var(row) for row in flip]
    self.drop_args = []
    for i in range(len(flip_var)):
      if(flip_var[i]==0.0): self.drop_args.append(i)
    self.data_matrix = np.delete(flip,self.drop_args,axis=0).T

  ## Updates the mu and sigma lists
  def Z_transform(self):
    self.mu_list =    np.array([ np.mean(i) for i in self.data_matrix.T])
    self.sigma_list = np.array([ np.std(i) for i in self.data_matrix.T])

  def min_max_transform(self):
    self.min_list = np.array([ min(i) for i in self.data_matrix.T])
    self.max_list = np.array([ max(i) for i in self.data_matrix.T])

  ## Use the current transformation on the input data to output the principle values
  def transform_data_to_PC(self, input_matrix):
    ## Drop useless columns
    temp = np.delete(input_matrix.T,self.drop_args,axis=0).T
    ## Normalise
    temp = np.array([ np.divide(i-self.mu_list,self.sigma_list) for i in temp ])
    ## Get the first few eigenvectors
    output_matrix = np.matmul(temp,self.eigenvectors.T)
    return output_matrix

  ## Get the significant eigenvalues
  def get_eigen(self):
    matrix_norm = np.array([ np.divide(self.data_matrix[i]-self.mu_list,self.sigma_list) for i in range(self.num_obs) ])
    
    #matrix_norm = self.data_matrix
    correlation_matrix = np.matmul(matrix_norm.T,matrix_norm)/(self.num_obs) 
    self.eigenvalues, self.eigenvectors = np.linalg.eig(correlation_matrix)
    self.eigenvalues = np.real(self.eigenvalues) 
    self.plotting_eigenvalues = self.eigenvalues

    sig_eig_index = []
    sig_eig = []
    sig_eigv = []
    for i in range(self.num_desc):
      if(self.eigenvalues[i] > self.MP_threshold): sig_eig_index.append(i)
    for i in sig_eig_index:
      u = self.eigenvectors[:,i]
      sig_eigv.append(np.real(u))
      sig_eig.append(self.eigenvalues[i])

    self.eigenvalues = np.array(sig_eig)
    self.eigenvectors = np.array(sig_eigv)

  ## Get the significant eigenvalues
  def get_svd(self):
    matrix_norm = np.array([ np.divide(self.data_matrix[i]-self.mu_list,self.sigma_list) for i in range(self.num_obs) ])
    u,s,vh = np.linalg.svd(matrix_norm)
    sd = np.zeros(shape=[u.shape[0],vh.shape[0]])
    for i in range(len(s)): sd[i,i]=s[i]
    #print(np.matmul(np.matmul(u,sd),vh) - matrix_norm)
    
    self.singularvalues = s
    self.singularuvectors = u
    self.singularvhvectors = vh

    return 
    
    
    self.eigenvalues, self.eigenvectors = np.linalg.eig(correlation_matrix)
    self.eigenvalues = np.real(self.eigenvalues) 
    self.plotting_eigenvalues = self.eigenvalues

    sig_eig_index = []
    sig_eig = []
    sig_eigv = []
    for i in range(self.num_desc):
      if(self.eigenvalues[i] > self.MP_threshold): sig_eig_index.append(i)
    for i in sig_eig_index:
      u = self.eigenvectors[:,i]
      sig_eigv.append(np.real(u))
      sig_eig.append(self.eigenvalues[i])

    self.eigenvalues = np.array(sig_eig)
    self.eigenvectors = np.array(sig_eigv)

  ## Given a new row of the dropped dataset make a prediction
  def get_diff(self, row):
    transformed_val  = np.divide(row-self.mu_list,self.sigma_list)
    temp = np.array([ np.dot(eigv,transformed_val)*eigv for eigv in self.eigenvectors])
    projected_val = np.sum(temp,axis=0) 
    diff = np.sqrt( ssd(transformed_val,projected_val) ) 
    return diff

  ## Return True or False if predicted to belong to the set
  def predict(self, row):
    row = np.delete(row, self.drop_args)
    transformed_val  = np.divide(row-self.mu_list,self.sigma_list)
    temp = np.array([ np.dot(eigv,transformed_val)*eigv for eigv in self.eigenvectors])
    projected_val = np.sum(temp,axis=0) 
    diff = np.sqrt( ssd(transformed_val,projected_val) )
    confidence = diff/abs(self.threshold)
    if(diff < self.threshold): return True, confidence
    return False, confidence

  ## This assumes a 95% threshold as in the paper
  def determine_threshold_from_internal_data(self):
    epsilon_spectrum = [ self.get_diff(row) for row in self.data_matrix ]
    self.threshold = sorted(epsilon_spectrum)[int(np.floor(0.95*len(epsilon_spectrum)))]
  
  ## Use some traning data to get a good value for the threshold
  def determine_threshold_from_training_data(self, training_data, training_labels):
    ## Drop the required columns of the matrix
    input_data = np.delete(training_data.T, self.drop_args, axis=0).T 
    ## Get the spectrum of epsilon values
    epsilon_spectrum = [ self.get_diff(row) for row in input_data ]
    
    ## Find the value of epsilon that maximises sum(true_positives) - sum(false_positives) as a function of epsilon
    high_score = 0
    high_eps = 0
    Q = list(zip(epsilon_spectrum,training_labels))
    for trial in sorted(epsilon_spectrum):
      true_positives  = np.sum([ 1.0 if (ez[0] <=trial and ez[1]==True) else 0.0 for ez in Q])
      false_positives = np.sum([ 1.0 if (ez[0] <=trial and ez[1]==False) else 0.0 for ez in Q])
      score = true_positives - false_positives
      if(score > high_score):
        high_score = score
        high_eps = trial

    self.threshold = high_eps

  def plot(self,eigenvalues_to_plot,title):
    counts, bins, bars = plt.hist(eigenvalues_to_plot,density=True,bins=self.data_matrix.shape[1])  # arguments are passed to np.histogram
    print(len(counts))
    print(len(bins))
    bins = [ (bins[i]+bins[i+1])/2.0  for i in range(len(bins)-1)]
    #plt.plot(bins,counts,'o',color="black")
    plt.title(title)
    x = np.linspace((1.0-np.sqrt(self.gamma))**2,(1.0+np.sqrt(self.gamma))**2,100)
    y = mp_dist(self.gamma,x)
    plt.plot(x,y,'-',color='red')
    x = np.linspace((1.0-np.sqrt(self.gamma_dashed))**2,(1.0+np.sqrt(self.gamma_dashed))**2,100)
    y = mp_dist(self.gamma_dashed,x)
    plt.plot(x,y,'-',color='blue')
    #modulator = 5.0
    #x = np.linspace(abs(1.0-np.sqrt(modulator*self.gamma)),abs(1.0+np.sqrt(modulator*self.gamma)),100)
    #y = svd_dist(modulator*self.gamma,x)
    #plt.plot(x,y,'-',color='blue')
    plt.show()
    return bins, counts
    
##########
## MAIN ##
##########
## Get files
file_name = sys.argv[1]
validation_name = sys.argv[2]
train_df = pd.read_csv(file_name)
test_df  = pd.read_csv(validation_name)

key = "2B7"

## Sort the training file by the activity of choice
train_df = train_df.sort_values(by=key,ascending=False)

train_labels = train_df[key].values
print(train_labels.astype(float))
test_labels = test_df[key]

drop_labels = [key, "Substrate"]

train_df = train_df.drop(columns = drop_labels)
test_df  = test_df.drop(columns  = drop_labels)

true_matrix = []
false_matrix = []

## Get the matrix of training values
## Sift into different classes (true and false)
matrix = train_df.values
train_headers = train_df.columns

test_matrix = test_df.values

## Make a shuffled up version of the matrix with similar elements but no correlation
shuffled_matrix = []
for i in matrix.copy().T:
  random.shuffle(i)
  shuffled_matrix.append(i)
shuffled_matrix = np.array(shuffled_matrix).T

print("Generating RMT Model")
RMTmodel = RMTclassifier(matrix) ## Initiate a model using the training matrix
RMTmodel.Generate_Variational_Method(matrix,train_labels)

print("Generating Randomized model for comparison")
randomRMTmodel = RMTclassifier(shuffled_matrix) ## Initiate a random model using the training matrix
randomRMTmodel.Generate_Variational_Method(shuffled_matrix,train_labels)


for power in [i/4 for i in range(200)]:
  index = 0
  true_values = []
  pred_values = []
  for row in test_matrix:
    #print(row)
    #print("Expected Activity: {}".format(test_labels[index]))
    plot_points = RMTmodel.Variational_Prediction(row)
    random_points = randomRMTmodel.Variational_Prediction(row)
    #plt.title("RMT Derived Activity Estimation")
    #plt.xlabel("Distances [RMT Projection]")
    #plt.ylabel("Activities")
  
    rmt_xs = []
    rmt_ys = []
    for xe, ye in plot_points:
      rmt_xs.append(xe)
      rmt_ys.append(ye)
    #print(rmt_xs)
    #print(rmt_ys)
    
    rand_xs = []
    rand_ys = []
    for xe, ye in random_points:
      rand_xs.append(xe)
      rand_ys.append(ye)
    #print(rand_xs)
    #print(rand_ys)
    
    min_rand_dist = min(rand_xs)
    #print("Minimum Random Distance = {}".format(min_rand_dist))
  
    ## Estimator using mean value
    prediction_value = 0.0
    weight_sum = 0.0
    for x,y_vec in zip(rmt_xs,rmt_ys):
      if(x<min_rand_dist):
        weight = 1.0/(x**power)
        mode_results = float(scs.mode(y_vec)[0][0])
        mean_results = round(np.mean(y_vec))
        prediction_value += weight*mean_results
        weight_sum += weight
    prediction_value /= weight_sum
    prediction_value = round(prediction_value)
  
    true_values.append(test_labels[index])
    pred_values.append(prediction_value)
  
    #for xe, ye in plot_points:
    #  plt.plot([xe] * len(ye), ye,'o',color='black',alpha=0.1)
    #for xe, ye in random_points:
    #  plt.plot([xe] * len(ye), ye,'o',color='red',alpha=0.1)
    #plt.show()
    index+=1
 
  true_values_float = np.array(true_values).astype(float)
  R2    = sklearn.metrics.r2_score(true_values_float,pred_values)
  kappa = sklearn.metrics.cohen_kappa_score(true_values_float,pred_values)
  print("{},{},{}".format(power,R2,kappa))
  

exit()

index = 0 
for row in test_matrix:
  print(row)
  print("Expected Activity: {}".format(test_labels[index]))
  RMTmodel.Variational_Prediction(row)
  index+=1


exit()




for i in range(len(train_labels)):
  if(train_labels[i]==False): false_matrix.append(matrix[i])
  elif(train_labels[i]==True):  true_matrix.append(matrix[i])
true_matrix = np.array(true_matrix)
false_matrix = np.array(false_matrix)

print("Creating Positive Model")
## Generate two random matrix theory models
true_model  = RMTclassifier(true_matrix) 
print("Creating Negative Model")
false_model = RMTclassifier(false_matrix) 
print("Creating Combined Model")
combined_model = RMTclassifier(matrix) 


print("Before:")
print(matrix)
print("After:")
recon = combined_model.reconstruct_svd_data()

## Assemble a print out df
output_df = pd.DataFrame(train_SMILES_col)
output_df["1A1"] = train_labels
output_df = output_df.join(pd.DataFrame(recon,columns = np.delete(train_headers,combined_model.drop_args)))
output_df.to_csv("Output_recon_train.csv")

#exit()

## Get eigenvectors for all data out 
true_PC_matrix = true_model.transform_data_to_PC(matrix)
false_PC_matrix = false_model.transform_data_to_PC(matrix)
comb_PC_matrix = combined_model.transform_data_to_PC(matrix)

## Assemble a print out df
output_df = pd.DataFrame(train_SMILES_col)
output_df["1A1"] = train_labels
output_df = output_df.join(pd.DataFrame(true_PC_matrix,columns = ["T{}".format(i) for i in range(true_PC_matrix.shape[1])]))
output_df = output_df.join(pd.DataFrame(false_PC_matrix,columns = ["F{}".format(i) for i in range(false_PC_matrix.shape[1])]))
output_df = output_df.join(pd.DataFrame(comb_PC_matrix,columns = ["C{}".format(i) for i in range(comb_PC_matrix.shape[1])]))
output_df.to_csv("Output_train.csv")

test_matrix = test_df.values

## Get eigenvectors for all data out 
true_PC_matrix = true_model.transform_data_to_PC(test_matrix)
false_PC_matrix = false_model.transform_data_to_PC(test_matrix)
comb_PC_matrix = combined_model.transform_data_to_PC(test_matrix)

## Assemble a print out df
output_df = pd.DataFrame(test_SMILES_col)
output_df["1A1"] = test_labels
output_df = output_df.join(pd.DataFrame(true_PC_matrix,columns = ["T{}".format(i) for i in range(true_PC_matrix.shape[1])]))
output_df = output_df.join(pd.DataFrame(false_PC_matrix,columns = ["F{}".format(i) for i in range(false_PC_matrix.shape[1])]))
output_df = output_df.join(pd.DataFrame(comb_PC_matrix,columns = ["C{}".format(i) for i in range(comb_PC_matrix.shape[1])]))
output_df.to_csv("Output_test.csv")



## Get the best value of the threshold parameter from the training data and labels (one parameter model)
#true_model.determine_threshold_from_internal_data()
#false_model.determine_threshold_from_internal_data()
#print("Determining Positive Model Threshold")
true_model.determine_threshold_from_training_data(matrix,train_labels.values)

#print("Determining Negative Model Threshold")
false_model.determine_threshold_from_training_data(matrix,np.invert(train_labels.values))  ## We put not train labels in

print("Determining Positive Model Threshold")
#true_model.determine_threshold_from_internal_data()

print("Determining Negative Model Threshold")
#false_model.determine_threshold_from_internal_data()  ## We put not train labels in

print(true_model.threshold)
print(false_model.threshold)


print("Training Scores:")
true_model_score = 0
false_model_score = 0
comb_model_score = 0

print(matrix)

for i in zip(matrix,train_labels):
  true_pred, true_conf = true_model.predict(i[0])
  false_pred, false_conf = false_model.predict(i[0])
  false_pred = not false_pred
  
  
  #if(true_pred != false_pred):
  #  if( true_conf < false_conf ): pred = true_pred
  #  else: pred = false_pred
  
  label = i[1]
  if(true_pred==label): true_model_score+=1
  if(false_pred==label): false_model_score+=1
  #if(pred==label): comb_model_score+=1

print("True Model: {}".format(true_model_score/len(train_labels)))
print("False Model: {}".format(false_model_score/len(train_labels)))
print("Comb Model: {}".format(comb_model_score/len(train_labels)))


print("Test Scores:")
true_model_score = 0
false_model_score = 0
comb_model_score = 0

for i in zip(test_matrix,test_labels):
  true_pred, true_conf = true_model.predict(i[0])
  false_pred, false_conf = false_model.predict(i[0])
  false_pred = not false_pred
  
 # print(true_conf,false_conf)

  #if(true_pred != false_pred):
  #  if( true_conf < false_conf ): pred = true_pred
  #  else: pred = false_pred
  
  label = i[1]
  if(true_pred==label): true_model_score+=1
  if(false_pred==label): false_model_score+=1
  #if(pred==label): comb_model_score+=1

print("True Model: {}".format(true_model_score/len(train_labels)))
print("False Model: {}".format(false_model_score/len(train_labels)))
#print("Comb Model: {}".format(comb_model_score/len(train_labels)))

exit()


