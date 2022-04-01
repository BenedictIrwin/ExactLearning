import numpy as np
import sys
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem             ## rdkit will be replaced with stardrop descriptors/models
from rdkit.Chem import AllChem     ## ""
from rdkit.Chem import Draw        ## ""
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from PIL import Image              ## ""

import os
import subprocess

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
    self.drop_args = np.array([])
    self.drop_redundant_columns()
    
    ## Define key variables
    self.num_obs = self.data_matrix.shape[0] 
    self.num_desc = self.data_matrix.shape[1]
    self.gamma = self.num_desc/self.num_obs
    self.MP_threshold = (1 + np.sqrt(self.gamma))**2
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
    stack_count = 3
    eigenvalue_stack = []
    for i in range(stack_count):
      eigenvalue_stack.append(self.generate_randomized_samples())
    self.eigenbins, self.eigencounts = self.plot(np.array(eigenvalue_stack).flatten(),"Randomised Eigenvaue Stack")
    
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
    counts, bins, bars = plt.hist(eigenvalues_to_plot,density=True,bins=self.num_desc)  # arguments are passed to np.histogram
    print(len(counts))
    print(len(bins))
    bins = [ (bins[i]+bins[i+1])/2.0  for i in range(len(bins)-1)]
    #plt.plot(bins,counts,'o',color="black")
    plt.title(title)
    x = np.linspace((1.0-np.sqrt(self.gamma))**2,(1.0+np.sqrt(self.gamma))**2,100)
    y = mp_dist(self.gamma,x)
    plt.plot(x,y,'-',color='red')
    #x = np.linspace(abs(1.0-np.sqrt(self.gamma)),abs(1.0+np.sqrt(self.gamma)),100)
    #y = svd_dist(self.gamma,x)
    #plt.plot(x,y,'-',color='blue')
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
train_SMILES_col = train_df["SMILES"]
test_SMILES_col  = test_df["SMILES"]
train_prodSMILES_col = train_df["Product SMILES"]
test_prodSMILES_col  = test_df["Product SMILES"]

train_fingerprints = []
test_fingerprints = []
train_prod_fingerprints = []
test_prod_fingerprints = []
for i in train_SMILES_col:     train_fingerprints.append(get_fingerprint(i))
for i in test_SMILES_col:      test_fingerprints.append(get_fingerprint(i))
for i in train_prodSMILES_col: train_prod_fingerprints.append(get_fingerprint(i))
for i in test_prodSMILES_col:  test_prod_fingerprints.append(get_fingerprint(i))

train_fingerprints = np.array(train_fingerprints)
test_fingerprints = np.array(test_fingerprints)
train_prod_fingerprints = np.array(train_prod_fingerprints)
test_prod_fingerprints = np.array(test_prod_fingerprints)

## Get the XOR of the fingerprints
xor_train = np.logical_xor(train_fingerprints,train_prod_fingerprints).astype(np.float)
xor_test =  np.logical_xor(test_fingerprints,test_prod_fingerprints).astype(np.float)

## Get the OR of the fingerprints
or_train = np.logical_or(train_fingerprints,train_prod_fingerprints).astype(np.float)
or_test =  np.logical_or(test_fingerprints,test_prod_fingerprints).astype(np.float)

## Get the AND of the fingerprints
and_train = np.logical_and(train_fingerprints,train_prod_fingerprints).astype(np.float)
and_test =  np.logical_xor(test_fingerprints,test_prod_fingerprints).astype(np.float)

## Assemble dataframes with labels
xor_train_df = pd.DataFrame(xor_train,columns = ["FP{}".format(i) for i in range(fingerprint_length)])
or_train_df = pd.DataFrame(or_train,columns = ["FP{}".format(i) for i in range(fingerprint_length)])
and_train_df = pd.DataFrame(and_train,columns = ["FP{}".format(i) for i in range(fingerprint_length)])



## Get fp data
drop_names = ["SMILES","Product SMILES","Site of Metabolism"]

train_df = train_df.drop(columns = drop_names)
#train_df = train_df.join(pd.DataFrame(xor_train,columns = ["FP{}".format(i) for i in range(1024)]))
#train_df = train_df.join(pd.DataFrame(train_prod_fingerprints,columns = ["pFP{}".format(i) for i in range(1024)]))

test_df = test_df.drop(columns = drop_names)
#test_df = test_df.join(pd.DataFrame(xor_test,columns = ["FP{}".format(i) for i in range(1024)]))
#test_df = test_df.join(pd.DataFrame(test_prod_fingerprints,columns = ["pFP{}".format(i) for i in range(1024)]))

all_train_df = train_df.copy()
all_test_df = test_df.copy()

all_train = all_train_df.drop(columns=["1A1"]).values
all_test = all_test_df.drop(columns=["1A1"]).values

key = sys.argv[3]

## Take the labels out of the data set
train_labels = train_df[key]
test_labels = test_df[key]

## Join the labels to these datasets
xor_train_df = xor_train_df.join(train_labels)
or_train_df = or_train_df.join(train_labels)
and_train_df = and_train_df.join(train_labels)

## Write the datasets to file
xor_train_df.to_csv("xor.csv",index=False)
or_train_df.to_csv("or.csv",index=False)
and_train_df.to_csv("and.csv",index=False)
all_train_df.to_csv("all.csv",index=False)

## Run the calculation for the various vector elements that maximally project these spaces
## Run these in independet shells with the tf conda environment loaded

## Open up the file and get the parameters
def get_model_params_from_files(file_string):
  prefixes = ["V1_","V2_","b1_","b2_"]
  model_params = [[] for i in prefixes]
  for pair in zip(prefixes,model_params):
    with open(pair[0]+file_string) as f:
      for line in f: pair[1].append(line.strip().split(","))
  model_params = [ np.array(q).astype(np.float) for q in model_params]
  return tuple(model_params)

## For a numpy array
def ReLU(x): return np.maximum(x, 0)

## A single layer NN essentially
def apply_model(input_data, params):
  temp_model = np.matmul(input_data,params[0]) + params[2].T
  temp_model = ReLU(temp_model)
  temp_model = np.matmul(temp_model,params[1]) + params[3].T
  return temp_model
first_pass = True

output_train_df = train_df.copy()
output_test_df = test_df.copy()

num_repeats = 100
## Loop over the model generation to create many additional columns
for qq in range(num_repeats):
  
  ## Send a random sample to file (to train varied models)
  xor_train_df.sample(frac=0.6).to_csv("xor.csv",index=False)
  or_train_df.sample(frac=0.6).to_csv("or.csv",index=False)
  and_train_df.sample(frac=0.6).to_csv("and.csv",index=False)
  all_train_df.sample(frac=0.6).to_csv("all.csv",index=False)
  
  if(first_pass == True):
    subprocess.run('activate tf && python TFSeperator\Seperator.py xor.csv temp_out_xor.csv', shell=True)
    subprocess.run('activate tf && python TFSeperator\Seperator.py or.csv temp_out_or.csv', shell=True)
    subprocess.run('activate tf && python TFSeperator\Seperator.py and.csv temp_out_and.csv', shell=True)
    subprocess.run('activate tf && python TFSeperator\Seperator.py all.csv temp_out_all.csv', shell=True)
  
  xor_params = get_model_params_from_files("temp_out_xor.csv")
  or_params  = get_model_params_from_files("temp_out_or.csv")
  and_params = get_model_params_from_files("temp_out_and.csv")
  all_params = get_model_params_from_files("temp_out_all.csv")
  
  ## Apply the vectors to the relevant fingerprints
  xor_proj_train = apply_model(xor_train,xor_params)
  or_proj_train  = apply_model(or_train,or_params)
  and_proj_train = apply_model(and_train,and_params)
  all_proj_train = apply_model(all_train,all_params)
  
  xor_proj_test = apply_model(xor_test,xor_params)
  or_proj_test  = apply_model(or_test,or_params)
  and_proj_test = apply_model(and_test,and_params)
  all_proj_test = apply_model(all_test,all_params)
  
  ## Write out the training and test dataframes
  num_proj = 1
  xorcols = ["XOR Projection {}_{}".format(i,qq) for i in range(num_proj)]
  orcols =  ["OR Projection {}_{}".format(i,qq) for i in range(num_proj)]
  andcols = ["AND Projection {}_{}".format(i,qq) for i in range(num_proj)]
  allcols = ["ALL Projection {}_{}".format(i,qq) for i in range(num_proj)]
  output_train_df = output_train_df.join(pd.DataFrame(xor_proj_train, columns=xorcols))
  output_train_df = output_train_df.join(pd.DataFrame(or_proj_train, columns = orcols))
  output_train_df = output_train_df.join(pd.DataFrame(and_proj_train, columns = andcols))
  output_train_df = output_train_df.join(pd.DataFrame(all_proj_train, columns = allcols))
  output_test_df = output_test_df.join(pd.DataFrame(xor_proj_test, columns = xorcols))
  output_test_df = output_test_df.join(pd.DataFrame(or_proj_test, columns = orcols))
  output_test_df = output_test_df.join(pd.DataFrame(and_proj_test, columns= andcols))
  output_test_df = output_test_df.join(pd.DataFrame(all_proj_test, columns= allcols))


output_train_df = output_train_df.join(train_SMILES_col,how="left")
output_test_df = output_test_df.join(test_SMILES_col,how="left")
output_train_df.to_csv("TRN_3PROJ.csv",index=False)
output_test_df.to_csv("VAL_3PROJ.csv",index=False)

exit()

train_df = train_df.drop(columns=[key])
test_df  = test_df.drop(columns=[key])

true_matrix = []
false_matrix = []

## Get the matrix of training values
## Sift into different classes (true and false)
matrix = train_df.values
train_headers = train_df.columns

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


