## This script is an efficient tensorflow port of the Mellin learning idea.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import random
import string
import pandas as pd
import datetime

## These may come in handy
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA

## Don't output tf warnings about possible speedups etc.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)
#tf.enable_eager_execution()

## A list of names so we can recognise stardrop descriptors
#stardrop_descriptor_names = []
#with open("stardrop_descriptors") as f:
#  for line in f:
#    line = line.strip()
#    stardrop_descriptor_names.append(line)

def main():
  argc=len(sys.argv)
  if(argc!=2):
    print("Usage: [Data.csv]")
    exit()

  ## Read Data from file into a data frame
  DATA = sys.argv[1]
  data_df=pd.read_csv(DATA)
  num_data_columns = len(data_df.columns)
  num_dim = num_data_columns 
  hypergeom_p = 1
  hypergeom_q = hypergeom_p - 1
  session_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  print("Data appears to have {} columns. Assuming multivariate probability distribution with {} varaibles.".format(num_data_columns,num_data_columns))
  log_file_string = "MELL_log_{}.txt".format(session_string)
  ## A random string for this session deatils

  ## Set to floating point data
  data = data_df.astype('float64')
  num_rows = data.shape[0]
  print("There are {} rows".format(num_rows))

  ##########################
  ## Inputs and constants ##
  ##########################
  EPOCHS = 1000
  n_batches = 50
  model_weight         = 1.0
  normalisation_weight = 5.0
  R_squared_weight     = 1.0
  intercept_weight     = 1.0
  gradient_weight      = 100.0
  
   
  # Create a placeholder to dynamically switch between batch sizes
  batch_size = tf.placeholder(tf.int64)
  drop_prob = tf.placeholder(tf.float32)

  ## Training Boolean
  training_bool = tf.placeholder(tf.bool)

  ##################
  ## Initializers ##
  ##################
  weight_initialiser_mean = 1.0
  weight_initialiser_std = 0.01

  ## Question do we need a separate initializer for everything?
  #weight_initer = tf.truncated_normal_initializer(mean=weight_initialiser_mean, stddev=weight_initialiser_std)
  #weight_initer = tf.constant_initializer(np.eye(num_gamma,num_dim),dtype=tf.float32)
  #avec_initer = tf.random_uniform_initializer(minval=0.90,maxval=1.1,dtype=tf.float32)
  #alpha_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)
  #beta_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)
  #q_initer = tf.constant_initializer(np.eye(num_dim),dtype=tf.float32)
  
  #matrix_M_init = tf.constant_initializer(np.eye(num_dim,num_dim),dtype=tf.float32)
  matrix_V_init = tf.truncated_normal_initializer(mean=weight_initialiser_mean, stddev=weight_initialiser_std)
  matrix_W_init = tf.truncated_normal_initializer(mean=weight_initialiser_mean, stddev=weight_initialiser_std)
  param_a_init  = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
  param_b_init  = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
  eta_init      = tf.truncated_normal_initializer(mean=1.0, stddev=0.1)

  ########################
  ## Input Placeholders ##
  ########################
  ## The data objects, x input data, S set of input exponents
  x = tf.placeholder(tf.float32, shape=[None,num_dim]) 
  S = tf.placeholder(tf.float32, shape=[None,num_dim])

  ##########################
  ## Restart placeholders ## For continuing and restarting if errors occur
  ##########################
  #eta_reset_input = tf.placeholder( tf.float32, shape=[num_dim])
  #q_reset_input = tf.placeholder( tf.float32, shape=[num_dim,num_dim])
  #w_reset_input = tf.placeholder( tf.float32, shape=[num_gamma, num_dim])
  #alpha_reset_input = tf.placeholder( tf.float32, shape=[num_gamma])
  #beta_reset_input = tf.placeholder( tf.float32, shape=[num_dim])
  #a_reset_input = tf.placeholder( tf.float32, shape=[num_gamma])
  
  ############################
  ## Iterators and features ##
  ############################
  ds_buffer_constant = 1000
  dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=ds_buffer_constant).batch(batch_size).repeat()
  iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes) 
  features = iter.get_next()
  dataset_init_op = iter.make_initializer(dataset)

  print("TO ADD: Scale parameter exponenets must equal variable exponenets in the series expansion via GRMT!")
  print("TO ADD: Write best models to file as we go (periodically to not loose data) : Checkpointing")
  print("TO ADD: Parameter files : Check it works first!")
  print("TO ADD: Add initialiser constants to log")
  print("Put a condition on the sign of the moment?")
  print("Num gammas must be greater or equal to num dim")
  print("Treat some of the gammas as a subset which are a function of the scale weights")
  print("How does GMRT det A come into this? will there be an additional scaling factor?")
  print("Complex network and ability to minimise the complex parts in real predictions? Then we can include the (-1)^(q.s+beta) kind of terms")
  print("Start with a product of marginals (which may be quicker/easier to train) then start with an expression that represents that product for the full distribution")
  print("#####")
  print("Check with simple function! How close can loss get to zero? Can we model this with a noise distribution?")
  print("Can we find the s which maximises the error for a given function? Can we work on that s only?")
  print("Consider a loss function that is signed for small errors such that they cancel if they are noise but do not if they are large?")
  print("For syymetry use the sum of the above function flipped!?")
  print("Check Autonorm is working correctly")
  print("Check Error expectations are working correctly")
  print("Check range 0-1 in s is acceptable?")

  ## Make a log of parameters
  with open(log_file_string, "w") as f:
    f.write("Session {} created {}\n".format(session_string,datetime.datetime.now()))
    f.write("Data of shape {}\n".format(data.shape))
    f.write("Number of dimensions = {}\n".format(num_dim))
    f.write("Hypergeometric p = {}\n".format(hypergeom_p))
    f.write("Hypergeometric q = {}\n".format(hypergeom_q))
    f.write("EPOCHS = {}\n".format(EPOCHS))
    f.write("n_batches = {}\n".format(EPOCHS))
    f.write("normalisation_weight = {}\n".format(normalisation_weight))
    f.write("intercept_weight = {}\n".format(intercept_weight))
    f.write("gradient_weight = {}\n".format(gradient_weight))
    f.write("R_squared_weight = {}\n".format(R_squared_weight))
    f.write("ds_buffer_constant = {}\n".format(ds_buffer_constant))
  
  #####################
  ## Model Variables ##
  #####################
  eta      = tf.get_variable(name="eta", dtype=tf.float32, shape = [num_dim], initializer = eta_init)
  matrix_V = tf.get_variable(name="V", dtype=tf.float32, shape=[hypergeom_p, num_dim], initializer=matrix_V_init)
  matrix_W = tf.get_variable(name="W", dtype=tf.float32, shape=[hypergeom_q, num_dim], initializer=matrix_W_init)
  param_a  = tf.get_variable(name="a", dtype=tf.float32, shape=[hypergeom_p], initializer=param_a_init) 
  param_b  = tf.get_variable(name="b", dtype=tf.float32, shape=[hypergeom_q], initializer=param_b_init) 
   
  eta_input      = tf.placeholder(dtype=tf.float32, shape=[num_dim]) 
  param_a_input  = tf.placeholder(dtype=tf.float32, shape=[hypergeom_p]) 
  param_b_input  = tf.placeholder(dtype=tf.float32, shape=[hypergeom_q]) 
  matrix_V_input = tf.placeholder(dtype=tf.float32, shape=[hypergeom_p,num_dim]) 
  matrix_W_input = tf.placeholder(dtype=tf.float32, shape=[hypergeom_q,num_dim]) 
  
  print("CHECK: use locking status and effect?")
  eta_assign_op      = tf.assign(eta,      eta_input,      use_locking=False)
  param_a_assign_op  = tf.assign(param_a,  param_a_input,  use_locking=False)
  param_b_assign_op  = tf.assign(param_b,  param_b_input,  use_locking=False)
  matrix_V_assign_op = tf.assign(matrix_V, matrix_V_input, use_locking=False)
  matrix_W_assign_op = tf.assign(matrix_W, matrix_W_input, use_locking=False)
  
  #####################
  ## Model Equations ##
  #####################
  
  ## Print (possible issue)
  print("Possible issue where the constant terms do not have the right dimension?")
  
  logXi_a = tf.reduce_sum(tf.lgamma(param_a))  ## Hypergeometric p normalisation coeffs
  logXi_b = tf.reduce_sum(tf.lgamma(param_b))  ## Hypergeometric q normalisation coeffs
  
  ## Things that depend on the exponent sample
  logXi_s           = tf.map_fn( lambda s : tf.reduce_sum(tf.lgamma(s)), S, dtype=tf.float32)
  logXi_a_minus_V_s = tf.map_fn( lambda s : tf.reduce_sum(tf.lgamma(param_a - tf.tensordot(matrix_V,s,axes=[[1],[0]]))), S, dtype=tf.float32) 
  logXi_b_minus_W_s = tf.map_fn( lambda s : tf.reduce_sum(tf.lgamma(param_b - tf.tensordot(matrix_W,s,axes=[[1],[0]]))), S, dtype=tf.float32) 
  
  ## Calculate the log of the analytic moments (as a vector over samples in S)
  logM = logXi_b + logXi_a_minus_V_s + logXi_s - logXi_a - logXi_b_minus_W_s
  
  ## A function to increarse the impact of large weights to offset the log penalty
  def train_func(x): return x

  logMfunc = train_func(logM)
  
  s_norm = tf.ones(num_dim)
  print("Optimisation: lgamma(snorm) = np.zeroes([num_dim])")
  print("Optimisation: tf.reduce_sum(np.zeroes([num_dim])) = 0")
  print("NOTE: param_a elements must be bigger than the row sums of V for positive arguments to loggamma(x)!")
  print("NOTE: param_b elements must be bigger than the row sums of W for positive arguments to loggamma(x)!")
  logXi_s_norm           = tf.reduce_sum(tf.lgamma(s_norm))
  logXi_a_minus_V_s_norm = tf.reduce_sum(tf.lgamma(param_a - tf.tensordot(matrix_V,s_norm,axes=[[1],[0]]))) 
  logXi_b_minus_W_s_norm = tf.reduce_sum(tf.lgamma(param_b - tf.tensordot(matrix_W,s_norm,axes=[[1],[0]]))) 

  norm_logM = logXi_b + logXi_a_minus_V_s_norm + logXi_s_norm - logXi_a - logXi_b_minus_W_s_norm
  
  norm_logMfunc = train_func(norm_logM)
  norm_target = train_func(tf.constant(0.0,dtype=tf.float32))


  ## Calculate the autonormalised quantity
  #logM_diff = logM - norm_logM
  ## WARNING THIS IS NOT LOG M BUT LOGM-LOGMnorm
  #logM = logXi_a_minus_V_s + logXi_s - logXi_b_minus_W_s - logXi_a_minus_V_s_norm - logXi_s_norm + logXi_b_minus_W_s_norm

  #logMfunc = train_func(logM)

  #########################################################################################
  ## Values Derived From Data (The things we are trying to fit the analytic function to) ##
  #########################################################################################
  data_exponents = tf.map_fn(lambda s : tf.subtract(s,tf.constant(1.0)), S)
  data_moments = tf.map_fn(lambda s: tf.pow(features, s), data_exponents)
  E = tf.map_fn( lambda mom : tf.reduce_mean(tf.reduce_prod(mom,axis=1)), data_moments, dtype = tf.float32 ) ## Careful of overflow here
  logE = tf.log(E)
  logEfunc = train_func(logE)

  ###################
  ## Loss function ##
  ###################
  
  ##Caluclate the R^2 coefficient of the set of S datapoints
  mean_log_E = tf.reduce_mean(logEfunc)
  mean_log_M = tf.reduce_mean(logMfunc)
  total_error = tf.reduce_sum(tf.square(tf.subtract(logEfunc,mean_log_E)))
  unexplained_error = tf.reduce_sum(tf.square(tf.subtract(logEfunc, logMfunc)))
  R_squared = tf.subtract(tf.constant(1.0,dtype=tf.float32), tf.divide(unexplained_error, total_error))

  logM_error = tf.reduce_sum(tf.square(tf.subtract(logMfunc,mean_log_M)))
  derived_gradient  = tf.divide(tf.reduce_sum(tf.multiply(logM_error,total_error)),tf.reduce_sum(tf.multiply(logM_error,logM_error)))
  derived_intercept = mean_log_E - tf.multiply(derived_gradient, mean_log_M) 

  ## Define the losses here ##
  intercept_loss = intercept_weight     * tf.losses.mean_squared_error(derived_intercept,tf.constant(0.0,dtype=tf.float32))
  gradient_loss  = gradient_weight      * tf.losses.mean_squared_error(derived_gradient,tf.constant(1.0,dtype=tf.float32))
  R_squared_loss = R_squared_weight     * tf.losses.mean_squared_error(R_squared,tf.constant(1.0,dtype=tf.float32)) 
  model_loss     = model_weight         * tf.losses.mean_squared_error(logMfunc, logEfunc)
  norm_loss      = normalisation_weight * tf.losses.mean_squared_error(norm_logMfunc,norm_target)
  #gamma_loss = integer_gamma_weight*tf.reduce_sum(tf.map_fn(lambda i : tf.abs(tf.abs(i)*(tf.abs(i)-1)),a))

  ## These require a definition of s which are allowed (can this be for s in 1 to 2?)
  #positive_argument_a_V_loss = tf.    tf.max() 
  #positive_argument_b_W_loss = ... 

  ## Define the total loss
  loss = model_loss + norm_loss + R_squared_loss + intercept_loss + gradient_loss
  
  print("Potential issue using one optimizer for different routines")
  optimiser = tf.train.AdamOptimizer()
  #optimiser = tf.train.GradientDescentOptimizer(0.001)

  ## Define training operations for independant sub-losses
  train_op     = optimiser.minimize(loss)
  norm_op      = optimiser.minimize(norm_loss)
  model_op     = optimiser.minimize(model_loss)
  R_squared_op = optimiser.minimize(R_squared_loss)
  gradient_op  = optimiser.minimize(gradient_loss)
  intercept_op = optimiser.minimize(intercept_loss)

  num_best_models = 10
  best_models_list = []
  largest_min_loss = 1e90
  def queue_compare(current_list, contender):
    clip_num = num_best_models
    if(len(current_list)<clip_num):
      current_list.append(contender)
    else:
      ## Seach for the correct place, add the contender in there and pop the end of the list till clip num elements
      index = clip_num
      for i in range(len(current_list)):
        if(contender[0] < current_list[i][0]):
          index = i
          break
      if(i<clip_num): current_list.insert(index,contender)
      while(len(current_list)>clip_num): current_list.pop(-1)
    return sorted(current_list, key = lambda x : x[0])
       
  ## TF graph write for chart viewing
  writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
  
  num_train_exponent_samples = 50
  
  #############
  ## Session ##
  #############
  epoch_min_max_thresholds = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      if(epoch%5==0): s_list = np.ones([num_dim])+np.random.uniform(0.0,1.0,size=[num_train_exponent_samples,num_dim])
      #s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[200,num_dim])
      sess.run(dataset_init_op, feed_dict = {x: data, S: s_list, batch_size: num_rows, drop_prob: 0.5, training_bool : True})
      tot_loss = 0.0
      for _ in range(n_batches):
        _, loss_value, prlogE, prlogM, prnorm_logM, ml, nl, rl, il, grl = sess.run([train_op, loss, logE, logM, norm_logM, model_loss, norm_loss, R_squared_loss, intercept_loss, gradient_loss], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        tot_loss += loss_value
      print("Iter: {}, Loss: {:.6f}".format(epoch, tot_loss / n_batches))
      print("norm_logM = {}".format(prnorm_logM))
      print("model, norm, R^2(loss), intecept, gradient = {},{},{},{},{}".format(ml,nl,rl,il,grl))
      maxl = max(ml,nl,rl,il,grl)
      if(not np.isnan(tot_loss) and tot_loss/n_batches < largest_min_loss):
        b_eta, b_W, b_V, b_a, b_b = sess.run([eta,matrix_W,matrix_V,param_a,param_b])
        best_models_list = queue_compare(best_models_list,[tot_loss/n_batches,[b_eta,b_W,b_V,b_a,b_b]])
        largest_min_loss = max([element[0] for element in best_models_list])
        smallest_min_loss = min([element[0] for element in best_models_list])
        print("BEST VALUE ADDED! Threshold = {} to {}".format(smallest_min_loss,largest_min_loss))
        epoch_min_max_thresholds.append((epoch,smallest_min_loss,largest_min_loss))
      if(np.isnan(tot_loss)):
        print("\n\n RESETTING \n\n")
        sess.run(tf.global_variables_initializer())
        sess.run(eta_assign_op, feed_dict = {eta_input : best_models_list[0][1][0]})
        sess.run(matrix_W_assign_op, feed_dict = {matrix_W_input : best_models_list[0][1][1]})
        sess.run(matrix_V_assign_op, feed_dict = {matrix_V_input : best_models_list[0][1][2]})
        sess.run(param_a_assign_op, feed_dict = {param_a_input : best_models_list[0][1][3]})
        sess.run(param_b_assign_op, feed_dict = {param_b_input : best_models_list[0][1][4]})
      elif(maxl==nl):
        for _ in range(10) : sess.run([norm_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
      elif(maxl==rl):
        for _ in range(10) : sess.run([R_squared_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
      elif(maxl==ml):
        for _ in range(10) : sess.run([model_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
      elif(maxl==il):
        for _ in range(10) : sess.run([intercept_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
      elif(maxl==grl):
        for _ in range(10) : sess.run([gradient_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})

    print("Run finished Writing Files...")
    with open("Output_Mell_{}.txt".format(session_string), "w") as f:
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        logE_obs, logM_obs = sess.run([logE, logM], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for sval in s_list[0]: f.write("{},".format(sval))
        f.write("{},{}\n".format(logE_obs[0],logM_obs[0]))
    with open("Output_Mell_{}_Best_100_Models.txt".format(session_string),"w") as f:
      for i in best_models_list:
        f.write("{}:{}\n".format(i[0],i[1]))
    
    ## Load the best model back in
    model=0
    sess.run(eta_assign_op, feed_dict = {eta_input : best_models_list[model][1][0]})
    sess.run(matrix_W_assign_op, feed_dict = {matrix_W_input : best_models_list[model][1][1]})
    sess.run(matrix_V_assign_op, feed_dict = {matrix_V_input : best_models_list[model][1][2]})
    sess.run(param_a_assign_op, feed_dict = {param_a_input : best_models_list[model][1][3]})
    sess.run(param_b_assign_op, feed_dict = {param_b_input : best_models_list[model][1][4]})
    with open("Output_Best_Mell_{}.txt".format(session_string),"w") as f:
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        logE_obs, logM_obs = sess.run([logE, logM], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for sval in s_list[0]: f.write("{},".format(sval))
        f.write("{},{}\n".format(logE_obs[0],logM_obs[0]))

    ### Make a huge array of predictions using the best 100 models
    with open("Output_Best_Models_Mell_{}.txt".format(session_string),"w") as f:
      for i in range(num_dim): f.write("s{},".format(i))
      f.write("logE,")
      for i in range(num_best_models): f.write("logM{},".format(i))
      f.write("\n")
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        for sval in s_list[0]: f.write("{},".format(sval))
        logE_obs = sess.run(logE, feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        f.write("{},".format(logE_obs[0]))
        for model in range(num_best_models):
          sess.run(eta_assign_op, feed_dict = {eta_input : best_models_list[model][1][0]})
          sess.run(matrix_W_assign_op, feed_dict = {matrix_W_input : best_models_list[model][1][1]})
          sess.run(matrix_V_assign_op, feed_dict = {matrix_V_input : best_models_list[model][1][2]})
          sess.run(param_a_assign_op, feed_dict = {param_a_input : best_models_list[model][1][3]})
          sess.run(param_b_assign_op, feed_dict = {param_b_input : best_models_list[model][1][4]})
          logM_obs = sess.run(logM, feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
          f.write("{},".format(logM_obs[0]))
        f.write("\n")

    ### Print parameters of best model into a file for reading



if __name__ == "__main__":
    main()
