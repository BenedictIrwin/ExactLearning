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
  num_gamma = 3
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
  EPOCHS = 5000
  n_batches = 50
  model_weight = 0.0
  integer_gamma_weight = 0.0
  normalisation_weight = 20.0
  R_squared_weight = 1.0
  intercept_weight = 0.0
  gradient_weight = 0.0
  
   
  # Create a placeholder to dynamically switch between batch sizes
  batch_size = tf.placeholder(tf.int64)
  drop_prob = tf.placeholder(tf.float32)

  ## Training Boolean
  training_bool = tf.placeholder(tf.bool)

  ##################
  ## Initializers ##
  ##################
  weight_initialiser_mean = 0.0
  weight_initialiser_std = 0.5

  ## Question do we need a separate initializer for everything?
  #weight_initer = tf.truncated_normal_initializer(mean=weight_initialiser_mean, stddev=weight_initialiser_std)
  weight_initer = tf.constant_initializer(np.eye(num_gamma,num_dim),dtype=tf.float32)
  avec_initer = tf.random_uniform_initializer(minval=0.90,maxval=1.1,dtype=tf.float32)
  eta_initer = tf.truncated_normal_initializer(mean=1.0, stddev=0.5)
  alpha_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)
  beta_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)
  q_initer = tf.constant_initializer(np.eye(num_dim),dtype=tf.float32)



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

  ## Make a log of parameters
  with open(log_file_string, "w") as f:
    f.write("Session {} created {}\n".format(session_string,datetime.datetime.now()))
    f.write("Data of shape {}\n".format(data.shape))
    f.write("Number of dimensions = {}\n".format(num_dim))
    f.write("Number of Gamma functions = {}\n".format(num_gamma))
    f.write("EPOCHS = {}\n".format(EPOCHS))
    f.write("n_batches = {}\n".format(EPOCHS))
    f.write("integer_gamma_weight = {}\n".format(integer_gamma_weight))
    f.write("normalisation_weight = {}\n".format(normalisation_weight))
    f.write("intercept_weight = {}\n".format(intercept_weight))
    f.write("gradient_weight = {}\n".format(gradient_weight))
    f.write("R_squared_weight = {}\n".format(R_squared_weight))
    f.write("ds_buffer_constant = {}\n".format(ds_buffer_constant))
  
  #####################
  ## Model Variables ##
  #####################
  ## The dimension scale matrix
  q = tf.get_variable(name="Q-matrix", dtype=tf.float32, shape=[num_dim, num_dim], initializer=q_initer)
  ## The gamma weight matrix
  w = tf.get_variable(name="W-matrix", dtype=tf.float32, shape=[num_gamma, num_dim], initializer=weight_initer)
  ## The gamma weight constant vector
  alpha = tf.get_variable(name="alpha-vector", dtype=tf.float32, shape=[num_gamma], initializer=alpha_initer)
  ## The dimension scale constant vector
  beta = tf.get_variable(name="beta-vector", dtype=tf.float32, shape=[num_dim], initializer=beta_initer)
  ## The gamma power (integer?) vector
  a = tf.get_variable(name="a-vector", dtype=tf.float32, shape=[num_gamma], initializer=avec_initer)
  ## The scale parameter vector (one per dimension)
  eta = tf.get_variable(name="eta-vector", dtype=tf.float32, shape=[num_dim], initializer=eta_initer) 
  ## Make sure these scale parameters are positive? (Because we use the log expectation?)
  #eta = tf.abs(eta)

  eta_input = tf.placeholder(dtype=tf.float32, shape=[num_dim]) 
  q_input = tf.placeholder(dtype=tf.float32, shape=[num_dim,num_dim]) 
  w_input = tf.placeholder(dtype=tf.float32, shape=[num_gamma,num_dim]) 
  alpha_input = tf.placeholder(dtype=tf.float32, shape=[num_gamma]) 
  beta_input = tf.placeholder(dtype=tf.float32, shape=[num_dim]) 
  a_input = tf.placeholder(dtype=tf.float32, shape=[num_gamma]) 
  print("CHECK: use locking status and effect?")
  eta_assign_op = tf.assign(eta, eta_input, use_locking=False)
  q_assign_op = tf.assign(q, q_input, use_locking=False)
  w_assign_op = tf.assign(w, w_input, use_locking=False)
  alpha_assign_op = tf.assign(alpha, alpha_input, use_locking=False)
  beta_assign_op = tf.assign(beta, beta_input, use_locking=False)
  a_assign_op = tf.assign(a, a_input, use_locking=False)
  #####################
  ## Model Equations ##
  #####################
  epsilon = tf.map_fn(lambda s: tf.tensordot(q,s,axes=[[1],[0]]) + beta, S, dtype=tf.float32)
  kappa = tf.map_fn(lambda s: tf.tensordot(w,s,axes=[[1],[0]]) + alpha, S, dtype=tf.float32)
  
  def train_func(x): return x

  gamma = tf.map_fn(lambda kapp : tf.multiply(a,tf.lgamma(kapp)), kappa)
  sigma = tf.map_fn(lambda eps : tf.multiply(eps,tf.log(eta)), epsilon)
  
  ## Get the estimate of the moment for this selection of exponents
  logM = tf.map_fn(lambda gam : tf.reduce_sum(gam), gamma, dtype=tf.float32) + tf.map_fn(lambda sig : tf.reduce_sum(sig), sigma, dtype = tf.float32)
  logMfunc = train_func(logM)

  #print("logM = {}".format(Mfunc))
  norm_epsilon = tf.tensordot(q,tf.ones([num_dim]),axes=[[1],[0]]) + beta
  norm_kappa = tf.tensordot(w,tf.ones([num_dim]),axes=[[1],[0]]) + alpha
  norm_gamma = tf.multiply(a,tf.lgamma(norm_kappa))
  norm_sigma = tf.multiply(norm_epsilon,tf.log(eta))
  #norm_logM = tf.map_fn(lambda gam : tf.reduce_sum(gam), norm_gamma, dtype=tf.float32) + tf.map_fn(lambda sig : tf.reduce_sum(sig), norm_sigma, dtype=tf.float32)
  norm_logM = tf.reduce_sum(norm_gamma) + tf.reduce_sum(norm_sigma)
  norm_logMfunc = train_func(norm_logM)
  norm_target = train_func(tf.constant(1.0,dtype=tf.float32))

  ##############################
  ## Values Derived From Data ##
  ##############################
  data_exponents = tf.map_fn(lambda s : tf.subtract(s,tf.constant(1.0)), S)
  data_moments = tf.map_fn(lambda s: tf.pow(features, s), data_exponents)
  E = tf.map_fn( lambda mom : tf.reduce_mean(tf.reduce_prod(mom,axis=1)), data_moments, dtype = tf.float32 )
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

  ## Define the losses here
  intercept_loss = intercept_weight*tf.losses.mean_squared_error(derived_intercept,tf.constant(1.0,dtype=tf.float32))
  gradient_loss = gradient_weight*tf.losses.mean_squared_error(derived_gradient,tf.constant(1.0,dtype=tf.float32))
  R_squared_loss = R_squared_weight*tf.losses.mean_squared_error(R_squared,tf.constant(1.0,dtype=tf.float32)) 
  model_loss = model_weight*tf.losses.mean_squared_error(logMfunc, logEfunc)
  norm_loss  = normalisation_weight*tf.losses.mean_squared_error(norm_logMfunc,norm_target)
  gamma_loss = integer_gamma_weight*tf.reduce_sum(tf.map_fn(lambda i : tf.abs(tf.abs(i)*(tf.abs(i)-1)),a))
  
  ## Define the total loss
  loss = model_loss + norm_loss + gamma_loss + R_squared_loss + intercept_loss + gradient_loss
 
  ## Define training operations for independant sub-losses
  train_op = tf.train.AdamOptimizer().minimize(loss)
  norm_op = tf.train.AdamOptimizer().minimize(norm_loss)
  gamma_op = tf.train.AdamOptimizer().minimize(gamma_loss)
  model_op = tf.train.AdamOptimizer().minimize(model_loss)
  R_squared_op = tf.train.AdamOptimizer().minimize(R_squared_loss)
  gradient_op = tf.train.AdamOptimizer().minimize(gradient_loss)
  intercept_op = tf.train.AdamOptimizer().minimize(intercept_loss)

  best_100_list = []
  largest_min_loss = 1e90
  def queue_compare(current_list, contender):
    clip_num = 100
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
       

  writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
  
  num_train_exponent_samples = 5
  
  #############
  ## Session ##
  #############
  epoch_min_max_thresholds = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      #s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[200,num_dim])
      s_list = [[1.0]]+np.random.uniform(1.0,10.0,size=[num_train_exponent_samples,num_dim])
      sess.run(dataset_init_op, feed_dict = {x: data, S: s_list, batch_size: num_rows, drop_prob: 0.5, training_bool : True})
      tot_loss = 0.0
      for _ in range(n_batches):
        _, loss_value, prlogE, prlogM, prnorm_logM, ml, nl, gl, rl, il, grl = sess.run([train_op, loss, logE, logM, norm_logM, model_loss, norm_loss, gamma_loss, R_squared_loss, intercept_loss, gradient_loss], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        tot_loss += loss_value
      print("Iter: {}, Loss: {:.6f}".format(epoch, tot_loss / n_batches))
      print("norm_logM = {}".format(prnorm_logM))
      print("model, norm, gamma, R^2(loss), intecept, gradient = {},{},{},{},{},{}".format(ml,nl,gl,rl,il,grl))
      if(not np.isnan(tot_loss) and tot_loss < largest_min_loss):
        b_eta, b_q, b_w, b_alpha, b_beta, b_a = sess.run([eta,q,w,alpha,beta,a])
        best_100_list = queue_compare(best_100_list,[tot_loss,[b_eta,b_q,b_w,b_alpha,b_beta,b_a]])
        largest_min_loss = max([bb[0] for bb in best_100_list])
        smallest_min_loss = min([bb[0] for bb in best_100_list])
        print("BEST VALUE ADDED! Threshold = {} to {}".format(smallest_min_loss,largest_min_loss))
        epoch_min_max_thresholds.append((epoch,smallest_min_loss,largest_min_loss))
      if(np.isnan(tot_loss)):
        print("\n\n RESETTING... \n\n")
        print("Current eta ={}".format(sess.run(eta)))
        sess.run(tf.global_variables_initializer())
        print("Init eta ={}".format(sess.run(eta)))
        print("Best eta = {}".format(best_100_list[0][1][0]))
        sess.run(eta_assign_op, feed_dict = {eta_input : best_100_list[0][1][0]})
        print("Reset eta ={}".format(sess.run(eta)))
        sess.run(q_assign_op, feed_dict = {q_input : best_100_list[0][1][1]})
        sess.run(w_assign_op, feed_dict = {w_input : best_100_list[0][1][2]})
        sess.run(alpha_assign_op, feed_dict = {alpha_input : best_100_list[0][1][3]})
        sess.run(beta_assign_op, feed_dict = {beta_input : best_100_list[0][1][4]})
        sess.run(a_assign_op, feed_dict = {a_input : best_100_list[0][1][5]})
        #with tf.variable_scope('',reuse=True):
        #  eta = tf.get_variable(name="eta-vector", dtype=tf.float32, shape=[num_dim], initializer=tf.constant_initializer(best_100_list[0][1][0])) 
        #  q = tf.get_variable(name="Q-matrix", dtype=tf.float32, shape=[num_dim, num_dim], initializer=tf.constant_initializer(best_100_list[0][1][1]))
        #  w = tf.get_variable(name="W-matrix", dtype=tf.float32, shape=[num_gamma, num_dim], initializer=tf.constant_initializer(best_100_list[0][1][2]))
        #  alpha = tf.get_variable(name="alpha-vector", dtype=tf.float32, shape=[num_gamma], initializer=tf.constant_initializer(best_100_list[0][1][3]))
        #  beta = tf.get_variable(name="beta-vector", dtype=tf.float32, shape=[num_dim], initializer=tf.constant_initializer(best_100_list[0][1][4]))
        #  a = tf.get_variable(name="a-vector", dtype=tf.float32, shape=[num_gamma], initializer=tf.constant_initializer(best_100_list[0][1][5]))
        print("UHOH! nan detected: Restarting from best seen varaibles...")
      else:
        for _ in range(10) : sess.run([norm_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for _ in range(10) : sess.run([R_squared_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for _ in range(10) : sess.run([gamma_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for _ in range(10) : sess.run([model_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for _ in range(10) : sess.run([intercept_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for _ in range(10) : sess.run([gradient_op], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})

    print("Final Parameters:")
    print("eta = {}".format(sess.run(eta)))
    print("q = {}".format(sess.run(q)))
    print("w = {}".format(sess.run(w)))
    print("alpha = {}".format(sess.run(alpha)))
    print("beta = {}".format(sess.run(beta)))
    print("a = {}".format(sess.run(a)))
    a=sess.run(a)
    print("a ~ {}".format(np.round(a)))
    num_pred_gamma = 0
    for i in np.round(a):
      if(i!=0): num_pred_gamma +=1
    print("Looks like a model with {} gamma functions is best!".format(num_pred_gamma))
    with open("Output_Mell_{}.txt".format(session_string), "w") as f:
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        logE_obs, logM_obs = sess.run([logE, logM], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for sval in s_list[0]: f.write("{},".format(sval))
        f.write("{},{}\n".format(logE_obs[0],logM_obs[0]))
    with open("Output_Mell_{}_Best_100_Models.txt".format(session_string),"w") as f:
      for i in best_100_list:
        f.write("{}:{}\n".format(i[0],i[1]))
    
    ## Load the best model back in
    model=0
    sess.run(eta_assign_op, feed_dict = {eta_input : best_100_list[model][1][0]})
    sess.run(q_assign_op, feed_dict = {q_input : best_100_list[model][1][1]})
    sess.run(w_assign_op, feed_dict = {w_input : best_100_list[model][1][2]})
    sess.run(alpha_assign_op, feed_dict = {alpha_input : best_100_list[model][1][3]})
    sess.run(beta_assign_op, feed_dict = {beta_input : best_100_list[model][1][4]})
    sess.run(a_assign_op, feed_dict = {a_input : best_100_list[model][1][5]})
    #with tf.variable_scope('',reuse=True):
    #  eta = tf.get_variable(name="eta-vector", dtype=tf.float32, shape=[num_dim], initializer=tf.constant_initializer(best_100_list[0][1][0])) 
    #  q = tf.get_variable(name="Q-matrix", dtype=tf.float32, shape=[num_dim, num_dim], initializer=tf.constant_initializer(best_100_list[0][1][1]))
    #  w = tf.get_variable(name="W-matrix", dtype=tf.float32, shape=[num_gamma, num_dim], initializer=tf.constant_initializer(best_100_list[0][1][2]))
    #  alpha = tf.get_variable(name="alpha-vector", dtype=tf.float32, shape=[num_gamma], initializer=tf.constant_initializer(best_100_list[0][1][3]))
    #  beta = tf.get_variable(name="beta-vector", dtype=tf.float32, shape=[num_dim], initializer=tf.constant_initializer(best_100_list[0][1][4]))
    #  a = tf.get_variable(name="a-vector", dtype=tf.float32, shape=[num_gamma], initializer=tf.constant_initializer(best_100_list[0][1][5])) 
    with open("Output_Best_Mell_{}.txt".format(session_string),"w") as f:
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        logE_obs, logM_obs = sess.run([logE, logM], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        for sval in s_list[0]: f.write("{},".format(sval))
        f.write("{},{}\n".format(logE_obs[0],logM_obs[0]))

    ### Make a huge array of predictions using the best 100 models
    with open("Output_Best_100_Mell_{}.txt".format(session_string),"w") as f:
      for i in range(num_dim): f.write("s{},".format(i))
      f.write("logE,")
      for i in range(100): f.write("logM{},".format(i))
      f.write("\n")
      for i in range(500):
        s_list = [[1.0]]+np.random.exponential(scale=2.0,size=[1,num_dim])
        for sval in s_list[0]: f.write("{},".format(sval))
        logE_obs = sess.run(logE, feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        f.write("{},".format(logE_obs[0]))
        for model in range(100):
          sess.run(eta_assign_op, feed_dict = {eta_input : best_100_list[model][1][0]})
          sess.run(q_assign_op, feed_dict = {q_input : best_100_list[model][1][1]})
          sess.run(w_assign_op, feed_dict = {w_input : best_100_list[model][1][2]})
          sess.run(alpha_assign_op, feed_dict = {alpha_input : best_100_list[model][1][3]})
          sess.run(beta_assign_op, feed_dict = {beta_input : best_100_list[model][1][4]})
          sess.run(a_assign_op, feed_dict = {a_input : best_100_list[model][1][5]})
          logM_obs = sess.run(logM, feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
          f.write("{},".format(logM_obs[0]))
        f.write("\n")



if __name__ == "__main__":
    main()
