## This script is an efficient tensorflow port of the Mellin learning idea.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import random
import pandas as pd

## These may come in handy
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA

## Don't output tf warnings about possible speedups etc.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()

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
  print("Data appears to have {} columns. Assuming multivariate probability distribution with {} varaibles.".format(num_data_columns,num_data_columns))


  ## Set to floating point data
  data = data_df.astype('float64')
  print(data)
  num_rows = data.shape[0]
  print("There are {} rows".format(num_rows))

  ##########################
  ## Inputs and constants ##
  ##########################
  EPOCHS = 100
  n_batches = 200
  integer_gamma_weight = 0.0
  normalisation_weight = 0.0
  # Create a placeholder to dynamically switch between batch sizes
  
  #batch_size = tf.placeholder(tf.int64)
  #drop_prob = tf.placeholder(tf.float32)
  batch_size = 1
  drop_prob = 0.5

  ## Training Boolean
  training_bool = True
  #training_bool = tf.placeholder(tf.bool)

  ##################
  ## Initializers ##
  ##################
  ## Question do we need a separate initializer for everything?
  weight_initer = tf.truncated_normal_initializer(mean=-1.0, stddev=1.0)
  #exponent_initer = tf.truncated_normal_initializer(mean=2.0, stddev=0.4)
  avec_initer = tf.random_uniform_initializer(minval=-1,maxval=1,dtype=tf.int64)
  eta_initer = tf.truncated_normal_initializer(mean=1.0, stddev=0.1)

  ########################
  ## Input Placeholders ##
  ########################
  ## The data object
  #x = tf.placeholder(tf.float32, shape=[None,num_dim]) 
  x = tf.constant([[1.0,2.0,3.0]])
  print("x = {}".format(x))
  ## The exponents draw
  #S = tf.placeholder(tf.float32, shape=[None,num_dim])
  
  #S = tf.constant([[1.0,1.0,1.0],[0.0,0.0,0.0],[2.0,2.0,2.0]],dtype=tf.float32)
  s = tf.constant([1.0,1.0,1.0],dtype=tf.float32)

  print("S = {}".format(s))



  ############################
  ## Iterators and features ##
  ############################
  #dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=1000).batch(batch_size).repeat()
  #iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes) 
  #features = iter.get_next()
  #dataset_init_op = iter.make_initializer(dataset)
 
  #####################
  ## Model Variables ##
  #####################
  ## The dimension scale matrix
  q = tf.get_variable(name="Q-matrix", dtype=tf.float32, shape=[num_dim, num_dim], initializer=weight_initer)
  print("q_matrix = {}".format(q))
  ## The gamma weight matrix
  w = tf.get_variable(name="W-matrix", dtype=tf.float32, shape=[num_gamma, num_dim], initializer=weight_initer)
  print("w_matrix = {}".format(w))
  ## The gamma weight constant vector
  alpha = tf.get_variable(name="alpha-vector", dtype=tf.float32, shape=[num_gamma], initializer=weight_initer)
  print("alpha_matrix = {}".format(alpha))
  ## The dimension scale constant vector
  beta = tf.get_variable(name="beta-vector", dtype=tf.float32, shape=[num_dim], initializer=weight_initer)
  print("beta_matrix = {}".format(beta))
  ## The gamma power (integer?) vector
  a = tf.get_variable(name="a-vector", dtype=tf.float32, shape=[num_gamma], initializer=avec_initer)
  print("a_matrix = {}".format(a))
  ## The scale parameter vector (one per dimension)
  eta = tf.get_variable(name="eta-vector", dtype=tf.float32, shape=[num_dim], initializer=eta_initer) 
  print("eta_matrix = {}".format(eta))
  ## Make sure these scale parameters are positive? (Because we use the log expectation?)
  ##eta = tf.abs(eta)

  ###########
  ## Model ##
  ###########
  epsilon = tf.tensordot(q,s,axes=[[1],[0]])
  print("epsilon = {}".format(epsilon))
  
  #kappa = tf.map_fn(lambda s : tf.tensordot(w,s,axes=[[1],[0]]) + alpha, S)
  kappa = tf.tensordot(w,s,axes=[[1],[0]]) + alpha
  print("kappa = {}".format(kappa))
   
  ## Work out the analytic continuation for negative Gamma in these varaibles
  ## Get a buffer of the smallest integer n to bring the argument positive
  kappa_floor = tf.floor(kappa)
  print("kappa_floor = {}".format(kappa_floor))
  
  kappa_min = tf.minimum(kappa_floor,tf.constant(0.0))
  print("kappa_min = {}".format(kappa_min))
  kappa_n = tf.abs(kappa_min)
  print("kappa_n = {}".format(kappa_n))
  
  z_n_pairs = (kappa, kappa_n)
  print(z_n_pairs)

  def gen_log_coef(xx,nn):
    print("GEN LOG COEFF: Inputs xx={}, nn={}".format(xx,nn))
    range_vars = tf.range(nn+1)
    print("Range Vars (up to n) = {}".format(range_vars))
    series_vars = tf.map_fn(lambda n_it : (xx+n_it), range_vars, dtype=tf.float32)
    print("Series Vars (n+x) = {}".format(series_vars))
    series_vars = tf.log(series_vars)
    print("Log Series Vars (n+x) = {}".format(series_vars))
    return tf.reduce_sum(series_vars)

  kappa_log_series = tf.map_fn( lambda x : gen_log_coef(x[0],x[1]), z_n_pairs, dtype=tf.float32)
  print("kappa_log_series = {}".format(kappa_log_series)) 
  exit()
  kappa_gamma_args = tf.map_fn( lambda x : x[0]+x[1]+tf.constant(1.0), z_n_pairs, dtype=tf.float32)
  kappa_gamma = tf.lgamma(kappa_gamma) - kappa_log_series

  
  gamma = tf.map_fn(lambda kapp : tf.multiply(a,tf.lgamma(kapp)), kappa)
  sigma = tf.map_fn(lambda eps : tf.multiply(epsilon,tf.log(eta)), epsilon)
  ## Get the estimate of the moment for this selection of exponents
  logM = tf.reduce_sum(gamma) + tf.reduce_sum(sigma)
  Mfunc = logM

  norm_epsilon = tf.tensordot(q,tf.ones([num_dim]),axes=[[1],[0]])
  norm_kappa = tf.tensordot(w,tf.ones([num_dim]),axes=[[1],[0]])
  norm_gamma = tf.multiply(a,tf.lgamma(norm_kappa))
  norm_sigma = tf.multiply(norm_epsilon,tf.log(eta))
  norm_logM = tf.reduce_sum(norm_gamma) + tf.reduce_sum(norm_sigma)
  norm_M = tf.exp(norm_logM)

  ###################################
  ## Values Derived From Iterators ##
  ###################################
  data_exponents = tf.map_fn(lambda s : tf.subtract(s,tf.constant(1.0)), S)
  data_moments = tf.map_fn(lambda s: tf.pow(features, s), data_exponents)
  logE = tf.log(tf.reduce_mean(data_moments))
  Efunc = logE
  #print("Consider activating |logE|")

  ###################
  ## Loss function ##
  ###################
  loss = tf.losses.mean_squared_error(Mfunc, Efunc) + normalisation_weight*tf.losses.mean_squared_error(norm_M,1.0) + integer_gamma_weight*tf.reduce_sum(tf.map_fn(lambda i : tf.abs(tf.abs(i)*(tf.abs(i)-1)),a))
  train_op = tf.train.AdamOptimizer().minimize(loss)

  #############
  ## Session ##
  #############
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      s_list = [[1.0,1.0,1.0]]+np.random.exponential(scale=4.0,size=[1,num_dim])
      sess.run(dataset_init_op, feed_dict = {x: data, S: s_list, batch_size: num_rows, drop_prob: 0.5, training_bool : True})
      tot_loss = 0
      for _ in range(n_batches):
        _, loss_value = sess.run([train_op, loss], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(epoch, tot_loss / n_batches))

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
    with open("Output_Mell.txt", "w") as f:
      for i in range(500):
        s_list = [[1.0,1.0,1.0]]+np.random.exponential(scale=4.0,size=[1,num_dim])
        logE_obs, logM_obs = sess.run([logE, logM], feed_dict={ S:s_list, drop_prob : 0.5, training_bool : True})
        f.write("{}: {} {}\n".format(s_list,logE_obs,logM_obs))

if __name__ == "__main__":
    main()
