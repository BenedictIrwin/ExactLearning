import numpy as np
string_list = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")

def get_random_key():
  """
  Generate a random 10 letter key for encoding functions
  """
  return "".join([np.random.choice(string_list) for i in range(10)])

## A helper function to assign pk parameter labels to constant spaces and vice versa
def parse_terms(terms):
  terms = terms.replace("_s_","")
  term_array = np.array(list(terms))
  underscores = np.argwhere( term_array == "_")
  undercount = len(underscores)
  if(undercount%2 != 0):
    print("Error! : Bad underscore count in {}".format(terms))
    exit()
  underscores = np.reshape(underscores,(undercount//2,2))
  strings = [terms[a:b+1] for a,b in underscores]
  param_type_dict = { strings[k] : "p{}".format(k) for k in range(len(strings)) }
  return strings , ["p{}".format(k) for k in range(undercount//2)]


# Constants
twopi = 2*np.pi
twopi_rec = 1/twopi
pi_rec = 1/np.pi

def wr(x): 
  """
  A function to map a result to the principle branch
  """
  return x - np.sign(x)*np.ceil(np.abs(twopi_rec*x)-0.5)*twopi
wrap = np.vectorize(wr)


## TODO: What is this? Add Descriptions
def deal(p0, states):
  """
  TODO:
  """
  p0 = list(p0)
  itr = 0
  mx = len(states)
  vec = []
  while(itr<mx):
    if(states[itr]):vec.append(0)
    else: vec.append(p0.pop(0))
    itr+=1
  return np.array(vec)
