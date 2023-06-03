from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.nn import relu
from scipy.optimize import linear_sum_assignment
from weight_matching import PermutationSpec, mlp_permutation_spec, weight_matching, apply_permutation

from utils import rngmix

epsilon = 1e-12

# apply the permutations from perm to the input weights
# axis indicates which side to apply the permutation to...
# except_axis = 0 returns W_l P_l
# except_axis = 1 returns P_{l-1}^{-1} W_l
def get_permuted_param(ps: PermutationSpec, perm, weight_label, weights, except_axis = None):
  """Get parameter `k` from `params`, with the permutations applied."""
  
  w = weights
  layer_list = ps.axes_to_perm[weight_label] # list of two layers the weights go across
  # choose either the first one or the second one, depending on except_axis
  # if applying it to axis 0, use previous layer's permutation, apply to left side
  # if applying to axis 1, use current layer's permutation, apply to right side
  # if axis_to_apply is None, then apply to both sides
  
  # print("layer_list: ", layer_list, "weight_label: ", weight_label, ", weight shape: ", w.shape)
  
  for axis, current_layer_label in enumerate(layer_list): #e.g. if weights go between P_1 and P_2, then (0, P_1), (1, P2)
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue
  
    # print("axis: ", axis, ", current layer label: ", current_layer_label)
  
    if current_layer_label is not None:
        w = jnp.take(w, perm[current_layer_label], axis=axis)
        # if len(w.shape) == 1: # weights are actually biases #TODO: how does sam get around this?
        #     if axis == 0: #only allow one-sided permutation (only occurs when applying permutation)
        #       w = jnp.take(w, perm[current_layer_label])
        # else:
        #     if axis == 0:
        #         w = jnp.take(w.T, perm[current_layer_label], axis=1).T
        #     else:
        #         w = jnp.take(w, perm[current_layer_label], axis=1)
  return w


# 2nd and 3rd parameters are the permutation and the model parameters
def apply_permutation_new(ps: PermutationSpec, final_permutation, model_params):
  """Apply a `perm` to `params`."""
  # return {key: get_permuted_param(ps, final_permutation, key, model_params) for key in model_params.keys()}
  return {weight_label: get_permuted_param(ps, final_permutation, weight_label, weights) for weight_label, weights in model_params.items()}
 
def find_norms(ps: PermutationSpec, params_a, params_b):
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
  # dictionary of permutations, key = layer label, value = permutation for the layer
  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
  
  a_norms = defaultdict(lambda : None) # dict with keys being the weight_labels, value being a vector indexed by neurons
  b_norms = defaultdict(lambda : None)
  
  for layer_label in perm.keys(): #normalize at every layer at every neuron
    n = perm_sizes[layer_label]
    a_norm = jnp.zeros(n)
    b_norm = jnp.zeros(n)
    # for each layer_label e.g. P_0, find the weights and biases. what's the easiest way?
    # ps.perm_to_axes[P_0] gives (weight_0, _), (bias_0, _), (weight_1, _)
    for weight_label, axis in ps.perm_to_axes[layer_label]: 
      w_a = params_a[weight_label]
      w_b = params_b[weight_label]
      # print("layer label: ", layer_label, "weight label: ", weight_label, "axis: ", axis, "weight shape: ", w_a.shape)
      if axis == 1 or (len(w_a.shape) == 1): # hacky way to get weight_0 and bias_0
        w_a = w_a.reshape((-1, n)) #if the weight is the bias vector, turn into 1xn matrix
        w_b = w_b.reshape((-1, n))
        a_norm += np.linalg.norm(w_a, axis=0)**2 #compute norm for each column, output is a vector of column norms
        b_norm += np.linalg.norm(w_b, axis=0)**2
        
    for weight_label, axis in ps.perm_to_axes[layer_label]:
      w_a = params_a[weight_label] #need to get the dimension of the weights
      if (axis == 1 or len(w_a.shape) == 1): #second part of the condition is extremely hacky way to get biases
        a_norms[weight_label] = np.sqrt(a_norm)
        b_norms[weight_label] = np.sqrt(b_norm)
      # else:
      #   a_norms[weight_label] = jnp.ones(n) #default no scaling (only relevant for final layer stuff)
      #   b_norms[weight_label] = jnp.ones(n) #default no scaling  covered by defaultdict

  # for weight_labels, norm in a_norms.items():
  #   print(weight_labels, norm.shape)
    
  return a_norms, b_norms

def weight_matching_new(rng,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    use_scales=False,
                    init_perm=None,
                    silent=False):
  """Find a permutation of `params_b` to make them match `params_a`."""
  
  # this is a horrible naming goddamn. 
  # key are weight types, values are the 2 layers it's involved in, earlier layer first
  # print(ps.axes_to_perm)
  # ps.perm_to_axes is a dict of layer labels mapping to a list of weight labels.. 
  # (each layer has weights, biases, and weights to next layer)
  # print(ps.perm_to_axes)
  
  # dictionary of sizes, key = "P0", "P1", "P2", value = size of layer (also size of permutation)
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
  
  # dictionary of permutations, key = layer label, value = permutation for the layer
  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  # list of layer labels
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
      layer_label = perm_names[p_ix] #layer label, i.e. P0, P1, P2
      n = perm_sizes[layer_label] #number of neurons in the layer
      A = jnp.zeros((n, n))
      for weight_label, axis in ps.perm_to_axes[layer_label]:
        w_a = params_a[weight_label]
        w_b = params_b[weight_label]
        
        #need this so we take outer product of the biases later
        w_a = w_a.reshape((w_a.shape[0],-1)) #only makes a difference when it's the biases
        w_b = w_b.reshape((w_b.shape[0],-1)) #in case it is the biases, turn into column vector as 2darray
        w_b = get_permuted_param(ps, perm, weight_label, w_b, except_axis=axis)
        
        # for the biases, this conveniently works out, because
        # we reshape the weights as a column vector first
        if axis == 0:
          A += w_b @ w_a.T
        else: 
          A += w_b.T @ w_a

      # ri is just the list of row indices, sorted
      ri, ci = linear_sum_assignment(A.T, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()

      # print("old perm: ", perm[layer_label])
      # print("new perm: ", ci)
      oldL = jnp.vdot(A.T, jnp.eye(n)[perm[layer_label]])
      newL = jnp.vdot(A.T, jnp.eye(n)[ci, :])
      if not silent: print(f"new {iteration}/{layer_label}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12
      
      perm[layer_label] = jnp.array(ci)
    if not progress:
      break

  return perm

def test_weight_matching(seed=123):
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=1)
  rng = random.PRNGKey(seed)
  num_hidden = 7
  shapes = {
      "Dense_0/kernel": (2, num_hidden),
      "Dense_0/bias": (num_hidden, ),
      "Dense_1/kernel": (num_hidden, 3),
      "Dense_1/bias": (3, )
  }
  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
  sim_orig = sim(params_a, params_b)
  
  perm, scale = weight_matching(rng, ps, params_a, params_b)
  clever_b = apply_permutation(ps, perm, params_b)
  sim_old = sim(params_a, clever_b)
  
  a_norms, b_norms = find_norms(ps, params_a, params_b)
  #params_a = normalize(params_a, a_norms)
  #params_b = normalize(params_b, b_norms)
  
  perm_alt = weight_matching_new(rng, ps, params_a, params_b)
  clever_b_alt = apply_permutation_new(ps, perm_alt, params_b)
  sim_new = sim(params_a, clever_b_alt)
  
  print("orig score: ", sim_orig, "old sim score: ", sim_old, "new sim score: ", sim_new)
  if (sim_old > sim_new):
    return 1
  return 0

def normalize(params, norms):
  normed_params = {}
  for weight_label, w in params.items():
    norm = norms[weight_label]
    if norm is not None:
      if len(w.shape) == 1: #weights are actually biases, indexed by neurons
        w = w / norm
      else: #weights are weights, divide by norms column-wise (columns are indexed by neurons at layer)
        w = w / norm[np.newaxis,:]
    normed_params[weight_label] = w
  return normed_params
  
def sim(params_a, params_b):
  sim = 0
  for weight_label, a_weights in params_a.items():
    b_weights = params_b[weight_label]
    sim += np.sum(a_weights * b_weights)
  return sim

if __name__ == "__main__":
  
  test_once = True
  
  if test_once:
    test_weight_matching()
  
  else:
    old_wins_count = 0
    for i in range(10):
      old_wins_count += test_weight_matching(i)
      
    print(old_wins_count)
