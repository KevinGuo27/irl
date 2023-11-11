from jax import numpy as jnp
from jax import jit
from jax import grad, value_and_grad
import timeit
from jax import vmap
from jax import random
import numpy as np
key = random.PRNGKey(2023)
def linear_comb(params, x):
  y = jnp.dot(params, x)
  return y

params = jnp.array([1., -1])
x = jnp.array([0.5, 0.5])

grad_linear_comb = grad(linear_comb)
cal_grad_linear_comb = value_and_grad(linear_comb)
jit_grad_linear_comb = jit(grad_linear_comb)
print(linear_comb(params, x))
print(grad_linear_comb(params, x))
print(cal_grad_linear_comb(params, x))
normal_key, key = random.split(key)
batched_inp = random.normal(normal_key, (50, 2))

@jit
def naive_batched_linear_comb(params, batched_inp):
  return jnp.stack([linear_comb(params, inp) for inp in batched_inp])

@jit
def batched_linear_comb(params, batched_inp):
  return batched_inp @ params

@jit
def vmap_linear_comb(params, batched_inp):
  return vmap(linear_comb, in_axes=[None, 0])(params, batched_inp)