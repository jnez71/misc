#!/usr/bin/env python3
"""
On the Differentiability of the Solution to Convex Optimization Problems
https://arxiv.org/abs/1804.05098

x(q) := argmin_x[f(x,q)]
g(x,q) := Df_x(x,q)
-> g(x(q),q) = 0.0
   -> Dg_x*Dx_q + Dg_q = 0.0
      -> Dx_q = -Dg_x^-1 * Dg_q

"""
import jax
from scipy.optimize import minimize

np = jax.numpy
np.set_printoptions(suppress=True)
jax.config.update('jax_enable_x64', True)

################################################################################

# f = lambda x, q: (x-4)**2 + q*x
# x_q = lambda q: 4.0 - q/2.0  # dx_q = -1/2

f = lambda x, q: (q*x**2)/100+1/2*x**2*np.sin(x)**2-1/2*x**2*np.cos(x)**2-(102*x)/25-4*x*np.sin(0.02*q*x)**2+(17*np.sin(x)**2)/2+2*x*np.sin(x)-8*np.sin(x)+4*x*np.cos(x)**2-(17*np.cos(x)**2)/2+483/50
x_q = lambda q: minimize(f, 4.6, q, 'Nelder-Mead').x.item()

################################################################################

g = jax.grad(f, argnums=0)
dg_x = jax.jacfwd(g, argnums=0)
dg_q = jax.jacfwd(g, argnums=1)

dx_q = lambda q: -dg_q(x_q(q), q) / dg_x(x_q(q), q)
aprx = lambda q, d=0.01: (x_q(q+d) - x_q(q)) / d

################################################################################

q = 51.0
print("   q:", q)
print(" x_q:", x_q(q))
print("dx_q:", dx_q(q))
print("aprx:", aprx(q))

################################################################################
