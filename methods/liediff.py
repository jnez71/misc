#!/usr/bin/env python3
"""
Extending vector automatic differentiation to Lie
groups without manually defining new chain-rules.

"""
from typing import NamedTuple
import jax
from numpy import random, testing

np = jax.numpy
np.set_printoptions(suppress=True)
jax.config.update('jax_enable_x64', True)

################################################################################

tol = 1e-6

def jacobian_numerical(f):
    return lambda x: np.column_stack([(f(x+tol*d) - f(x)) / tol for d in np.identity(3, float)])

def jacobian_automatic(f):
    return jax.jit(lambda x: jax.jacfwd(lambda d: f(x+d) - f(x))(np.zeros(len(x), float)))

################################################################################

class SO3(NamedTuple):
    # Internal representation
    w: float
    x: float
    y: float
    z: float

    # Dimensionality
    def __len__(self):
        return 3

    # Group composition or action
    def __mul__(self, other):
        if type(other) == SO3:
            return SO3(-self[1]*other[1] - self[2]*other[2] - self[3]*other[3] + self[0]*other[0],
                        self[1]*other[0] + self[2]*other[3] - self[3]*other[2] + self[0]*other[1],
                       -self[1]*other[3] + self[2]*other[0] + self[3]*other[1] + self[0]*other[2],
                        self[1]*other[2] - self[2]*other[1] + self[3]*other[0] + self[0]*other[3])
        else:
            return np.array((self * SO3(0.0, *other) * -self)[1:], float)

    # Group inverse
    def __neg__(self):
        return SO3(self.w, -self.x, -self.y, -self.z)

    # "Box-plus"
    def __add__(self, v):
        return self * SO3.exp(v)

    # "Box-minus"
    def __sub__(self, other):
        return SO3.log(-other * self)

    def normalized(self):
        c = np.array([self.w, self.x, self.y, self.z], float)
        c = np.where(self.w >= 0.0, c, -c)
        return SO3(*c/np.linalg.norm(c))

    def matrix(self):
        return np.column_stack((self*b for b in np.identity(3, float)))

    def log(self):
        a2 = 1.0 - self.w**2
        t = a2 < tol
        a = np.sqrt(np.where(t, 1.0, a2))
        s = np.arctan2(np.copysign(a, self.w), np.abs(self.w))
        s = np.where(np.abs(self.w) < tol, np.copysign(np.pi/a, self.w), 2.0*s/a)
        s = np.where(t, 2.0/self.w-2.0/3.0*a2/self.w**3, s)
        return s*np.array([self.x, self.y, self.z], float)
        # a = np.copysign(2.0, self.w) / np.sinc(np.arccos(np.clip(np.abs(self.w), -1.0, 1.0)) / np.pi)
        # return  a*np.array([self.x, self.y, self.z], float)

    @staticmethod
    def exp(v):
        a2 = v.dot(v)
        a4 = a2 * a2
        t = a2 < tol
        a = np.sqrt(np.where(t, 1.0, a2))
        h = a / 2.0
        w = np.where(t, 1.0-a2/8.0+a4/384.0, np.cos(h))
        s = np.where(t, 0.5-a2/48.0+a4/3840.0, np.sin(h)/a)
        return SO3(w, *(s*v)).normalized()
        # a = np.linalg.norm(v) / 2.0
        # return SO3(np.cos(a), *(np.sinc(a/np.pi)/2.0)*v).normalized()

    @staticmethod
    def identity():
        return SO3(w=1.0, x=0.0, y=0.0, z=0.0)

    @staticmethod
    def random():
        return SO3(*random.normal(size=4)).normalized()

################################################################################

q = SO3.random()
v = random.normal(size=3)

testing.assert_allclose((q+v)-q, (-q*q)*v, atol=tol)

################################################################################

def f(q):  # SO3 -> R3
    q0 = SO3.exp(np.array([0, np.pi/2, 0], float))
    return q - q0

testing.assert_allclose(jacobian_automatic(f)(q), jacobian_numerical(f)(q), atol=tol)

################################################################################

def f(q):  # SO3 -> SO3
    v0 = np.array([0, np.pi/2, 0], float)
    return q + v0

testing.assert_allclose(jacobian_automatic(f)(q), jacobian_numerical(f)(q), atol=tol)

################################################################################

print("All asserts passed!")
