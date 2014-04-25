from pointer import SemanticPointer
from nengo.utils.distributions import UniformHypersphere
import numpy as np

def rand_sphere(d):
    '''Draw one random vector from a unit hypershpere of dimension n'''
    sphere = UniformHypersphere(d)
    return np.ravel(sphere.sample(1))

def initial_zero(N):
    '''Randomly generate a symbol for ZERO'''
    zero = SemanticPointer(N=N, data=rand_sphere(N))
    return zero

def gen_numerals(zero):
    '''Successively convolve ZERO with itself'''
    i = 0
    state = zero
    while True:
        yield (i, state)
        state = state.convolve(zero)
        state.normalize()
        i += 1

def similarity(x, y):
    return x.compare(y)

def numerals(D, cap, zero=None):
    if zero is None: zero = initial_zero(D)
    _gen_numerals = gen_numerals(zero)
    return [next(_gen_numerals) for i in xrange(cap)]

def batch_similarity(symbol, nums):
    return [(i, similarity(x, symbol)) for i, x in nums]
