from pointer import SemanticPointer
from nengo.utils.distributions import UniformHypersphere
import numpy as np

def rand_sphere(d):
    '''Draw one random vector from a unit hypershpere of dimension n'''
    sphere = UniformHypersphere(d)
    return np.ravel(sphere.sample(1))

def initial_values(N):
    '''Randomly generate symbols for ZERO and f'''
    zero = SemanticPointer(N=N, data=np.abs(rand_sphere(N)))
    f = SemanticPointer(N=N, data=np.abs(rand_sphere(N)))
    zero.normalize()
    f.normalize()
    return zero, f

def gen_numerals(zero, f):
    '''Successively convolve f with the state symbol'''
    i = 0
    state = zero
    while True:
        yield (i, state)
        state = state.convolve(f)
        state.normalize()
        i += 1

def similarity(x, y):
    return x.compare(y)

def numerals(D, cap, zero=None, f=None):
    if zero is None or f is None: zero, f = initial_values(D)
    _gen_numerals = gen_numerals(zero, f)
    return [next(_gen_numerals) for i in xrange(cap)]

def batch_similarity(symbol, nums):
    return [(i, similarity(x, symbol)) for i, x in nums]




