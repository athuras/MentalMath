import nengo
from nengo import Connection, LIF, Ensemble
from nengo.networks import CircularConvolution as CC
from collections import namedtuple
from nengo.utils.distributions import Uniform, UniformHypersphere
import numpy as np
from integrator import Integrator  # local variant of nengo.networks.integrator




#  HELPERS   ##################################################################
def rand_hypersphere(d, n):
    '''Return n d-dimensional vectors on the surface of a unit-hypersphere'''
    sphere = UniformHypersphere(d)
    return sphere.sample(n)


def new_positive_integrator(label, d, n, seed, tau=0.02):
    '''Builds an integrator that has positive encoders, and intercepts
    between (0, 1)'''
    enc = np.abs(rand_hypersphere(d, n))
    I = Integrator(recurrent_tau=tau,
                    neurons=LIF(n_neurons=n),
                    radius=1.5,
                    dimensions=d,
                    seed=seed,
                    label=label,
                    intercepts=Uniform(0, 1),
                    max_rates=Uniform(40, 200),
                    encoders=enc)
    return I

def new_integrator(label, d, n, seed, tau=0.1):
    return Integrator(recurrent_tau=tau, neurons=LIF(n_neurons=n),
            radius=1.5, dimensions=d, seed=seed, label=label)

def labelled_integrators(labels, dimensions, n_neurons, seed=42, tau=0.1, positive=False):
    '''Generates a bunch of integrators, each with the given label'''
    f = new_positive_integrator if positive else new_integrator
    return [f(label, d=dimensions, n=n_neurons, seed=seed, tau=tau)
            for label in labels]
