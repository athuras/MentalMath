import nengo
from nengo.utils.distributions import Uniform
import numpy as np
from integrator import Integrator  # local variant of nengo.networks.integrator


def new_integrator(label, d, n, seed, tau=0.1, **ens_kwargs):
    '''Builds a 'slow' integrator'''
    return Integrator(recurrent_tau=tau, neurons=LIF(n_neurons=n, tau_rc=0.08),
            radius=1, dimensions=d, seed=seed, label=label,
            max_rates=Uniform(40, 200), **ens_kwargs)

def labelled_integrators(labels, dimensions, n_neurons, seed=42, tau=0.1, **kwargs):
    '''Generates a bunch of integrators, each with the given label'''
    return [new_integrator(label, d=dimensions, n=n_neurons,
                            seed=seed, tau=tau, **kwargs)
            for label in labels]
