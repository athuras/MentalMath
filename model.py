import numpy as np
import nengo
import symbols
import constructs
import nengo.networks as networks
from nengo.utils.distributions import Uniform
import dot_product

# Local version of nengo.networks.integrator.Integrator
from integrator import Integrator

# Simulation Control, Nengo2 is slow on my machine ... really slow
D = 24
n_per_d = 32
N = D * n_per_d

# Base Symbols
ZERO = symbols.initial_zero(D)
model = nengo.Network()
numeral_table = dict(symbols.numerals(D, 4, zero=ZERO))
def hold_for_t(t, value):
    '''yields function that outputs value for time <= t'''
    default = np.zeros_like(value)
    def f(tau):
        return value if tau <= t else default
    return f

def symbolic_decode(numeral_table, x):
    similarities = {k: np.dot(v.v, x.T)
                    for k, v in numeral_table.iteritems()}
    return np.vstack([d for _, d in
                      sorted([(k, v)
                              for k, v in similarities.iteritems()])])

with model:
    A_in = nengo.Node(label='A',
            output=hold_for_t(0.05, 1.3 * numeral_table[1].v),
            size_out=D)
    B_in = nengo.Node(label='B',
            output=hold_for_t(0.05, 1.3 * numeral_table[2].v),
            size_out=D)
    # Held Constant
    Fun = nengo.Node(label='Fun', output=ZERO.v, size_out=D)

    # Integrators, in general, Fun, and Ref would be as well,
    # but it involves growing the model too much, KISS etc.
    A, B = [Integrator(recurrent_tau=0.1,
                       neurons=nengo.LIF(n_neurons=N),
                       dimensions=D,
                       radius=1.,
                       max_rates=Uniform(40, 200))
            for i in xrange(2)]


    # Function Application
    NG1 = networks.CircularConvolution(nengo.LIF(N),
                                       dimensions=D)
    NG2 = networks.CircularConvolution(nengo.LIF(N),
                                       dimensions=D,
                                       invert_a=True)

    # Connect the stuff
    nengo.Connection(A.output, NG1.B, synapse=0.01)
    nengo.Connection(B.output, NG2.B, synapse=0.01)
    # simulate coming from integrator
    nengo.Connection(Fun, NG1.A, synapse=0.01)
    nengo.Connection(Fun, NG2.A, synapse=0.01)
    nengo.Connection(A_in, A.input, synapse=None)
    nengo.Connection(B_in, B.input, synapse=None)

    Similarity = dot_product.MapReduceDotProduct(
                             leo_neurons=nengo.LIF(n_per_d),
                             leo_dimensions=D,
                             reducer_neurons=nengo.LIF(2 * n_per_d),
                             reducer_radius=1.2)
    # Now we compare 'Ref' to A_prime out of NG1.
    nengo.Connection(NG1.output, Similarity.A, synapse=0.002)

    # The utilities for updating the integrators are based on 1 - similarity
    # The return utility is simply the value of similarity.
    # The idea is once we've become similar to the 'zero' symbol, we're done.
    Q = nengo.Ensemble(neurons=nengo.LIF(n_per_d * 2),
                       dimensions=2,
                       radius=1)
    def utility(sim):
        '''if similarity is high, utility associated
        with looping should be low'''
        x =  sim[0]
        z = np.abs(1 - x)
        return np.array([z, 1 - z])
    _intermediary = nengo.Ensemble(neurons=nengo.LIF(n_per_d * 2),
                                   dimensions=2, radius=1)
    nengo.Connection(Similarity.output, _intermediary,
                     transform=np.ones((2, 1)), synapse=None)
    # Necessary because Pre is a nengo.Node, and therefore can't
    # be involved in function computing ...
    nengo.Connection(_intermediary, Q, function=utility, synapse=0.002)


    # Now for the Basal Ganglia, which will suppress the input to the integrators
    # This only works because the integrators represent positive values (I think)
    BG = networks.BasalGanglia(dimensions=2, output_weight=-3)
    nengo.Connection(Q, BG.input, synapse=0.008)

    # Generate the 'Gates', these are controlled by the 'iterate' action
    A_gate, B_gate = [dot_product.ThalamusGate(signal, inhibit=BG.output,
                                      inhibit_transform=np.array([[1., 0]]),
                                      mapper_neurons=nengo.LIF(n_per_d),
                                      control_neurons=nengo.LIF(n_per_d))
                      for signal in (NG1.output, NG2.output)
                      ]

    nengo.Connection(A_gate.output, A.input, synapse=0.008)
    nengo.Connection(B_gate.output, B.input, synapse=0.008)

    # Ignore the 'return state', too many neurons already ...

    # System Output
    Output = nengo.Node(label="Return", size_in=D, size_out=D)
    nengo.Connection(B.output, Output)
