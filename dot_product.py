from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.network import with_self
from nengo.utils.distributions import Uniform
import nengo
import numpy as np

def corners(n):
    c = np.array([[1., 1], [-1., 1], [1, -1], [-1, -1]])
    return np.tile(c, (n // 4, 1))

class BinaryElementwiseOperation(nengo.Network):
    '''Handles splitting high-dimensions stuff into ensemble arrays
    which each handle one 'dimension' or 'element'.
    the x.output object is of the same dimensions as the inputs'''
    def __init__(self, neurons, dimensions, kernel=None,
            input_transform=None):
        self.kernel = kernel if kernel is not None else self.product
        self.dimensions = dimensions
        self.A = nengo.Node(size_in=dimensions, label='A')
        self.B = nengo.Node(size_in=dimensions, label='B')
        self.ensemble = EnsembleArray(neurons,
                n_ensembles=dimensions,
                dimensions=2,
                radius=1,
                label='elementwise_operation')
        self.output = nengo.Node(size_in=dimensions, label='output')

        if input_transform is None:
            self.input_transform = self.elementwise_comparator_transform
        else:
            self.input_transform = input_transform


        for ens in self.ensemble.ensembles:
            if not isinstance(neurons, nengo.Direct):
                ens.encoders = corners(ens.n_neurons)
        nengo.Connection(self.A, self.ensemble.input, synapse=None,
                transform=self.input_transform)
        nengo.Connection(self.B, self.ensemble.input, synapse=None,
                transform=self.input_transform)
        nengo.Connection(self.ensemble.add_output('result', self.kernel),
                self.output)

    @property
    def elementwise_comparator_transform(self):
        I = np.eye(self.dimensions)
        return np.vstack((I, I))

    @staticmethod
    def product(x):
        return x[0] * x[1]



class MapReduceDotProduct(nengo.Network):
    '''A combination of LinearElementwiseOperation (with product),
    and a reducer phase that simply adds everything up'''
    def __init__(self, leo_neurons, leo_dimensions, reducer_neurons, reducer_radius):
        self.map_phase = BinaryElementwiseOperation(neurons=leo_neurons,
                                                dimensions=leo_dimensions)
        self.input_dimensions = leo_dimensions
        self.output_dimensions = 1
        self.reducer = nengo.Ensemble(neurons=reducer_neurons, dimensions=1,
                radius=reducer_radius)
                #encoders=np.ones((reducer_neurons.n_neurons, 1)))

        # Even though the high-dimensional result is available
        # from LinearElementwiseOperation.output, we'll go directly
        # to the ensembles for free addition!
        nengo.Connection(self.map_phase.output, self.reducer,
                synapse=0.002, transform=np.ones((1, leo_dimensions)))

        self.A = self.map_phase.A
        self.B = self.map_phase.B
        self.output = nengo.Node(size_in=1, label='output')
        nengo.Connection(self.reducer, self.output)

class ThalamusGate(nengo.Network):
    '''elementwise multiplication of high-dimensional signal A with scalar
    inhibitory signal :inhibit'''
    def __init__(self, A, inhibit, inhibit_transform, mapper_neurons, control_neurons):
        self.main_input = A
        self.map_phase = BinaryElementwiseOperation(neurons=mapper_neurons,
                                                    dimensions=A.size_out)
        self.control = nengo.Ensemble(neurons=control_neurons, dimensions=1,
                                      radius=1.,
                                      encoders=np.ones((control_neurons.n_neurons, 1)),
                                      max_rates=Uniform(8, 128))
        nengo.Connection(inhibit, self.control, transform=inhibit_transform, synapse=0.006)
        nengo.Connection(self.main_input, self.map_phase.A, synapse=0.002)
        nengo.Connection(self.control, self.map_phase.B, synapse=0.002,
                transform=np.ones((A.size_in, 1)))

        self.output = self.map_phase.output
