import dot_product
import nengo

tester = nengo.Network()
with tester2:
    in1 = nengo.Node(output=lambda t: [np.sin(t), np.sin(2*t), np.sin(t)])
    in2 = nengo.Node(output=lambda t: [np.sin(2*t), np.cos(t), -np.sin(t)])
    Z = dot_product.MapReduceDotProduct(leo_neurons=nengo.LIF(32), leo_dimensions=3,
                                        reducer_neurons=nengo.LIF(100), reducer_radius=1.)

    nengo.Connection(in1, Z.A)
    nengo.Connection(in2, Z.B)
    nengo.Connection(Z.map_phase.output,
            Z.reducer, synapse=0.005, transform=np.ones((1, 3)))

    # Probe
    out = nengo.Probe(Z.output, 'output', synapse=0.02)
    map_out = nengo.Probe(Z.map_phase.output, 'output', synapse=0.02)

sim = nengo.Simulator(tester)
sim.run(3)

import pylab
pylab.plot(sim.trange(), sim.data[out])
pylab.plot(sim.trange(), sim.data[map_out])
pylab.show()
