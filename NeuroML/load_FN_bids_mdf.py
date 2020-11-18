import FN
import psyneulink as pnl

FN = pnl.Composition(name='FN')

FNpop_0 = pnl.IntegratorMechanism(name='FNpop_0', function=pnl.FitzHughNagumoIntegrator(name='Function_FitzHughNagumoIntegrator', d_v=1, initial_v=-1))

FN.add_node(FNpop_0)
