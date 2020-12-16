import psyneulink as pnl

comp = pnl.Composition(name='comp')

fn = pnl.IntegratorMechanism(name='fn', function=pnl.FitzHughNagumoIntegrator(name='FitzHughNagumoIntegrator Function-0', d_v=1, initial_v=-1, initializer=[[0]], default_variable=[[0]]))

comp.add_node(fn)


comp.scheduler.add_condition(fn, pnl.Always())

comp.scheduler.termination_conds = {pnl.TimeScale.RUN: pnl.Never(), pnl.TimeScale.TRIAL: pnl.AfterNCalls(fn, 2000)}
comp.show_graph()
