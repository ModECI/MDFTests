import psyneulink as pnl

comp = pnl.Composition(name='comp')
# A = pnl.IntegratorMechanism(name='A', function=pnl.SimpleIntegrator(rate=2.))
# B = pnl.IntegratorMechanism(name='B', function=pnl.SimpleIntegrator(rate=2.))
# C = pnl.IntegratorMechanism(name='C', function=pnl.SimpleIntegrator(rate=2.))
# D = pnl.IntegratorMechanism(name='D', function=pnl.SimpleIntegrator(rate=2.))
A = pnl.TransferMechanism(name='A', function=pnl.Linear(slope=2.))
B = pnl.TransferMechanism(name='B', function=pnl.Linear(slope=2.))
C = pnl.TransferMechanism(name='C', function=pnl.Linear(slope=2.))
D = pnl.TransferMechanism(name='D', function=pnl.Linear(slope=2.))

comp.add_linear_processing_pathway([A, B, C])
comp.add_linear_processing_pathway([A, B, D])

comp.scheduler.add_condition_set({
    A: pnl.AtNCalls(A, 0),
    B: pnl.Always(),
    C: pnl.EveryNCalls(B, 5),
    D: pnl.EveryNCalls(B, 10),
})

comp.run(inputs={A: 1})

print(comp.results)

# A, B, B, B, B, B, C, B, B, B, B, B, {C, D}
print(*comp.scheduler.execution_list[comp.default_execution_id], sep="\n")

with open(__file__.replace('.py', '.json'), 'w') as f:
    f.write(comp.json_summary)

#comp.show_graph()