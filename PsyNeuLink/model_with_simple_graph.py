import psyneulink as pnl

comp = pnl.Composition(name='comp')

A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.TransferMechanism(function=pnl.Exponential, name='C')

for m in [A, B, C]:
    comp.add_node(m)

comp.add_projection(pnl.MappingProjection(), A, B)
comp.add_projection(pnl.MappingProjection(), A, C)

comp.run(inputs={A: 0}, log=True, num_trials=1)

print('Finished running model')

print(comp.results)
for node in comp.nodes:
    print(f'{node} {node.name}: {node.parameters.value.get(comp)}')


with open('model_with_simple_graph.json', 'w') as outfi:
    outfi.write(comp.json_summary)

with open('model_with_simple_graph.converted.py', 'w') as outfi:
    outfi.write(pnl.generate_script_from_json(comp.json_summary))
    outfi.write('\ncomp.show_graph()')
    
comp.show_graph()