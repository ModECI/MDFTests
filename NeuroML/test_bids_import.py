import psyneulink as pnl

net_ids = ['ABCD']

for net_id in net_ids:
    conv_json_filename = '%s.bids-mdf.json'%net_id
    with open(conv_json_filename, 'r') as infile:
        json = infile.read()

    #print(json)
    py_filename = '%s.bids-mdf.py'%net_id
    with open(py_filename, 'w') as outfi:
        outfi.write(pnl.generate_script_from_json(json))
        
        run_plot = '''
{0}.run(inputs={1}, log=True, num_trials=50)
        
print('Finished running model')
        
print({0}.results)
for node in {0}.nodes:
    print(f'{{node}} {{node.name}}: {{node.parameters.value.get({0})}}')
    
{0}.show_graph()
        
print('Done!')
        '''.format(net_id,'{A_input_0: 0}')
        
        outfi.write(run_plot)
        
    print('Written JSON file: %s and python to load it: %s'%(conv_json_filename, py_filename))
    
