import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
from onnx.defs import get_schema

from ast import literal_eval

#function_map = {
#    'linear': 'Linear',
#    'sin': 'Sin',
#    'exponential': 'Exp'
#}

def get_function_name(name):
    prefix = 'onnx::'
    if name.startswith(prefix):
        return name[len(prefix):]
    raise NotImplementedError('Function not defined in ONNX')

def generate_onnx_graph(graph, sorted_nodes):
    print('Generating ONNX graph for ', graph.id)
    onnx_inputs = []
    onnx_outputs = []
    onnx_nodes = []
    onnx_initializer = []
    for node_name in sorted_nodes:
        node = graph.nodes[node_name]
        in_edges = [edge for edge in graph.edges.values() if edge.receiver == node_name]
        out_edges = [edge for edge in graph.edges.values() if edge.sender == node_name]

        # get function name
        # assumes node has only 1 function
        assert len(node.functions) == 1
        func = next(iter(node.functions.values()))
        func_name = get_function_name(func.function)
        schema = get_schema(func_name)
        input_names = [func.args[inp.name] for inp in schema.inputs]

        # get list of input values
        name_actual_map = {}
        for edge in in_edges:
            name_actual_map[edge.receiver_port] = edge.sender + '_' + edge.sender_port

        for param, value in node.parameters.items():
            name = node.id + '_' + param
            constant = helper.make_tensor(name, data_type=TensorProto.FLOAT, dims=[], vals=[value])
            onnx_initializer.append(constant)
            name_actual_map[param] = name

        inputs = [name_actual_map[name] if name in name_actual_map else name for name in input_names]

        parameters = {}

        outputs = [node.id + '_' + port for port in node.output_ports]

        onnx_node = helper.make_node(func_name,
                                inputs,
                                outputs,
                                name=node.id,
                                **parameters)

        onnx_nodes.append(onnx_node)

        # Find and add graph input ports
        defined_input_ports = [e.receiver_port for e in in_edges]
        graph_input_ports = [p for p in node.input_ports.values() if p.id not in defined_input_ports]
        if graph_input_ports:
            for input_port in graph_input_ports:
                shape = literal_eval(input_port.shape)
                value_info = helper.make_tensor_value_info(input_port.id,
                                                           TensorProto.FLOAT,
                                                           shape)
                onnx_inputs.append(value_info)

        # Find and add graph output ports
        used_output_ports = [e.sender_port for e in out_edges]
        graph_output_ports = [p for p in node.output_ports.values() if p.id not in used_output_ports]
        if graph_output_ports:
            for output_port in graph_output_ports:
                # output shapes are determined by ONNX type inference
                value_info = helper.make_empty_tensor_value_info(node.id + '_' + output_port.id)
                onnx_outputs.append(value_info)

    # Make the final graph
    graph = helper.make_graph(
        onnx_nodes,
        graph.id,
        onnx_inputs,
        onnx_outputs,
        onnx_initializer
    )

    return graph

def generate_onnx_model(model, sorted_nodes):
    onnx_models = []
    for graph, nodes in zip(model.graphs.values(), sorted_nodes):
        onnx_graph = generate_onnx_graph(graph, nodes)
        onnx_model = helper.make_model(onnx_graph)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(onnx_model)
        onnx_models.append(onnx_model)
    return onnx_models

def write_onnx_models(models, savepath, name):
    for model in models:
        file_name = savepath + name + '-m2o.onnx'
        onnx.save(model, file_name)
        print('ONNX output saved in ',file_name)
