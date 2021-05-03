import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

from ast import literal_eval

function_map = {
    'linear': 'Linear',
    'sin': 'Sin',
    'exponential': 'Exp'
}

def get_function_name(name):
    if name in function_map:
        return function_map[name]
    else:
        return name

def generate_onnx_graph(graph, sorted_nodes):
    print('Generating ONNX graph for ', graph.id)

    def get_inedges(node_name):
        return [edge for edge in graph.edges.values() if edge.receiver == node_name]

    def get_outedges(node_name):
        return [edge for edge in graph.edges.values() if edge.sender == node_name]

    def get_graph_outputs(node, out_edges):
        used_output_ports = [e.sender_port for e in out_edges]
        graph_outputs = [p for p in node.output_ports.values() if p.id not in used_output_ports]
        return graph_outputs

    def get_graph_inputs(node, in_edges):
        defined_input_ports = [e.receiver_port for e in in_edges]
        graph_inputs = [p for p in node.input_ports.values() if p.id not in defined_input_ports]
        return graph_inputs

    def create_input_value_info(node, input_ports):
        onnx_value_infos = []
        for output_port in node.output_ports.values():
            value_info = helper.make_tensor_value_info(node.id + '_' + output_port.id,
                                                       TensorProto.FLOAT,
                                                       (1,)) # TODO get shape
            onnx_value_infos.append(value_info)
        return onnx_value_infos

    def create_onnx_node(node, in_edges):
        # get function name
        # assumes node has only 1 function
        assert len(node.functions) == 1
        func = next(iter(node.functions.values()))
        func_name = get_function_name(func.function)

        # get list of input values
        src_name_for_port = {}
        for edge in in_edges:
            src_name_for_port[edge.receiver_port] = edge.sender + '_' + edge.sender_port
        inputs = [src_name_for_port[port] if port in src_name_for_port else port for port in node.input_ports]
        outputs = [node.id + '_' + port for port in node.output_ports]

        parameters = {}

        return helper.make_node(func_name,
                                inputs,
                                outputs,
                                name=node.id,
                                **parameters)

    def create_input_value_info(node, input_ports):
        onnx_value_infos = []
        for input_port in input_ports:
            shape = literal_eval(input_port.shape)
            value_info = helper.make_tensor_value_info(input_port.id,
                                                       TensorProto.FLOAT,
                                                       shape)
            onnx_value_infos.append(value_info)
        return onnx_value_infos

    def create_output_value_info(node, output_ports):
        onnx_value_infos = []
        for output_port in output_ports:
            value_info = helper.make_tensor_value_info(node.id + '_' + output_port.id,
                                                       TensorProto.FLOAT,
                                                       (1,))  # TODO get shape
            onnx_value_infos.append(value_info)
        return onnx_value_infos

    onnx_inputs = []
    onnx_outputs = []
    onnx_nodes = []
    for node_name in sorted_nodes:
        node = graph.nodes[node_name]
        in_edges = get_inedges(node_name)
        out_edges = get_outedges(node_name)
        # if not in_edges:
        #     onnx_inputs.extend(create_input_value_info(node))
        # else:
        #     print("Creating node: " + node_name)
        #     onnx_nodes.append(create_onnx_node(node, in_edges))
        onnx_nodes.append(create_onnx_node(node, in_edges))
        graph_input_ports = get_graph_inputs(node, in_edges)
        if graph_input_ports:
            onnx_inputs.extend(create_input_value_info(node, graph_input_ports))
        graph_output_ports = get_graph_outputs(node, out_edges)
        if graph_output_ports:
            onnx_outputs.extend(create_output_value_info(node, graph_output_ports))

    # print(onnx_inputs)
    # print(onnx_nodes)
    # print(onnx_outputs)

    graph = helper.make_graph(
        onnx_nodes,
        graph.id,
        onnx_inputs,
        onnx_outputs
    )

    return graph

def generate_onnx_model(model, sorted_nodes):
    onnx_models = []
    for graph, nodes in zip(model.graphs.values(), sorted_nodes):
        onnx_graph = generate_onnx_graph(graph, nodes)
        onnx_model = helper.make_model(onnx_graph)
        onnx.checker.check_model(onnx_model)
        onnx_models.append(onnx_model)
    return onnx_models

def write_onnx_models(models, savepath, name):
    for model in models:
        file_name = savepath + name + '-m2o.onnx'
        onnx.save(model, file_name)
        print('ONNX output saved in ',file_name)
