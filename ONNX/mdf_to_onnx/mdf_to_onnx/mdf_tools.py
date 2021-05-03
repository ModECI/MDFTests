from .mdf import Model, Node, Edge, InputPort, OutputPort, Function
from toposort import toposort_flatten
from collections import OrderedDict


def load_mdf_json(filename):
    '''
    Load an MDF JSON file
    '''

    from neuromllite.utils import load_json, _parse_element

    data = load_json(filename)

    print("Loaded graph from %s"%filename)

    model = Model()
    model = _parse_element(data, model)
    convert_to_ordered_dict(model)
    # model = Base.to_dict_format(model)
    return model


def convert_to_ordered_dict(model):
    from neuromllite.BaseTypes import Base

    def convert_list(items):
        return OrderedDict([(item.id, item) for item in items])

    def convert(obj):
        if not isinstance(obj, Base):
            return obj
        for allowed_child in obj.allowed_children:
            if allowed_child in obj.children:
                values = [convert(v) for v in obj.children[allowed_child]]
                obj.children[allowed_child] = convert_list(values)
        return obj

    convert(model)


def flatten(model):
    '''
    MAYBE also extract subgraph nodes and edges into the main graph?
    If an edge has functions, it converts edge to edge0 - Node - edge1
    If a node has multiple functions, it converts into Node0 - edge0 - Node1 ..
    '''

    for graph in model.graphs:
        pass

def fill_output_shapes(model):
    '''
    Add shape information for output ports
    '''
    for graph in model.graphs.values():
        for edge in graph.edges.values():
            receiver_port = graph.nodes[edge.receiver].input_ports[edge.receiver_port]
            sender_port = graph.nodes[edge.sender].output_ports[edge.sender_port]
            sender_port.shape = receiver_port.shape

def build_dependency_graph(graph):
    '''
    Build a dependency graph from an mdf graph
    '''

    dependency_graph = {}
    for edge in graph.edges.values():
        if edge.receiver not in dependency_graph:
            dependency_graph[edge.receiver] = set()
        dependency_graph[edge.receiver].add(edge.sender)

    # TODO add condition WhenFinished once conditions are included in mdf

    return dependency_graph


def get_sorted_nodes(model):
    sorted_nodes = []
    for graph in model.graphs.values():
        dependency_graph = build_dependency_graph(graph)
        #sorted = list(toposort(dependency_graph))
        sorted = toposort_flatten(dependency_graph)
        sorted_nodes.append(sorted)

    return sorted_nodes
