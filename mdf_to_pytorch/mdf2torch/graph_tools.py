import json

def load_json(model_input):

    # Take either .json or json string
    try:
        model_input = open(model_input, 'r').read()
    except (FileNotFoundError, OSError):
        pass
    model_input = json.loads(model_input)

    return model_input


def get_graphs(mdf_dict):
    """
    Return main and nested graphs
    """
    if 'graphs' not in mdf_dict.keys():
        model_name = list(mdf_dict.keys())[0]
        mdf_dict = mdf_dict[model_name]
    graphs = mdf_dict["graphs"]
    return graphs


def get_elements(graph_dict, element, prefix=""):
    """
    Returns specified graph element: nodes, edges, conditions
    """
    # Recursively look for nodes in a given graph
    elements = {}

    if element not in graph_dict:
        return elements

    for element_name, element_dict in graph_dict[element].items():

        # If recursing on subgraph include subgraph name as prefix
        if prefix:
            elements["{}.{}".format(prefix, element_name)] = element_dict
        else:
            elements[element_name] = element_dict

        # Check if node is itself a sub-graph
        element_keys = set(element_dict.keys())
        if "type" in element_keys and element_dict["type"]["generic"] == "graph":
            sub_graph_prefix = element_name
            elements = {**elements, **get_elements(element_dict, element, sub_graph_prefix)}

    return elements


def flatten_edges(nodes, edges):
    """
    If an edge has functions, convert egde to (edgeA, node, edgeB)
    """
    # Get edge dictionary keys in case need to delete while iterating
    edge_names = list(edges.keys())

    for edge_name in edge_names:
        edge_dict = edges[edge_name]
        if "functions" in edge_dict:

            nodes[edge_name] = {
                                'input_ports':{""
                                               "in_{}".format(edge_name):{
                                                                        "shape":nodes[edge_dict["sender"]]["output_\
                                                                            ports"][edge_dict["sender_port"]]["shape"],
                                                                          }
                                                },
                                'functions':edge_dict["functions"],
                                'output_ports':{
                                                "out_{}".format(edge_name):{
                                                                        "shape":nodes[edge_dict["receiver"]]["input_\
                                                                        ports"][edge_dict]["receiver_port"]["shape"]
                                                                            }
                                                }
                                }

            edges[edge_name+"_A"] = {
                                        "sender":edge_dict["sender"],
                                        "receiver":edge_name,
                                        "sender_port": edge_dict["sender_port"],
                                        "receiver_port": "in_"+edge_name
                                    }

            edges[edge_name+"_B"] = {
                                        "sender": edge_name,
                                        "receiver": edge_dict["receiver"],
                                        "sender_port": "out_"+edge_name,
                                        "receiver_port": edge_dict["receiver_port"]
                                    }
            del edges[edge_name]

    return nodes, edges


def make_dependency_graph(edges, conditions=None):
    """
    Return simple dict of node names and which nodes they depend on.

    Some nodes such as WhenFinished can be better represented by position
    in the graph hierarchy. Handle these conditions here as part of the
    graph hierarchy, and other call-based conditions later.
    """
    dependency_graph = {}

    for edge_name, edge_dict in edges.items():

        sender_name = edge_dict["sender"]
        receiver_name = edge_dict["receiver"]

        if receiver_name in dependency_graph:
            dependency_graph[receiver_name].add([sender_name])
        else:
            dependency_graph[receiver_name] = {sender_name}

    if conditions:

        for node_with_condition, condition in conditions.items():

            if condition["type"] == "WhenFinished":
                depends_on = condition["kwargs"]["dependency"]

                if node_with_condition not in dependency_graph:
                    dependency_graph[node_with_condition] = set()
                dependency_graph[node_with_condition].add(depends_on)

            # Handle other hierarchical conditions here...


    return dependency_graph
