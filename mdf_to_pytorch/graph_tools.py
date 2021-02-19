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
    if 'graphs' not in mdf_dict.keys():
        model_name = list(mdf_dict.keys())[0]
        mdf_dict = mdf_dict[model_name]
    graphs = mdf_dict["graphs"]
    return graphs

# TODO: get_nodes and get_edges are same, should be refactored with
#   node / edge identifier adn generic recursive function

def get_nodes(graph_dict, prefix=""):

    # Recursively look for nodes in a given graph
    nodes = {}

    for node_name, node_dict in graph_dict["nodes"].items():

        # If recursing on subgraph node include subgraph name as prefix
        if prefix:
            nodes["{}.{}".format(prefix, node_name)] = node_dict
        else:
            nodes[node_name] = node_dict

        # Check if node is itself a sub-graph
        node_keys = set(node_dict.keys())
        if "type" in node_keys and node_dict["type"]["generic"] == "graph":
            sub_graph_prefix = node_name
            nodes = {**nodes, **get_nodes(node_dict, sub_graph_prefix)}

    return nodes


def get_edges(graph_dict, prefix=""):

    # Recursively look for nodes in a given graph
    edges = {}

    for edge_name, edge_dict in graph_dict["edges"].items():

        # If recursing on subgraph node include subgraph name as prefix
        if prefix:
            edges["{}.{}".format(prefix, edge_name)] = edge_dict
        else:
            edges[edge_name] = edge_dict

        # Check if node is itself a sub-graph
        edge_keys = set(edge_dict.keys())
        if "type" in edge_keys and edge_dict["type"]["generic"] == "graph":
            sub_graph_prefix = edge_name
            edges = {**edges, **get_edges(edge_dict, sub_graph_prefix)}

    return edges

def make_dependency_graph(edges):

    dependency_graph = {}

    for edge_name, edge_dict in edges.items():

        sender_name = edge_dict["sender"]
        receiver_name = edge_dict["receiver"]

        if receiver_name in dependency_graph:
            dependency_graph[receiver_name].add(sender_name)
        else:
            dependency_graph[receiver_name] = set([sender_name])

    return dependency_graph


def _get_declared_identifiers(graph_list):
    """
    graph_list: [dict]: list of mdf graphs
    return: {name_of_graph_or_subgraph : graph_or_subgraph}
    """
    names = {}

    # Iterate on all graph names including nested graphs
    for graph_dict in graph_list:

        # is graph a nested dict or are we in the dict?
        if "nodes" not in graph_dict:
            # We aren't in dict, name is the key to the graph dict
            name = list(graph_dict.keys())[0]

            # Go into graph dict level before proceeding
            graph_dict = graph_dict[name]

        names[name] = graph_dict

        for node in graph_dict["nodes"]:

            node_values = list(graph_dict["nodes"][node])

            for node_value in node_values:
                if type(node_value) == dict and 'type' in node_value.keys():
                    if node_value['type']['generic'] == 'graph':
                        names = {**names, **_get_declared_identifiers([node])}

    return names