import os
from importlib import reload
from toposort import toposort

from .text_tools import build_script
from .graph_tools import load_json, get_graphs, get_elements, make_dependency_graph, flatten_edges


def _generate_scripts_from_json(model_input):
    """
    Parse elements from MDF and use text_tools module to make script
    """

    # Load mdf json into dict
    mdf_dict, weights = load_json(model_input)

    # Get all top-level (non-nested) graphs
    graphs = get_graphs(mdf_dict)
    scripts = {}

    for graph_name, graph_dict in graphs.items():

        # Get nodes, including those nested in subgraphs
        nodes = get_elements(graph_dict, "nodes")

        # Get edges
        edges = get_elements(graph_dict, "edges")

        # Check if any of the edges contain functions.
        # If so, break out as node and augment edges
        nodes, edges = flatten_edges(nodes, edges)

        # Get conditions, for now only consider node-specific
        conditions = get_elements(graph_dict, "conditions")
        if conditions:
            conditions = conditions["node_specific"]

        # Construct simple dependency graph. If there are no conditions,
        # this will solely specify a model. If conditions, augment graph.
        dependency_graph = make_dependency_graph(edges, conditions=conditions)

        # Get the top bottom of the dependency graph for top-level model call
        ordered_dependency_graph = list(toposort(dependency_graph))

        # Set top level nodes to depend on input
        dependency_graph[next(iter(ordered_dependency_graph[0]))] = {"input"}

        # Build script
        script = build_script(nodes, dependency_graph, ordered_dependency_graph, conditions=conditions, weights=weights)
        scripts[graph_name] = script

    return scripts

def _script_to_model(script):
    """
    Convert script to pytorch object.

    Should find a cleaner way to do this but did not want to use exec
    """
    with open("module.py", mode="w") as f:
        f.write(script)
    import module
    reload(module)
    my_model = module.model
    os.remove("module.py")
    return my_model

def load(model_input, eval_models=True):
    """
    Load and return all models specified in an MDF graph
    """
    scripts = _generate_scripts_from_json(model_input)
    models = {}

    for script_name, script in scripts.items():
        model = _script_to_model(script)

        if eval_models:
            model.eval()

        models[script_name] = model

    return models

__all__ = ["load"]
