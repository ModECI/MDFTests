import os
import sys
import h5py
import numpy as np

from .text_tools import build_script

from modeci_mdf.utils import load_mdf
from modeci_mdf.scheduler import EvaluableGraph

def _generate_scripts_from_json(model_input):
    """
    Parse elements from MDF and use text_tools module to make script
    """
    file_name = model_input.split("/")[-1].split(".")[0]
    file_dir = "/".join(model_input.split("/")[:-1])

    model = load_mdf(model_input)

    scripts = {}



    for graph in model.graphs:
        nodes = graph.nodes
        # Read weights.h5 if exists
        if "weights.h5" in os.listdir(file_dir):
            weight_dict = h5py.File(os.path.join(file_dir, "weights.h5"), 'r')

            # Hack to fix problem with HDF5 parameters
            for node in graph.nodes:
                if node.parameters:
                    for param_key, param_val in node.parameters.items():
                        if param_key in ["weight", "bias"] and type(param_val)==str:
                            # Load and reassign
                            array = weight_dict[param_val][:]
                            np.set_printoptions(threshold=sys.maxsize)
                            node.parameters[param_key] = np.array2string(array, separator=", ")

        evaluable_graph = EvaluableGraph(graph, False)
        #root = evaluable_graph.root_nodes[0]
        enodes = evaluable_graph.enodes
        edges = evaluable_graph.ordered_edges
        try:
            conditions = evaluable_graph.conditions
        except AttributeError:
            conditions = {}

        # Use edges and nodes to construct execution order
        execution_order = []
        for idx, edge in enumerate(edges):
            if idx==0:
                execution_order.append(edge.sender)
            execution_order.append(edge.receiver)

        # Build script
        script = build_script(nodes, execution_order, conditions=conditions)
        scripts[graph.id] = script

    return scripts

def _script_to_model(script):
    """
    Convert script to pytorch object.
    Should find a cleaner way to do this but did not want to use exec
    """
    with open("module.py", mode="w") as f:
        f.write(script)
    import module
    from importlib import reload
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
