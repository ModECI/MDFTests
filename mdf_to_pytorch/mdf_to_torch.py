from toposort import toposort

from graph_tools import load_json, get_graphs, get_nodes, get_edges, make_dependency_graph
from text_tools import build_script

"""
First Prototype Restrictions:
    * One root node per graph, one output node per graph
    * More complicated UDFs are slotted in via an environment-specific implementation
    * One input port / one output port per node
    * No controllers
    * No conditions / conditional branching
"""

def embed_in_module_text(name, text):
    module_text = """class {}(torch.nn.Module):\n    def __init__(self):\n        super({}, self).__init__()""".format(name, name)
    module_text += "\n\n    def forward(self):\n        return {}".format(text)
    return module_text

def _generate_scripts_from_json(model_input):

    # Load mdf json into dict
    mdf_dict = load_json(model_input)

    # Get all top-level (non-nested) graphs
    graphs = get_graphs(mdf_dict)

    scripts = {}

    for graph_name, graph_dict in graphs.items():

        # Get nodes, including those nested in subgraphs
        nodes = get_nodes(graph_dict)

        # Get edges
        edges = get_edges(graph_dict)

        # TODO: Flatten functional edges
        # TODO: Flatten functional input/output

        # Create a dependency graph representing nodes & children relationships
        dependency_graph = make_dependency_graph(edges)

        # Get the top bottom of the dependency graph for top-level model call
        ordered_dependency_graph = list(toposort(dependency_graph))
        top_call_node = next(iter(ordered_dependency_graph[-1]))

        # Build script
        script = build_script(nodes, top_call_node, dependency_graph)
        #scripts[graph_name] = script
        print(script)

    return scripts

def load(model_input, model_eval=True):
    """
    Load and return all models specified in an MDF graph
    """
    scripts = _generate_scripts_from_json(model_input)

    models = {}

    for script_name, script in scripts.items():

        exec(script)
        model = locals()[script_name]

        if model_eval:
            model.eval()

        models[script_name] = model

    return models

if __name__=="__main__":

    name = "linear_test"
    address = "{}.json".format(name)
    models = _generate_scripts_from_json(address), model_eval=True)

    model = models["linear_test"]
    
    # Create sample data and feed forward
    data = torch.randn(3)
    output = model(data)
