import torch.nn as nn
from collections import defaultdict
from inspect import getmembers, isclass, signature

from . import auxiliary_functions as af
from .mapping import function_map, argument_map
from .mdf2torch_errors import TorchNameError, TorchArgumentError

def generate_constructor_call(function, params):
    """
    Generate the constructor for a given torch module
    """
    # Map the name of the function to torch.nn or another lib if specified
    function_name = list(function.keys())[0]  # Only handle one at a time
    function_dict = function[function_name]

    aux_func_names = set([item[0] for item in getmembers(af, isclass)])

    mdf_name = function_dict["function"].lower()

    if mdf_name in function_map:
        torch_module = True
        torch_function_name = function_map[mdf_name]
        torch_function = getattr(nn, torch_function_name)

    # Check if the name is instead in auxiliary functions
    elif mdf_name in aux_func_names:
        torch_module = False
        torch_function_name = mdf_name
        torch_function = getattr(nn, torch_function_name)

    else:
        raise TorchNameError(function_name)

    # Using a proper function name, find its constructor arguments
    constructor_args = signature(torch_function.__init__).parameters

    args = []
    kwargs = []

    satisfied_args = {}

    for argname, argspec in constructor_args.items():
        if argname=='self':
            pass
        else:
            if "=" in str(argspec):
                default = str(argspec).split("=")[-1]
                kwargs.append(argname)
                satisfied_args[argname] = default
            else:
                args.append(argname)

    # These will be the possible inputs to the constructor
    potential_args = {}
    if "args" in function_dict:
        potential_args = function_dict["args"]
    potential_args = {**potential_args, **params}

    # Match arguments to parameters we are given in the mdf
    # Unless the mdf is created with pytorch as a primary target,
    # it is likely we need to use a map between potential mdf arg
    # names and the arg name the torch constructor looks for
    for arg in [*args, *kwargs]:
        # Direct match
        if arg in potential_args:
            satisfied_args[arg] = potential_args[arg]

        # Otherwise use map
        else:
            accepted_arg_names = argument_map[torch_function_name][arg]

            found_match = False

            for accepted_name in accepted_arg_names:
                if accepted_name in potential_args:
                    satisfied_args[arg] = potential_args[accepted_name]
                    found_match = True

            if not found_match and arg not in kwargs:
                raise TorchArgumentError(torch_function_name)

    # complete the constructor call
    argstring = ""
    for arg, val in satisfied_args.items():
        argstring+="{}={},".format(arg, val)
    argstring = argstring[:-1]

    if torch_module:
        call = "{}.{}({})".format("nn",torch_function_name, argstring)
    else:
        call = "{}.{}({})".format("af", torch_function_name, argstring)
    return call


def get_module_declaration_text(name, node_dict, dependency_graph):
    """
    Create text representing
    """
    declaration_text = ("\nclass {}(nn.Module):"
                        "\n\tdef __init__(self):"
                        "\n\t\tsuper().__init__()"
                        "\n\t\tself.calls = 0"
                        ).format(name, name)

    # Build parameter dictionary that can be used to make constructor call
    params = {}
    if "parameters" in node_dict.keys():
        params = node_dict["parameters"]


    for key in ["input_ports", "output_ports"]:
        shapes = []
        if key in node_dict.keys():
            port = list(node_dict[key].keys())[0]
            params[key] = node_dict[key][port]["shape"]

    # Single function of multi-function
    functions = node_dict["functions"]

    if len(functions) == 1:

        constructor_call = generate_constructor_call(functions[0], params)
        text = "\n\t\tself.function = {}".format(constructor_call)

        forward_call, forward_signature = generate_module_forward_call(name, dependency_graph)
        text += forward_call

    else:
        text = "\n\t\tfunction_list = []"
        for function in functions:
            constructor_call = generate_constructor_call(function, params)
            text += "\n\t\tfunction_list.append()".format(constructor_call)
        text+="\n\t\tself.function = nn.Sequential(*function_list)"

        forward_call, forward_signature = generate_module_forward_call(name, dependency_graph)
        text += forward_call

    declaration_text += text

    return declaration_text, forward_signature

def generate_module_forward_call(name, dependency_graph):
    """
    Make the string representing forward call in torch.nn Module
    """
    # Constrain to a single output for now, determine # inputs
    depends_on = dependency_graph[name]

    argstring = ""
    for dependency in depends_on:
        argstring += "from_{},".format(dependency)
    argstring = argstring[:-1]

    forward_call = ("\n\tdef forward(self, {}):"
                    "\n\t\tself.calls+=1"
                    "\n\t\treturn self.function({})"
                    ).format(argstring, argstring)

    module_signature = argstring

    return forward_call, module_signature


def generate_condition_text(node_name, condition, simple_call="pass", indent="", logic_only=False):
    """
    Create text representing given conditions. Can be recursively called for composite components.
    """
    if condition["type"] == "EveryNCalls":
        """
        use modulo operator to check calls remainder
        """
        params = condition["kwargs"]
        dependency = params["dependency"]
        calls = params["calls"]

        if not logic_only:
            call = (
                "\n{}if self.{}.calls%{} == 0:"
                "\n{}\t{}"
                "\n{}else:"
                "\n{}\treturn None"
            ).format(indent, dependency, calls,
                     indent, simple_call,
                     indent,
                     indent)
        else:
            call = "(self.{}.calls%{} == 0)".format(dependency, calls)

    elif condition["type"] == "Threshold":
        """
        access parameter of function nested in class
        """
        parameter = condition["kwargs"]["parameter"]
        threshold = condition["kwargs"]["threshold"]
        direction = condition["kwargs"]["direction"]

        if not logic_only:
            call = (
                    "\n{}if {}.function.{}{}{}:"
                    "\n{}\t"
                    "\n{}else:"
                    "\n{}\treturn False"
                    ).format(indent, node_name, parameter, direction, threshold,
                             indent,
                             indent,
                             indent)
        else:
            call = "({}.function.{}{}{})".format(node_name, parameter, direction, threshold)

    # Composite conditions
    elif condition["type"] in ["and", "any", "all", "or"]:

        # Get list of sub_condition logical operators
        sub_conditions = []

        for sub_condition in condition["args"]:
            sub_condition_logic = generate_condition_text(node_name, sub_condition,
                                                          simple_call="", indent="",
                                                          logic_only=True)
            sub_conditions.append(sub_condition_logic)

        if condition["type"] in ["and", "all"]:
            logic_string = "and".join(sub_conditions)

        elif condition["type"] in ["or", "any"]:
            logic_string = "or".join(sub_conditions)

        if not logic_only:
            call = (
                "\n{}if {}:"
                "\n{}\t{}"
                "\n{}else:"
                "\n{}\treturn None"
            ).format(indent, logic_string,
                     indent, simple_call,
                     indent,
                     indent)
        else:
            call = "({})".format(logic_string)

    else:
        call = "\n\t\t{}".format(simple_call)

    return call


def generate_main_forward(ordered_dependency_graph, module_signatures, conditions=None):
    """
    Use graph hierarchy and conditions to specify forward function for main Model call
    """
    # Iterate through the ordered dependency graph, and place text elements in script
    # Add conditions if necessary

    # Index intermediate variables
    var_idx = 0

    # Map intermediate variable name to the module that produced it
    return_vars = defaultdict(list)

    main_forward = "\n\tdef forward(self, input):"

    # Define in order specified by toposort
    for node_set in ordered_dependency_graph:

        for node in node_set:

            # Create simple call, which absent of conditions is whole call
            simple_call = "var_{} = self.{}()".format(var_idx, node)
            return_vars[node].append("var_{}".format(var_idx))

            # Insert arguments into simple call in proper order
            args_call_depends_on = module_signatures[node]

            if "," not in args_call_depends_on:
                args_call_depends_on = [args_call_depends_on]
            else:
                args_call_depends_on = args_call_depends_on.split(",")

            argstring = ""
            for arg in args_call_depends_on:
                node = arg.split("from_")[-1]
                if node in return_vars:
                    argstring += "{}={},".format(arg, return_vars[node][0])
                elif "input" in node:
                    argstring+="{},".format(node)
            if argstring.endswith(","):
                argstring = argstring[:-1]

            simple_call = simple_call.split(")")[0] + argstring + ")"

            # Check if any node-specific conditions apply to the node
            if node in conditions:

                condition = conditions[node]
                call = generate_condition_text(node, condition, simple_call, indent="\t\t")

            else:
                call = "\n\t\t{}".format(simple_call)

            var_idx += 1
            main_forward += call


    main_forward+="\n\t\treturn var_{}".format(var_idx-1)
    main_forward+="\nmodel = Model()"
    return main_forward


def build_script(nodes, dependency_graph, ordered_dependency_graph, conditions=None):
    """
    Create and assemble following components for a complete script:

        * Module declarations
            * Initialization of functions
            * Definition of forward function

        * Model main call declaration:
            * Initialization of subcomponents
            * Forward function logic
    """
    script = ""
    imports_string = "import torch\nimport torch.nn as nn"

    # Declarations string
    modules_declaration_text = ""
    module_signatures = {}
    for node_name, node_dict in nodes.items():
        declaration_text, module_signature = get_module_declaration_text(node_name, node_dict, dependency_graph)
        modules_declaration_text += declaration_text
        module_signatures[node_name] = module_signature


    # Build Main call
    main_call_declaration = ("\nclass Model(nn.Module):"
                            "\n\tdef __init__(self):"
                            "\n\t\tsuper().__init__()")

    for class_declaration in ["\n\t\tself.{} = {}()".format(n,n) for n in nodes]:
        main_call_declaration += class_declaration

    # Build Main forward
    main_call_forward = generate_main_forward(ordered_dependency_graph, module_signatures, conditions=conditions)

    # Compose script
    script += imports_string
    script += modules_declaration_text
    script += main_call_declaration
    script += main_call_forward
    return script