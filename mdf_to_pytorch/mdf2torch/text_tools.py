import torch
import torch.nn as nn
import numpy as np
import sys
from collections import defaultdict
from inspect import getmembers, signature, getsource

from .function import udfs as udf
from .function import mod_torch_builtins as torch_builtins
from .function.alias import nn_module_map, nn_module_argument_map
from .mdf2torch_errors import TorchNameError, TorchArgumentError

def generate_constructor_call(function_info, params):
    """
    Generate the constructor for a given torch module
    """

    # Map the name of the function to torch.nn or another lib if specified


    function_name, function_dict = function_info
    function_type = function_dict["function"]

    # Check if function name maps to a nn.Module module
    if function_type in nn_module_map:
        nn_module = True
        torch_function_name = nn_module_map[function_type]
        torch_function = getattr(nn, torch_function_name)

    # Check if the name is instead in udfs
    elif function_type in udf.__all__:
        nn_module = False
        torch_function_name = function_name
        torch_function = getattr(udf, torch_function_name)

    elif function_type in torch_builtins.__all__:
        nn_module = False
        torch_function_name = function_name
        torch_function = getattr(torch_builtins, torch_function_name)

    else:
        raise TorchNameError(function_name)

    # Using a proper function name, find its constructor arguments
    constructor_args = signature(torch_function.__init__).parameters

    ca = {}

    for arg in constructor_args:
        if str(arg) not in ['self', "args", "kwargs"]:
            typ = str(constructor_args[arg]).split(":")[-1]
            if "=" in typ:
                typ = typ.split("=")[0]
            ca[arg] = (typ, constructor_args[arg])


    args = []
    kwargs = []

    satisfied_args = {}

    for argname, argspec in ca.items():

        typ, arg_desc = argspec

        if "=" in str(arg_desc):
            default = str(arg_desc).split("=")[-1]
            kwargs.append((typ,argname))
            satisfied_args[argname] = default
        else:
            args.append((typ,argname))

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

        arg_type, arg_name = arg

        # Direct match
        if arg_name in potential_args:
            if type(potential_args[arg_name])=="<class '{}'>".format(arg_type):
                satisfied_args[arg_name] = potential_args[arg_name]

        # Otherwise use map
        else:
            accepted_arg_names = nn_module_argument_map[torch_function_name][arg_name]

            found_match = False

            for accepted_name in accepted_arg_names:
                if accepted_name in potential_args:
                    satisfied_args[arg_name] = potential_args[accepted_name]
                    found_match = True

            if not found_match and arg not in kwargs:
                raise TorchArgumentError(torch_function_name)

    # complete the constructor call
    argstring = ""
    for arg, val in satisfied_args.items():
        argstring+="{}={},".format(arg, val)
    argstring = argstring[:-1]

    if nn_module:
        call = "{}.{}({})".format("nn",torch_function_name, argstring)

    else:
        call = "{}({})".format(torch_function_name, argstring)
    return call, torch_function

def generate_initializer_call(func_class, params, idx=False):

    settable_params = get_instance_params(func_class)

    text = ""

    for param in settable_params:
        if param in params:

            param_text = "nn.Parameter(torch.Tensor({}))".format(params[param])

            if not idx:
                text += "\n\t\tself.function.{} = {}".format(param, param_text)
            else:
                text += "\n\t\tself.function_list[-1].{} = {}".format(param, param_text)

    return text

def get_instance_params(funcname):

    # Want to make a dummy instance to introspect
    sig = signature(funcname)
    args = []
    for param in sig.parameters.values():
        if "=" not in str(param):
            arg_type = param.annotation
            if arg_type == int:
                args.append(1)
            elif arg_type == list:
                args.append([1])
            elif arg_type == torch.Tensor:
                args.append(torch.Tensor([1]))
    dummy = funcname(*args)

    params = []

    for member in getmembers(dummy):
        name, member_type = member
        if type(member_type) == nn.parameter.Parameter:
            params.append(name)
    del dummy

    return params

def get_module_declaration_text(name, node_dict, dependency_graph, declared_module_types=None):
    """
    Create script specifying classes with forward methods that will be
    used in the main forward call.

    Two circumstances will arise here:
        1. We must use one or more nn modules, in which case we construct
           the declaration component-wise.
        2. We must use a torch builtin or user specified function, perhaps
           from another python library, in which case the class definition
           is slotted in wholesale.
    """

    # TODO: Some repeated logic, could be trimmed

    # Determine if making a custom module, or inserting text
    functions = node_dict["functions"]

    declaration_text = ("\nclass {}(nn.Module):"
                        "\n\tdef __init__(self):"
                        "\n\t\tsuper().__init__()"
                        "\n\t\tself.calls = 0"
                        ).format(name)

    # Strictly mdf parameters dict
    parameters = {}
    if "parameters" in node_dict.keys():
        parameters = node_dict["parameters"]

    # Shape parameters since some torch modules used them as constructor args
    constructor_parameters = {}
    for key in ["input_ports", "output_ports"]:
        if key in node_dict.keys():
            port = list(node_dict[key].keys())[0]
            constructor_parameters[key] = node_dict[key][port]["shape"]


    # Single function node
    if len(functions) == 1:

        function_name = list(functions.keys())[0]
        function_type = functions[function_name]["function"]

        # Place in existing definition
        if function_type in udf.__all__ or function_type in torch_builtins.__all__:
            if function_type in udf.__all__:
                function_object = getattr(udf, function_type)
            else:
                function_object = getattr(torch_builtins, function_type)

            # Grab source code and prepend to text
            declaration_text = "\n" + getsource(function_object) + declaration_text
            if declared_module_types:
                declared_module_types.add(function_type)
            else:
                declared_module_types = {function_type}
            declaration_text += "\n\t\tself.function = {}()".format(function_type)

            # Add forward call to declaration
            forward_call, forward_signature = generate_module_forward_call(name, dependency_graph)
            declaration_text += "\n{}".format(forward_call)


        # Build module
        else:
            constructor_call, func_class = generate_constructor_call((function_name, functions[function_name]), constructor_parameters)
            declaration_text += "\n\t\tself.function = {}".format(constructor_call)

            initializer_call = generate_initializer_call(func_class, parameters, idx=False)
            declaration_text += "\n{}".format(initializer_call)

            forward_call, forward_signature = generate_module_forward_call(name, dependency_graph)
            declaration_text += "\n{}".format(forward_call)

    # Multi function node
    else:
        declaration_text += "\n\t\tself.function_list = []"
        for function in functions:
            function_name = next(iter(function.keys()))
            function_type = function[function_name]["function"]

            # Function is predefined
            if (function_type in udf.__all__ or function_type in torch_builtins.__all__):

                if declared_module_types and function_type not in declared_module_types:

                    if function_type in udf.__all__:
                        function_object = getattr(udf, function_type)
                    else:
                        function_object = getattr(torch_builtins, function_type)

                    declaration_text = getsource(function_object) + declaration_text

                    if declared_module_types:
                        declared_module_types.add(function_type)
                    else:
                        declared_module_types = {function_type}

                declaration_text += "\n\t\tself.function_list.append({}())".format(function_type)

            else:
                constructor_call, func_class = generate_constructor_call(function, constructor_parameters)
                declaration_text += "\n\t\tself.function_list.append({})".format(constructor_call)

                initializer_call = generate_initializer_call(func_class, parameters, idx=True)
                declaration_text += "\n{}".format(initializer_call)

        declaration_text += "\n\t\tself.function = nn.Sequential(*self.function_list)"
        forward_call, forward_signature = generate_module_forward_call(name, dependency_graph)
        declaration_text+="\n{}".format(forward_call)

    return declaration_text, forward_signature, declared_module_types

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

def build_script(nodes, dependency_graph, ordered_dependency_graph, conditions=None, weights=None):
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
    imports_string = ("import torch"
                      "\nimport torch.nn as nn")

    # Declarations string
    modules_declaration_text = ""
    module_signatures = {}
    declared_module_types = None
    for node_name, node_dict in nodes.items():

        # Check here if we have a parameter represented by a larger weight matrix and if so, expand
        if "parameters" in node_dict:
            kv_pairs = [(k,v) for k,v in node_dict["parameters"].items()]

            for k,v in kv_pairs:
                if v.startswith("weights"):
                    # Load and set
                    np.set_printoptions(threshold=sys.maxsize)
                    node_dict["parameters"][k] = np.array2string(weights[v], separator=", ")


        declaration_text, module_signature, declared_module_types = \
                                        get_module_declaration_text(node_name,
                                                                    node_dict,
                                                                    dependency_graph,
                                                                    declared_module_types=declared_module_types)
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
