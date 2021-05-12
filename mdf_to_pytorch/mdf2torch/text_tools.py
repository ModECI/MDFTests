import torch
import torch.nn as nn
import numpy as np
import sys
from collections import defaultdict
from inspect import getmembers, signature, getsource, isclass

from .function import mod_torch_builtins as torch_builtins
from .function.alias import alias
from .function.function_tools import generate_aux_text

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


def get_module_declaration_text(name, node_dict, execution_order, declared_module_types):

    declaration_text = ("\nclass {}(nn.Module):"
                        "\n\tdef __init__(self):"
                        "\n\t\tsuper().__init__()"
                        "\n\t\tself.calls = 0"
                        ).format(name)

    functions = node_dict["functions"]
    parameters = node_dict["parameters"]

    # Single function node
    if len(functions) == 1:

        current_function = functions[0]
        function_name = current_function.id
        function_type = current_function.function
        function_args = current_function.args

        function_type_alias = alias(str(function_type).lower())
        if function_type_alias is not None:
            function_type = function_type_alias

        # For torch builtins implemented as modules
        if function_type in torch_builtins.__all__:
            function_object = getattr(torch_builtins, function_type)

            # Grab source code and prepend to text
            if function_type not in declared_module_types:
                declaration_text = "\n" + getsource(function_object) + declaration_text
                declared_module_types.add(function_type)
            declaration_text += "\n\t\tself.function = {}()".format(function_type)


            # Get the signature to call this module
            source_string = getsource(function_object)
            args = source_string.split("forward(")[1].split("):")[0].split(", ")[1:]
            #annotated_args = ["{}:{}".format(k, function_args[k]) for k in args]
            #call_signature = "{}({})".format(function_name, ",".join(args))

            constructor_info = ("builtin", function_name, function_type, args)

            # Make a forward call
            declaration_text += "\n\tdef forward(self, {}):".format(",".join(args))
            declaration_text += "\n\t\treturn self.function({})".format(",".join(args))

        else:
            # Get torch.nn module
            function_object = None
            for class_tup in getmembers(nn, isclass):
                if class_tup[0].lower()==function_type.lower():
                    function_type = class_tup[0]
                    function_object = class_tup[1]
                    break
            constructor_info = ("nn", function_name, function_type, [])

            declaration_text += "\n\t\tself.function = nn.{}()".format(str(function_object).split(".")[-1].split("'")[0])

            # TODO: Resolve ordering of args
            args = list(function_args.keys())
            declaration_text += "\n\tdef forward(self, {}):".format(",".join(args))
            declaration_text += "\n\t\treturn self.function({})".format(",".join(args))

    # TODO: Fix for Multi function nodes

    return declaration_text, constructor_info


def generate_main_forward(nodes, execution_order, constructor_calls):
    #
    # for node in nodes:
    #     print(node.id)

    node_dict = {node.id:node for node in nodes}

    # Index intermediate variables
    std_var_idx = 0
    nstd_var_idx = 0

    # Map intermediate variable name to the module that produced it
    return_vars = defaultdict(list)

    main_forward = "\n\tdef forward(self, input):"

    # TODO: Handle multi-input graphs
    for idx, node_name in enumerate(execution_order):
        if idx==0:
            standard_arg = "input"
        else:
            standard_arg = "svar_{}".format(std_var_idx-1)

        node = node_dict[node_name]
        non_standard_args = []
        if node.parameters:
            for param_key in list(node.parameters.keys()):
                # TODO: Resolve ordering
                pre_expression = "\n\t\tnsvar_{} = torch.Tensor({})".format(nstd_var_idx, node.parameters[param_key])
                main_forward += pre_expression
                non_standard_args.append("nsvar_{}".format(nstd_var_idx))
                nstd_var_idx+=1

        args = [standard_arg]
        args.extend(non_standard_args)

        expression = "\n\t\tsvar_{} = self.{}({})".format(std_var_idx, node_name, ",".join(args))
        main_forward += expression
        std_var_idx+=1
    main_forward+="\n\t\treturn {}".format("svar_{}".format(std_var_idx-1))


    return main_forward


def build_script(nodes, execution_order, conditions=None):
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
    constructor_calls = {}
    declared_module_types = set()


    for node in nodes:

        id, funcs, params = node.id, node.functions, node.parameters
        node_dict = {"functions":funcs, "parameters":params}

        declaration_text, constructor_call = get_module_declaration_text(id,node_dict,execution_order,declared_module_types)

        modules_declaration_text += declaration_text
        constructor_calls[id] = constructor_call

    # Build Main call
    main_call_declaration = ("\nclass Model(nn.Module):"
                            "\n\tdef __init__(self):"
                            "\n\t\tsuper().__init__()")


    for node in execution_order:
        main_call_declaration += "\n\t\tself.{} = {}()".format(node, node)

    # Build Main forward
    main_call_forward = generate_main_forward(nodes, execution_order, constructor_calls)

    # Compose script
    script += imports_string
    script += modules_declaration_text
    script += main_call_declaration
    script += main_call_forward

    script += "\nmodel = Model()"
    return script
