import re
import torch.nn as nn
import mapping as tm
from mapping import map_func_mdf2torch
from collections import OrderedDict
from inspect import getmembers, isclass, signature

def get_node_io_shapes(node):
    input_shapes, output_shapes = OrderedDict(), OrderedDict()
    if "input_ports" in node.keys():
        for name, spec in node["input_ports"].items():
            shape = tuple(int(i) for i in spec["shape"][1:-1].split(","))
            input_shapes[name] = shape

    if "output_ports" in node.keys():
        for name, spec in node["output_ports"].items():
            shape = tuple(int(i) for i in spec["shape"][1:-1].split(","))
            output_shapes[name] = shape

    return input_shapes, output_shapes

def _get_node_func_name(node):
    func_names = OrderedDict()
    for function_spec in node["functions"]:
        name = list(function_spec.keys())[0]

        if "function" in function_spec[name]:
            torch_name = map_func_mdf2torch(function_spec[name]["function"])
            func_names[name] = torch_name

    return func_names

def generate_class_text(name, node_dict):
    # {nn.torch_class_name: nn object}
    torch_modules = {"nn.{}".format(k): v for k, v in getmembers(nn, isclass)}

    # Get the name of the function used in the mdf
    mdf2torch_name_map = _get_node_func_name(node_dict)  # {common_name:torch_func or UDF name}

    # Determine what arguments the torch function or UDF takes
    arg_dicts = OrderedDict()

    # Iterate on common name, class name
    for func_name, torch_func in mdf2torch_name_map.items():

        # Get the argument signature if class is a UDF
        if torch_func.startswith("pt."):
            key = torch_func.split("pt.")[-1]
            user_def_functions = {k: v for k, v in getmembers(ptu, isclass)}

            # Need parameters for __init__() and forward()
            udf_constructor_args = list(signature(user_def_functions[key].__init__).parameters)[1:]  # Exclude self
            udf_forward_args = list(signature(user_def_functions[key].forward).parameters)[1:]

            arg_list = OrderedDict()

            # Use dot notation to dictate if it is constructor or forward arg
            for arg in [*udf_constructor_args, *udf_forward_args]:
                print(arg)

        # If class is a nn module member
        else:
            args = signature(torch_modules[torch_func]).parameters
            arg_list = OrderedDict()
            for arg in args.items():
                arg_name = arg[0]
                argtype = str(arg[1]).split(":")[1]
                arg_default = ""
                if "=" in argtype:
                    argtype, arg_default = argtype.split("=")
                arg_list[arg_name] = (argtype, arg_default)
            arg_dicts[func_name] = arg_list

    # Parse node parameters and shapes, add to potential arguments
    in_shape, out_shape = get_node_io_shapes(node_dict)

    potential_args = {"in_shape": in_shape, "out_shape": out_shape}

    if "parameters" in node_dict:
        for param in node_dict["parameters"].keys():
            potential_args[param] = node_dict["parameters"][param]

    torch_class_components = []

    for item in arg_dicts.items():
        func, args = item
        if not args:
            args = None  # If nn class takes no arguments

        torch_class_text = TorchClassText(name,
                                          mdf2torch_name_map[name],
                                          arg_dict=args,
                                          potential_arg_dict=potential_args)
        torch_class_components.append(torch_class_text)

    return torch_class_components

class TorchClassText(object):
    def __init__(self, name, torch_class=None, arg_dict=None, potential_arg_dict=None):

        self.name = name
        self.torch_class = torch_class
        self.arg_dict = arg_dict
        self.potential_arg_dict = potential_arg_dict

        self.text = ""

        if self.torch_class:
            self.text = "{}(".format(torch_class)

        else:
            # TODO: Need logic for building & naming custom nn module
            pass

        if not self.arg_dict:
            self.complete_class_text()
        else:
            self.find_arg_match()
            self.complete_class_text()

    def find_arg_match(self):
        # Make a set of the args we have
        args_available = set([key for key, value in self.potential_arg_dict.items()])

        # Iterate over args we need, do one of 3 things:
        #   1. Get a match using our mapping
        #       a. put necessary arg name into map
        #       b. see if map result is in args set
        #       c. if it is, use arg set key to key potential_arg_dict, put that in text, return
        #       d. if it is not, got to 2
        #   2. Set default if available
        #       a. if mapping failed, check if potential arg dict has a default
        #           this is the order we would prefer anyway, since if there is a non-default
        #           passed value, we want to use that
        #       b. if has default, put it, return
        #       c. if not, go to 3
        #   3. Raise insufficient info error

        for arg in self.arg_dict:
            valid_names = tm.argument_map[self.torch_class][arg]

            default = None

            if self.arg_dict[arg][1] != "":
                default = self.arg_dict[arg][1]

            matched = False

            for valid_name in valid_names:
                if valid_name in args_available:

                    # If valid name found, get the value and add to expression
                    # TODO: This should be a list of different parser / look for expression
                    # for now just do simple index
                    ans = self.potential_arg_dict[valid_name]
                    if type(ans) == OrderedDict:
                        ans = ans[list(ans.keys())[0]][0]

                    if not default:
                        self.add_positional_arg(ans)
                    else:
                        self.add_keyword_arg(arg, ans)

                    matched = True
                    break

                else:
                    if default is not None:
                        self.add_keyword_arg(arg, default)
                        matched = True
                        break
            if not matched:
                raise InsufficientArgsError(self.torch_class)

    def add_positional_arg(self, arg_value):
        self.text += "{}, ".format(arg_value)

    def add_keyword_arg(self, arg_name, arg_value):
        self.text += "{}={}, ".format(arg_name, arg_value)

    def complete_class_text(self):
        if self.text.endswith(", "):
            self.text = self.text[:-2]
        self.text += ")"

    def set_text(self, text):
        self.text = text

    def get_text(self):
        return self.text

def wrap_class_text(name, class_text_objects):

    # If length 1, simple assignment
    if len(class_text_objects)==1:
        text =  "{} = {}".format(name, class_text_objects[0].get_text())

    # TODO: Explicit args vs passed forward args

    # If length > 1 assignments & sequence
    else:
        text = ""
        seq_list = []
        for classname in class_text_objects:
            seq_list.append(classname.name)
            text +="\n{} = {}".format(classname.name, classname.get_text())
        text+="\nmodule_list = {}".format(str(seq_list))
        text+="\n{} = nn.Sequential(*module_list)".format(name)
    return text


def generate_main_call_text(nested_call, nodes):

    module_text = """\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super(Model, self).__init__()"""

    # TODO: Node initializers
    for name in nodes:
        module_text += "\n        self.{} = {}".format(name, name)

    # TODO: manage args --> forward
    module_text += "\n\n    def forward(self):\n        return {}".format(nested_call)
    return module_text

def generate_nested_call(top_call_node, dependency_graph):

    nested = ""

    tops = [top_call_node]

    while tops:
        top_call_node = tops.pop(0)

        try:
            dependencies = dependency_graph[top_call_node]
        except KeyError:
            dependencies = ["input"]

        if nested == "":
            nested = "{}()".format(top_call_node)

        # Make args
        arg_string = ""
        for dependency in dependencies:

            if dependency!="input":
                tops.append(dependency)
                arg_string += "{}(), ".format(dependency)
            else:
                arg_string += "{}, ".format("input")
        arg_string = arg_string[:-2]

        literal = "{}\(\)".format(top_call_node)
        nested = re.sub(literal, "self.{}({})".format(top_call_node, arg_string), nested)

    return nested

def build_script(nodes, top_call_node, dependency_graph):
    script = ""
    imports_string = "import torch.nn as nn"

    # TODO: Error if top call node not len(1)

    # TODO: Refactor generate_class_text
    #   * If not udf, standalone module
    #   * If 2 function node, sequence
    #   * If UDF, module

    # TODO: Enhance generate_class_text with
    #   parameters and target-specific

    # object instantiation texts
    for node_name, node_dict in nodes.items():
        node_text = generate_class_text(node_name, node_dict)
        text = wrap_class_text(node_name, node_text)
        script += "\n{}".format(text)

    # TODO: Hash out parameter passing to main call

    # main call text
    nested_call = generate_nested_call(top_call_node, dependency_graph)
    main_call = generate_main_call_text(nested_call, nodes)
    script += "\n{}".format(main_call)


    return script