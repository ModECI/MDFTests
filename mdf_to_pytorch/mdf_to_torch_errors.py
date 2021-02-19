class TorchNameError(Exception):
    def __init__(self, function_name):
        error_string = "No valid torch function mapping or user-defined module found for {}"
        self.function_name = function_name
        self.message = error_string.format(self.function_name)
        super().__init__(self.message)


class InsufficientArgsError(Exception):
    def __init__(self, torch_function_name):
        self.torch_function_name = torch_function_name
        self.message = "Not enough arguments to construct {}".format(self.torch_function_name)
        super().__init__(self.message)

class NonUnaryOutputError(Exception):
    def __init__(self):
        self.message = "Graph has multiple output nodes, not currently supported"
        super().__init__(self.message)

class NonUnaryInputError(Exception):
    def __init__(self):
        self.message = "Graph has multiple input nodes, not currently supported"
        super().__init__(self.message)