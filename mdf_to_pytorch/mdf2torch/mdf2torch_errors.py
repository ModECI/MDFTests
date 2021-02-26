class TorchNameError(Exception):
    def __init__(self, function_name):
        error_string = "No valid torch function mapping or user-defined module found for {}"
        self.function_name = function_name
        self.message = error_string.format(self.function_name)
        super().__init__(self.message)


class TorchArgumentError(Exception):
    def __init__(self, function_name):
        error_string = ("\nValid arguments required to construct torch module: {} not found." 
                        "Consider updating mapping.argument_map for this function.")
        self.function_name = function_name
        self.message = error_string.format(self.function_name)
        super().__init__(self.message)
