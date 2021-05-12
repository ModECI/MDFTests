
def generate_aux_text(function_type, parameters):

    aux_text = ""

    # Add more..
    if function_type=='matmul':
        weight = parameters["weight"]
        aux_text += "\n\t\tself.weight = torch.Tensor({})".format(weight)

    elif function_type=='add':
        pass


    return aux_text