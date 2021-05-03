from .mdf_tools import load_mdf_json, flatten, get_sorted_nodes
from.onnx_tools import generate_onnx_model, write_onnx_models

import os

def convert_mdf_to_onnx(filename, savedir):
    basename, _ = os.path.splitext(os.path.basename(filename))
    #print(basename)
    model = load_mdf_json(filename)
    flatten(model)
    sorted_nodes = get_sorted_nodes(model)
    onnx_models = generate_onnx_model(model, sorted_nodes)
    write_onnx_models(onnx_models, savedir, basename)



