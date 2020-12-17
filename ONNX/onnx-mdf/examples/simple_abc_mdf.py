
import torch
import onnx

from onnx import helper

from onnx_mdf.tools import onnx_to_mdf


class A(torch.nn.Module):
    def forward(self, x):
        return x + 1


@torch.jit.script
def loop_b(x, y):
    for i in range(int(y)):
        x = x / 10
    return x


class B(torch.nn.Module):
    def forward(self, x, y):
        return loop_b(x, y)


class C(torch.nn.Module):
    def forward(self, x):
        return x * 100


class ABC(torch.nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.A = A()
        self.B = B()
        self.C = C()

    def forward(self, x, B_loop_count):

        # Run A
        y = self.A(x)

        # Run B (loop_count times)
        y = self.B(y, B_loop_count)

        # Run C
        y = self.C(y)

        return y


model = ABC()
dummy_input = torch.zeros(2, 3)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model,
                  (dummy_input, loop_count),
                  'examples/abc.onnx',
                  verbose=True,
                  input_names=['input', 'B_loop_count'])


# Load it back in using ONNX package
onnx_model = onnx.load("examples/abc.onnx")
onnx.checker.check_model(onnx_model)

# Extract the loop or if body as a sub-model, this is just because I want
# to view it in netron and sub-graphs can't be rendered
for node in [node for node in onnx_model.graph.node if node.op_type in ["Loop", 'If']]:

    # Get the GraphProto of the body
    body_graph = node.attribute[0].g

    # Turn it into a model
    model_def = helper.make_model(body_graph, producer_name='simple_abc_mdf.py')

    onnx.save(model_def, f'examples/{node.name}_body.onnx')


mdf_model = onnx_to_mdf(onnx_model)

mdf_model.to_json_file('examples/abc-onnx_mdf.json')

# Lets convert to YAML
try:
    import json
    import yaml

    with open(r'abc-mdf.yml', 'w') as file:
        yaml.dump(json.loads(mdf_model.to_json()), file, default_flow_style=None, width=120)
except ImportError as ex:
    print("Couldn't load pyaml, skipping YAML output.")

