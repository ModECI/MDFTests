"""
This file does three things:
    - It implements a simple PyTorch model.
    - Exports in to ONNX using a combination of tracing and scripting
    - Converts it to MDF
"""
import torch
import onnx

from onnx import helper

from onnx_mdf.tools import onnx_to_mdf, find_subgraphs

class SimpleIntegrator(torch.nn.Module):
    def __init__(self, shape, rate):
        super(SimpleIntegrator, self).__init__()
        self.previous_value = torch.zeros(shape)
        self.rate = rate

    def forward(self, x):
        value = self.previous_value + (x * self.rate)
        self.previous_value = value
        return value

class Linear(torch.nn.Module):
    def __init__(self, slope=1.0, intercept=0.0):
        super(Linear, self).__init__()
        self.slope = slope
        self.intercept = intercept

    def forward(self, x):
        return self.slope*x + self.intercept


class ABCD(torch.nn.Module):
    def __init__(self, A, B, C, D):
        super(ABCD, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def forward(self, x):

        # Since we are implementing conditions that reference the number of calls
        # to A and B, we need to keep track of this.
        num_A_calls = 0
        num_B_calls = 0

        # We need to initialize outputs, torchscript jit complains if c and d
        # are not defined in the FALSE branches of our conditionals.
        a = torch.zeros_like(x)
        b = torch.zeros_like(x)
        c = torch.zeros_like(x)
        d = torch.zeros_like(x)

        for i in range(10):

            # A: pnl.AtNCalls(A, 0),
            if num_A_calls == 0:
                a = self.A(x)
                num_A_calls = num_A_calls + 1

            # B: pnl.Always()
            b = self.B(a)
            num_B_calls = num_B_calls + 1

            # C: pnl.EveryNCalls(B, 5),
            if num_B_calls % 5 == 0:
                c = self.C(b)

            # D: pnl.EveryNCalls(B, 10)
            if num_B_calls % 10 == 0:
                d = self.D(b)

        return c, d


def main():

    # model = ABCD(A=SimpleIntegrator(shape=(1,), rate=2.0),
    #              B=SimpleIntegrator(shape=(1,), rate=2.0),
    #              C=SimpleIntegrator(shape=(1,), rate=2.0),
    #              D=SimpleIntegrator(shape=(1,), rate=2.0))

    slope = torch.ones((1,1))*2.0
    intercept = torch.ones((1, 1)) * 2.0
    model = ABCD(A=Linear(slope=slope, intercept=intercept),
                 B=Linear(slope=slope, intercept=intercept),
                 C=Linear(slope=slope, intercept=intercept),
                 D=Linear(slope=slope, intercept=intercept))

    model = torch.jit.script(model)

    output = model(torch.ones((1,1)))

    print(output)

    dummy_input = torch.ones((1,1))
    torch.onnx.export(model,
                      (dummy_input),
                      'examples/abcd/abcd.onnx',
                      verbose=True,
                      input_names=['input'],
                      example_outputs=output)


    # Load it back in using ONNX package
    onnx_model = onnx.load("examples/abcd/abcd.onnx")
    onnx.checker.check_model(onnx_model)

    import onnx_mdf
    onnx_mdf.tools.convert_file("examples/abcd/abcd.onnx")

    # Extract the loop or if body as a sub-model, this is just because I want
    # to view it in netron and sub-graphs can't be rendered
    for key, graph in find_subgraphs(onnx_model.graph).items():

        # Turn it into a model
        model_def = helper.make_model(graph, producer_name='simple_abcd.py')
        onnx.save(model_def, f'examples/abcd/{key}_graph.onnx')


if __name__ == "__main__":
    main()
