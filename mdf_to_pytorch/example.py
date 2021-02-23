import torch
import torch.nn
import mdf2torch

models = mdf2torch.load("examples/linear_test.json", eval_models=True)
model = models["linear_test"]