import numpy as np
import torch.nn
import mdf2torch

# Load Model
models = mdf2torch.load("example_mdfs/prototype.json", eval_models=True)
model = models["prototype"]

# Iterate on training data, feed forward and log accuracy
imgs = np.load("example_data/imgs.npy")
labels = np.load("example_data/labels.npy")

for i in range(len(imgs)):
    img = torch.Tensor(imgs[i,:,:]).view(-1, 14*14)
    target = labels[i]
    prediction = model(img)
    print(target, prediction)


