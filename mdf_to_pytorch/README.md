## _Work in Progress_ MDF > PyTorch

### Description
This is  a prototype parser for MDF > PyTorch and has the following limitations for the first pass:
* Best-suited for specifying feed-forward torch models
* Controllers are not used to dynamically change parameters of other components
* Simplistic variable passing, does not fully use input/output ports and edges.
* The following conditions are implemented:
    * WhenFinished
    * EveryNCalls
    * Threshold
    * and / all
    * or / any
    

### Example Usage

The following implements a simple MLP to classify (14x14) handwritten digit images.   
Pre-trained weights are loaded in as MDF parameters, so there is no need to train the model.
```python3
import torch
import torch.nn
import numpy as np

import mdf2torch

# Load Model
models = mdf2torch.load("example_mdfs/mlp_prototype.json", eval_models=True)
model = models["mlp_prototype"]

# Iterate on training data, feed forward and log accuracy
imgs = np.load("example_data/imgs.npy")
labels = np.load("example_data/labels.npy")

for i in range(len(imgs)):
    img = torch.Tensor(imgs[i,:,:]).view(-1, 14*14)
    target = labels[i]
    prediction = model(img)
    print(target, prediction)

```

### Known Issues & To-Do (Rough Order of Priority to Add)
* Variable passing is too simple and does not make use of the specifications
  provided by MDF input/output ports. Rather, edges are used
  to specify graph hierarchy and dictate module arguments & returns. 
  The return value is simply fed forward to the next module.
* Implementing more conditions
* Standardizing / improving function & argument naming


