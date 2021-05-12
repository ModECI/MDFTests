"""
Expand...
"""
lookup = {
  "matmul":"matmul",
  "conv_2d":"conv2d",
  "add":"add",
  "argmax":"argmax"
}

def alias(name):
    if name in lookup:
        return lookup[name]
    else:
        return None