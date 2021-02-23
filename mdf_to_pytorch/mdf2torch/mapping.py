"""
This file contains the following dictionaries to be used in the construction of
TorchClassText objects.

    * `function_map` is used to map commonly-occurring generic function names that may be used
    to identify functions in the mdf, to a corresponding torch.nn class.

    * `argument_map` is used to map commonly-occurring mdf function arguments to corresponding
    torch.nn class constructor arguments. MDF elements that are not specifically function args,
    such as inputshape, are considered in the mapping, as these are often required when
    initializing a torch.nn class object.

   {
    nn class1 : {
            nn class signature arg1 : [potential, names, from, mdf],
            nn class signature arg2 : [potential, names, from, mdf]
            }

    nn class2 : {
            nn class signature arg1 : [potential, names, from, mdf],
            nn class signature arg2 : [potential, names, from, mdf]
            }
   }
"""

function_map = {
  "adaptiveavgpool1d": "AdaptiveAvgPool1d",
  "adaptiveavgpool2d": "AdaptiveAvgPool2d",
  "adaptiveavgpool3d": "AdaptiveAvgPool3d",
  "adaptivelogsoftmaxwithloss": "AdaptiveLogSoftmaxWithLoss",
  "adaptivemaxpool1d": "AdaptiveMaxPool1d",
  "adaptivemaxpool2d": "AdaptiveMaxPool2d",
  "adaptivemaxpool3d": "AdaptiveMaxPool3d",
  "alphadropout": "AlphaDropout",
  "avgpool1d": "AvgPool1d",
  "avgpool2d": "AvgPool2d",
  "avgpool3d": "AvgPool3d",
  "bceloss": "BCELoss",
  "bcewithlogitsloss": "BCEWithLogitsLoss",
  "batchnorm1d": "BatchNorm1d",
  "batchnorm2d": "BatchNorm2d",
  "batchnorm3d": "BatchNorm3d",
  "bilinear": "Bilinear",
  "celu": "CELU",
  "ctcloss": "CTCLoss",
  "constantpad1d": "ConstantPad1d",
  "constantpad2d": "ConstantPad2d",
  "constantpad3d": "ConstantPad3d",
  "container": "Container",
  "conv1d": "Conv1d",
  "conv2d": "Conv2d",
  "conv3d": "Conv3d",
  "convtranspose1d": "ConvTranspose1d",
  "convtranspose2d": "ConvTranspose2d",
  "convtranspose3d": "ConvTranspose3d",
  "cosineembeddingloss": "CosineEmbeddingLoss",
  "cosinesimilarity": "CosineSimilarity",
  "crossentropyloss": "CrossEntropyLoss",
  "crossmaplrn2d": "CrossMapLRN2d",
  "dataparallel": "DataParallel",
  "dropout": "Dropout",
  "dropout2d": "Dropout2d",
  "dropout3d": "Dropout3d",
  "elu": "ELU",
  "embedding": "Embedding",
  "embeddingbag": "EmbeddingBag",
  "featurealphadropout": "FeatureAlphaDropout",
  "flatten": "Flatten",
  "fold": "Fold",
  "fractionalmaxpool2d": "FractionalMaxPool2d",
  "fractionalmaxpool3d": "FractionalMaxPool3d",
  "gelu": "GELU",
  "glu": "GLU",
  "gru": "GRU",
  "grucell": "GRUCell",
  "groupnorm": "GroupNorm",
  "hardshrink": "Hardshrink",
  "hardsigmoid": "Hardsigmoid",
  "hardswish": "Hardswish",
  "hardtanh": "Hardtanh",
  "hingeembeddingloss": "HingeEmbeddingLoss",
  "identity": "Identity",
  "instancenorm1d": "InstanceNorm1d",
  "instancenorm2d": "InstanceNorm2d",
  "instancenorm3d": "InstanceNorm3d",
  "kldivloss": "KLDivLoss",
  "l1loss": "L1Loss",
  "lppool1d": "LPPool1d",
  "lppool2d": "LPPool2d",
  "lstm": "LSTM",
  "lstmcell": "LSTMCell",
  "layernorm": "LayerNorm",
  "leakyrelu": "LeakyReLU",
  "linear": "Linear",
  "localresponsenorm": "LocalResponseNorm",
  "logsigmoid": "LogSigmoid",
  "logsoftmax": "LogSoftmax",
  "mseloss": "MSELoss",
  "marginrankingloss": "MarginRankingLoss",
  "maxpool1d": "MaxPool1d",
  "maxpool2d": "MaxPool2d",
  "maxpool3d": "MaxPool3d",
  "maxunpool1d": "MaxUnpool1d",
  "maxunpool2d": "MaxUnpool2d",
  "maxunpool3d": "MaxUnpool3d",
  "module": "Module",
  "moduledict": "ModuleDict",
  "modulelist": "ModuleList",
  "multilabelmarginloss": "MultiLabelMarginLoss",
  "multilabelsoftmarginloss": "MultiLabelSoftMarginLoss",
  "multimarginloss": "MultiMarginLoss",
  "multiheadattention": "MultiheadAttention",
  "nllloss": "NLLLoss",
  "nllloss2d": "NLLLoss2d",
  "prelu": "PReLU",
  "pairwisedistance": "PairwiseDistance",
  "parameter": "Parameter",
  "parameterdict": "ParameterDict",
  "parameterlist": "ParameterList",
  "pixelshuffle": "PixelShuffle",
  "poissonnllloss": "PoissonNLLLoss",
  "rnn": "RNN",
  "rnnbase": "RNNBase",
  "rnncell": "RNNCell",
  "rnncellbase": "RNNCellBase",
  "rrelu": "RReLU",
  "relu": "ReLU",
  "relu6": "ReLU6",
  "reflectionpad1d": "ReflectionPad1d",
  "reflectionpad2d": "ReflectionPad2d",
  "replicationpad1d": "ReplicationPad1d",
  "replicationpad2d": "ReplicationPad2d",
  "replicationpad3d": "ReplicationPad3d",
  "selu": "SELU",
  "sequential": "Sequential",
  "silu": "SiLU",
  "sigmoid": "Sigmoid",
  "smoothl1loss": "SmoothL1Loss",
  "softmarginloss": "SoftMarginLoss",
  "softmax": "Softmax",
  "softmax2d": "Softmax2d",
  "softmin": "Softmin",
  "softplus": "Softplus",
  "softshrink": "Softshrink",
  "softsign": "Softsign",
  "syncbatchnorm": "SyncBatchNorm",
  "tanh": "Tanh",
  "tanhshrink": "Tanhshrink",
  "threshold": "Threshold",
  "transformer": "Transformer",
  "transformerdecoder": "TransformerDecoder",
  "transformerdecoderlayer": "TransformerDecoderLayer",
  "transformerencoder": "TransformerEncoder",
  "transformerencoderlayer": "TransformerEncoderLayer",
  "tripletmarginloss": "TripletMarginLoss",
  "tripletmarginwithdistanceloss": "TripletMarginWithDistanceLoss",
  "unflatten": "Unflatten",
  "unfold": "Unfold",
  "upsample": "Upsample",
  "upsamplingbilinear2d": "UpsamplingBilinear2d",
  "upsamplingnearest2d": "UpsamplingNearest2d",
  "zeropad2d": "ZeroPad2d"
}

argument_map = {
  "AdaptiveAvgPool1d": {
    "output_size": [
      "output_size"
    ]
  },
  "AdaptiveAvgPool2d": {
    "output_size": [
      "output_size"
    ]
  },
  "AdaptiveAvgPool3d": {
    "output_size": [
      "output_size"
    ]
  },
  "AdaptiveLogSoftmaxWithLoss": {
    "in_features": [
      "in_features"
    ],
    "n_classes": [
      "n_classes"
    ],
    "cutoffs": [
      "cutoffs"
    ],
    "div_value": [
      "div_value"
    ],
    "head_bias": [
      "head_bias"
    ]
  },
  "AdaptiveMaxPool1d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "AdaptiveMaxPool2d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "AdaptiveMaxPool3d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "AlphaDropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "AvgPool1d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "ceil_mode": [
      "ceil_mode"
    ],
    "count_include_pad": [
      "count_include_pad"
    ]
  },
  "AvgPool2d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "ceil_mode": [
      "ceil_mode"
    ],
    "count_include_pad": [
      "count_include_pad"
    ],
    "divisor_override": [
      "divisor_override"
    ]
  },
  "AvgPool3d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "ceil_mode": [
      "ceil_mode"
    ],
    "count_include_pad": [
      "count_include_pad"
    ],
    "divisor_override": [
      "divisor_override"
    ]
  },
  "BCELoss": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "BCEWithLogitsLoss": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ],
    "pos_weight": [
      "pos_weight"
    ]
  },
  "BatchNorm1d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "BatchNorm2d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "BatchNorm3d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "Bilinear": {
    "in1_features": [
      "in1_features"
    ],
    "in2_features": [
      "in2_features"
    ],
    "out_features": [
      "out_features"
    ],
    "bias": [
      "bias"
    ]
  },
  "CELU": {
    "alpha": [
      "alpha"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "CTCLoss": {
    "blank": [
      "blank"
    ],
    "reduction": [
      "reduction"
    ],
    "zero_infinity": [
      "zero_infinity"
    ]
  },
  "ConstantPad1d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "ConstantPad2d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "ConstantPad3d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "Container": {
    "kwargs": [
      "kwargs"
    ]
  },
  "Conv1d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "Conv2d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "Conv3d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "ConvTranspose1d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "output_padding": [
      "output_padding"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "dilation": [
      "dilation"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "ConvTranspose2d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "output_padding": [
      "output_padding"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "dilation": [
      "dilation"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "ConvTranspose3d": {
    "in_channels": [
      "in_channels"
    ],
    "out_channels": [
      "out_channels"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "output_padding": [
      "output_padding"
    ],
    "groups": [
      "groups"
    ],
    "bias": [
      "bias"
    ],
    "dilation": [
      "dilation"
    ],
    "padding_mode": [
      "padding_mode"
    ]
  },
  "CosineEmbeddingLoss": {
    "margin": [
      "margin"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "CosineSimilarity": {
    "dim": [
      "dim"
    ],
    "eps": [
      "eps"
    ]
  },
  "CrossEntropyLoss": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "ignore_index": [
      "ignore_index"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "CrossMapLRN2d": {
    "size": [
      "size"
    ],
    "alpha": [
      "alpha"
    ],
    "beta": [
      "beta"
    ],
    "k": [
      "k"
    ]
  },
  "DataParallel": {
    "module": [
      "module"
    ],
    "device_ids": [
      "device_ids"
    ],
    "output_device": [
      "output_device"
    ],
    "dim": [
      "dim"
    ]
  },
  "Dropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Dropout2d": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Dropout3d": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "ELU": {
    "alpha": [
      "alpha"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Embedding": {
    "num_embeddings": [
      "num_embeddings"
    ],
    "embedding_dim": [
      "embedding_dim"
    ],
    "padding_idx": [
      "padding_idx"
    ],
    "max_norm": [
      "max_norm"
    ],
    "norm_type": [
      "norm_type"
    ],
    "scale_grad_by_freq": [
      "scale_grad_by_freq"
    ],
    "sparse": [
      "sparse"
    ],
    "_weight": [
      "_weight"
    ]
  },
  "EmbeddingBag": {
    "num_embeddings": [
      "num_embeddings"
    ],
    "embedding_dim": [
      "embedding_dim"
    ],
    "max_norm": [
      "max_norm"
    ],
    "norm_type": [
      "norm_type"
    ],
    "scale_grad_by_freq": [
      "scale_grad_by_freq"
    ],
    "mode": [
      "mode"
    ],
    "sparse": [
      "sparse"
    ],
    "_weight": [
      "_weight"
    ],
    "include_last_offset": [
      "include_last_offset"
    ]
  },
  "FeatureAlphaDropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Flatten": {
    "start_dim": [
      "start_dim"
    ],
    "end_dim": [
      "end_dim"
    ]
  },
  "Fold": {
    "output_size": [
      "output_size"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "dilation": [
      "dilation"
    ],
    "padding": [
      "padding"
    ],
    "stride": [
      "stride"
    ]
  },
  "FractionalMaxPool2d": {
    "kernel_size": [
      "kernel_size"
    ],
    "output_size": [
      "output_size"
    ],
    "output_ratio": [
      "output_ratio"
    ],
    "return_indices": [
      "return_indices"
    ],
    "_random_samples": [
      "_random_samples"
    ]
  },
  "FractionalMaxPool3d": {
    "kernel_size": [
      "kernel_size"
    ],
    "output_size": [
      "output_size"
    ],
    "output_ratio": [
      "output_ratio"
    ],
    "return_indices": [
      "return_indices"
    ],
    "_random_samples": [
      "_random_samples"
    ]
  },
  "GLU": {
    "dim": [
      "dim"
    ]
  },
  "GRU": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "GRUCell": {
    "input_size": [
      "input_size"
    ],
    "hidden_size": [
      "hidden_size"
    ],
    "bias": [
      "bias"
    ]
  },
  "GroupNorm": {
    "num_groups": [
      "num_groups"
    ],
    "num_channels": [
      "num_channels"
    ],
    "eps": [
      "eps"
    ],
    "affine": [
      "affine"
    ]
  },
  "Hardshrink": {
    "lambd": [
      "lambd"
    ]
  },
  "Hardsigmoid": {
    "inplace": [
      "inplace"
    ]
  },
  "Hardswish": {
    "inplace": [
      "inplace"
    ]
  },
  "Hardtanh": {
    "min_val": [
      "min_val"
    ],
    "max_val": [
      "max_val"
    ],
    "inplace": [
      "inplace"
    ],
    "min_value": [
      "min_value"
    ],
    "max_value": [
      "max_value"
    ]
  },
  "HingeEmbeddingLoss": {
    "margin": [
      "margin"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "Identity": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "InstanceNorm1d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "InstanceNorm2d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "InstanceNorm3d": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ]
  },
  "KLDivLoss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ],
    "log_target": [
      "log_target"
    ]
  },
  "L1Loss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "LPPool1d": {
    "norm_type": [
      "norm_type"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "ceil_mode": [
      "ceil_mode"
    ]
  },
  "LPPool2d": {
    "norm_type": [
      "norm_type"
    ],
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "ceil_mode": [
      "ceil_mode"
    ]
  },
  "LSTM": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "LSTMCell": {
    "input_size": [
      "input_size"
    ],
    "hidden_size": [
      "hidden_size"
    ],
    "bias": [
      "bias"
    ]
  },
  "LayerNorm": {
    "normalized_shape": [
      "normalized_shape"
    ],
    "eps": [
      "eps"
    ],
    "elementwise_affine": [
      "elementwise_affine"
    ]
  },
  "LeakyReLU": {
    "negative_slope": [
      "negative_slope"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Linear": {
    "in_features": [
      "in_features","in_shape", "in", "input_ports"
    ],
    "out_features": [
      "out_features", "out_shape", "out", "output_ports"
    ],
    "bias": [
      "bias", "offset"
    ]
  },
  "LocalResponseNorm": {
    "size": [
      "size"
    ],
    "alpha": [
      "alpha"
    ],
    "beta": [
      "beta"
    ],
    "k": [
      "k"
    ]
  },
  "LogSoftmax": {
    "dim": [
      "dim"
    ]
  },
  "MSELoss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "MarginRankingLoss": {
    "margin": [
      "margin"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "MaxPool1d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "return_indices": [
      "return_indices"
    ],
    "ceil_mode": [
      "ceil_mode"
    ]
  },
  "MaxPool2d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "return_indices": [
      "return_indices"
    ],
    "ceil_mode": [
      "ceil_mode"
    ]
  },
  "MaxPool3d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ],
    "dilation": [
      "dilation"
    ],
    "return_indices": [
      "return_indices"
    ],
    "ceil_mode": [
      "ceil_mode"
    ]
  },
  "MaxUnpool1d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ]
  },
  "MaxUnpool2d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ]
  },
  "MaxUnpool3d": {
    "kernel_size": [
      "kernel_size"
    ],
    "stride": [
      "stride"
    ],
    "padding": [
      "padding"
    ]
  },
  "ModuleDict": {
    "modules": [
      "modules"
    ]
  },
  "ModuleList": {
    "modules": [
      "modules"
    ]
  },
  "MultiLabelMarginLoss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "MultiLabelSoftMarginLoss": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "MultiMarginLoss": {
    "p": [
      "p"
    ],
    "margin": [
      "margin"
    ],
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "MultiheadAttention": {
    "embed_dim": [
      "embed_dim"
    ],
    "num_heads": [
      "num_heads"
    ],
    "dropout": [
      "dropout"
    ],
    "bias": [
      "bias"
    ],
    "add_bias_kv": [
      "add_bias_kv"
    ],
    "add_zero_attn": [
      "add_zero_attn"
    ],
    "kdim": [
      "kdim"
    ],
    "vdim": [
      "vdim"
    ]
  },
  "NLLLoss": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "ignore_index": [
      "ignore_index"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "NLLLoss2d": {
    "weight": [
      "weight"
    ],
    "size_average": [
      "size_average"
    ],
    "ignore_index": [
      "ignore_index"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "PReLU": {
    "num_parameters": [
      "num_parameters"
    ],
    "init": [
      "init"
    ]
  },
  "PairwiseDistance": {
    "p": [
      "p"
    ],
    "eps": [
      "eps"
    ],
    "keepdim": [
      "keepdim"
    ]
  },
  "Parameter": {
    "data": [
      "data"
    ],
    "requires_grad": [
      "requires_grad"
    ]
  },
  "ParameterDict": {
    "parameters": [
      "parameters"
    ]
  },
  "ParameterList": {
    "parameters": [
      "parameters"
    ]
  },
  "PixelShuffle": {
    "upscale_factor": [
      "upscale_factor"
    ]
  },
  "PoissonNLLLoss": {
    "log_input": [
      "log_input"
    ],
    "full": [
      "full"
    ],
    "size_average": [
      "size_average"
    ],
    "eps": [
      "eps"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "RNN": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "RNNBase": {
    "mode": [
      "mode"
    ],
    "input_size": [
      "input_size"
    ],
    "hidden_size": [
      "hidden_size"
    ],
    "num_layers": [
      "num_layers"
    ],
    "bias": [
      "bias"
    ],
    "batch_first": [
      "batch_first"
    ],
    "dropout": [
      "dropout"
    ],
    "bidirectional": [
      "bidirectional"
    ]
  },
  "RNNCell": {
    "input_size": [
      "input_size"
    ],
    "hidden_size": [
      "hidden_size"
    ],
    "bias": [
      "bias"
    ],
    "nonlinearity": [
      "nonlinearity"
    ]
  },
  "RNNCellBase": {
    "input_size": [
      "input_size"
    ],
    "hidden_size": [
      "hidden_size"
    ],
    "bias": [
      "bias"
    ],
    "num_chunks": [
      "num_chunks"
    ]
  },
  "RReLU": {
    "lower": [
      "lower"
    ],
    "upper": [
      "upper"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "ReLU": {
    "inplace": [
      "inplace"
    ]
  },
  "ReLU6": {
    "inplace": [
      "inplace"
    ]
  },
  "ReflectionPad1d": {
    "padding": [
      "padding"
    ]
  },
  "ReflectionPad2d": {
    "padding": [
      "padding"
    ]
  },
  "ReplicationPad1d": {
    "padding": [
      "padding"
    ]
  },
  "ReplicationPad2d": {
    "padding": [
      "padding"
    ]
  },
  "ReplicationPad3d": {
    "padding": [
      "padding"
    ]
  },
  "SELU": {
    "inplace": [
      "inplace"
    ]
  },
  "Sequential": {
    "args": [
      "args"
    ]
  },
  "SiLU": {
    "inplace": [
      "inplace"
    ]
  },
  "SmoothL1Loss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ],
    "beta": [
      "beta"
    ]
  },
  "SoftMarginLoss": {
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "Softmax": {
    "dim": [
      "dim"
    ]
  },
  "Softmin": {
    "dim": [
      "dim"
    ]
  },
  "Softplus": {
    "beta": [
      "beta"
    ],
    "threshold": [
      "threshold"
    ]
  },
  "Softshrink": {
    "lambd": [
      "lambd"
    ]
  },
  "SyncBatchNorm": {
    "num_features": [
      "num_features"
    ],
    "eps": [
      "eps"
    ],
    "momentum": [
      "momentum"
    ],
    "affine": [
      "affine"
    ],
    "track_running_stats": [
      "track_running_stats"
    ],
    "process_group": [
      "process_group"
    ]
  },
  "Threshold": {
    "threshold": [
      "threshold"
    ],
    "value": [
      "value"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "Transformer": {
    "d_model": [
      "d_model"
    ],
    "nhead": [
      "nhead"
    ],
    "num_encoder_layers": [
      "num_encoder_layers"
    ],
    "num_decoder_layers": [
      "num_decoder_layers"
    ],
    "dim_feedforward": [
      "dim_feedforward"
    ],
    "dropout": [
      "dropout"
    ],
    "activation": [
      "activation"
    ],
    "custom_encoder": [
      "custom_encoder"
    ],
    "custom_decoder": [
      "custom_decoder"
    ]
  },
  "TransformerDecoder": {
    "decoder_layer": [
      "decoder_layer"
    ],
    "num_layers": [
      "num_layers"
    ],
    "norm": [
      "norm"
    ]
  },
  "TransformerDecoderLayer": {
    "d_model": [
      "d_model"
    ],
    "nhead": [
      "nhead"
    ],
    "dim_feedforward": [
      "dim_feedforward"
    ],
    "dropout": [
      "dropout"
    ],
    "activation": [
      "activation"
    ]
  },
  "TransformerEncoder": {
    "encoder_layer": [
      "encoder_layer"
    ],
    "num_layers": [
      "num_layers"
    ],
    "norm": [
      "norm"
    ]
  },
  "TransformerEncoderLayer": {
    "d_model": [
      "d_model"
    ],
    "nhead": [
      "nhead"
    ],
    "dim_feedforward": [
      "dim_feedforward"
    ],
    "dropout": [
      "dropout"
    ],
    "activation": [
      "activation"
    ]
  },
  "TripletMarginLoss": {
    "margin": [
      "margin"
    ],
    "p": [
      "p"
    ],
    "eps": [
      "eps"
    ],
    "swap": [
      "swap"
    ],
    "size_average": [
      "size_average"
    ],
    "reduce": [
      "reduce"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "TripletMarginWithDistanceLoss": {
    "distance_function": [
      "distance_function"
    ],
    "margin": [
      "margin"
    ],
    "swap": [
      "swap"
    ],
    "reduction": [
      "reduction"
    ]
  },
  "Unflatten": {
    "dim": [
      "dim"
    ],
    "unflattened_size": [
      "unflattened_size"
    ]
  },
  "Unfold": {
    "kernel_size": [
      "kernel_size"
    ],
    "dilation": [
      "dilation"
    ],
    "padding": [
      "padding"
    ],
    "stride": [
      "stride"
    ]
  },
  "Upsample": {
    "size": [
      "size"
    ],
    "scale_factor": [
      "scale_factor"
    ],
    "mode": [
      "mode"
    ],
    "align_corners": [
      "align_corners"
    ]
  },
  "UpsamplingBilinear2d": {
    "size": [
      "size"
    ],
    "scale_factor": [
      "scale_factor"
    ]
  },
  "UpsamplingNearest2d": {
    "size": [
      "size"
    ],
    "scale_factor": [
      "scale_factor"
    ]
  },
  "ZeroPad2d": {
    "padding": [
      "padding"
    ]
  }
}
