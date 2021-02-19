from inspect import getmembers, isclass, signature

import pytorch_udfs as ptu

def map_func_mdf2torch(mdf_func_name):

    # Check if name corresponds with a udf for this environment
    aux_functions = [item[0] for item in getmembers(ptu, isclass)]

    if mdf_func_name in aux_functions:
        return "pt.{}".format(mdf_func_name)

    mdf_func_name = mdf_func_name.lower()

    if mdf_func_name not in function_map:
        raise TorchNameError(mdf_func_name)
    else:
        return function_map[mdf_func_name]

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
  "adaptiveavgpool1d": "nn.AdaptiveAvgPool1d",
  "adaptiveavgpool2d": "nn.AdaptiveAvgPool2d",
  "adaptiveavgpool3d": "nn.AdaptiveAvgPool3d",
  "adaptivelogsoftmaxwithloss": "nn.AdaptiveLogSoftmaxWithLoss",
  "adaptivemaxpool1d": "nn.AdaptiveMaxPool1d",
  "adaptivemaxpool2d": "nn.AdaptiveMaxPool2d",
  "adaptivemaxpool3d": "nn.AdaptiveMaxPool3d",
  "alphadropout": "nn.AlphaDropout",
  "avgpool1d": "nn.AvgPool1d",
  "avgpool2d": "nn.AvgPool2d",
  "avgpool3d": "nn.AvgPool3d",
  "bceloss": "nn.BCELoss",
  "bcewithlogitsloss": "nn.BCEWithLogitsLoss",
  "batchnorm1d": "nn.BatchNorm1d",
  "batchnorm2d": "nn.BatchNorm2d",
  "batchnorm3d": "nn.BatchNorm3d",
  "bilinear": "nn.Bilinear",
  "celu": "nn.CELU",
  "ctcloss": "nn.CTCLoss",
  "constantpad1d": "nn.ConstantPad1d",
  "constantpad2d": "nn.ConstantPad2d",
  "constantpad3d": "nn.ConstantPad3d",
  "container": "nn.Container",
  "conv1d": "nn.Conv1d",
  "conv2d": "nn.Conv2d",
  "conv3d": "nn.Conv3d",
  "convtranspose1d": "nn.ConvTranspose1d",
  "convtranspose2d": "nn.ConvTranspose2d",
  "convtranspose3d": "nn.ConvTranspose3d",
  "cosineembeddingloss": "nn.CosineEmbeddingLoss",
  "cosinesimilarity": "nn.CosineSimilarity",
  "crossentropyloss": "nn.CrossEntropyLoss",
  "crossmaplrn2d": "nn.CrossMapLRN2d",
  "dataparallel": "nn.DataParallel",
  "dropout": "nn.Dropout",
  "dropout2d": "nn.Dropout2d",
  "dropout3d": "nn.Dropout3d",
  "elu": "nn.ELU",
  "embedding": "nn.Embedding",
  "embeddingbag": "nn.EmbeddingBag",
  "featurealphadropout": "nn.FeatureAlphaDropout",
  "flatten": "nn.Flatten",
  "fold": "nn.Fold",
  "fractionalmaxpool2d": "nn.FractionalMaxPool2d",
  "fractionalmaxpool3d": "nn.FractionalMaxPool3d",
  "gelu": "nn.GELU",
  "glu": "nn.GLU",
  "gru": "nn.GRU",
  "grucell": "nn.GRUCell",
  "groupnorm": "nn.GroupNorm",
  "hardshrink": "nn.Hardshrink",
  "hardsigmoid": "nn.Hardsigmoid",
  "hardswish": "nn.Hardswish",
  "hardtanh": "nn.Hardtanh",
  "hingeembeddingloss": "nn.HingeEmbeddingLoss",
  "identity": "nn.Identity",
  "instancenorm1d": "nn.InstanceNorm1d",
  "instancenorm2d": "nn.InstanceNorm2d",
  "instancenorm3d": "nn.InstanceNorm3d",
  "kldivloss": "nn.KLDivLoss",
  "l1loss": "nn.L1Loss",
  "lppool1d": "nn.LPPool1d",
  "lppool2d": "nn.LPPool2d",
  "lstm": "nn.LSTM",
  "lstmcell": "nn.LSTMCell",
  "layernorm": "nn.LayerNorm",
  "leakyrelu": "nn.LeakyReLU",
  "linear": "nn.Linear",
  "localresponsenorm": "nn.LocalResponseNorm",
  "logsigmoid": "nn.LogSigmoid",
  "logsoftmax": "nn.LogSoftmax",
  "mseloss": "nn.MSELoss",
  "marginrankingloss": "nn.MarginRankingLoss",
  "maxpool1d": "nn.MaxPool1d",
  "maxpool2d": "nn.MaxPool2d",
  "maxpool3d": "nn.MaxPool3d",
  "maxunpool1d": "nn.MaxUnpool1d",
  "maxunpool2d": "nn.MaxUnpool2d",
  "maxunpool3d": "nn.MaxUnpool3d",
  "module": "nn.Module",
  "moduledict": "nn.ModuleDict",
  "modulelist": "nn.ModuleList",
  "multilabelmarginloss": "nn.MultiLabelMarginLoss",
  "multilabelsoftmarginloss": "nn.MultiLabelSoftMarginLoss",
  "multimarginloss": "nn.MultiMarginLoss",
  "multiheadattention": "nn.MultiheadAttention",
  "nllloss": "nn.NLLLoss",
  "nllloss2d": "nn.NLLLoss2d",
  "prelu": "nn.PReLU",
  "pairwisedistance": "nn.PairwiseDistance",
  "parameter": "nn.Parameter",
  "parameterdict": "nn.ParameterDict",
  "parameterlist": "nn.ParameterList",
  "pixelshuffle": "nn.PixelShuffle",
  "poissonnllloss": "nn.PoissonNLLLoss",
  "rnn": "nn.RNN",
  "rnnbase": "nn.RNNBase",
  "rnncell": "nn.RNNCell",
  "rnncellbase": "nn.RNNCellBase",
  "rrelu": "nn.RReLU",
  "relu": "nn.ReLU",
  "relu6": "nn.ReLU6",
  "reflectionpad1d": "nn.ReflectionPad1d",
  "reflectionpad2d": "nn.ReflectionPad2d",
  "replicationpad1d": "nn.ReplicationPad1d",
  "replicationpad2d": "nn.ReplicationPad2d",
  "replicationpad3d": "nn.ReplicationPad3d",
  "selu": "nn.SELU",
  "sequential": "nn.Sequential",
  "silu": "nn.SiLU",
  "sigmoid": "nn.Sigmoid",
  "smoothl1loss": "nn.SmoothL1Loss",
  "softmarginloss": "nn.SoftMarginLoss",
  "softmax": "nn.Softmax",
  "softmax2d": "nn.Softmax2d",
  "softmin": "nn.Softmin",
  "softplus": "nn.Softplus",
  "softshrink": "nn.Softshrink",
  "softsign": "nn.Softsign",
  "syncbatchnorm": "nn.SyncBatchNorm",
  "tanh": "nn.Tanh",
  "tanhshrink": "nn.Tanhshrink",
  "threshold": "nn.Threshold",
  "transformer": "nn.Transformer",
  "transformerdecoder": "nn.TransformerDecoder",
  "transformerdecoderlayer": "nn.TransformerDecoderLayer",
  "transformerencoder": "nn.TransformerEncoder",
  "transformerencoderlayer": "nn.TransformerEncoderLayer",
  "tripletmarginloss": "nn.TripletMarginLoss",
  "tripletmarginwithdistanceloss": "nn.TripletMarginWithDistanceLoss",
  "unflatten": "nn.Unflatten",
  "unfold": "nn.Unfold",
  "upsample": "nn.Upsample",
  "upsamplingbilinear2d": "nn.UpsamplingBilinear2d",
  "upsamplingnearest2d": "nn.UpsamplingNearest2d",
  "zeropad2d": "nn.ZeroPad2d"
}

argument_map = {
  "nn.AdaptiveAvgPool1d": {
    "output_size": [
      "output_size"
    ]
  },
  "nn.AdaptiveAvgPool2d": {
    "output_size": [
      "output_size"
    ]
  },
  "nn.AdaptiveAvgPool3d": {
    "output_size": [
      "output_size"
    ]
  },
  "nn.AdaptiveLogSoftmaxWithLoss": {
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
  "nn.AdaptiveMaxPool1d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "nn.AdaptiveMaxPool2d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "nn.AdaptiveMaxPool3d": {
    "output_size": [
      "output_size"
    ],
    "return_indices": [
      "return_indices"
    ]
  },
  "nn.AlphaDropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.AvgPool1d": {
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
  "nn.AvgPool2d": {
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
  "nn.AvgPool3d": {
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
  "nn.BCELoss": {
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
  "nn.BCEWithLogitsLoss": {
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
  "nn.BatchNorm1d": {
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
  "nn.BatchNorm2d": {
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
  "nn.BatchNorm3d": {
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
  "nn.Bilinear": {
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
  "nn.CELU": {
    "alpha": [
      "alpha"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.CTCLoss": {
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
  "nn.ConstantPad1d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "nn.ConstantPad2d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "nn.ConstantPad3d": {
    "padding": [
      "padding"
    ],
    "value": [
      "value"
    ]
  },
  "nn.Container": {
    "kwargs": [
      "kwargs"
    ]
  },
  "nn.Conv1d": {
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
  "nn.Conv2d": {
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
  "nn.Conv3d": {
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
  "nn.ConvTranspose1d": {
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
  "nn.ConvTranspose2d": {
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
  "nn.ConvTranspose3d": {
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
  "nn.CosineEmbeddingLoss": {
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
  "nn.CosineSimilarity": {
    "dim": [
      "dim"
    ],
    "eps": [
      "eps"
    ]
  },
  "nn.CrossEntropyLoss": {
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
  "nn.CrossMapLRN2d": {
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
  "nn.DataParallel": {
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
  "nn.Dropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.Dropout2d": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.Dropout3d": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.ELU": {
    "alpha": [
      "alpha"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.Embedding": {
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
  "nn.EmbeddingBag": {
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
  "nn.FeatureAlphaDropout": {
    "p": [
      "p"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.Flatten": {
    "start_dim": [
      "start_dim"
    ],
    "end_dim": [
      "end_dim"
    ]
  },
  "nn.Fold": {
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
  "nn.FractionalMaxPool2d": {
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
  "nn.FractionalMaxPool3d": {
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
  "nn.GLU": {
    "dim": [
      "dim"
    ]
  },
  "nn.GRU": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "nn.GRUCell": {
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
  "nn.GroupNorm": {
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
  "nn.Hardshrink": {
    "lambd": [
      "lambd"
    ]
  },
  "nn.Hardsigmoid": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.Hardswish": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.Hardtanh": {
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
  "nn.HingeEmbeddingLoss": {
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
  "nn.Identity": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "nn.InstanceNorm1d": {
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
  "nn.InstanceNorm2d": {
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
  "nn.InstanceNorm3d": {
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
  "nn.KLDivLoss": {
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
  "nn.L1Loss": {
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
  "nn.LPPool1d": {
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
  "nn.LPPool2d": {
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
  "nn.LSTM": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "nn.LSTMCell": {
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
  "nn.LayerNorm": {
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
  "nn.LeakyReLU": {
    "negative_slope": [
      "negative_slope"
    ],
    "inplace": [
      "inplace"
    ]
  },
  "nn.Linear": {
    "in_features": [
      "in_features","in_shape", "in"
    ],
    "out_features": [
      "out_features", "out_shape", "out"
    ],
    "bias": [
      "bias", "offset", "intercept"
    ]
  },
  "nn.LocalResponseNorm": {
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
  "nn.LogSoftmax": {
    "dim": [
      "dim"
    ]
  },
  "nn.MSELoss": {
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
  "nn.MarginRankingLoss": {
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
  "nn.MaxPool1d": {
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
  "nn.MaxPool2d": {
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
  "nn.MaxPool3d": {
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
  "nn.MaxUnpool1d": {
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
  "nn.MaxUnpool2d": {
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
  "nn.MaxUnpool3d": {
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
  "nn.ModuleDict": {
    "modules": [
      "modules"
    ]
  },
  "nn.ModuleList": {
    "modules": [
      "modules"
    ]
  },
  "nn.MultiLabelMarginLoss": {
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
  "nn.MultiLabelSoftMarginLoss": {
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
  "nn.MultiMarginLoss": {
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
  "nn.MultiheadAttention": {
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
  "nn.NLLLoss": {
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
  "nn.NLLLoss2d": {
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
  "nn.PReLU": {
    "num_parameters": [
      "num_parameters"
    ],
    "init": [
      "init"
    ]
  },
  "nn.PairwiseDistance": {
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
  "nn.Parameter": {
    "data": [
      "data"
    ],
    "requires_grad": [
      "requires_grad"
    ]
  },
  "nn.ParameterDict": {
    "parameters": [
      "parameters"
    ]
  },
  "nn.ParameterList": {
    "parameters": [
      "parameters"
    ]
  },
  "nn.PixelShuffle": {
    "upscale_factor": [
      "upscale_factor"
    ]
  },
  "nn.PoissonNLLLoss": {
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
  "nn.RNN": {
    "args": [
      "args"
    ],
    "kwargs": [
      "kwargs"
    ]
  },
  "nn.RNNBase": {
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
  "nn.RNNCell": {
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
  "nn.RNNCellBase": {
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
  "nn.RReLU": {
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
  "nn.ReLU": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.ReLU6": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.ReflectionPad1d": {
    "padding": [
      "padding"
    ]
  },
  "nn.ReflectionPad2d": {
    "padding": [
      "padding"
    ]
  },
  "nn.ReplicationPad1d": {
    "padding": [
      "padding"
    ]
  },
  "nn.ReplicationPad2d": {
    "padding": [
      "padding"
    ]
  },
  "nn.ReplicationPad3d": {
    "padding": [
      "padding"
    ]
  },
  "nn.SELU": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.Sequential": {
    "args": [
      "args"
    ]
  },
  "nn.SiLU": {
    "inplace": [
      "inplace"
    ]
  },
  "nn.SmoothL1Loss": {
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
  "nn.SoftMarginLoss": {
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
  "nn.Softmax": {
    "dim": [
      "dim"
    ]
  },
  "nn.Softmin": {
    "dim": [
      "dim"
    ]
  },
  "nn.Softplus": {
    "beta": [
      "beta"
    ],
    "threshold": [
      "threshold"
    ]
  },
  "nn.Softshrink": {
    "lambd": [
      "lambd"
    ]
  },
  "nn.SyncBatchNorm": {
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
  "nn.Threshold": {
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
  "nn.Transformer": {
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
  "nn.TransformerDecoder": {
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
  "nn.TransformerDecoderLayer": {
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
  "nn.TransformerEncoder": {
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
  "nn.TransformerEncoderLayer": {
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
  "nn.TripletMarginLoss": {
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
  "nn.TripletMarginWithDistanceLoss": {
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
  "nn.Unflatten": {
    "dim": [
      "dim"
    ],
    "unflattened_size": [
      "unflattened_size"
    ]
  },
  "nn.Unfold": {
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
  "nn.Upsample": {
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
  "nn.UpsamplingBilinear2d": {
    "size": [
      "size"
    ],
    "scale_factor": [
      "scale_factor"
    ]
  },
  "nn.UpsamplingNearest2d": {
    "size": [
      "size"
    ],
    "scale_factor": [
      "scale_factor"
    ]
  },
  "nn.ZeroPad2d": {
    "padding": [
      "padding"
    ]
  }
}
