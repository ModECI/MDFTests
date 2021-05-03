import tensorflow.keras as keras
from tensorflow.keras.models import Model
from spektral.layers import *
from tensorflow.keras.layers import *

def custom_softmax_axis__1(x):
    return keras.activations.softmax(x, axis=-1)
def custom_linear_31(x):
    return keras.activations.linear(x, )
def custom_linear_30(x):
    return keras.activations.linear(x, )
def custom_linear_29(x):
    return keras.activations.linear(x, )
def custom_linear_28(x):
    return keras.activations.linear(x, )
def custom_linear_27(x):
    return keras.activations.linear(x, )
def custom_linear_26(x):
    return keras.activations.linear(x, )
def custom_linear_25(x):
    return keras.activations.linear(x, )
def custom_linear_24(x):
    return keras.activations.linear(x, )
def custom_linear_23(x):
    return keras.activations.linear(x, )
def custom_linear_22(x):
    return keras.activations.linear(x, )
def custom_linear_21(x):
    return keras.activations.linear(x, )
def custom_linear_20(x):
    return keras.activations.linear(x, )
def custom_linear_19(x):
    return keras.activations.linear(x, )
def custom_linear_18(x):
    return keras.activations.linear(x, )
def custom_linear_17(x):
    return keras.activations.linear(x, )
def custom_linear_16(x):
    return keras.activations.linear(x, )
def custom_linear_15(x):
    return keras.activations.linear(x, )
def custom_linear_14(x):
    return keras.activations.linear(x, )
def custom_linear_13(x):
    return keras.activations.linear(x, )
def custom_linear_12(x):
    return keras.activations.linear(x, )
def custom_linear_11(x):
    return keras.activations.linear(x, )
def custom_linear_10(x):
    return keras.activations.linear(x, )
def custom_linear_9(x):
    return keras.activations.linear(x, )
def custom_linear_8(x):
    return keras.activations.linear(x, )
def custom_linear_7(x):
    return keras.activations.linear(x, )
def custom_linear_6(x):
    return keras.activations.linear(x, )
def custom_linear_5(x):
    return keras.activations.linear(x, )
def custom_linear_4(x):
    return keras.activations.linear(x, )
def custom_linear_3(x):
    return keras.activations.linear(x, )
def custom_linear_2(x):
    return keras.activations.linear(x, )
def custom_linear_(x):
    return keras.activations.linear(x, )

ebv_output = Input(shape=(1,), batch_shape=None, dtype=None, sparse=False, tensor=None)
galaxy_images_output = Input(shape=[64, 64, 5], batch_shape=None, dtype=None, sparse=False, tensor=None)
conv2d = Conv2D(filters=64, kernel_size=(5, 5), padding="same", data_format=None, activation=custom_linear_, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output = conv2d(inputs=galaxy_images_output)
prelu = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output = prelu(inputs=conv2d_output)
averagepooling2d = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same", data_format=None)
averagepooling2d_output = averagepooling2d(inputs=prelu_output)
conv2d2 = Conv2D(filters=48, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_2, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output2 = conv2d2(inputs=averagepooling2d_output)
prelu2 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output2 = prelu2(inputs=conv2d_output2)
conv2d3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format=None, activation=custom_linear_3, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output3 = conv2d3(inputs=prelu_output2)
prelu3 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output3 = prelu3(inputs=conv2d_output3)
conv2d4 = Conv2D(filters=48, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_4, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output4 = conv2d4(inputs=averagepooling2d_output)
prelu4 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output4 = prelu4(inputs=conv2d_output4)
averagepooling2d2 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same", data_format=None)
averagepooling2d_output2 = averagepooling2d2(inputs=prelu_output4)
conv2d5 = Conv2D(filters=64, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_5, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output5 = conv2d5(inputs=averagepooling2d_output)
prelu5 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output5 = prelu5(inputs=conv2d_output5)
conv2d6 = Conv2D(filters=48, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_6, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output6 = conv2d6(inputs=averagepooling2d_output)
prelu6 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output6 = prelu6(inputs=conv2d_output6)
conv2d7 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", data_format=None, activation=custom_linear_7, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output7 = conv2d7(inputs=prelu_output6)
prelu7 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output7 = prelu7(inputs=conv2d_output7)
concatenate = Concatenate(axis=-1)
concatenate_output = concatenate(inputs=[prelu_output5, prelu_output3, prelu_output7, averagepooling2d_output2])
conv2d8 = Conv2D(filters=64, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_8, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output8 = conv2d8(inputs=concatenate_output)
conv2d9 = Conv2D(filters=92, kernel_size=(1, 1), padding="valid", data_format=None, activation=custom_linear_9, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output9 = conv2d9(inputs=concatenate_output)
conv2d10 = Conv2D(filters=64, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_10, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output10 = conv2d10(inputs=concatenate_output)
prelu8 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output8 = prelu8(inputs=conv2d_output10)
conv2d11 = Conv2D(filters=92, kernel_size=(5, 5), padding="same", data_format=None, activation=custom_linear_11, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output11 = conv2d11(inputs=prelu_output8)
prelu9 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output9 = prelu9(inputs=conv2d_output11)
prelu10 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output10 = prelu10(inputs=conv2d_output8)
averagepooling2d3 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same", data_format=None)
averagepooling2d_output3 = averagepooling2d3(inputs=prelu_output10)
conv2d12 = Conv2D(filters=64, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_12, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output12 = conv2d12(inputs=concatenate_output)
prelu11 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output11 = prelu11(inputs=conv2d_output12)
conv2d13 = Conv2D(filters=92, kernel_size=(3, 3), padding="same", data_format=None, activation=custom_linear_13, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output13 = conv2d13(inputs=prelu_output11)
prelu12 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output12 = prelu12(inputs=conv2d_output13)
prelu13 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output13 = prelu13(inputs=conv2d_output9)
concatenate2 = Concatenate(axis=-1)
concatenate_output2 = concatenate2(inputs=[prelu_output13, prelu_output12, prelu_output9, averagepooling2d_output3])
averagepooling2d4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same", data_format=None)
averagepooling2d_output4 = averagepooling2d4(inputs=concatenate_output2)
conv2d14 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_14, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output14 = conv2d14(inputs=averagepooling2d_output4)
prelu14 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output14 = prelu14(inputs=conv2d_output14)
conv2d15 = Conv2D(filters=128, kernel_size=(5, 5), padding="same", data_format=None, activation=custom_linear_15, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output15 = conv2d15(inputs=prelu_output14)
prelu15 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output15 = prelu15(inputs=conv2d_output15)
conv2d16 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_16, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output16 = conv2d16(inputs=averagepooling2d_output4)
prelu16 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output16 = prelu16(inputs=conv2d_output16)
conv2d17 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_17, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output17 = conv2d17(inputs=averagepooling2d_output4)
prelu17 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output17 = prelu17(inputs=conv2d_output17)
averagepooling2d5 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same", data_format=None)
averagepooling2d_output5 = averagepooling2d5(inputs=prelu_output17)
conv2d18 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_18, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output18 = conv2d18(inputs=averagepooling2d_output4)
prelu18 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output18 = prelu18(inputs=conv2d_output18)
conv2d19 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format=None, activation=custom_linear_19, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output19 = conv2d19(inputs=prelu_output18)
prelu19 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output19 = prelu19(inputs=conv2d_output19)
concatenate3 = Concatenate(axis=-1)
concatenate_output3 = concatenate3(inputs=[prelu_output16, prelu_output19, prelu_output15, averagepooling2d_output5])
conv2d20 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_20, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output20 = conv2d20(inputs=concatenate_output3)
prelu20 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output20 = prelu20(inputs=conv2d_output20)
conv2d21 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format=None, activation=custom_linear_21, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output21 = conv2d21(inputs=prelu_output20)
prelu21 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output21 = prelu21(inputs=conv2d_output21)
conv2d22 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_22, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output22 = conv2d22(inputs=concatenate_output3)
prelu22 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output22 = prelu22(inputs=conv2d_output22)
averagepooling2d6 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same", data_format=None)
averagepooling2d_output6 = averagepooling2d6(inputs=prelu_output22)
conv2d23 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_23, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output23 = conv2d23(inputs=concatenate_output3)
prelu23 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output23 = prelu23(inputs=conv2d_output23)
conv2d24 = Conv2D(filters=128, kernel_size=(5, 5), padding="same", data_format=None, activation=custom_linear_24, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output24 = conv2d24(inputs=prelu_output23)
prelu24 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output24 = prelu24(inputs=conv2d_output24)
conv2d25 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_25, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output25 = conv2d25(inputs=concatenate_output3)
prelu25 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output25 = prelu25(inputs=conv2d_output25)
concatenate4 = Concatenate(axis=-1)
concatenate_output4 = concatenate4(inputs=[prelu_output25, prelu_output21, prelu_output24, averagepooling2d_output6])
averagepooling2d7 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same", data_format=None)
averagepooling2d_output7 = averagepooling2d7(inputs=concatenate_output4)
conv2d26 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_26, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output26 = conv2d26(inputs=averagepooling2d_output7)
prelu26 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output26 = prelu26(inputs=conv2d_output26)
averagepooling2d8 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same", data_format=None)
averagepooling2d_output8 = averagepooling2d8(inputs=prelu_output26)
conv2d27 = Conv2D(filters=92, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_27, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output27 = conv2d27(inputs=averagepooling2d_output7)
prelu27 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output27 = prelu27(inputs=conv2d_output27)
conv2d28 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format=None, activation=custom_linear_28, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output28 = conv2d28(inputs=prelu_output27)
prelu28 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output28 = prelu28(inputs=conv2d_output28)
conv2d29 = Conv2D(filters=128, kernel_size=(1, 1), padding="same", data_format=None, activation=custom_linear_29, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
conv2d_output29 = conv2d29(inputs=averagepooling2d_output7)
prelu29 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output29 = prelu29(inputs=conv2d_output29)
concatenate5 = Concatenate(axis=-1)
concatenate_output5 = concatenate5(inputs=[prelu_output29, prelu_output28, averagepooling2d_output8])
flatten = Flatten()
flatten_output = flatten(inputs=concatenate_output5)
concatenate6 = Concatenate(axis=-1)
concatenate_output6 = concatenate6(inputs=[flatten_output, ebv_output])
dense = Dense(units=1096, activation=custom_linear_30, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
dense_output = dense(inputs=concatenate_output6)
prelu30 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output30 = prelu30(inputs=dense_output)
dense2 = Dense(units=1096, activation=custom_linear_31, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
dense_output2 = dense2(inputs=prelu_output30)
prelu31 = PReLU(alpha_initializer=keras.initializers.Zeros(), shared_axes=None)
prelu_output31 = prelu31(inputs=dense_output2)
dense3 = Dense(units=180, activation=custom_softmax_axis__1, use_bias=True, kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.Zeros())
dense_output3 = dense3(inputs=prelu_output31)

custom_objects = {}
custom_objects['custom_linear_'] = custom_linear_
custom_objects['custom_linear_2'] = custom_linear_2
custom_objects['custom_linear_3'] = custom_linear_3
custom_objects['custom_linear_4'] = custom_linear_4
custom_objects['custom_linear_5'] = custom_linear_5
custom_objects['custom_linear_6'] = custom_linear_6
custom_objects['custom_linear_7'] = custom_linear_7
custom_objects['custom_linear_8'] = custom_linear_8
custom_objects['custom_linear_9'] = custom_linear_9
custom_objects['custom_linear_10'] = custom_linear_10
custom_objects['custom_linear_11'] = custom_linear_11
custom_objects['custom_linear_12'] = custom_linear_12
custom_objects['custom_linear_13'] = custom_linear_13
custom_objects['custom_linear_14'] = custom_linear_14
custom_objects['custom_linear_15'] = custom_linear_15
custom_objects['custom_linear_16'] = custom_linear_16
custom_objects['custom_linear_17'] = custom_linear_17
custom_objects['custom_linear_18'] = custom_linear_18
custom_objects['custom_linear_19'] = custom_linear_19
custom_objects['custom_linear_20'] = custom_linear_20
custom_objects['custom_linear_21'] = custom_linear_21
custom_objects['custom_linear_22'] = custom_linear_22
custom_objects['custom_linear_23'] = custom_linear_23
custom_objects['custom_linear_24'] = custom_linear_24
custom_objects['custom_linear_25'] = custom_linear_25
custom_objects['custom_linear_26'] = custom_linear_26
custom_objects['custom_linear_27'] = custom_linear_27
custom_objects['custom_linear_28'] = custom_linear_28
custom_objects['custom_linear_29'] = custom_linear_29
custom_objects['custom_linear_30'] = custom_linear_30
custom_objects['custom_linear_31'] = custom_linear_31
custom_objects['custom_softmax_axis__1'] = custom_softmax_axis__1


model = Model(inputs=[galaxy_images_output,ebv_output], outputs=[dense_output3])
result = model
model.custom_objects = custom_objects