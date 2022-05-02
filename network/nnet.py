#%%
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
#%%
class residual_block(keras.layers.Layer):

    def __init__(self,kernel_size,filters):
        super().__init__()
        self.layers = [
            layers.Conv2D(filters,kernel_size,padding = "same"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(filters,kernel_size,padding = "same"),
            layers.BatchNormalization(),
        ]

    def call(self,inputs):
        x = inputs 
        for i in self.layers:
            x = i(x)
        x = layers.Add()([x,inputs])
        return layers.ReLU()(x)


def create_c4_model(board_shape, filters, residual_layers, kernel_size, action_size):
    board_input = layers.Input(shape=board_shape)
    board_input = layers.Reshape((board_shape[0],board_shape[1],1))(board_input)
    conv1 = layers.Conv2D(filters,kernel_size,padding = "same")(board_input)
    relu1 = layers.ReLU()(conv1)
    x = layers.BatchNormalization()(relu1)
    residual_blocks = [residual_block(kernel_size,filters) for i in range(residual_layers)]
    for l in residual_blocks:
        x = l(x)

    #Policy network
    pi_conv1 = layers.Conv2D(2,(1,1))(x)
    pi_bn1 = layers.BatchNormalization()(pi_conv1)
    pi_relu1 = layers.ReLU()(pi_bn1)
    pi_flat = layers.Flatten()(pi_relu1)
    pi = layers.Dense(action_size,activation="softmax",name="pi")(pi_flat)

    #Value network
    v_conv1 = layers.Conv2D(1,(1,1))(x)
    v_bn1 = layers.BatchNormalization()(v_conv1)
    v_relu1 = layers.ReLU()(v_bn1)
    v_flat = layers.Flatten()(v_relu1)
    v_hidden = layers.Dense(256,activation="relu")(v_flat)
    v = layers.Dense(1,activation="tanh",name="v")(v_hidden)

    model = keras.Model(inputs = board_input,outputs = [pi,v])
    model.compile(optimizer = keras.optimizers.Adam(),loss = [keras.losses.categorical_crossentropy,keras.losses.mean_squared_error])
    return model


class nnet():

    def __init__(self, board_shape, residual_layers, filters,kernel_size):
        self.net = create_c4_model(board_shape,filters,residual_layers,kernel_size,board_shape[0])
        self.board_shape =board_shape
        self.action_size = board_shape[0]
        self.filters = filters
        self.layers = layers
    
    def pi(self,board):
        b = np.expand_dims(board,1)
        p = self.net.predict(b)[0]
        return p

    def value(self,board):
        b = np.expand_dims(board,1)
        v = self.net.predict(b)[1]
        return v


# %%
# %%
