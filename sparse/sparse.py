#needed to specific version 2.x when both tensorflow 1.x and 2.x are installed
# import sys
# sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages/')

# make sure these modules are on the search path
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops.nn import bias_add
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_array_ops


def shuffle_2d(nin, nout=None, din=8, dout=1, rseed=1):
  """Specify a sparsely-connected NN layer with uniform sparse connections in both input and output
  neurons.
  Example:
  ```python
  # as first layer in a sequential model:
  indices, cmat_shape, nout = shuffle_2d(nin=6, nout=3, dout=1)
 ```
  Arguments:
    nin: Positive integer, dimensionality of the input space.
    nout: Positive integer, dimensionality of the output space.
    din: degree of ingoing connections or number of connections per output neuron.
    dout: degree of outgoing connections or number of connections per input neuron.
    rseed, random seed used when specifying the sparse connections, default to None, i.e. the
      connections or sparse network topology. is specified randomly. Otherwise, a fixed or
      predefined network topology set by rseed will be used.

  Returns:
    A tuple of 3 elements: a two-column matrix mapping connections between input and output
    neurons, i.e. input index /t output index; a matrix with nin rows and dout columns,
    with each row specifying the output indices connecting to each of the input neuron; and the
    number of actual output neurons, usually nout and occassionally <nout due to dropout nodes.
  """

    if nout is not None:
        din=math.ceil(nin*dout/nout)
    else:
        nout=math.ceil(nin*dout/din)
    if nout>=nin:
        n1=nout
        n2=nin
    else:
        n1=nin
        n2=nout
    if rseed is not None:
        np.random.seed(rseed)
    a=np.arange(n2)
    m=math.ceil(n1/n2)
    a1=[np.random.choice(a, n2, replace=False) for j in range(m)]
    newi=[np.roll(a1[j], -i) for j in range(m) for i in range(n2)]
    newi=np.stack(newi, axis=0)
    ridx=np.random.choice(m*n2, n1, replace=False)
    newi1=newi[ridx]
    ncon=nin*dout
    newi1=newi1.T.reshape([-1,])[:ncon]#np.transpose(newi1).reshape([-1,])
    oldi=np.tile(np.arange(n1),n2)[:ncon]
    if nout<nin:
        imat=np.stack([oldi,newi1], axis=1)
    else:
        imat=np.stack([newi1,oldi], axis=1)
    nshape=[nin,dout]
    nn=nin
    nn1=np.max(newi) + 1
    return (imat, nshape, nn1)  


def shuffle_1d(nin, nout=None, din=8, dout=1, rseed=1):
  """Specify a sparsely-connected NN layer with uniform sparse connections in input neurons and
  random sparse connections in output neurons. 
  Example:
  ```python
  # as first layer in a sequential model:
  indices, cmat_shape, nout = shuffle_1d(nin=6, nout=3, dout=1)
 ```
  Arguments:
    nin: Positive integer, dimensionality of the input space.
    nout: Positive integer, dimensionality of the output space.
    din: degree of ingoing connections or number of connections per output neuron.
    dout: degree of outgoing connections or number of connections per input neuron.
    rseed, random seed used when specifying the sparse connections, default to None, i.e. the
      connections or sparse network topology. is specified randomly. Otherwise, a fixed or
      predefined network topology set by rseed will be used.

  Returns:
    A tuple of 3 elements: a two-column matrix mapping connections between input and output
    neurons, i.e. input index /t output index; a matrix with nin rows and dout columns,
    with each row specifying the output indices connecting to each of the input neuron; and the
    number of actual output neurons, usually nout and occassionally <nout due to dropout nodes.
  """

    if nout is not None:
        din=math.ceil(nin*dout/nout)
    else:
        nout=math.ceil(nin*dout/din)
    m=nin#math.ceil(nin/nout)
    a=np.arange(nout)
    if rseed is not None:
        np.random.seed(rseed)
    newi=[np.random.choice(a, nout, replace=False) for j in range(m)]
    # newi=[np.roll(a1[j], -i) for j in range(m) for i in range(nout)]
    newi=np.stack(newi, axis=0)
    # ridx=np.random.choice(m*nout, nin, replace=False)
    newi1=newi[:,:dout]
    newi1=np.transpose(newi1).reshape([-1,])
    oldi=np.repeat(np.arange(nin),dout)
    # print((newi1.shape, oldi.shape))
    imat=np.stack([oldi,newi1], axis=1)
    nshape=[nin,dout]
    nn=nin
    nn1=np.max(newi) + 1
    return (imat, nshape, nn1)


def random_2d(nin, nout=None, din=8, dout=1, rseed=1):
  """Specify a sparsely-connected NN layer with random sparse connections in both input and output
  neurons.
  Example:
  ```python
  # as first layer in a sequential model:
  indices, cmat_shape, nout = shuffle_2d(nin=6, nout=3, dout=1)
 ```
  Arguments:
    nin: Positive integer, dimensionality of the input space.
    nout: Positive integer, dimensionality of the output space.
    din: degree of ingoing connections or number of connections per output neuron.
    dout: degree of outgoing connections or number of connections per input neuron.
    rseed, random seed used when specifying the sparse connections, default to None, i.e. the
      connections or sparse network topology. is specified randomly. Otherwise, a fixed or
      predefined network topology set by rseed will be used.

  Returns:
    A tuple of 3 elements: a two-column matrix mapping connections between input and output
    neurons, i.e. input index /t output index; a matrix with nin rows and dout columns,
    with each row specifying the output indices connecting to each of the input neuron; and the
    number of actual output neurons, usually nout and occassionally <nout due to dropout nodes.
  """

    if nout is not None:
        din=math.ceil(nin*dout/nout)
    else:
        nout=math.ceil(nin*dout/din)
    m=int(round(nin*dout))
    a=np.arange(nout)
    b=np.arange(nin)
    oldi=np.repeat(b,nout)
    newi=np.tile(a,nin)
    if rseed is not None:
        np.random.seed(rseed)
    imat=np.stack([oldi,newi], axis=1)
    ridx=np.random.choice(nin*nout, m, replace=False)
    imat=imat[ridx,:]
    nshape=[nin,dout]
    nn=nin
    nn1=np.max(newi) + 1
    return (imat, nshape, nn1)


class sparse(Layer):
  """The sparsely-connected NN layer, just like the regular Dense layer in TensorFlow.
  `sparse` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).
  Note: currently, we do not allow the input to the layer has a rank greater than 2, 
  input has dimensions `(batch_size, d)`, but not `(batch_size, d0, d1)`. Input rank greater than 2 will be supported soon if neeeded.
  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  Example:
  ```python
  # as first layer in a sequential model:
  model = Sequential()
  model.add(sparse(16, dout=1, input_shape=(64,)))
  # now the model will take as input arrays of shape (*, 64)
  # and output arrays of shape (*, 16), i.e. each input only connects to 1 of 16 output units
  # after the first layer, you don't need to specify
  # the size of the input anymore, and you can either continue with sparse layer:
  model.add(sparse(4, dout=2))
  # or you can do Dense layer
  model.add(Dense(4))
 ```
  Arguments:
    units: Positive integer, dimensionality of the output space.
    din: degree of ingoing connections or number of connections per output neuron.
    dout: degree of outgoing connections or number of connections per input neuron.
    density: connection density, i.e. the ratio of present connections over all possible connections.
      This argument is redundant with din and dout, default to be None. When specified, the latter two will be overridden. 
    share_kernel: whether the same weight is used for all connections of a input neuron.
    weight_type: the type of tensor used for the kernel or weight matrix, 1 sparse tensor and 2 regular dense tensor.
    confun: connection pattern specifying function, i.e. "shuffle_2d", "shuffle_1d", or random_2d
      corresponding to uniform sparse connection in both input and output layers, uniform sparse
      connection in input but random sparse in output layer, or random sparse connection for for
      both input and output layers.
    rseed, random seed used in confun when specifying the sparse connections, default to None, i.e.
      the connections or sparse network topology. is specified randomly. Otherwise, a fixed or
      predefined network topology set by rseed will be used.
    
    All the remaining arguments are the same as in Dense layer.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    (not implemented yet) N-D tensor with shape: `(batch_size, ..., input_dim)`.
    Currently implemented: 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    (not implemented yet) N-D tensor with shape: `(batch_size, ..., units)`.
    Currently implemented: for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

    def __init__(self,
                 units=None,
                 din=8,
                 dout=1,
                 density=None,
                 share_kernel=False,
                 weight_type=None,
                 confun="shuffle_2d",
                 rseed=None,
                 activation=None,
                 use_bias=True, 
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.units=units
        self.din=din
        self.dout=dout
        self.density=density
        self.share_kernel=share_kernel and (confun is not "random_2d") and (density is None)
        self.weight_type=weight_type
        self.confun=confun
        self.rseed=rseed
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        activity_regularizer = regularizers.get(activity_regularizer)
        
        super(sparse, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
      

    def build(self, input_shape):
        if (self.density is not None) and (self.units is not None):
            self.dout=math.ceil(self.units*self.density)
            ncon=int(round(input_shape[1]*self.units*self.density))
            kernel_shape=(ncon, 1)
        else:
            kernel_shape=None
        # old_rs=np.random.get_state()
        confun=globals()[self.confun]#eval(self.confun)#random_2d#shuffle_1d#
        condata=confun(input_shape[1], nout=self.units, din=self.din, dout=self.dout, rseed=self.rseed)
        # np.random.set_state(old_rs)
        indices, self.nshape, self.nn1=condata
        self.nn = self.nshape[0]
        if self.units is not None:
            self.nn1=self.units
        if kernel_shape is None:
            kernel_shape=self.nshape
            ncon=self.nshape[0]*self.nshape[1]
            if self.share_kernel: 
                kernel_shape=(self.nshape[0], 1)
        npars=kernel_shape[0]*kernel_shape[1]#self.nn*self.dout
        if (self.weight_type == None and npars<=50000) or self.share_kernel:
          self.weight_type = 1
        if self.weight_type == None and npars>50000:
          self.weight_type = 2
        print("weight_type used: ", self.weight_type)
        if self.weight_type==1:#elf.dout/self.nn1 < 0.7:
            self.kernel = self.add_weight(
                                      name="kernel",
                                      shape=[npars],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
            self.indices = tf.constant(indices[:,1::-1])[:ncon]#[:npars]
        else: #if self.dout/self.nn1 >= 0.3:
            self.mask=tf.Variable(lambda : tf.zeros(shape=(self.nn, self.nn1)), trainable=False)
            self.indices = tf.constant(indices)[:ncon]#[:npars]
            tf.compat.v1.scatter_nd_update(self.mask, self.indices, tf.ones(shape=(kernel_shape[0]*kernel_shape[1])))
            self.wmat = self.add_weight(
                                    name="kernel",
                                    shape=(self.nn, self.nn1),#[kernel_shape[0]*kernel_shape[1]],
                                    initializer=self.kernel_initializer,#tf.initializers.Zeros,#
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
            #not work: self.wmat=tf.multiply(self.wmat, self.mask)
            tf.compat.v1.assign(self.wmat, tf.multiply(self.wmat, self.mask))

        if self.use_bias:
            self.bias = self.add_weight(
                                        name='bias',
                                        shape=[self.nn1,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        super(sparse, self).build(input_shape)


    def call(self, inputs):
        nn = self.nn #len(self.newi)
        nn1 = self.nn1 # max(self.newi) + 1
        nj = self.dout
        
        if self.weight_type==1:#nj/nn1 < 0.7: 
            if self.share_kernel: 
                k1=ragged_array_ops.tile(self.kernel, [nj])
            else:
                k1=self.kernel
            self.wmat=tf.sparse.SparseTensor(indices=self.indices, values=k1, dense_shape=[self.nn1, self.nn])
            outputs = tf.transpose(tf.sparse.sparse_dense_matmul(self.wmat, tf.transpose(inputs)))
        else:
            outputs = math_ops.matmul(inputs, tf.multiply(self.wmat,self.mask))#self.wmat)#            

        if self.use_bias:
            outputs = bias_add(outputs, self.bias)            
        if self.activation is not None:
            return self.activation(outputs) #
        return outputs
  
  
    def compute_output_shape(self, input_shape):
        # return (self.nn1, input_shape[2])
        return input_shape[:-1].concatenate(self.nn1)


    def get_config(self):
        config = {
            'units': self.units,
            'din': self.din, 
            'dout': self.dout,
            'density': self.density,
            'share_kernel': self.share_kernel,
            'weight_type': self.weight_type,
            'confun': self.confun,
            'rseed': self.rseed,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(sparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
dense=tf.keras.layers.Dense
def model_generator(layers=["sparse", "dense"], 
                    largs=[None, None],
                    inshape=(28,28), 
                    nout=10,  
                    droprate=0.2, 
                    l2rate=1e-3,
                    do_compile=True, 
                    optimizer='Nadam', 
                    lr=2e-3, 
                    dtype=None, 
                    # indim=None, 
                    conv=False):
  """Generate NN model with sparse or dense layer(s).
  Example:
  ```python
  ns=12000
  mnist = tf.keras.datasets.mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train, X_test = X_train / 255.0, X_test / 255.0
  X_train, y_train = X_train[:ns].reshape((ns,-1)), y_train[:ns]
  X_test=X_test.reshape((X_test.shape[0],-1))
  largs1={"units": 64, "dout": 16, "confun": "shuffle_2d", "activation": "relu"}
  largs2={"units": 10, "activation": "softmax"}
  layers=["sparse", "dense"]
  mod=model_generator1(layers=layers, largs=[largs1, largs2], inshape=(784,), 
    optimizer='Nadam', lr=2e-3)
  hist=mod.fit(X_train, y_train, validation_data=(X_test, y_test))
  ```
  Arguments:
    layers: a list of sparse or dense layers, element value can be "sparse" or "dense".
    largs: a list of dictionaries specifying the layer settings, each dictionary contains the
    argument names and values to use when calling sparse() or Dense() function. See Examples for
    more.
    inshape: the shape of input data without the first dimension (or sample or batch size).
    nout: Positive integer, dimensionality of the output space.
    droprate: dropout rate when dropout is used, positive real value between 0 and 1. 
    l2rate: l2 rate when the regularization is used, float. 
    do_compile: boolean, whether compile the model or  not.
    optimizer: String (name of optimizer) or optimizer instance. See tf.keras.optimizers. 
    lr: A Tensor or a floating point value. The learning rate. 
    dtype: Optional datatype of the input. When not provided, the Keras default float type will be 
      used.
    conv: boolean, whether to add convoluational layers for two-dimensional imput data.

  Returns:
    A NN model with sparse or dense layer(s) built and compiled as specified.
  """


    nlayers=len(layers)
    if (type(droprate) is float):
        droprate = [droprate]*(nlayers-1)
    elif (type(droprate) is list) and (len(droprate)==1):
        droprate = droprate*(nlayers-1)

    # for i in range(nlayers):
    #     if layers[i]=="dense":
    #           largs[i]={"units": largs[i]["units"], "activation": largs[i]["activation"]}

    input=tf.keras.layers.Input(shape=inshape, dtype=dtype)
    if len(inshape)>1 and not conv:
        x=tf.keras.layers.Flatten()(input)
    elif len(inshape)>1:
        x=tf.keras.layers.Reshape([inshape[0], inshape[1], 1], input_shape=inshape)(input)
        x=tf.keras.layers.Conv2D(32, kernel_size=4, padding="SAME", activation="relu")(x)
        x=tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x=tf.keras.layers.Conv2D(64, kernel_size=4, padding="SAME", activation="relu")(x)
        x=tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x=tf.keras.layers.Conv2D(64*2, kernel_size=4, padding="SAME", activation="relu")(x)
        x=tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x=tf.keras.layers.Flatten()(x)
    # elif len(inshape)<1:
    #     hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=False)#True)
    #     if type(indim)==type(None):
    #         x=hub_layer(input)
    #     elif type(indim)==int:
    #         x=hub_layer(input)[:,:indim]
    #     else:
    #         indim=tf.convert_to_tensor(indim, dtype=tf.int32)
    #         x=tf.gather(hub_layer(input), indim, axis=1) #hub_layer(input)hub_layer(input)[:,indim]
    else:
        x=input
    for i in range(nlayers-1):
        if layers[i]=="dense":
              largs[i]={"units": largs[i]["units"], "activation": largs[i]["activation"], 
                        "kernel_regularizer": largs[i]["kernel_regularizer"]}
        x=globals()[layers[i]](**largs[i])(x)
        x=tf.keras.layers.Dropout(droprate[i])(x)
    if layers[-1]=="dense":
        largs[-1]={"units": largs[-1]["units"], "activation": largs[-1]["activation"],
                   "kernel_regularizer": largs[-1]["kernel_regularizer"]}
    output=globals()[layers[-1]](**largs[-1])(x)#, kernel_regularizer=tf.keras.regularizers.l2(l2rate))(x)
    model=tf.keras.models.Model(input, output)

    if do_compile:
      optimizer = optimizers.get({"class_name": optimizer,
                               "config": {"learning_rate": lr}})
      model.compile(optimizer=optimizer,
                    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model



