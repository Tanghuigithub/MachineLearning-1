# keras

 Keras是一个极度简化、高度模块化的神经网络第三方库。基于Python+Theano开发，充分发挥了GPU和CPU操作。其开发目的是为了更快的做神经网络实验。适合前期的网络原型设计、支持卷积网络和反复性网络以及两者的结果、支持人工设计的其他网络、在GPU和CPU上运行能够无缝连接。

## 怎么保存Keras模型？


### PyYAML==3.11
 YAML is a data serialization format designed for human readability and interaction with scripting languages.
 
 如果只保存模型结构，代码如下：
 
 ```python
 # save as JSON  
json_string = model.to_json()  
# save as YAML  
yaml_string = model.to_yaml()  
# model reconstruction from JSON:  
from keras.modelsimport model_from_json  
model = model_from_json(json_string)  
   
# model reconstruction from YAML  
model = model_from_yaml(yaml_string)  
 ```
### h5py==2.5.0
h5py：将数据储存在hdf5文件中。
如果需要保存数据：
```python
model.save_weights('my_model_weights.h5')  
model.load_weights('my_model_weights.h5')  
```
    sudo apt-get install libhdf5-dev
    sudo apt-get install python-h5py

## Layer

layers模块包含了core、convolutional、recurrent、advanced_activations、normalization、embeddings这几种layer。

其中core里面包含了flatten(CNN的全连接层之前需要把二维特征图flatten成为一维的)、reshape（CNN输入时将一维的向量弄成二维的）、dense(就是隐藏层，dense是稠密的意思),还有其他的就不介绍了。convolutional层基本就是Theano的Convolution2D的封装。

- Convolution1D
    - `nb_filter`: Number of convolution kernels to use (dimensionality of the output). 
    - `filter_length`: The extension (spatial or temporal) of each filter.
- MaxPooling1D
    - `pool_length`: factor by which to downscale. 2 will halve the input. 
    
### core

- Dense
- Dropout
- Flatten
- Lambda层
  - keras.layers.core.Lambda(function, output_shape=None, arguments={})
### Lambda层

本函数用以对上一层的输入实现任何Theano/TensorFlow表达式

- function：要实现的函数，该函数仅接受一个变量，即上一层的输出
- output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
- arguments：可选，字典，用来记录向函数中传递的其他关键字参数
全连接网络

# Preprocessing

这是预处理模块，包括序列数据的处理，文本数据的处理，图像数据的处理。重点看一下图像数据的处理，keras提供了ImageDataGenerator函数,实现data augmentation，数据集扩增，对图像做一些弹性变换，比如水平翻转，垂直翻转，旋转等。

## Sequential

```
from keras.models import Sequential

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

增加layer：

```
model = Sequential()
 ngram_layer.add(Convolution1D(
            NB_FILTER,
            ngram_length,
            input_dim=embedding_size,
            input_length=SAMPLE_LENGTH,
            init='lecun_uniform',
            activation='tanh',
        ))
```
第一层需要知道输入的形状

#### Compilation

- an `optimizer`. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class. See: optimizers.
- a loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: objectives.
- a list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric (only accuracy is supported at this point), or a custom metric function.
```python
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        class_mode='binary',
    )
       
```

# 优化器(Optimizer)

机器学习包括两部分内容，一部分是如何构建模型，另一部分就是如何训练模型。训练模型就是通过挑选最佳的优化器去训练出最优的模型。
        Keras包含了很多优化方法。比如最常用的随机梯度下降法(SGD)，还有Adagrad、Adadelta、RMSprop、Adam等,一些新的方法以后也会被不断添加进来。
```
keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.9, nesterov=False)
```

上面的代码是SGD的使用方法，lr表示学习速率,momentum表示动量项，decay是学习速率的衰减系数(每个epoch衰减一次),Nesterov的值是False或者True，表示使不使用Nesterov momentum。其他的请参考文档。
        
## SGD（随机梯度下降优化器，性价比最好的算法）

keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
     参数：
lr :float>=0，学习速率
momentum :float>=0 参数更新的动量
decay : float>=0 每次更新后学习速率的衰减量
nesterov :Boolean 是否使用Nesterov动量项

## Adagrad（参数推荐使用默认值） 

keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
     参数：
lr : float>=0，学习速率
epsilon :float>=0

# 目标函数(Objective)

这是目标函数模块，keras提供了mean_squared_error，mean_absolute_error ，squared_hinge，hinge，binary_crossentropy，categorical_crossentropy这几种目标函数。

这里binary_crossentropy 和 categorical_crossentropy也就是logloss
```
model.compile(loss='mean_squared_error', optimizer='sgd')
```  
这段代码已经见过很多次了。可以通过传递一个函数名。也可以传递一个为每一块数据返回一个标量的Theano symbolic function。而且该函数的参数是以下形式：
y_true : 实际标签。类型为Theano tensor
y_pred: 预测结果。类型为与y_true同shape的Theanotensor
        其实想一下很简单，因为损失函数的作用就是返回预测结果与实际值之间的差距。然后优化器根据差距进行参数调整。不同的损失函数之间的区别就是对这个差距的度量方式不

- mean_squared_error mse均方误差，常用的目标函数，公式为：
$
(y_{pred}-y_{true})^2.mean(axis=-1)
$
就是把预测值与实际值差的平方累加求均值。
- mean_absolute_error / mae绝对值均差，公式为
$|y_{pred}-y_{true}|.mean(axis=-1)$
就是把预测值与实际值差的绝对值累加求和。
- mean_absolute_percentage_error / mape公式为：
$
|\frac {y_{true} - y_{pred}}{ clip((|y_true|),\epsilon, infinite)|.mean(axis=-1)} * 100
$，和mae的区别就是，累加的是（预测值与实际值的差）除以（剔除不介于epsilon和infinite之间的实际值)，然后求均值。
mean_squared_logarithmic_error / msle公式为： (log(clip(y_pred, epsilon, infinite)+1)- log(clip(y_true, epsilon,infinite)+1.))^2.mean(axis=-1)，这个就是加入了log对数，剔除不介于epsilon和infinite之间的预测值与实际值之后，然后取对数，作差，平方，累加求均值。
squared_hinge公式为：(max(1-y_true*y_pred,0))^2.mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的平方的累加均值。
hinge公式为：(max(1-y_true*y_pred,0)).mean(axis=-1)，取1减去预测值与实际值乘积的结果与0比相对大的值的的累加均值。
binary_crossentropy:常说的逻辑回归.
categorical_crossentropy:多分类的逻辑回归注意：using this objective requires that your labels are binary arrays ofshape (nb_samples, nb_classes).

# 激活函数(Activation)

这是激活函数模块，keras提供了linear、sigmoid、hard_sigmoid、tanh、softplus、relu、softplus，另外softmax也放在Activations模块里(我觉得放在layers模块里更合理些）。此外，像LeakyReLU和PReLU这种比较新的激活函数，keras在keras.layers.advanced_activations模块里提供。

## 自定义激活函数

可以采用以下几种方法：

from keras import backend as K
from keras.layers.core import Lambda
from keras.engine import Layer

### 使用theano/tensorflow的内置函数简单地编写激活函数
```
def sigmoid_relu(x):
    """
    f(x) = x for x>0
    f(x) = sigmoid(x)-0.5 for x<=0
    """
    return K.relu(x)-K.relu(0.5-K.sigmoid(x))
```

### 使用Lambda

```
def lambda_activation(x):
    """
    f(x) = max(relu(x),sigmoid(x)-0.5)
    """
    return K.maximum(K.relu(x),K.sigmoid(x)-0.5)
```

### 编写自己的layer（这个例子中参数theta,alpha1,alpha2是固定的超参数，不需要train，故没定义build）

```
class My_activation(Layer):
    """
    f(x) = x for x>0
    f(x) = alpha1 * x for theta<x<=0
    f(x) = alpha2 * x for x<=theta
    """
    def __init__(self,theta=-5.0,alpha1=0.2,alpha2=0.1,**kwargs):
        self.theta = theta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super(My_activation,self).__init__(**kwargs)
    def call(self,x,mask=None):
        fx_0 = K.relu(x) #for x>0
        fx_1 = self.alpha1*x*K.cast(x>self.theta,K.floatx())*K.cast(x<=0.0,K.floatx()) #for theta<x<=0
        fx_2 = self.alpha2*x*K.cast(x<=self.theta,K.floatx())#for x<=theta
        return fx_0+fx_1+fx_2
    def get_output_shape_for(self, input_shape):
        #we don't change the input shape
        return input_shape
#alpha1,alpha2是可学习超参数
class Trainable_activation(Layer):
    """
    f(x) = x for x>0
    f(x) = alpha1 * x for theta<x<=0
    f(x) = alpha2 * x for x<=theta
    """
    def __init__(self,init='zero',theta=-5.0,**kwargs):
        self.init = initializations.get(init)
        self.theta = theta
        super(Trainable_activation,self).__init__(**kwargs)
    def build(self,input_shape):
        self.alpha1 = self.init(input_shape[1:],name='alpha1')#init alpha1 and alpha2 using ''zero''
        self.alpha2 = self.init(input_shape[1:],name='alpha2')
        self.trainable_weights = [self.alpha1,self.alpha2]
    def call(self,x,mask=None):
        fx_0 = K.relu(x) #for x>0
        fx_1 = self.alpha1*x*K.cast(x>self.theta,K.floatx())*K.cast(x<=0.0,K.floatx()) #for theta<x<=0
        fx_2 = self.alpha2*x*K.cast(x<=self.theta,K.floatx())#for x<=theta
        return fx_0+fx_1+fx_2
    def get_output_shape_for(self, input_shape):
        #we don't change the input shape
        return input_shape
```

使用时：

```
model.add(Activation(sigmoid_relu))
model.add(Lambda(lambda_activation))
model.add(My_activation(theta=-5.0,alpha1=0.2,alpha2=0.1))
model.add(Trainable_activation(init='normal',theta=-5.0))
```

# 初始化（Initializations）

这是参数初始化模块，在添加layer的时候调用init进行初始化。keras提供了uniform、lecun_uniform、normal、orthogonal、zero、glorot_normal、he_normal这几种。

## callbacks

### ModelCheckpoint

```
keras.callbacks.ModelCheckpoint(filepath,verbose=0, save_best_only=False)  
```
用户每次epoch之后保存模型数据。如果save_best_only=True，则最近验证误差最好的模型数据会被保存下来。filepath是由epoch和logs的键构成的。比如filepath=weights.{epoch:02d}-{val_loss:.2f}.hdf5，那么会保存很多带有epoch和val_loss信息的文件；当然也可以是某个路径。
## history

Returns a history object. Its `history` attribute is a record of
        training loss values at successive epochs,
        as well as validation loss values (if applicable).

# Arguments
            X: data, as a numpy array.
            y: labels, as a numpy array.
            batch_size: int. Number of samples per gradient update.
            nb_epoch: int.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: `keras.callbacks.Callback` list.
                List of callbacks to apply during training.
                See [callbacks](callbacks.md).
            validation_split: float (0. < x < 1).
                Fraction of the data to use as held-out validation data.
            validation_data: tuple (X, y) to be used as held-out
                validation data. Will override validation_split.
            shuffle: boolean or str (for 'batch').
                Whether to shuffle the samples at each epoch.
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
            show_accuracy: boolean. Whether to display
                class accuracy in the logs to stdout at each epoch.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: list or numpy array of weights for
                the training samples, used for scaling the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
