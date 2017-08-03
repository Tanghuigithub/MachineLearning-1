# TensorFlow

TensorFlow是一个开源 C ++ / Python 软件库，用于使用数据流图的数值计算，尤其是深度神经网络。它是由谷歌创建的。在设计方面，它最类似于 Theano，但比  Caffe 或 Keras 更低级。


## 第一步很重要：session

像是一个容器一样，创建之后使用，使用完就close，这货还是个context manager

```
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(c))
```

ConfigProto

```
# Launch the graph in a session that allows soft device placement and
# logs the placement decisions.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))
```

## 第二步：在tensorflow中调用Keras层

```
from keras import backend as K
K.set_session(sess)
```

### 上下文管理

Use with the `with` keyword to specify that calls to `Operation.run()` or `Tensor.eval()` should be executed in this session.

```
with sess.as_default():
  assert tf.get_default_session() is sess
  print(c.eval())
```
## 第三步：用tensorflow构建模型 **Placeholders**


为输入图像和目标输出类创建节点，从而构造计算图：
```
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

这儿的`x`和`y_`只是占个位子，并不是实值，只有在用Tensorflow计算时才有值。

- 输入图像`x`为浮点数组成的二维张量（tensor）,`shape`是它的形状，其中784=28*28像素的一维向量，而`None`指的是`batch size`，可以是任意数量。 
- 目标输出`y_`类似，只不过这儿的一位向量是one-hot代表的分类结果。就像这里以mnist为例，是10个数字。

**placeholder**的`shape`参数可选项, but it allows TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes.

返回：A **Tensor** that may be used as a **handle** for **feeding a value**, but not evaluated directly.

### 共享变量

#### get_variable()

根据`reuse`这个flag分两种情况

#### variable_scope

`tf.variable_scope(<scope_name>)`: 通过 `tf.get_variable()`为变量名指定命名空间。有点类似目录的概念

```
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

## LSTM

```
lstm = rnn_cell.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)
```

A simplified version of the code for the graph creation for truncated backpropagation:
```
# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = rnn_cell.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
```

### Dynamic RNN decoder
for a sequence-to-sequence model specified by RNNCell and decoder function.
is similar to the tf.python.ops.rnn.dynamic_rnn as the decoder does not make any assumptions of sequence length and batch size of the input.

### Input

```
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```