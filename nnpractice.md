# 第3章 神经网络编程

---

## Theano

### 机器学习中矩阵约定

行是水平的列是垂直的。每一行是一个实例。因此，输入[10,5]是一个10个实例，每个实例的维数是5的一个矩阵。如果这个成为一个神经网络的输入，那么从输入到第一隐藏层的权重将表示一个矩阵的大小（5,#hid）。

```
>>> numpy.asarray([[1.,2],[3,4],[5,6]])

array([[1.,
2.],


[3.,
4.],


[5.,
6.]])

>>> numpy.asarray([[1.,
2],[3,4],[5,6]]).shape

(3,2)
```

在Theano中，所有的符号必须是有**类型**的。特别地，`T.dscalar`是我们分配给双精度（doubles）的"0-维"数组（标量）的类型。它是一个Theano类型

## ANN

```
class Network():
    def init(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]] #
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
```
- sizes=[2,3,1]
- 权值$w$和偏置$b$是需要初始化,梯度下降算法是在某个点在梯度方向上开始不断迭代计算最优的w和b，所以w,b必须有一个初始值作为起始迭代点。
随机地初始化它们，我们调用了numpy中的函数生成符合**高斯分布**的数据。
```python
        # Either we have a random seed or the WTS for each layer from a
        # previously trained NeuralNet
        if allwts is None:
            self.rand_gen = np.random.RandomState(training_params['SEED'])
        else:
            self.rand_gen = None
```
其次这里的w，b表示成向量形式，原因是矢量化编程可以在线性代数库中加快速度，那么到底该怎么表示w，和b呢？让我们从最简单的问题开始，看看最简单的单个神经元：
![此处输入图片的描述][1]

### 激活函数

```
def feedforward(self,a):
for b,w in zip(self.biases,self.weights):
a=sigmoid_vec(np.dot(w,a)+b)
return a
```

```
def sigmoid(z):
return 1.0 / (1.0 + np.exp(-z))
sigmoid_vec=np.vectorize(sigmoid)
```

### 随机梯度下降算法

```
def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
```
随机梯度下降的思想不是迭代所有的训练样本，而是**挑一部分**出来来代表所有的训练样本，这样可以加快训练速度。换句话说就是，原始梯度下降算法是一个一个地训练，而SGD是**一批一批**地训练，而这个批的大小就是`mini_batch_size`，并且这个批是**随机**挑出来的，而等这些批都训练完了我们叫一次迭代，并且给了它一个更好听的名字叫`epochs`，`eta`是学习率$\eta$，`test_data`是测试数据。看代码！
```
if test_data: 
    n_test=len(test_data)
    n=len(training_data)
for j in xrange(epochs):#开始迭代
    random.shuffle(training_data)#为了是随机地挑选出来的批次，先调用这个函数扰乱训练样本，这样就可以制造随机了
    mini_batches=[training_data[k:k+mini_batch_size]
    for k in xrange(0,n,mini_batch_size)] #索引分批的数据
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch,eta) #根据规则更新权值w和偏置b
    if test_data:#在每次迭代结束时，我们都会利用测试数据来测试一下当前参数的准确率
        print "Epoch {0}:{1} / {2}".format(j,self.evaluate(test_data),n_test)
    else:
        print "Epoch {0} complete".format(j)
```
这里又用到了update_mini_batch方法，它直接把更新w和b的过程单独抽象了出来，这个函数是梯度下降的代码，而只有加上这个函数前面的代码才能叫随机梯度下降。过会我们再来分析这个函数，在迭代完一次（一个epoch）之后，我们使用了测试数据来检验我们的神经网络（已经根据mini-batch学习到了权值和偏置）的识别数字准确率。这里调用了一个函数叫evaluate，它定义为：
```
def evaluate(self,test_data):
    test_results=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
    return sum(int(x==y) for (x,y) in test_results)
```
这个函数的主要作用就是返回识别正确的样本个数。其中np.argmax(a)是返回a中最大值的下标，这里我们可以看到self.feedforward(x)，其中x是test_data中的输入图像，然后神经网络计算出最终的结果是分类数字的标签，它是一个shape=(1,10)矩阵，比如数字5表示为[0,0,0,0,0,1,0,0,0,0]，这时1最大，就返回下标5，其实也就表示了数字5。然后将这个结果和test_data中的y作比较，如果相等就表示识别正确，sum就是用来计数的。

### 梯度下降算法

$\Delta \nabla_b,\Delta \nabla_w, \nabla_b,\nabla_w$
下面来看看梯度下降的代码update_mini_batch:
```
def update_mini_batch(self,mini_batch,eta):
    nabla_b=[np.zeros(b.shape) for b in self.biases] #迭代的初值当然是0了，记住这和偏置b的初始值不一样额
    nabla_w=[np.zeros(w.shape) for w in self.weights]

for x,y in mini_batch:#扫描mini_batch中的每个样本，由于算的是平均梯度
    delta_nabla_b,delta_nabla_w=self.backprop(x,y) #反向传播计算梯度，每个样本(x,y)

nabla_b=[nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)] #计算一个mini_batch中的所有样本的梯度和，因为我们要算的是平均梯度

nabla_w=[nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)] #

self.biases=[b-(eta/len(mini_batch)) * nb for b,nb in zip(self.biases,nabla_b)]
```

### BP算法

原始梯度下降是针对每个样本都计算出一个梯度，然后沿着这个梯度移动。

而随机梯度下降是针对多个样本（一个mini_batch）计算出一个平均梯度，然后这个梯度移动，那么这个epochs就是权值和偏置更新的次数了。

随机梯度下降学习算法的重点又在这个BP算法了：
```
def backprop(self,x,y):#它是用来计算单个训练样本的梯度的
    nabla_b=[np.zeros(b.shape) for b in self.biases] #梯度值初始化为0
    nabla_w=[np.zeros(w.shape) for w in self.weights]
```
#### 前向计算

```
activation=x
activations=[x] #存储所有的激活，一层一层的
zs=[]#存储所有的z向量，一层一层的，这个z是指每一层的输入计算结果（z=wx+b）
for b,w in zip(self.biases,self.weights):
    z=np.dot(w,activation) + b
    zs.append(z)
activation=sigmoid_vec(z)
activations.append(activation)
```

#### 后向传递

```
delta=self.cost_derivative(activations[-1],y) * sigmoid_prime_vec(zs[-1])
```
因为这里的$C=\frac12\sum_j(y_j-a_j)^2$,（单个训练样本的损失函数），所以$\frac{\partial C}{\partial y_j^L}=a_j-y_j$
于是下面的cost_derivative就直接返回了这个式子。

这里我们使用了2个辅助函数：
```
def cost_derivative(self,output_activations,y):
    return output_activations – y

def sigmoid_prime(z): #sigmoid函数的导数
    return sigmoid(z) * (1-sigmoid(z))
```
其中zs[-1]表示最后一层神经元的输入，上述delta对应BP算法中的式子：
$
\delta_j^L=\frac{\partial C}{\partial y_j^L}\sigma'z_j^L
$
delta就是指最后一层的残差。代码接下来：
```
nabla_b[-1]=delta #对应式子BP3：
```
$\frac{\partial C}{\partial b_j^L}=\delta_j^l$
```
nabla_w[-1]=np.dot(delta,activations[-2].transpose())#对应式子BP4：
```
$
\frac{\partial C}{\partial a_{jk}^L}=a_k^{l-1}\delta_j^l
$
以上算的是最后一层的相关变量。下面是反向计算前一层的梯度根据最后一层的梯度。
```
for l in xrange(2,self.num_layers):
z=zs[-l]
spv=sigmoid_prime_vec(z)
delta=np.dot(self.weights[-l+1].transpose(),delta) * spv
nabla_b[-l]=delta
nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
return (nabla_b,nabla_w)
```
这便是BP算法的全部了。详细代码请看这里的[network.py][2]。
## 关于纹理基元和特征抽取

### CIELAB
![此处输入图片的描述][3]
CIE L*a*b*（CIELAB）是慣常用來描述人眼可見的所有顏色的最完備的色彩模型，具有視覺上的均勻性。名稱是由國際照明委員會（Commission Internationale d'Eclairage的首字母是CIE)提出的。

### SLIC Superpixel 超像素
![此处输入图片的描述][4]
使像素聚类为五维颜色和图像层，用来生成简洁整齐的超像素,对像素迭代聚类
$(L,A,B)+(x,y)$
CIELAB color values
It is not possible to simply use the Euclidean distance in this 5D space without normalizing the spatial distances.
In order to cluster pixels in this 5D space, we therefore introduce a new distance measure that considers superpixel size. Using it, we enforce color similarity as well as pixel proximity in this 5D space such that the expected cluster sizes and their spatial extent are approximately equal.
#### 1. 距离测量方法

输入一个超像素数目的参数$K$。那么对于一张$N$个像素的图像来说，每个超像素大小约为$\frac{N}{K}$个像素。那么，每两个相邻的超像素块之间的距离为$s=\sqrt{\frac{N}{K}}$。
算法开始时，我们选择聚类的中心$C_k =\{l_k,a_k,b_k,x_k,y_k\}$,$k$属于$[1,K]$。每个超像素的面积大约为$s^2$（近似于超像素的面积）。我们可以安全地假设：像素在聚类中心的$(2s)^2$范围内。这个范围就是每个聚类中心的**搜寻范围**。

**欧式距离**在**亮度**空间内小距离是很有意义的。如果空间像素距离超过了颜色距离，那么图像上的边界不再有效（文章的意思是这样是不对的）。因此，在5D空间中取代简单的欧式距离，我们采用$d_s$如下：
$
d_s=d_{lab}+\frac{m}{s}d_{xy}
$
其中$d_s$是$lab$距离和归一化后的$xy$距离之和。其中变量$m$用来控制超像素的紧密度。$m$的值在$[1,20]$之间。在接下来的文章中，我们统一选择$m=10$。这个数值既能在感官经验上满足颜色距离最大化，又能很好的在**颜色相似度**和**空间相似度**的平衡。

### Histogram intersection(直方图交叉核,Pyramid Match Kernel)

假设图像或其他数据的特征可以构成直方图，根据直方图间距的不同可以得到多种类型的直方图：

![此处输入图片的描述][5]
 a)里的y和z代表两种数据分布，三幅图代表三层金字塔，每一层里有间距相等的虚线，意思和我之前说的2cm，4cm的宽度一样。可以看到红点蓝点的位置是固定的，但是根据直方图宽度的不同可以划到不同的直方图里，如(b)所示。(c)图就是L的计算结果，是通过(b)里两种直方图取交集得来的，不过直方图的高度忽略不计，只计算交集后的数目，(c)图每个图的下方都给出了交集数目，比如x0=2,x1=4,x2=3（原图里是5，是不是错了？）。
L得到了，就算N就是通过，也就是通过Ni=Li-Li-1得到（看公式是能取负数的，比如上图里的N0=2，N1=2，N2=-1）。



  [1]: http://www.gumpcs.com/wp-content/uploads/2015/07/070915_0246_1.png
  [2]: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
  [3]: http://4.bp.blogspot.com/-GJsnhAWlYqY/U0erTUv5p0I/AAAAAAAAA3Q/Ac6SeCEYgts/s1600/LAB_COLOR1.png
  [4]: http://ivrl.epfl.ch/files/content/sites/ivrg/files/research/images/RK_SLICSuperpixels/intro_pics/54082_combo.jpg
  [5]: http://img.blog.csdn.net/20140408110748640