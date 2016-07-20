# 神经网络入门


[TOC]

![1.gif-3kB][1]
![2.gif-2.9kB][2]
![3.gif-4.2kB][3]



![此处输入图片的描述][4]
High dimension：

- SVM ,拟合能力随维度增加线性增长
- kenel-based,内积
- 易收敛
- convex
- low computational cost

Going deep：神经网络，拟合能力随深度增加指数增长

- 复杂boundary
- non-convex
- Black-magic
- overfitting
- High computational cost
- 矩阵联乘

1.神经网络获取的知识是从外界环境中学习得来的；
2.互连神经元的连接强度，即突触权值，用于储存获取的知识。

**神经网络的功能**
|功能 |神经网络  |
| :------:   | :----- | 
|分类|单层感知器、线性神经网络、BP神经网络、RBF神经网络、CNN、DBN、LVQ神经网络、Elman神经网络|
|拟合|BP神经网络、RBF神经网络、DBN、Elman神经网络|
|聚类|通用型竞争神经网络、SOM神经网络|
|特征表示|RBM、Word2vec、CNN|
|记忆|离散型Hopfield网络、Elman神经网络|
|最优化求解|连续型Hopfield神经网络|

# 神经元 {#ID1}

![ScreenShot_20160426150621.png-73.5kB][5]


## 线性神经元

$$
y=\varphi(\sum _{i=0}^mw_ix_i )=\varphi( \mathbf w^T \mathbf x)\\
\varphi(v)=v
$$

## 非线性神经元——感知器 (perceptron)

$$
\varphi(v)=\frac 1{1+e^{-v}}
$$

- 人工神经元模型：
1940年McCulloch与Pitts提出仿造神经元的人工神经元模型。神经元网络的输入端接收多维输入向量（如模式识别中常用的特征向量），输出端只有一个。该模型由两部分功能组成，一部分功能将输入端信号进行线性迭加，另一部分是对该迭加值进行非线性映射及从输出端输出。
McCulloch-Pitts模型中的非线性映射是**阈值函数**，输入值从负无穷到某个值内输出为0，输入一旦达到该阈值，输出为1。这种0—1输出也可改成-1—+1输出。近来人们根据需要, 在人工神经元模型中也采用其它类型的非线性映射。

    - McCulloch-Pitts模型：就是McCulloch和Pitts俩人在1940提出的人工神经元模型。

- 单层感知器：
一个McCulloch-Pitts模型可实现的功能是**两类别**线性分类器，输入向量的线性求和是**判别函数**，非线性映射及输出的0、1值表示了分类结果。这种线性分类器的系数若采用感知准则函数方法确定，它就是一个感知器。这种感知器与多层感知器相比，只有一层，因而称为单层感知器。

![ScreenShot_20160426150732.png-61.8kB][6]

- 突触（权值）$w_{ki}$
每一个都由其权值或者强度作为特征。特别是，在连接到神经元$k$的突触$j$上的输入信号被乘以$k$的突触权值。
- 偏置$w_{k0}$,  $x_{0}=+1$
         用来调节值域的。
- 加法器
是一个线性组合器，求输入信号与神经元相应突触权值的加权和，并与偏置求和。

$$
net_{k} = \sum_{i=1}^{m}x_{i}w_{ki}+w_{k0}=\sum_{i=0}^{m}x_{i}w_{ki}=w_{k}^{t}x
$$

### 激活函数
用来限制神经元的输入输出振幅。激活函数也称为压制函数，因为它的输出信号压制（限制）到允许范围之内的一定值。
常用激活函数有

- 阈值函数
$$
\varphi(v)=
\begin{cases}
1, &v\geq0 \cr 0, &v<0
\end{cases}
$$
![ScreenShot_20160426153615.png-8.5kB][7]
- 2. sigmoid函数
$$
\varphi(v)=\frac{1}{1+\exp(-av)}
$$
![ScreenShot_20160426153626.png-26.2kB][8]

非线性且处处可微

- Logistic函数
- 双曲正切

- Logistic函数

逻辑函数或逻辑曲线是一种常见的S形函数，它是皮埃尔·弗朗索瓦·韦吕勒在1844或1845年在研究它与人口增长的关系时命名的。广义Logistic曲线可以模仿一些情况人口增长（P）的S形曲线。起初阶段大致是指数增长；然后随着开始变得饱和，增加变慢；最后，达到成熟时增加停止。

一个简单的Logistic函数可用下式表示：$P(t)=\frac 1{1+e^-t}$
![此处输入图片的描述][9]

- 3. tanh函数

$$
\varphi(v)=\tanh(v)
$$

#### Why use？

Activation functions for the hidden units are needed to introduce **nonlinearity** into the network. Without nonlinearity, hidden units would not make nets more powerful than just plain perceptrons (which do not have any hidden units, just input and output units). The reason is that a linear function of linear functions is again a linear function. However, it is the nonlinearity (i.e, the capability to represent nonlinear functions) that makes multilayer networks so powerful. Almost any nonlinear function does the job, except for polynomials. For backpropagation learning, the activation function must be differentiable, and it helps if the function is bounded; the sigmoidal functions such as logistic and tanh and the Gaussian function are the most common choices. Functions such as tanh or arctan that produce both positive and negative values tend to yield faster training than functions that produce only positive values such as logistic, because of better numerical conditioning (see ftp://ftp.sas.com/pub/neural/illcond/illcond.html).
For hidden units, sigmoid activation functions are usually preferable to threshold activation functions. Networks with threshold units are difficult to train because the error function is stepwise constant, hence the gradient either does not exist or is zero, making it impossible to use backprop or more efficient gradient-based training methods. Even for training methods that do not use gradients--such as simulated annealing and genetic algorithms--sigmoid units are easier to train than threshold units. With sigmoid units, a small change in the weights will usually produce a change in the outputs, which makes it possible to tell whether that change in the weights is good or bad. With threshold units, a small change in the weights will often produce no change in the outputs.

For the output units, you should choose an activation function suited to the distribution of the target values:

- For binary (0/1) targets, the logistic function is an excellent choice (Jordan, 1995).
- For categorical targets using 1-of-C coding, the softmax activation function is the logical extension of the logistic function.
- For continuous-valued targets with a bounded range, the logistic and tanh functions can be used, provided you either scale the outputs to the range of the targets or scale the targets to the range of the output activation function ("scaling" means multiplying by and adding appropriate constants).
- If the target values are positive but have no known upper bound, you can use an exponential output activation function, but beware of overflow.
- For continuous-valued targets with no known bounds, use the identity or "linear" activation function (which amounts to no activation function) unless you have a very good reason to do otherwise.

# 多层前馈神经网络(Multi-Layer Neural Network)

特点：

- 每层神经元与下一层神经元全互联
- 解决非线性可分问题——**多层**功能神经元
- 前馈（feedforward）并不意味着网络中信号不能后传，而是指网络拓扑结构上不存在环或回路。
- 输入层仅是接受输入，不进行函数处理，隐层与输出层包含功能神经元
- 隐层结点：由两层及两层以上结点组成的前馈网络中处于非输出层的结点称为隐层结点，一个只含**一个隐含层**的前馈网络称为**双层前馈网络**。含两个隐含层的称三层前馈网络。一般使用至多两个隐含层的网络。

网络中传递的两种信号：

- 函数信号
    - 来自于输入端，到达输出端
- 误差信号
    - 产生于输出端，反向传播

每个神经元完成两种计算：

- 计算输出函数信号
    - 输入信号和权值的连续非线性函数
- 计算梯度向量
    - 误差曲面对于权值的梯度的估计


# 学习

神经网络的学习过程，就是根据训练数据来调整神经元之间的连接权（connection weight）以及每个功能神经元的阈值；换言之，神经网络“学”到的东西，蕴含在连接权与阈值中。

## 最小均方算法(LMS)

- 定义**代价函数**(cost function)为均方误差
$$
\varepsilon(w)=\frac 1{2N}\sum_{n=1}^N e^2(n)
$$
其中$e(n)$为时刻$n$测得的误差。
定义权值更新规则
$$
w\leftarrow w+\Delta w
$$
基于最速下降法则
$$
\Delta w=-\frac {\partial \varepsilon (w)}{\partial w}=-\frac 1N\sum_{n=1}^N e(
n)\frac {\partial e(n)}{\partial w}
$$
要求激活函数可导。

批处理方式（batch）
称为集中方式、批处理方式
训练集内的全部样本都呈现一遍之后，才进行权值更新。这样称为一个回合（epoch）

- 定义**代价函数**(cost function)为瞬时误差的平方
$$
\varepsilon(w)=\frac 12 e^2(n)
$$
在线学习（on-line）
称为串行方式、在线方式、随机方式
每个训练样本呈现之后，即进行权值更新. 训练集内的全部N个样本都呈现一次，称为一个回合(epoch)；

对于单层感知器来说：
$C$是输出层的所有神经元

- 集中方式
$$
\varepsilon(w)=\frac 1{2N} \sum _{i=1}^N\sum _{k \in C}e_k^2(n)
$$

- 在线方式
$$
\varepsilon(w)=\frac 12 \sum _{k \in C}e_k^2(n)
$$

在单层感知器中误差信号是直接可以测量的
在多层感知器中， 隐层神经元是不可见的，其误差信号无法直接测量

## 感知机学习规则

对训练样例$(x,y)$,若当前感知机的输出为$\hat y$,则感知机权重将这样调整：
$
\begin{eqnarray}
w_i \leftarrow w_i+\Delta w_i  \\
\Delta w_i=\eta(y-\hat y)x_i
\end{eqnarray}
$
其中$\eta \in (0,1)$称为学习率（learning rate），通常设置一个较小正数，如0.1.

感知机只有输出层神经元进行激活函数处理，即只拥有一层**功能神经元**（functional neuron），学习能力非常有限。
若两类模式是线性可分的，即存在一个线性超平面能将他们分开，则感知机的学习过程一定会**收敛**（converge）而求得适当的权值向量$w=(w_1;w_2;\dots;w_{n+1})$，否则感知机学习过程将发生**震荡**（fluctuation），$w$难以稳定下来，不能求得合适解。
![此处输入图片的描述][10]







## 误差逆传播算法(error BackPropagation)

欲训练**多层**网络，简单感知机学习规则显然不够，需要更强大的学习算法，BP算法是其中最杰出的代表。
有**串行**方式（on-line）和**集中**方式(batch)两种。
The popularity of on-line learning for the supervised training of multilayer perceptrons has been further enhanced by the development of the back-propagation algorithm. 
它是基于**误差**修正学习规则的。误差反向传播学习由两次经过网络不同层的通过组成：一次前向通过和一次后向通过。在前向通过中，一个输入向量作用于网络感知节点，它的影响通过网络一层接一层的传播，最后产生一个输出作为网络的实际输出。在前行通过中，网络的突触权值全为**固定**的。另一方面，在反向通过中，网络的突触权值全部根据误差规则来**调整**。特别是从目标响应减去网络的实际响应为误差信号。这个误差信号反向传播经过整个网络，与突触连接方向相反因此叫“误差反向传播”。

![此处输入图片的描述][12]

工作流程
         A 初始化
         B 训练样本的呈现
         C 信息前向计算
         D 误差反向传播
         E 判断终止条件并迭代

### 计算梯度


给定训练集$D=\{(x_1,y_1)，(x_2,y_2)，\dots,(x_m,y_m)\}$,$x_i \in \mathbb R^d$,$y_i \in \mathbb R^c$，即输入实例由d个属性描述，输出$c$维实值向量.

下图给出了一个拥有d个输入神经元、m个隐层神经元、c个输出神经元的多层前馈网络结构：

![此处输入图片的描述][11]

其中，

- 隐层第$h$个神经元的阈值用$\gamma _h$表示，
- 输出层第$j$个神经元的阈值用$\theta_j$表示。
- 输入层第$i$个神经元与隐层第$h$个神经元之间的连接权为$v_{ih}$;
- 隐层第$h$个神经元与输出层第$j$个神经元之间的连接权为$w_{hj}$;
- 记隐层第$h$个神经元接收到的输入为$\alpha_h=\sum_{i=1}^dv_{ih}x_i$;
- 输出层第$j$个神经元接收到的输入为$\beta_j=\sum_{h=1}^mw_{hj}b_h$, 其中$b_h$为隐层第$h$个神经元的输出;
- 假设隐层和输出层神经元都使用Logistic函数：
$$
P(t)=\frac 1{1+e^{-t}}
$$

#### 输出层

以串行为例，对训练样本$（x_k,y_k）$,假定神经网络的输出为$\hat y_k=（\hat y_1^k,\hat y_2^k,\dots,\hat y_c^k）$,即
$$
\hat y_j^k=f(\beta_j-\theta_j),
$$
则网络在$（x_k,y_k）$上的均方误差（LMS算法）为
$$
E_k=\frac 12\sum_{j=1}^c(\hat y_j^k-y_j^k)^2.
$$
根据广义的感知机学习规则对参数进行更新，任意参数$v$的更新估计式为
$$
v \leftarrow v+\Delta v
$$
BP算法基于梯度下降策略，对误差$E_k$,给定学习率$\eta$,有
$$
\Delta w_{hj}=-\eta \frac {\partial E_k}{\partial w_{hj}}
$$
考虑到$w_{hj}$先影响到第$j$个输出层神经元的输入值$\beta_j$，再影响到其输出值$\hat y_j^k$,最后影响到$E_k$,有
$$
\frac {\partial E_k}{\partial w_{hj}}=\frac {\partial E_k}{\partial \hat y_j^k}\frac {\partial \hat y_j^k}{\partial \beta _j}\frac {\partial \beta _j}{\partial w_{hj}}
$$
根据$\beta_j$的定义，显然有
$$
\frac {\partial \beta _j}{\partial w_{hj}}=b_h
$$
而Logistic函数有一个性质：
$$
f'(x)=f(x)(1-f(x))
$$
于是有
$$
g_j=-\frac {\partial E_k}{\partial \hat y_j^k}\frac {\partial \hat y_j^k}{\partial \beta _j} =-(\hat y_j^k-y_j^k)f'(\beta _j-\theta _j)=\hat y_j^k(1-\hat y_j^k)(y_j^k-\hat y_j^k).
$$
于是得到了BP算法中关于$w_{hj}$的更新公式
$$
\Delta w_{hj}=\eta g_j b_h
$$

#### 隐层

根据（2）中的推导
$$
\begin{aligned}
\Delta \theta_j&=-\eta\frac {\partial E_k}{\partial \hat y_j^k}\cdot\frac {\partial \hat y_j^k}{\partial \theta_j}\\
&=\eta(\hat y_j^k-y_j^k)f'(\beta _j-\theta _j)\\
&=-\eta g_j
\end{aligned}
$$
$b_h$为第$h$个隐层神经元的输出，即
$$
b_h=f(\alpha _h-\gamma _h)
$$
考虑到$v_{ih}$先影响到第$h$个隐层神经元的输入值$\alpha_h$，再影响到其输出值$b_h$,最后影响到$E_k$,有
$$
\frac {\partial E_k}{\partial v_{ih}}=\frac {\partial E_k}{\partial b_h}\cdot\frac {\partial b_h}{\partial \alpha _h}\cdot\frac {\partial \alpha_h}{\partial v_{ih}}
$$
类似（2）中的$g_j$，有
$$
\begin{aligned}
e_h&=-\frac {\partial E_k}{\partial b_h}\cdot\frac {\partial b_h}{\partial \alpha _h}\\
&=-\sum _{j=1}^c\frac {\partial E_k}{\partial \beta _j}\cdot \frac {\partial \beta _j}{\partial b_h}f'(\alpha_h-\gamma_h)\\
&=\sum _{j=1}^cw_{hj}g_jf'(\alpha_h-\gamma_h)\\
&=b_h(1-b_h)\sum_{j=1}^cw_{hj}g_j
\end{aligned}
$$
所以
$$
\Delta v_{ih}=\eta e_h x_i
$$
另外,类似$\Delta \theta _j$
$$
\Delta \gamma_h=-\eta e_h
$$

# 卷积神经网络(CNN)

## 概念

卷积神经网络是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。
它有5个特点：
1.局部感知；
2.参数共享；
3.采样；
4.多卷积核；
5.多卷积层。
其中前3个特征是CNN较少参数数目的利器。

## 网络结构

CNN由输入层、**卷积层**和**抽样层**的交替组合、**全连接层**、输出层组成。
其中：

- 输入层一般是二维的，如64*64的图像，输入量可以看做是一个矩阵，其大小等于图像的像素个数。
- 卷积层是依次将矩阵的区域与卷积核作卷积运算，从而产生一个新的矩阵；而矩阵的区域是从左上角开始依次向右或向下移动一维产生的，直到碰到右端或下方边界；而卷积核可以有多个，因此原矩阵经过卷积层的处理后会产生**多个新矩阵**。
- 采样层是将原矩阵的区域抽象成一个数字表示，从而极大减少特征的维数；而矩阵的区域的选择与卷积层相同；矩阵区域抽象成数字的方法：取**最大值，或最小值，或平均值**等；抽象层的处理不会引起矩阵个数的变化，而只是改变矩阵的**规模**。
- 全连接层是指将卷积层和抽样层的交替组合产生的矩阵映射成**一维**的向量，然后采用**BP神经网络**的结构和处理方式。输出层神经元的个数等于分类的个数，用于表示分类的输出值。
         
![此处输入图片的描述][13]

##学习过程

前向传播计算
卷积层：依次将矩阵的区域与卷积核作卷积运算，从而产生一个新的矩阵
![此处输入图片的描述][14]
![此处输入图片的描述][15]
采样层：将原矩阵的区域抽象成一个数字表示，从而极大减少特征的维数
![此处输入图片的描述][16]
![此处输入图片的描述][17]

# 其他常见神经网络

## 经向基函数网络

在多层感知器的人工神经元一般采用S型函数作非线性映射。而经向基函数网络则采用单峰型函数（如高斯函数）为非线性映射函数。在实现数据拟合等等应用中较多。训练也较简单。

## Hopfield模型

是人工神经元网络中一个典型网络，该网络中的所有神经元实行**全联接**，即任一个神经元的输入输出都联接至其它结点的输入端，因此每一个结点的状态对其它结点的状态都有牵扯作用。该网络的联接数值由**Hebb规则**确定。其功能可实现**联想记忆**, 是人工神经网络中的一种动力系统。网络具有**存储模式**功能，有自己的稳态，其原理与电子线路中的**双稳态触发器**相仿。

## 自适应共振理论

自适应共振理论是Carpeter和Grossberg在1987年提出的网络模型，它主要模仿人积累知识的过程。例如人们可以辩别当前见到的事物, 是已往见到过的事物，还是从未见到过的事物，如是没有见过的事物，则将其印入脑海中。自适应共振理论模型在网络内部设有**竞争网络**，计算当前输入信息与已存模型中哪个**最相似**。如最相似度超过一定阈值，则达到识别模式的目的。如当可输入信息与所有已存模型都不够相似，则判定为新模式，并为其建模与存储。
## RBF网络

## 竞争学习

竞争学习（competitive learning）是神经网络中一种常用的无监督学习策略，在使用该策略时，网络的输出神经元相互竞争，每一时刻只有一个竞争获胜的神经元被激活，其他神经元的状态被抑制。这种机制亦称“胜者通吃”（win-take-all）原则。
有了竞争机制，人工神经元网络可以实现聚类等功能，一些典型的网络模型如自适应共振理论（Adaptive Resonance Theory），自组织特征映射等都属于具有竞争功能的非监督学习人工神经元网络。

### ART网络

### SOM网络

自组织特征映射（Self Organization Feature mapping SOF）是Kohonen提出的一种将高维空间中的数据映射到低维空间的一种神经元网络，而实现映射的同时要保持数据在原高维空间的**相似性**。这是指原数据在高维空间较相近，则它们的映象在低维空间中也应相近。这种网络结构的特点是**输出结点数量众多**，并按一定规则排列。它们接受高维信号，并在某个输出结点中得到高响应。输入高维信号相近，则高响应的输出结点也应相近。
## Elman网络

递归神经网络（recurrent/recursive neural networks）允许网络中出现环形结构，从而可让一些神经元的输出反馈回来作为输入信号。使得网络在t时刻的输出状态不仅与t时刻的网络状态有关，还与t-1时刻有关，从而能处理与时间有关的动态变化。
Elman网络是最常用的递归神经网络之一。

### Hebb学习


>  我们外在的行为是神经网络接受刺激之后产生的反应。神经元与神经元之间的连接有强有弱，而这种强弱是可以通过学习（训练）来不断改变的。如果两个神经元总是相关连的受到刺激或者反应，他们的连接就会一次一次的被加强。这也就是Hebb学习法了。

- Hebb规则：
Hebb是神经生理学家, 在1949年提出一种假没，认为脑细胞中突触强度根据其感受到的经验改变。他因此提出这种突触强度的改变与神经在触发前后突触之间的相关成正比。由于在Hopfield模型中采用的联接规则与Hebb的修改相比，因而在人工神经之网络中广泛使用的规则以Hebb冠名。这种规则除了在Hopfield模型中使用外，在非监督学习方法的人工神经元网络中广泛采用。
- 非监督Hebb学习方法：人工神经元网络也用来分析数据集内的一些规律，如方差分析，主分量分析等，由于其训练方法采用了Hebb规则，因而称为非监督Hebb学习方法。


通俗描述：巴甫洛夫的狗
先摇铃铛，之后给一只狗喂食，久而久之，狗听到铃铛就会口水连连。这也就是狗的“听到”铃铛的神经元与“控制”流口水的神经元之间的连接被加强了。


  [1]: http://static.zybuluo.com/sixijinling/woavorx8bm6q43u2ku1jzsgl/1.gif
  [2]: http://static.zybuluo.com/sixijinling/weepf5yeg8l74ge72xldbz6h/2.gif
  [3]: http://static.zybuluo.com/sixijinling/17ia36g3oo6rw9bbb1gpjli1/3.gif
  [4]: http://hahack.com/images/ann2/MqLaO.png
  [5]: http://static.zybuluo.com/sixijinling/m34tru465md4fxh5ried45ni/ScreenShot_20160426150621.png
  [6]: http://static.zybuluo.com/sixijinling/rba2928lcj3o188b2i6zy5sy/ScreenShot_20160426150732.png
  [7]: http://static.zybuluo.com/sixijinling/zcs7oxqe75t3aebzyld5mmu1/ScreenShot_20160426153615.png
  [8]: http://static.zybuluo.com/sixijinling/os6tp2kzrwe1yfbgmzi42548/ScreenShot_20160426153626.png
  [9]: https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png
  [10]: http://hahack.com/images/ann2/XhmO7.png
  [11]: http://static.zybuluo.com/sixijinling/g6tyjdmia5ggil07421lr6k2/ann.png
  [12]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUoo8xKuQWmKx55vF67iaw6b9rLzD2q8nwWsNPBun3ic2ye6rFVnxuZoxhg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1
  [13]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUoYmI1QypB37Alz7U5QU8yricibFJDa0Jme3iaOmkoXQDialApLdD3KHVq4A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1
  [14]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUo7f3JEq29XdicBIGwTdNN372yvVEhTA95ibibmTIjZ6ibXsg4JD8PIciby3Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1
  [15]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUoLOSol5SsoVBDR3WhHHfwUWsbLJKFibG1iaf3nZEIq2Yybfst3BvQmD7w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1
  [16]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUo8HNWpyRE69Y7NVMKjACP7xL3eMKVnFFicKblf4VUfJz3fDx3eXhFG4w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&retryload=1
  [17]: http://mmbiz.qpic.cn/mmbiz/qX2AoDIKJ2WKKVBiaVQNB0KNKXH619EUofUcnGakvPqLuZa0wuYsZ5icYedvkmXnZwUL43q0BNYV8D3B6r1aSkDw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&retryload=1