# 函数家族

-------------

## 激活函数（Activation Function）

为了让神经网络能够学习复杂的决策边界（decision boundary），我们在其一些层应用一个非线性激活函数。最常用的函数包括  sigmoid、tanh、ReLU（Rectified Linear Unit 线性修正单元） 以及这些函数的变体。

 - **motivation**: to compose *simple transformations* in order to obtain 
*highly non-linear* ones
 - (MLPs compose affine transformations and element-wise non-linearities)
 - hyperbolic tangent activation functions:
 $$
 { h }^{ k }=tanh({ b }^{ k }+{ W }^{ k }{ h }^{ k-1 })
$$
 - the input of the neural net: ${ h }^{ 0 }=x$
 - theoutputofthe k-th hidden layer: ${ h }^{ k }$

 - affine transformation $a = b+Wx$ \, elementwise
$$
h=\phi (a)⇔{ h }_{ i }=\phi ({ a }_{ i })=\phi ({ b }_{ i }+{ W }_{ i,: }x)
$$

 - non-linear neural network activation functions:
 ######Rectifier or rectified linear unit (ReLU) or positive part
 ######Hyperbolic tangent
 ######Sigmoid
 ######Softmax
 ######Radial basis function or RBF
 ######Softplus
 ######Hard tanh
 ######Absolute value rectification
 ######Maxout

 
 - the structure (also called architecture) of the family of input-output functions can be varied in many ways: 
*convolutional networks*, 
*recurrent networks*


#### 6.3.2 Loss Function and Conditional Log-Likelihood
 - In the 80’s and 90’s the most commonly used loss function was the squared error
$$
L({ f }_{ θ }(x),y)={ ||fθ(x)−y|| }^{ 2 }
$$
 
 
 - if f is unrestricted (non- parametric),
$$
 f(x) = E[y | x = x]
$$

 - Replacing the squared error by an absolute value makes the neural network try to estimate not the conditional expectation but the conditional median
 
 - **交叉熵（cross entropy）目标函数 **: when y is a discrete label, i.e., for classification problems, other loss functions such as the Bernoulli negative log-likelihood4 have been found to be more appropriate than the squared error. ($$y∈{ \left\{ 0,1 \right\}  }$$)

$$
L({ f }_{ θ }(x),y)=−ylog{ f }_{ θ }(x)−(1−y)log(1−{ f }_{ θ }(x))
$$

- $${f}_{\theta}(x)$$ to be strictly between 0 to 1: use the sigmoid as non-linearity for the output layer(matches well with the binomial negative log-likelihood cost function)




#####Learning a Conditional Probability Model

- loss function as corresponding to a conditional log-likelihood, i.e., the negative log-likelihood (NLL) cost function
$$
{ L }_{ NLL }({ f }_{ \theta  }(x),y)=−logP(y=y|x=x;θ)
$$
- example) if y is a continuous random variable and we assume that, given x, it has a Gaussian distribution with mean ${f}_{θ}$(x) and variance ${\sigma}^{2}$
$$
−logP(y|x;θ)=\frac { 1 }{ 2 } { ({ f }_{ \theta  }(x)−y) }^{ 1 }/{ σ }^{ 2 }+log(2π{ σ }^{ 2 })
$$
- minimizing this negative log-likelihood is therefore equivalent to minimizing the squared error loss.

- for discrete variables, the binomial negative log-likelihood cost func- tion corresponds to the conditional log-likelihood associated with the Bernoulli distribution (also known as cross entropy) with probability $p = {f}_{θ}(x)$ of generating y = 1 given x =$ x$
$$
{L}_{NLL}=−logP(y|x;θ)={−1}_{y=1}{logp−1}_{y=0}log(1−p)\\ =−ylog{f}_{θ}(x)−(1−y)log(1−{f}_{θ}(x))
$$

##### Softmax

- designed for the purpose of specifying **multinoulli distributions**:
$$
p=softmax(a)\Longleftrightarrow { p }_{ i }=\frac { { e }^{ { a }_{ i } } }{ \sum { _{ j }^{  }{ { e }^{ { a }_{ j } } } }  } 
$$
- consider the gradient with respect to the scores $a$.
$$
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=\frac { ∂ }{ ∂{ a }_{ k } } (−log{ p }_{ y })=\frac { ∂ }{ ∂{ a }_{ k } } ({ −a }_{ y }+log\sum _{ j }^{  }{ { e }^{ { a }_{ j } } } )\\ ={ −1 }_{ y=k }+\frac { { e }^{ { a }_{ k } } }{ \sum _{ j }^{  }{ { e }^{ { a }_{ j } } }  } ={ p }_{ k }-{1}_{y=k}
$$
or
$$
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=(p-{e}_{y})
$$
#### Cost Functions For Neural Networks

- a good choice for the criterion is maximum likelihood regularized with dropout, possibly also with weight decay.

#### Optimization Procedure

- a good choice for the optimization algorithm for a feed-forward network is usually stochastic gradient descent with momentum.

