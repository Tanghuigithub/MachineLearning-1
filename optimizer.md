# 优化

## 算法

## 动量（Momentum）

动量是梯度下降算法（Gradient Descent Algorithm）的扩展，可以加速和阻抑参数更新。在实际应用中，在梯度下降更新中包含一个动量项可在深度网络中得到更好的收敛速度（convergence rate）。

论文：通过反向传播（back-propagating error）错误学习表征

### 随机梯度下降（SGD：Stochastic Gradient Descent）

随机梯度下降是一种被用在训练阶段学习网络参数的基于梯度的优化算法。梯度通常使用反向传播算法计算。在实际应用中，人们使用微小批量版本的 SGD，其中的参数更新基于批案例而非单个案例进行执行，这能增加计算效率。vanilla SGD 存在许多扩展，包括动量（Momentum）、Adagrad、rmsprop、Adadelta 或 Adam。

论文：用于在线学习和随机优化的自适应次梯度方法（Adaptive Subgradient Methods for Online Learning and Stochastic Optimization）
技术博客：斯坦福CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）


### Adadelta

Adadelta 是一个基于梯度下降的学习算法，可以随时间调整适应每个参数的学习率。它是作为 Adagrad 的改进版提出的，它比超参数（hyperparameter）更敏感而且可能会太过严重地降低学习率。Adadelta 类似于 rmsprop，而且可被用来替代 vanilla SGD。

论文：Adadelta：一种自适应学习率方法（ADADELTA: An Adaptive Learning Rate Method）
技术博客：斯坦福 CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

### Adagrad

Adagrad 是一种自适应学习率算法，能够随时间跟踪平方梯度并自动适应每个参数的学习率。它可被用来替代vanilla SGD (http://www.wildml.com/deep-learning-glossary/#sgd)；而且在稀疏数据上更是特别有用，在其中它可以将更高的学习率分配给更新不频繁的参数。

论文：用于在线学习和随机优化的自适应次梯度方法（Adaptive Subgradient Methods for Online Learning and Stochastic Optimization）
技术博客：斯坦福 CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

### Adam

Adam 是一种类似于 rmsprop 的自适应学习率算法，但它的更新是通过使用梯度的第一和第二时刻的运行平均值（running average）直接估计的，而且还包括一个偏差校正项。

论文：Adam：一种随机优化方法（Adam: A Method for Stochastic Optimization）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

### RMSProp

RMSProp 是一种基于梯度的优化算法。它与 Adagrad 类似，但引入了一个额外的衰减项抵消 Adagrad 在学习率上的快速下降。

PPT：用于机器学习的神经网络 讲座6a
技术博客：斯坦福CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）



## 问题

### 梯度消失问题（Vanishing Gradient Problem）

梯度消失问题出现在使用梯度很小（在 0 到 1 的范围内）的激活函数的非常深的神经网络中，通常是循环神经网络。因为这些小梯度会在反向传播中相乘，它们往往在这些层中传播时「消失」，从而让网络无法学习长程依赖。解决这一问题的常用方法是使用 ReLU 这样的不受小梯度影响的激活函数，或使用明确针对消失梯度问题的架构，如LSTM。这个问题的反面被称为梯度爆炸问题（exploding gradient problem）。

论文：训练循环神经网络的困难之处（On the difficulty of training Recurrent Neural Networks）
