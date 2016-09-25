# 优化算法

## Adadelta

Adadelta 是一个基于梯度下降的学习算法，可以随时间调整适应每个参数的学习率。它是作为 Adagrad 的改进版提出的，它比超参数（hyperparameter）更敏感而且可能会太过严重地降低学习率。Adadelta 类似于 rmsprop，而且可被用来替代 vanilla SGD。

论文：Adadelta：一种自适应学习率方法（ADADELTA: An Adaptive Learning Rate Method）
技术博客：斯坦福 CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

## Adagrad

Adagrad 是一种自适应学习率算法，能够随时间跟踪平方梯度并自动适应每个参数的学习率。它可被用来替代vanilla SGD (http://www.wildml.com/deep-learning-glossary/#sgd)；而且在稀疏数据上更是特别有用，在其中它可以将更高的学习率分配给更新不频繁的参数。

论文：用于在线学习和随机优化的自适应次梯度方法（Adaptive Subgradient Methods for Online Learning and Stochastic Optimization）
技术博客：斯坦福 CS231n：优化算法（http://cs231n.github.io/neural-networks-3/）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

## Adam

Adam 是一种类似于 rmsprop 的自适应学习率算法，但它的更新是通过使用梯度的第一和第二时刻的运行平均值（running average）直接估计的，而且还包括一个偏差校正项。

论文：Adam：一种随机优化方法（Adam: A Method for Stochastic Optimization）
技术博客：梯度下降优化算法概述（http://sebastianruder.com/optimizing-gradient-descent/）

