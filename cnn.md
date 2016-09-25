# 卷积神经网络（CNN/ConvNet：Convolutional Neural Network）

CNN 使用卷积连接从输入的局部区域中提取的特征。大部分 CNN 都包含了卷积层、池化层和仿射层的组合。CNN 尤其凭借其在视觉识别任务的卓越性能表现而获得了普及，它已经在该领域保持了好几年的领先。

技术博客：斯坦福CS231n类——用于视觉识别的卷积神经网络（http://cs231n.github.io/neural-networks-3/）
技术博客：理解用于自然语言处理的卷积神经网络（http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/）

## 平均池化（Average-Pooling）

平均池化是一种在卷积神经网络中用于图像识别的池化（Pooling）技术。它的工作原理是在特征的局部区域上滑动窗口，比如像素，然后再取窗口中所有值的平均。它将输入表征压缩成一种更低维度的表征。

## Alexnet

Alexnet 是一种卷积神经网络架构的名字，这种架构曾在 2012 年 ILSVRC 挑战赛中以巨大优势获胜，而且它还导致了人们对用于图像识别的卷积神经网络（CNN）的兴趣的复苏。它由 5 个卷积层组成。其中一些后面跟随着最大池化（max-pooling）层和带有最终 1000 条路径的 softmax (1000-way softmax)的 3个全连接层。Alexnet 被引入到了使用深度卷积神经网络的 ImageNet 分类中。

## Google LeNet

GoogleLeNet 是曾赢得了 2014 年 ILSVRC 挑战赛的一种卷积神经网络架构。这种网络使用 Inception 模块（Inception Module）以减少参数和提高网络中计算资源的利用率。

论文：使用卷积获得更深（Going Deeper with Convolutions）

## Deep Dream

这是谷歌发明的一种试图用来提炼深度卷积神经网络获取的知识的技术。这种技术可以生成新的图像或转换已有的图片从而给它们一种幻梦般的感觉，尤其是递归地应用时。

代码：Github 上的 Deep Dream（https://github.com/google/deepdream）
技术博客：Inceptionism：向神经网络掘进更深（https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html）