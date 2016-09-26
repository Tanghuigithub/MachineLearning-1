# 术语（2016/9/25 更新）



本文整理了一些深度学习领域的专业名词及其简单释义，同时还附加了一些相关的论文或文章链接。本文编译自 wildml，作者仍在继续更新该表，编译如有错漏之处请指正。文章中的论文与 PPT 读者可点击阅读原文下载。











## 自编码器（Autoencoder）

自编码器是一种神经网络模型，它的目标是预测输入自身，这通常通过网络中某个地方的「瓶颈（bottleneck）」实现。通过引入瓶颈，我们迫使网络学习输入更低维度的表征，从而有效地将输入压缩成一个好的表征。自编码器和 PCA 等降维技术相关，但因为它们的非线性本质，它们可以学习更为复杂的映射。目前已有一些范围涵盖较广的自编码器存在，包括 降噪自编码器（Denoising Autoencoders）、变自编码器（Variational Autoencoders）和序列自编码器（Sequence Autoencoders）。

降噪自编码器论文：Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion 
变自编码器论文：Auto-Encoding Variational Bayes
序列自编码器论文：Semi-supervised Sequence Learning


## 深度信念网络（DBN：Deep Belief Network）

DBN 是一类以无监督的方式学习数据的分层表征的概率图形模型。DBN 由多个隐藏层组成，这些隐藏层的每一对连续层之间的神经元是相互连接的。DBN 通过彼此堆叠多个 RBN（限制波尔兹曼机）并一个接一个地训练而创建。

论文：深度信念网络的一种快速学习算法（A fast learning algorithm for deep belief nets）








## Highway Layer

Highway Layer　是使用门控机制控制通过层的信息流的一种神经网络层。堆叠多个 Highway Layer 层可让训练非常深的网络成为可能。Highway Layer 的工作原理是通过学习一个选择输入的哪部分通过和哪部分通过一个变换函数（如标准的仿射层）的门控函数来进行学习。Highway Layer 的基本公式是 T * h(x) + (1 - T) * x；其中 T 是学习过的门控函数，取值在 0 到 1 之间；h(x) 是一个任意的输入变换，x 是输入。注意所有这些都必须具有相同的大小。

论文：Highway Networks

## ICML

即国际机器学习大会（International Conference for Machine Learning），一个顶级的机器学习会议。

## ILSVRC

即 ImageNet 大型视觉识别挑战赛（ImageNet Large Scale Visual Recognition Challenge），该比赛用于评估大规模对象检测和图像分类的算法。它是计算机视觉领域最受欢迎的学术挑战赛。过去几年中，深度学习让错误率出现了显著下降，从 30% 降到了不到 5%，在许多分类任务中击败了人类。



Keras

Kears 是一个基于 Python 的深度学习库，其中包括许多用于深度神经网络的高层次构建模块。它可以运行在 TensorFlow 或 Theano 上。



## MNIST

MNIST数据集可能是最常用的一个图像识别数据集。它包含 60,000 个手写数字的训练样本和 10,000 个测试样本。每一张图像的尺寸为 28×28像素。目前最先进的模型通常能在该测试集中达到 99.5% 或更高的准确度。







## 神经图灵机（NTM：Neural Turing Machine）

NTM 是可以从案例中推导简单算法的神经网络架构。比如，NTM 可以通过案例的输入和输出学习排序算法。NTM 通常学习记忆和注意机制的某些形式以处理程序执行过程中的状态。

论文：神经图灵机（Neural Turing Machines）

## 非线性（Nonlinearity）

参见激活函数（Activation Function）。

## 噪音对比估计（NCE：noise-contrastive estimation）

噪音对比估计是一种通常被用于训练带有大输出词汇的分类器的采样损失（sampling loss）。在大量的可能的类上计算 softmax 是异常昂贵的。使用 NCE，我们可以将问题降低成二元分类问题，这可以通过训练分类器区别对待取样和「真实」分布以及人工生成的噪声分布来实现。

论文：噪音对比估计：一种用于非标准化统计模型的新估计原理（Noise-contrastive estimation: A new estimation principle for unnormalized statistical models ）
论文：使用噪音对比估计有效地学习词向量（Learning word embeddings efficiently with noise-contrastive estimation）



## 受限玻尔兹曼机（RBN：Restricted Boltzmann Machine）

RBN 是一种可被解释为一个随机人工神经网络的概率图形模型。RBN 以无监督的形式学习数据的表征。RBN 由可见层和隐藏层以及每一个这些层中的二元神经元的连接所构成。RBN 可以使用对比散度（contrastive divergence）进行有效的训练，这是梯度下降的一种近似。

第六章：动态系统中的信息处理：和谐理论基础
论文：受限玻尔兹曼机简介（An Introduction to Restricted Boltzmann Machines）




## 残差网络（ResNet）

深度残差网络（Deep Residual Network）赢得了 2015 年的 ILSVRC 挑战赛。这些网络的工作方式是引入跨层堆栈的快捷连接，让优化器可以学习更「容易」的残差映射（residual mapping）而非更为复杂的原映射（original mapping）。这些快捷连接和 Highway Layer 类似，但它们与数据无关且不会引入额外的参数或训练复杂度。ResNet 在 ImageNet 测试集中实现了 3.57% 的错误率。

论文：用于图像识别的深度残差网络（Deep Residual Learning for Image Recognition）











