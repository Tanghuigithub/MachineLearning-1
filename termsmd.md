# 术语（2016/9/25 更新）



本文整理了一些深度学习领域的专业名词及其简单释义，同时还附加了一些相关的论文或文章链接。本文编译自 wildml，作者仍在继续更新该表，编译如有错漏之处请指正。文章中的论文与 PPT 读者可点击阅读原文下载。





## 仿射层（Affine Layer）

神经网络中的一个全连接层。仿射（Affine）的意思是前面一层中的每一个神经元都连接到当前层中的每一个神经元。在许多方面，这是神经网络的「标准」层。仿射层通常被加在卷积神经网络或循环神经网络做出最终预测前的输出的顶层。仿射层的一般形式为 y = f(Wx + b)，其中 x 是层输入，w 是参数，b 是一个偏差矢量，f 是一个非线性激活函数。





## 自编码器（Autoencoder）

自编码器是一种神经网络模型，它的目标是预测输入自身，这通常通过网络中某个地方的「瓶颈（bottleneck）」实现。通过引入瓶颈，我们迫使网络学习输入更低维度的表征，从而有效地将输入压缩成一个好的表征。自编码器和 PCA 等降维技术相关，但因为它们的非线性本质，它们可以学习更为复杂的映射。目前已有一些范围涵盖较广的自编码器存在，包括 降噪自编码器（Denoising Autoencoders）、变自编码器（Variational Autoencoders）和序列自编码器（Sequence Autoencoders）。

降噪自编码器论文：Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion 
变自编码器论文：Auto-Encoding Variational Bayes
序列自编码器论文：Semi-supervised Sequence Learning


## 深度信念网络（DBN：Deep Belief Network）

DBN 是一类以无监督的方式学习数据的分层表征的概率图形模型。DBN 由多个隐藏层组成，这些隐藏层的每一对连续层之间的神经元是相互连接的。DBN 通过彼此堆叠多个 RBN（限制波尔兹曼机）并一个接一个地训练而创建。

论文：深度信念网络的一种快速学习算法（A fast learning algorithm for deep belief nets）




## GloVe

Glove 是一种为话语获取矢量表征（嵌入）的无监督学习算法。GloVe 的使用目的和 word2vec 一样，但 GloVe 具有不同的矢量表征，因为它是在共现（co-occurrence）统计数据上训练的。

论文：GloVe：用于词汇表征（Word Representation）的全局矢量（Global Vector）（GloVe: Global Vectors for Word Representation ）



## GRU

GRU（Gated Recurrent Unit：门控循环单元）是一种 LSTM 单元的简化版本，拥有更少的参数。和 LSTM 细胞（LSTM cell）一样，它使用门控机制，通过防止梯度消失问题（vanishing gradient problem）让循环神经网络可以有效学习长程依赖（long-range dependency）。GRU 包含一个复位和更新门，它们可以根据当前时间步骤的新值决定旧记忆中哪些部分需要保留或更新。

论文：为统计机器翻译使用 RNN 编码器-解码器学习短语表征（Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation）
技术博客：循环神经网络教程，第 4 部分：用 Python 和 Theano 实现 GRU/LSTM RNN（http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/）

Highway Layer

Highway Layer　是使用门控机制控制通过层的信息流的一种神经网络层。堆叠多个 Highway Layer 层可让训练非常深的网络成为可能。Highway Layer 的工作原理是通过学习一个选择输入的哪部分通过和哪部分通过一个变换函数（如标准的仿射层）的门控函数来进行学习。Highway Layer 的基本公式是 T * h(x) + (1 - T) * x；其中 T 是学习过的门控函数，取值在 0 到 1 之间；h(x) 是一个任意的输入变换，x 是输入。注意所有这些都必须具有相同的大小。

论文：Highway Networks

## ICML

即国际机器学习大会（International Conference for Machine Learning），一个顶级的机器学习会议。

## ILSVRC

即 ImageNet 大型视觉识别挑战赛（ImageNet Large Scale Visual Recognition Challenge），该比赛用于评估大规模对象检测和图像分类的算法。它是计算机视觉领域最受欢迎的学术挑战赛。过去几年中，深度学习让错误率出现了显著下降，从 30% 降到了不到 5%，在许多分类任务中击败了人类。



Keras

Kears 是一个基于 Python 的深度学习库，其中包括许多用于深度神经网络的高层次构建模块。它可以运行在 TensorFlow 或 Theano 上。

LSTM

长短期记忆（Long Short-Term Memory）网络通过使用内存门控机制防止循环神经网络（RNN）中的梯度消失问题（vanishing gradient problem）。使用 LSTM 单元计算 RNN 中的隐藏状态可以帮助该网络有效地传播梯度和学习长程依赖（long-range dependency）。

论文：长短期记忆（LONG SHORT-TERM MEMORY）
技术博客：理解 LSTM 网络（http://colah.github.io/posts/2015-08-Understanding-LSTMs/）
技术博客：循环神经网络教程，第 4 部分：用 Python 和 Theano 实现 GRU/LSTM RNN（http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/）

最大池化（Max-Pooling）

池化（Pooling）操作通常被用在卷积神经网络中。一个最大池化层从一块特征中选取最大值。和卷积层一样，池化层也是通过窗口（块）大小和步幅尺寸进行参数化。比如，我们可能在一个 10×10 特征矩阵上以 2 的步幅滑动一个 2×2 的窗口，然后选取每个窗口的 4 个值中的最大值，得到一个 5×5 特征矩阵。池化层通过只保留最突出的信息来减少表征的维度；在这个图像输入的例子中，它们为转译提供了基本的不变性（即使图像偏移了几个像素，仍可选出同样的最大值）。池化层通常被安插在连续卷积层之间。

MNIST

MNIST数据集可能是最常用的一个图像识别数据集。它包含 60,000 个手写数字的训练样本和 10,000 个测试样本。每一张图像的尺寸为 28×28像素。目前最先进的模型通常能在该测试集中达到 99.5% 或更高的准确度。

动量（Momentum）

动量是梯度下降算法（Gradient Descent Algorithm）的扩展，可以加速和阻抑参数更新。在实际应用中，在梯度下降更新中包含一个动量项可在深度网络中得到更好的收敛速度（convergence rate）。

论文：通过反向传播（back-propagating error）错误学习表征

多层感知器（MLP：Multilayer Perceptron）

多层感知器是一种带有多个全连接层的前馈神经网络，这些全连接层使用非线性激活函数（activation function）处理非线性可分的数据。MLP 是多层神经网络或有两层以上的深度神经网络的最基本形式。

负对数似然（NLL：Negative Log Likelihood）

参见分类交叉熵损失（Categorical Cross-Entropy Loss）。

神经网络机器翻译（NMT：Neural Machine Translation）

NMT 系统使用神经网络实现语言（如英语和法语）之间的翻译。NMT 系统可以使用双语语料库进行端到端的训练，这有别于需要手工打造特征和开发的传统机器翻译系统。NMT 系统通常使用编码器和解码器循环神经网络实现，它可以分别编码源句和生成目标句。

论文：使用神经网络的序列到序列学习（Sequence to Sequence Learning with Neural Networks）
论文：为统计机器翻译使用 RNN 编码器-解码器学习短语表征（Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation）

神经图灵机（NTM：Neural Turing Machine）

NTM 是可以从案例中推导简单算法的神经网络架构。比如，NTM 可以通过案例的输入和输出学习排序算法。NTM 通常学习记忆和注意机制的某些形式以处理程序执行过程中的状态。

论文：神经图灵机（Neural Turing Machines）

非线性（Nonlinearity）

参见激活函数（Activation Function）。

噪音对比估计（NCE：noise-contrastive estimation）

噪音对比估计是一种通常被用于训练带有大输出词汇的分类器的采样损失（sampling loss）。在大量的可能的类上计算 softmax 是异常昂贵的。使用 NCE，我们可以将问题降低成二元分类问题，这可以通过训练分类器区别对待取样和「真实」分布以及人工生成的噪声分布来实现。

论文：噪音对比估计：一种用于非标准化统计模型的新估计原理（Noise-contrastive estimation: A new estimation principle for unnormalized statistical models ）
论文：使用噪音对比估计有效地学习词向量（Learning word embeddings efficiently with noise-contrastive estimation）

池化

参见最大池化（Max-Pooling）或平均池化（Average-Pooling）。

受限玻尔兹曼机（RBN：Restricted Boltzmann Machine）

RBN 是一种可被解释为一个随机人工神经网络的概率图形模型。RBN 以无监督的形式学习数据的表征。RBN 由可见层和隐藏层以及每一个这些层中的二元神经元的连接所构成。RBN 可以使用对比散度（contrastive divergence）进行有效的训练，这是梯度下降的一种近似。

第六章：动态系统中的信息处理：和谐理论基础
论文：受限玻尔兹曼机简介（An Introduction to Restricted Boltzmann Machines）




## 残差网络（ResNet）

深度残差网络（Deep Residual Network）赢得了 2015 年的 ILSVRC 挑战赛。这些网络的工作方式是引入跨层堆栈的快捷连接，让优化器可以学习更「容易」的残差映射（residual mapping）而非更为复杂的原映射（original mapping）。这些快捷连接和 Highway Layer 类似，但它们与数据无关且不会引入额外的参数或训练复杂度。ResNet 在 ImageNet 测试集中实现了 3.57% 的错误率。

论文：用于图像识别的深度残差网络（Deep Residual Learning for Image Recognition）











