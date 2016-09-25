# 循环神经网络（RNN：Recurrent Neural Network）

RNN 模型通过隐藏状态（或称记忆）连续进行相互作用。它可以使用最多 N 个输入，并产生最多 N 个输出。比如，一个输入序列可能是一个句子，其输出为每个单词的词性标注（part-of-speech tag）（N 到 N）；一个输入可能是一个句子，其输出为该句子的情感分类（N 到 1）；一个输入可能是单个图像，其输出为描述该图像所对应一系列词语（1 到 N）。在每一个时间步骤中，RNN 会基于当前输入和之前的隐藏状态计算新的隐藏状态「记忆」。其中「循环（recurrent）」这个术语来自这个事实：在每一步中都是用了同样的参数，该网络根据不同的输入执行同样的计算。

技术博客：了解 LSTM 网络（http://colah.github.io/posts/2015-08-Understanding-LSTMs/）
技术博客：循环神经网络教程第1部分——介绍 RNN （http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/）

## 递归神经网络（Recursive Neural Network）

递归神经网络是循环神经网络的树状结构的一种泛化（generalization）。每一次递归都使用相同的权重。就像 RNN 一样，递归神经网络可以使用向后传播（backpropagation）进行端到端的训练。尽管可以学习树结构以将其用作优化问题的一部分，但递归神经网络通常被用在已有预定义结构的问题中，如自然语言处理的解析树中。

论文：使用递归神经网络解析自然场景和自然语言（Parsing Natural Scenes and Natural Language with Recursive Neural Networks ）

## 序列到序列（Seq2Seq）

序列到序列（Sequence-to-Sequence）模型读取一个序列（如一个句子）作为输入，然后产生另一个序列作为输出。它和标准的 RNN 不同；在标准的 RNN 中，输入序列会在网络开始产生任何输出之前被完整地读取。通常而言，Seq2Seq 通过两个分别作为编码器和解码器的 RNN 实现。神经网络机器翻译是一类典型的 Seq2Seq 模型。

论文：使用神经网络的序列到序列学习（Sequence to Sequence Learning with Neural Networks）


## LSTM（Long short-time memory）

长短期记忆（Long Short-Term Memory）网络通过使用内存门控机制防止循环神经网络（RNN）中的梯度消失问题（vanishing gradient problem）。使用 LSTM 单元计算 RNN 中的隐藏状态可以帮助该网络有效地传播梯度和学习长程依赖（long-range dependency）。

论文：长短期记忆（LONG SHORT-TERM MEMORY）
技术博客：理解 LSTM 网络（http://colah.github.io/posts/2015-08-Understanding-LSTMs/）
技术博客：循环神经网络教程，第 4 部分：用 Python 和 Theano 实现 GRU/LSTM RNN（http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/）

Recurrent neural networks (RNNs) with long short-term memory (LSTM) units (Hochreiter and Schmidhuber, 1997) have been successfully applied to a wide range of NLP tasks, such as machine translation (Sutskever et al., 2014), constituency parsing (Vinyals et al., 2014), language modeling (Zaremba et al., 2014) and recently RTE (Bowman et al., 2015). LSTMs encompass memory cells that can store information for a long period of time, as well as three types of gates that control the flow of information into and out of these cells: input gates (Eq. 2), forget gates (Eq. 3) and output
gates (Eq. 4). Given an input vector xt at time step t, the previous output ht−1 and cell state ct−1, an LSTM with hidden size k computes the next output ht and cell state ct as

- 更新门
- 重置门

## GRU（gated recurrent unit）

GRU（Gated Recurrent Unit：门控循环单元）是一种 LSTM 单元的简化版本，拥有更少的参数。和 LSTM 细胞（LSTM cell）一样，它使用门控机制，通过防止梯度消失问题（vanishing gradient problem）让循环神经网络可以有效学习长程依赖（long-range dependency）。GRU 包含一个复位和更新门，它们可以根据当前时间步骤的新值决定旧记忆中哪些部分需要保留或更新。

论文：为统计机器翻译使用 RNN 编码器-解码器学习短语表征（Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation）
技术博客：循环神经网络教程，第 4 部分：用 Python 和 Theano 实现 GRU/LSTM RNN（http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/）

## 双向循环神经网络（Bidirectional RNN）

双向循环神经网络是一类包含两个方向不同的 RNN 的神经网络。其中的前向 RNN 从起点向终点读取输入序列，而反向 RNN 则从终点向起点读取。这两个 RNN 互相彼此堆叠，它们的状态通常通过附加两个矢量的方式进行组合。双向 RNN 常被用在自然语言问题中，因为在自然语言中我们需要同时考虑话语的前后上下文以做出预测。

论文：双向循环神经网络（Bidirectional Recurrent Neural Networks）

## 应用

### 神经网络机器翻译（NMT：Neural Machine Translation）

NMT 系统使用神经网络实现语言（如英语和法语）之间的翻译。NMT 系统可以使用双语语料库进行端到端的训练，这有别于需要手工打造特征和开发的传统机器翻译系统。NMT 系统通常使用编码器和解码器循环神经网络实现，它可以分别编码源句和生成目标句。

论文：使用神经网络的序列到序列学习（Sequence to Sequence Learning with Neural Networks）
论文：为统计机器翻译使用 RNN 编码器-解码器学习短语表征（Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation）