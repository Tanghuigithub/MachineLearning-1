# 循环神经网络

## 序列到序列（Seq2Seq）

序列到序列（Sequence-to-Sequence）模型读取一个序列（如一个句子）作为输入，然后产生另一个序列作为输出。它和标准的 RNN 不同；在标准的 RNN 中，输入序列会在网络开始产生任何输出之前被完整地读取。通常而言，Seq2Seq 通过两个分别作为编码器和解码器的 RNN 实现。神经网络机器翻译是一类典型的 Seq2Seq 模型。

论文：使用神经网络的序列到序列学习（Sequence to Sequence Learning with Neural Networks）


## LSTM（Long short-time memory）

Recurrent neural networks (RNNs) with long short-term memory (LSTM) units (Hochreiter and Schmidhuber, 1997) have been successfully applied to a wide range of NLP tasks, such as machine translation (Sutskever et al., 2014), constituency parsing (Vinyals et al., 2014), language modeling (Zaremba et al., 2014) and recently RTE (Bowman et al., 2015). LSTMs encompass memory cells that can store information for a long period of time, as well as three types of gates that control the flow of information into and out of these cells: input gates (Eq. 2), forget gates (Eq. 3) and output
gates (Eq. 4). Given an input vector xt at time step t, the previous output ht−1 and cell state ct−1, an LSTM with hidden size k computes the next output ht and cell state ct as
- 更新门
- 重置门

## GRU（gated recurrent unit）

## 双向循环神经网络（Bidirectional RNN）

双向循环神经网络是一类包含两个方向不同的 RNN 的神经网络。其中的前向 RNN 从起点向终点读取输入序列，而反向 RNN 则从终点向起点读取。这两个 RNN 互相彼此堆叠，它们的状态通常通过附加两个矢量的方式进行组合。双向 RNN 常被用在自然语言问题中，因为在自然语言中我们需要同时考虑话语的前后上下文以做出预测。

论文：双向循环神经网络（Bidirectional Recurrent Neural Networks）
