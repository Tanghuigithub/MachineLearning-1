# 循环神经网络

## LSTM（Long short-time memory）

Recurrent neural networks (RNNs) with long short-term memory (LSTM) units (Hochreiter and Schmidhuber, 1997) have been successfully applied to a wide range of NLP tasks, such as machine translation (Sutskever et al., 2014), constituency parsing (Vinyals et al., 2014), language modeling (Zaremba et al., 2014) and recently RTE (Bowman et al., 2015). LSTMs encompass memory cells that can store information for a long period of time, as well as three types of gates that control the flow of information into and out of these cells: input gates (Eq. 2), forget gates (Eq. 3) and output
gates (Eq. 4). Given an input vector xt at time step t, the previous output ht−1 and cell state ct−1, an LSTM with hidden size k computes the next output ht and cell state ct as
- 更新门
- 重置门

## GRU（gated recurrent unit）