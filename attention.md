# Attention

Attention是选择显著区域的人眼视觉过程，这方面的算法模型注重给出fixation-prediction，典型的有Itti和Koch 1998年给出的[视觉注意机制模型][1]，attention主要是从人眼产生注意的过程和视觉系统的特征研究算法模型。

## Machine Translation
最早将Attention-based model引入NLP的就是2015年ICLR，Bahdanau的[《Neural machine translation by jointly learning to align and translate》][2]。
>  The decoder decides parts of the source sentence to pay **attention** to. By letting the decoder have an **attention mechanism**, we relieve the encoder from the burden of having to encode all information in the source sentence into a fixedlength vector.

### Motivation
传统翻译模型将整句的信息压缩至定长向量，使得长句上的翻译效果差。
soft-align：在进行翻译时，虽然语序会变，但大体的语义部分是有对应关系的，如果能找到这个关系，就不用整句整句encode。
### Approach
新模型**RNNsearch**：
对每一个target word $y_i$,都计算一个context vector $c_i$——用softmax刻画“expected annotation”。$y_i$究竟对应哪一个annotation，可以看成一种分布（多分类），这样就可以用一个权重/概率的期望来刻画。
$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$
- 对齐（Align）与翻译同时进行。





  [1]: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=730558
  [2]: http://www.cl.uni-heidelberg.de/courses/ws14/deepl/BahdanauETAL14.pdf