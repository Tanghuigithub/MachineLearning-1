# 注意机制（Attention Mechanism）


## Neural Turing Machines

Instead of specifying a single location, the RNN gives “attention distribution” which describe how we spread out the amount we care about different memory positions. As such, the result of the read operation is a weighted sum.


## Attentional Interfaces

We’d like attention to be differentiable, so that we can learn where to focus. To do this, we use the same trick Neural Turing Machines use: we focus everywhere, just to different extents.

Attention can also be used on the interface between a convolutional neural network and an RNN. This allows the RNN to look at different position of an image every step. One popular use of this kind of attention is for image captioning. First, a conv net processes the image, extracting high-level features. Then an RNN runs, generating a description of the image. As it generates each word in the description, the RNN focuses on the conv nets interpretation of the relevant parts of the image. We can explicitly visualize this:

![](http://distill.pub/2016/augmented-rnns/assets/show-attend-tell.png)

注意机制是由人类视觉注意所启发的，是一种关注图像中特定部分的能力。注意机制可被整合到语言处理和图像识别的架构中以帮助网络学习在做出预测时应该「关注」什么。

技术博客：深度学习和自然语言处理中的注意和记忆[链接](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

Attention是选择显著区域的人眼视觉过程，这方面的算法模型注重给出fixation-prediction，典型的有Itti和Koch 1998年给出的[视觉注意机制模型][1]，attention主要是从人眼产生注意的过程和视觉系统的特征研究算法模型。

## 应用

### 1. Machine Translation

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

### 2. Image Caption

![](http://static.zybuluo.com/sixijinling/pu4yhiw0im7t8fj30ruxvkpv/image_1aocgvgqt1hn5l5k6jt1rbiei411.png)
**《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* 》**
主要思想还是和上次讲的机器翻译类似：

$$
\alpha _{ti}=\frac {\exp(e_{ti})}{\sum _{k-1}^L \exp(e_{tk})}
$$

看成是**multinoulli**分布
而这里的context vector计算为：
$$\hat z_t=\phi(\{a_i\},\{\alpha _i\})$$

## Adaptive Computation Time

## Neural Programmer

## Reference
---

- [Augmented-RNNs](http://distill.pub/2016/augmented-rnns/#neural-turing-machines)

  [1]: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=730558
  [2]: http://www.cl.uni-heidelberg.de/courses/ws14/deepl/BahdanauETAL14.pdf