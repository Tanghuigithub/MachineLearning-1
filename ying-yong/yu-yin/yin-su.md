# 语音关键词检出 (spoken term detection,STD)

语音关键词检出是从给定的语料数据中查询指定关键词是否出现的任务.该任务在语音检索、电话监控、舆情分析等领域具有广泛的应用.根据关键词的输入形式,该任务可以分为基于文本[１２]和基于语音样例[３１２]两种类型.传统的基于文本的关键词检出技术又称为关键词搜索(keyword search,KWS),通常基于一个大词汇量连续语音识别系统,在识别的NBest结果或者网格上进行搜索.近年来,**基于样例的语音关键词检出** (query-by-example spoken term detection,**QbyE-STD**)成为新的研究热点.
MediaEval多媒体评测组织从2011年起,已经连续４年进行了 QbyE-STD 方面的评测,瞄准跨语种、少资源或零资源(low or zero-resource)场景下的关键词检出任务.在没有任何针对待检语种的专家知识(如音素、词、发音字典等)和标注数据的情况下,无法建立一个有效的语音识别系统,传统基于文本的关键词检出技术无法直接应用.

QbyE-STD目前通常有３种实现方法[３４].

- 借鉴传统语音关键词检出的思路,称之为声学关键词检出(acoustic keyword spotting,AKWS)[３].
- 借助其他语种成熟的语音识别系统进行解码,将查询样例和待检语音转换成音素序列或音
素网格,而后进行字符串匹配,该类方法称之为字符搜索(symbolic search,SS)[５６].此类方法通常采用加权有限状态机(weighted finite state transducer,WFST)建立索引并进行快速查找.
- 基于经典的模板匹配的思路,采用动态时间规整(dynamictimewarping,DTW)[６１３]将查询样例和待查语句进行匹配.由于查询语句的时长通常远远大于样例时长,因此需要在语句上进行**滑动查找**,常用策略包括Segmental-DTW[８]、Subsequence-DTW
[１４]、SLN-DTW (segmental local normalized-DTW)[１５]等.

查询样例转换成音素序列后怎么查找
- 网格？
- 音素边界

## 少资源语言关键词检出评测(QUESST)

跑通了上周提到的完成QUESST任务的一个项目，其使用的是BUT大学提供的`phnrec`工具来提取音素后验概率（音素模型是已经训练好的）。目前在Quesst数据集上跑通了，利用`phnrec`提取了自己采集的数据集的后验概率，之后还要进一步探索。
[QUESST 2014 Multilingual Database for Query-by-Example Keyword Spotting][5]

> QUESST瞄准少资源语言的关键词检出，评测数据中包含多种少资源语言的未标注语音数据，语音文件来自不同录制环境和多种说话风格，部分语音文件带有较强的背景噪声。

> QUESST 2014任务提供了包含23 h的检索语料库，共12492句话。 用于算法调试的发展集关键词560个，最终评测集关键词555个。 语料涉及斯洛伐克语、 罗马尼亚语、 阿尔巴尼亚语、 捷克语、 巴斯克语和英语等6种语言，其中英语部分多来自非母语说话人。 在评测任务期间，语料库不提供任何语种信息。

> 评测任务分为3种不同类型的查询：

> - 1) 精确查询T1： 在语料库中找出与查询关键词精确匹配的地方。
> - 2) 近似查询T2： 允许前后缀不同的匹配，例如给定的关键词是“friend”，可以匹配到语料库中包含“friendly”和“friends”单词的句子。
> - 3) 近似查询T3： 在T2类型的基础上允许填充词和次序颠倒的匹配，例如给定的关键词为“white house”，在语料库中可以找到包含“house is whiter”的句子。

> 评测结果采用最小交叉熵(Cnxe)和TWV(term weighted value)来衡量。 为了计算这2项指标，参加评测的队伍需要提交每个关键词在语料库中每个句子上的打分，用以表示该关键词与该句子的匹配分数。

> - 1.1G
> - dev+valid

Query-by-example spoken term detection (QbE-STD) involves
two main modules: 
1. feature extraction （已解决）
2. detection by dynamic time warping.

输入是spoken query和audio segment，输出audio segment是否包含该query。

# 数据集介绍

采集了16人（8男8女）的语音数据，每人20句话，截选自《小王子》。



## Feature extraction

### 后验概率

The initial step was to run unconstrained phonetic recog- nition on all audio and extract frame-wise posterior prob- ability of phonemes.
基于DTW的语音关键词检出http://jst.tsinghuajournals.com/CN/rhhtml/20170104.htm
从上面的ROC图来看，还有很大提升空间。在尝试提取音素后验概率作为特征：
英文音素效果：
![roc_result.png-124.4kB][3]
2017.8.14 中文音素改進：
![roc_cn_result.png-123.8kB][4]

### CTC的tensorflow实现

[github项目地址](https://github.com/Rowl1ng/phoneme-ctc)

#### ctc_loss
```
ctc_loss(
    labels,
    inputs,
    sequence_length,
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False,
    time_major=True
)
```
最好用named argument

### 后验特征

使用基于Theano的RNN模型在TIMIT数据集（英文）上训练音素识别器：
![phoneme.png-551.3kB][6]
接下来会基于这个识别器在当前采集的数据集上提取特征，再来和MFCC比较效果。

---

# 音素后验概率

给定一帧语音的谱特征向量a，后验特征定义为在K个类别C1,C2,…,CK上的后验概率分布：
$$
P = [P(C_1|a),P(C_2|a),\dots,P(C_K|a)]
$$
$P(C_i|a)$谱特征($a$)在$C_i$(某个音素)上的后验概率，更好的鲁棒性（不同说话人）。 
例如，类别定义为音素，训练基于HMM的音素识别器对语音数据进行`解码`，得到在每个音素上的`打分`即为音素后验特征。 与此类似，针对语音数据训练具有K个成分Gauss混合模型(GMM)，然后对每帧语音数据在K个Gauss成分上打分，即可得到**GMM后验特征**(GMM posterior)。 
一般来说，音素后验特征在基于DTW的QbE-STD任务中的效果最好。 但是对于少资源语言来说，在没有该语种的专家知识和标注数据的情况下，通常无法训练音素识别器。 借用`其他语种`的音素识别器(不匹配音素识别)来获得音素后验特征，是一种普遍采用的做法[10-11, 16]。 此外，采用机器学习方法自动发现某种语言中的`“类音素”子词单元`，获得类音素后验特征也是一种可行的方法[17-18]。

$$
r(q_i,d_j)=\frac{U(q_{i}\cdot d_{j})-||q_i||||d_j||}{\sqrt {(U||q_i||^2-||q_i||)(U||d_j||^2-||d_j||)}}
$$

### Phoneme Unit Selection
$$
r(q_i,d_j,u)=\frac{Uq_{i,u}d_{j,u}-\frac 1U||q_i||||d_j||}{\sqrt {(U||q_i||^2-||q_i||)(U||d_j||^2-||d_j||)}}
$$

$$
R(P(Q,D),u)=\frac 1K \sum_{k=1}^K r(q_i,d_j,u)
$$

## 实验设置

在数据集上对上述3种策略进行了关键词检出测试，采用的语音特征包括：

1. MFCC：39维
2. EN音素后验
3. CN音素后验

### MFCC

使用MFCC特征来计算DTW，根据输出的结果计算的ROC图：

![roc_out.png-105.9kB][2]

发现在不同人之间差别比较大，比如下面的`01`代表第一句话，huang所说的01和所有人说的01计算距离的话浮动很大，尽管在另一个人说的全部20句话中，huang所说的01和另一个人01的距离是最小的：
```
boy_huang_01.wav,boy_liubingnan_01.wav,72.6871660638
boy_huang_01.wav,boy_lu_01.wav,75.8904701408
boy_huang_01.wav,boy_meng_01.wav,63.3867188977
boy_huang_01.wav,boy_wanglei_01.wav,107.501702788
boy_huang_01.wav,boy_zhang_01.wav,66.6690807097
boy_huang_01.wav,girl_chenjingjie_01.wav,101.776886946
boy_huang_01.wav,girl_chenxi_01.wav,95.786444278
boy_huang_01.wav,girl_dong_01.wav,68.7827225926
boy_huang_01.wav,girl_guojingyi_01.wav,98.9626036893
boy_huang_01.wav,girl_linyan_01.wav,104.338175707
```
不确定是否需要一些标准化的措施。

## Detection by dynamic time warping

Variants of dynamic time warping (DTW) have been used to
align two sequences of acoustic features for various tasks, such
as speech pattern discovery [1, 2, 3, 19, 20, 21], story segmentation
[5] and speech summarization [22]. In this paper, to match
通常音素后验的local distance定义为the negative log of inner product.

$$
\rho(x_i,y_j)=-\log(x_i\cdot y_j)
$$
Matlab上dtw的效果：
![dtw.jpg-40.8kB][1]



[2]: http://static.zybuluo.com/sixijinling/2owqberz9bis8rup6y9mbsp1/roc_out.png
  [1]: http://static.zybuluo.com/sixijinling/u36pwzvzmk8lqxp1sszwxo91/dtw.jpg
  [3]: http://static.zybuluo.com/sixijinling/k3qqkx2un8aucd9ntzkd3ac0/roc_result.png
  [4]: http://static.zybuluo.com/sixijinling/u3638nl6ck4q3145kkvuomiv/roc_cn_result.png
  [5]: http://speech.fit.vutbr.cz/software/quesst-2014-multilingual-database-query-by-example-keyword-spotting
  [6]: http://static.zybuluo.com/sixijinling/ynzwvhqqkceam69ye1g3scb0/phoneme.png