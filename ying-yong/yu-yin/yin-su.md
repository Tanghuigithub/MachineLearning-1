# 音素识别

采集了16人（8男8女）的语音数据

## DTW

Matlab上dtw的效果：
![dtw.jpg-40.8kB][1]

#### Quesst

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

### 后验概率
基于DTW的语音关键词检出http://jst.tsinghuajournals.com/CN/rhhtml/20170104.html
从上面的ROC图来看，还有很大提升空间。在尝试提取音素后验概率作为特征：
英文音素效果：
![roc_result.png-124.4kB][3]
2017.8.14 中文音素改進：
![roc_cn_result.png-123.8kB][4]
#### CTC的tensorflow实现
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
  [1]: http://static.zybuluo.com/sixijinling/u36pwzvzmk8lqxp1sszwxo91/dtw.jpg
  [3]: http://static.zybuluo.com/sixijinling/k3qqkx2un8aucd9ntzkd3ac0/roc_result.png
  [4]: http://static.zybuluo.com/sixijinling/u3638nl6ck4q3145kkvuomiv/roc_cn_result.png
  [5]: http://speech.fit.vutbr.cz/software/quesst-2014-multilingual-database-query-by-example-keyword-spotting
  [6]: http://static.zybuluo.com/sixijinling/ynzwvhqqkceam69ye1g3scb0/phoneme.png