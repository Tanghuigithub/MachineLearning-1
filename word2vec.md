# word2vec

NNLM第五周的编程作业是用Matlab写网络预测下一个词，输出词向量语言模型。之前虽然看过神经网络语言模型的论文，也用词向量模型跑过NLP任务，但从来没有自己动手实现过，也就更不可能思考一些实现上的细节计算问题。

这次编程作业提供好了大部分的关键模块，以ABCD选择的形式让学生自己选出正确的实现语句，可以说在很大程度上简化了实现过程，也打消了畏难心理。

首先来理解嵌入的概念：

## 嵌入（Embedding）

一个嵌入映射到一个输入表征，比如一个词或一句话映射到一个矢量。一种流行的嵌入是词语嵌入（word embedding，国内常用的说法是：词向量），如 word2vec 或 GloVe。我们也可以嵌入句子、段落或图像。比如说，通过将图像和他们的文本描述映射到一个共同的嵌入空间中并最小化它们之间的距离，我们可以将标签和图像进行匹配。嵌入可以被明确地学习到，比如在 word2vec 中；嵌入也可作为监督任务的一部分例如情感分析（Sentiment Analysis）。通常一个网络的输入层是通过预先训练的嵌入进行初始化，然后再根据当前任务进行微调（fine-tuned）。

## word2vec

word2vec 是一种试图通过预测文档中话语的上下文来学习词向量（word embedding）的算法和工具 (https://code.google.com/p/word2vec/)。最终得到的词矢量（word vector）有一些有趣的性质，例如vector('queen') ~= vector('king') - vector('man') + vector('woman') （女王~=国王-男人+女人）。两个不同的目标函数可以用来学习这些嵌入：Skip-Gram 目标函数尝试预测一个词的上下文，CBOW  目标函数则尝试从词上下文预测这个词。

论文：向量空间中词汇表征的有效评估（Efficient Estimation of Word Representations in Vector Space）
论文：分布式词汇和短语表征以及他们的组合性（Distributed Representations of Words and Phrases and their Compositionality）
论文：解释 word2vec 参数学习（word2vec Parameter Learning Explained）

## Abstract

Word2vec training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application.

##Introduction

The success of machine learning methods in NLP tasks depends much on word representation, since different representations may encode different explanatory factors of variation behind the word. With the rapid development of deep learning techniques,researchers have started to train complex and deep models on large amounts of text corpus, to learn distributed representations of words(also known as word embeddings) in the form of continuous vectors.
While conventional NLP techniques usually represent words as indices in a vocabulary causing no notion of relationship between words, word embeddings learned by deep learing approaches aim at explicitly encoding many semantic relationships as well as linguistic regularities and patterns into the new word embedding space.
 不同于传统one-hot，distributed词向量包含了词与词之间的关联
In this paper, we introduce a benchmark collection\cite{how}, which is built from several different data source, to measure quality of word embeddings from different aspects.
复现了若干个task，来衡量词向量的优劣。

##Models

### 1. NNLM

这里看的是Bengio的[论文][1]
Bengio et al. \cite{nnlm} first proposed a Neural Network Language Model (NNLM) that simultaneously learns a word embedding and a language model.The language model utilizes several previous words to predict the distribution of the next word.For each sample in the corpus ,we maximize the log-likelihood of the probability of the last word given the previous words.This model uses a concatenation of the previous words' embeddings as the input.The model structure is a feed-forward neural network with one hidden layer.

### 2. LBL

The Log-Bilinear Language Model(LBL) proposed by Mnih and Hinton combines Bengio's Hierachical NNLM and Log Bi-Linear.It uses a log-bilinear energy function that is almost equal to that of the NNLM and removes the non-linear activation function tanh. 

A previous study \cite{lbl} proposed a widely used model architecture for estimating neural network language model.

### GloVe

Glove 是一种为话语获取矢量表征（嵌入）的无监督学习算法。GloVe 的使用目的和 word2vec 一样，但 GloVe 具有不同的矢量表征，因为它是在共现（co-occurrence）统计数据上训练的。

论文：GloVe：用于词汇表征（Word Representation）的全局矢量（Global Vector）（GloVe: Global Vectors for Word Representation ）

## 2. Word Vector  {#ID12}  

在训练模型前，对语料构建合适的词向量模型。
使用评论训练词向量模型 理论上，可以使用不同的语料进行训练或者直接使用已经训练好的。


### 1.3 NLTK

#### 1、 Sentences Segment（分句）

可以使用NLTK中的 punkt sentence segmenter。
```python
nltk.set_proxy("**.com:80") //设置代理
nltk.download('punkt')
```



### 1.2 读取数据
    
```python
        train = []
        self.cur.execute("SELECT * FROM Bug_Report_Data")
        for row in self.cur:
            review = str(row[16])
            review= str(review.decode('utf-8', errors='ignore'))
            train.append((review, 'bug'))
```


#### Training 

##### Param
- `sg` defines the **training algorithm**. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
- `size` 
= the dimensionality of the feature vectors.
the size of the NN layers, which correspond to the “degrees” of freedom the training algorithm has.Bigger size values require more training data, but can lead to better (more accurate) models. Reasonable values are in the tens to hundreds.
- `window` is the maximum distance between the current and predicted word within a sentence.
- `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
- `seed` = for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed).
- `min_count` 
= ignore all words with total frequency lower than this.
It is for pruning the internal dictionary. Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage. In addition, there’s not enough data to make any meaningful training on those words, so it’s best to ignore them.
- `max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM. Set to `None` for no limit (default).
- `sample` = threshold for configuring which higher-frequency words are randomly downsampled; default is 1e-3, useful range is (0, 1e-5).
- `workers` 
= use this many worker threads to train the model (=faster training with multicore machines).for training parallelization, to speed up training.
The workers parameter has only effect if you have **Cython** installed. Without Cython, you’ll only be able to use one core because of the GIL (and word2vec training will be miserably slow).
- `hs` = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
- `negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
- `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean. Only applies when cbow is used.
- `hashfxn` = hash function to use to randomly **initialize** weights, for increased training reproducibility. Default is Python's rudimentary built in hash function.
- `iter` = number of iterations (epochs) over the corpus.
- `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count). Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and returns either `util.RULE_DISCARD`, `util.RULE_KEEP` or `util.RULE_DEFAULT`. Note: The rule, if given, is only used prune vocabulary during `build_vocab()` and is not stored as part of the model.
- `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before  assigning word indexes.
- `batch_words` = target size (in words) for batches of examples passed to worker threads (and thus cython routines). Default is 10000. (Larger batches can be passed if individual texts are longer, but the cython code may truncate.)

### 2.1 Construct Matrix

zero-pad or shortan 文本至固定大小 $n$.
从`sentences`迭代器初始化模型. Each sentence is a list of words (unicode strings) that will be used for training.
The `sentences` iterable can be simply a list, but for larger corpora,consider an iterable that streams the sentences directly from disk/network.
See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in this module for such examples.
If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it in some other way.



### 2.1.1 Build Vocabulary

Build vocabulary from a sequence of sentences (can be a once-only generator stream). Each sentence must be a list of **unicode** strings.

```python
self.scan_vocab(sentences, trim_rule=trim_rule)  # initial survey
self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
self.finalize_vocab()  # build tables & arrays
```

Create a binary **Huffman tree** using stored vocabulary word counts. Frequent words will have shorter binary codes. Called internally from `build_vocab()`

 Get all the word2vec vectors in a 2D matrix and fit the scaler on it. This scaler can be used afterwards for normalizing feature matrices.
     
```python
def fit_scaler(data_dir, word2vec_model=WORD2VEC_MODELPATH, batch_size=1024, persist_to_path=None):
```

#### 2.1.2 初始化matrix

```python
self.vocab = {}  # mapping from a word (string) to a Vocab object
self.index2word = []  # map from a word's matrix index (int) to word (string)
```

选取一个seed string，对vocabulary中的word逐个初始化random vector。

```python
once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
return (once.rand(self.vector_size) - 0.5) / self.vector_size
```
### 2.2 训练 word vector :

```
    >>> from magpie import MagpieModel
    >>> model = MagpieModel()
    >>> model.train_word2vec('/path/to/training-directory', vec_dim=100)
```

### 采集数据 


### 2.2 gensim.model.Word2Vec

```python
>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
```
$\not 0$
不继续训练的话，调用
```python
model.init_sims(replace=True)
```
L2 normalize：
$
y_i=\frac{x_i}{\sum_{i=1}^n{x_1}^2}
$

cpu_count=8

```
$ python train_word2vec_model.py wiki.en.txt wiki.en.text.model wiki.en.text.vector
```
 calling Word2Vec(sentences) will run two passes over the sentences iterator. 
 
- The first pass collects words and their frequencies to build an internal dictionary tree structure.
- The second pass trains the neural model.
```
2016-05-08 22:11:43,575: INFO: running train_word2vec_model.py wiki.en.txt wiki.en.text.model wiki.en.text.vector
2016-05-08 22:11:43,575: INFO: collecting all words and their counts
2016-05-08 22:11:43,599: INFO: PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2016-05-08 22:11:52,334: INFO: PROGRESS: at sentence #10000, processed 29237917 words, keeping 426680 word types
...
2016-05-08 22:25:14,292: INFO: PROGRESS: at sentence #4050000, processed 2197557235 words, keeping 8508124 word types
2016-05-08 22:25:14,760: INFO: collected 8511566 word types from a corpus of 2198527566 raw words and 4053349 sentences
2016-05-08 22:25:32,895: INFO: min_count=5 retains 2082765 unique words (drops 6428801)
2016-05-08 22:25:32,895: INFO: min_count leaves 2188708620 word corpus (99% of original 2198527566)
2016-05-08 22:25:44,397: INFO: deleting the raw counts dictionary of 8511566 items
2016-05-08 22:25:45,418: INFO: sample=0.001 downsamples 23 most-common words
2016-05-08 22:25:45,419: INFO: downsampling leaves estimated 1781672804 word corpus (81.4% of prior 2188708620)
2016-05-08 22:25:45,419: INFO: estimated required memory for 2082765 words and 100 dimensions: 2707594500 bytes
2016-05-08 22:25:52,354: INFO: resetting layer weights
2016-05-08 22:26:25,949: INFO: training model with 8 workers on 2082765 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5
2016-05-08 22:26:25,949: INFO: expecting 4053349 sentences, matching count from corpus used for vocabulary survey
...
2016-05-09 02:44:26,985: INFO: worker thread finished; awaiting finish of 4 more threads
2016-05-09 02:44:26,988: INFO: worker thread finished; awaiting finish of 3 more threads
2016-05-09 02:44:27,009: INFO: worker thread finished; awaiting finish of 2 more threads
2016-05-09 02:44:27,009: INFO: worker thread finished; awaiting finish of 1 more threads
2016-05-09 02:44:27,015: INFO: worker thread finished; awaiting finish of 0 more threads
2016-05-09 02:44:27,016: INFO: training on 10992637830 raw words (8908325719 effective words) took 15481.1s, 575434 effective words/s
2016-05-09 02:44:27,035: INFO: saving Word2Vec object under wiki.en.text.model, separately None
2016-05-09 02:44:27,035: INFO: storing numpy array 'syn1neg' to wiki.en.text.model.syn1neg.npy
2016-05-09 02:44:35,705: INFO: not storing attribute syn0norm
2016-05-09 02:44:35,705: INFO: storing numpy array 'syn0' to wiki.en.text.model.syn0.npy
2016-05-09 02:44:51,486: INFO: not storing attribute cum_table
2016-05-09 02:45:36,145: INFO: storing 2082765x100 projection weights into wiki.en.text.vector
```
得到2G的wiki.en.text.vector
```
2082765 100
the 3.006923 0.197016 1.821211 0.577468 1.200783 -2.143173 -2.189645 2.280789 -0.500885 -0.029055 -1.866612 -3.537239 4.109666 2.681008 1.685016 1.846582 -4.121732 3.391886 -2.395795 -2.229913 2.567938 -2.872733 -3.175062 1.440397 0.989027 3.137877 -4.718245 -0.139462 -0.581739 -1.701072 -2.628166 0.748065 -0.704107 2.452936 -3.509637 -2.165193 3.200839 2.084164 -0.610669 -7.839321 2.376417 3.354800 1.399709 2.877409 -6.148983 -1.288296 -1.481419 1.514872 1.643264 -0.084024 -0.421993 1.965687 2.487681 -0.201347 1.095493 1.406878 -1.837678 -2.597307 -3.908334 -0.431535 1.659305 -1.325693 -1.448273 0.911582 -0.185698 -1.997922 -1.835947 -1.374178 2.162223 -1.981623 -1.046288 0.450714 0.966067 1.800625 -0.736050 -1.816792 -3.730878 -1.215927 0.920658 2.609656 -2.049159 -0.126193 1.389721 0.238941 1.667714 -1.431075 -3.288007 0.063848 -2.413035 0.897759 -3.347217 -2.927267 2.743397 -1.251601 -0.985500 -1.804784 -2.669296 -2.585315 2.603862 2.024329
```

##### Online training / Resuming training

## Multi-Label
### Binary Relevance (BR) 


## 工具类

### 1.Document.py

#### stopwords

```python
return [w for w in self.get_all_words() if w not in STOPWORDS]

```
、stemmer处理之后得到关键词
```python
def get_all_words(self):
        """ Return all words tokenized, in lowercase and without punctuation """
        return [w.lower() for w in word_tokenize(self.text)
                if w not in PUNCTUATION]
```

利用nltk的tokenize:

```python
def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
```


  [1]: http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf