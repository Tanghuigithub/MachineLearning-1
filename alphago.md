# AlphaGo

开源
缩减

 - 广度：减少需要模拟的落子选项 
    - imitating expert moves(supervised learning)
    - 强化学习（self-plays）

、

 - 深度：Board Evaluation

> 搜索的本义，是寻找罗马的过程而非罗马本身！
……
从数学上讲，我们把迷宫的开始叫做“根”，每一个路口叫做“结点”，路口的每一个路叫做“分支”，每一个无路可走的状态叫做“叶”，那么走迷宫所有的变化就变成了一棵“树”。搜索的过程，就是按照某个顺序遍历这棵树，直到找到出口的叶子或者找遍所有的叶子。
……
在有限的时间和选择里，我们还能找到罗马吗？
——梅俏竹

深度优先搜索/广度优先搜索
A*search
minmax algorithm
Alpha-beta剪枝
精确的局势评估:静态->动态搜索
Crazy Stone 、Zen

## Monte Carlo Tree Search

>像这样，在确定时间内完成的随机算法，就叫做蒙特卡洛。
……
用模拟对局的最终胜率而不是评分值，来评价当前局面的好坏。

问题：
>exploration vs. exploitation

**multi-armed bandit**,UCB策略->UCT(UCB applied to trees)
 
### 4月9号更新
用Pycharm直接从

## 环境

### Cython==0.23.4

Cython的主要目的是： 简化python调用c语言程序的繁琐封装过程，提高python代码执行速度（C语言的执行速度比python快）

    sudo apt-get install cython
### scipy==0.17.0 numpy==1.10.4

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

#### 安装Keras==0.3.2
这就没什么好说的了，自己下载下来就行了，keras Github地址（https://github.com/fchollet/keras）。

    python setup.py develop --user



    pip install pyyaml
### six==1.10.0
Six is a Python 2 and 3 compatibility library. It provides utility functions for smoothing over the differences between the Python versions with the goal of writing Python code that is compatible on both Python versions. 
    pip install six

### wheel==0.29.0

    sudo apt-get install python-pip
    pip install wheel
### 安装Theano和keras
-e git://github.com/Theano/Theano.git@eab9cf5d594bac251df57885509394d2c52ccd1a#egg=Theano
    git clone git://github.com/Theano/Theano.git
    cd Theano
    python setup.py develop --user
    cd ..

执行之后，将Theano目录下的theano目录拷贝到python安装目录下的dist-package下就可以了，我的机器是/usr/lib/python2.7/dist-packages

    
#### 安装sublime

运行：

    subl
    

## 开始下棋辣
今天（2016.3.16）看纪录片《围棋》第五集才知道"气"的英文是“liberty”

## go.py
基本的设置
    WHITE = -1
    BLACK = +1
    EMPTY = 0
    PASS_MOVE = None


`self.liberty_sets` is a 2D array with the same indexes as `board` each entry points to a set of tuples - the liberties of a stone's connected block. By caching liberties in this way, we can directly optimize update functions (e.g. do_move) and in doing so indirectly speed up any function that queries liberties
neighbors and diagnals

- 矩阵liberty_count用-1初始化；
- group_sets保存和（x,y）相连的所有落子；

方法：

- get_group给出某个子相连的所有同色棋子
- get_groups_around返回某子周围孤立的子群
### 界面
` /interface/server`
### Alphgo/ 
#### models/


`preprocessing.py`
a class to convert from AlphaGo GameState objects to tensors of **one-hot** features for NN inputs
`game_converter.py`