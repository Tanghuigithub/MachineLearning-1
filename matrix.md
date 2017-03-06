# 矩阵

### 行列式

### 0.4.1 行列の定義

Dimension of matrix : number of rows\(行\) x number of columns\(列\)

$$\left(
  \begin{array}{cccc}
    a{11} & a{12} & \ldots & a{1n} \\
    a{21} & a{22} & \ldots & a{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a{m1} & a{m2} & \ldots & a_{mn}
  \end{array}
\right)$$

$$a_{ij}$$ :  第$$ i $$行，第$$ j $$列

常用大写字母表示矩阵，小写表示向量

### 0.4.2 向量の定義

An $$n \times 1$$ matrix  
$$\left(
  \begin{array}{c}
    x_1 \\
    x_2 \\
    \vdots \\
   x_n  
  \end{array}
\right)$$ のようものをベクトルといいます。ベクトルを文字で表す時は $${\bf a }$$ のようにアルファベット小文字の太字で表すことが多いです。  
また、単にベクトルと言ったときはたいてい数字を縦に並べた縦ベクトルを指します。$$n$$個の数を並べて作ったベクトルを$$n$$次元縦ベクトルと呼び、上から$$i$$番目の数を「第$$i$$成分」と呼びます。

縦ベクトルをスペースの関係で横に並べて表記する場合は後述する転置記号を用いて $$\left(
  \begin{array}{cccc}
    x_1 x_2 \ldots x_n  
  \end{array}
\right)^T$$ のように書くこともあります。

（縦）ベクトルの例:  
$$\left(
  \begin{array}{ccc}
    2 \\
    5
  \end{array}
\right),
\left(
  \begin{array}{ccc}
    5 \\
    8 \\
    9 \\
    2
  \end{array}
\right)$$

### 0.4.3 基本演算

この節では簡単のためにベクトルを横1列の行列の一種として扱います。

如果两个矩阵$$A=(a_{ij})$$和$$B=(b_{ij})$$$$的行数和列数分别相等（也称**同型矩阵**），且各对应元素也相等，即$$$$a_{ij}=b_{ij}
(1=1,2,…,m;j=1,2,…,n)$$,就称$$A$$和$$B$$相等，记作$$A=B$$.  
例:  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) =
\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right)$$  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) \ne
\left(
  \begin{array}{ccc}
    1 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right)$$  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) \ne
\left(
  \begin{array}{c}
    2 \\
    5
   \end{array}
\right)$$

2つの行列の足し算・引き算は次元が同じときにのみ行うことができ、結果は対応する次元同士の足し算・引き算となります。

例:  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) +
\left(
  \begin{array}{ccc}
    1 & 1 & 1\\
    2 & 2 & 2
   \end{array}
\right) =
\left(
  \begin{array}{ccc}
    3 & 4 & 5\\
    7 & 8 & 9
   \end{array}
\right)$$  
次の計算は次元が違うため実行できません。  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) +
\left(
  \begin{array}{cc}
    1 & 1 \\
    2 & 2
   \end{array}
\right)$$

行列にスカラー値（いわゆる普通の一つの数）を掛けるときは、各成分にその値を掛けます

例:  
$$2\cdot \left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) =
\left(
  \begin{array}{ccc}
    4 & 6 & 8\\
    10 & 12 & 14
   \end{array}
\right)$$

例:  
$$\left(
  \begin{array}{ccc}
    2 & 3 & 4\\
    5 & 6 & 7
   \end{array}
\right) \cdot
\left(
  \begin{array}{cc}
    1 & 2 \\
    3 & 4 \\
    5 & 6
   \end{array}
 \right) =
 \left(
   \begin{array}{cc}
     2\cdot 1+3\cdot 3+4\cdot 5 & 2\cdot 2+3\cdot 4+4\cdot 6 \\
     5\cdot 1+6\cdot 3+7\cdot 5 & 5\cdot 2+6\cdot 4+7\cdot 6
    \end{array}
  \right) \\
  =\left(
    \begin{array}{cc}
      31 & 40 \\
      58 & 76
     \end{array}
   \right)$$

**prediction = Datamatrix x Parameters**

$$行数と列数が同じ行列（正方行列）では繰り返し掛け算を行うことができ、Aを$$n$$回掛けて得られる行列を$$A^n$$と書きます。

行列の基本演算について、$$A+B=B+A$$のような普通の数の計算で成り立つ法則はだいたい成り立ちますが、$$AB = BA$$ **とは限らない**ことに注意してください。そもそも$$AB$$が計算できても$$BA$$が計算できない場合もあります。

### 0.4.4 いろいろな行列

行数と列数が同じ行列を**正方行列**と言います。

正方行列のうち、対角成分（行数と列数が等しい第$$(i, i)$$成分のこと）のみが1、その他の成分は全て0となる
$$ \left(
  \begin{array}{cccc}
    1 & 0 & \ldots & 0 \\
    0 & 1 & \ldots & 0 \\
     &  & \ddots &  \\
    0 & 0 & \ldots & 1
  \end{array}
 \right)$$$$

のような行列を**単位行列**と言います。単位行列は記号$$E$$で表すことが多いです。

正方行列$$A$$に対して、$$A$$に掛けると積が$$E$$になる行列をAの逆行列と呼び、$$A^{-1}$$ と書きます。  
$$ AA^{-1}=A^{-1}A=E $$

行列$$A$$に対して、行と列を入れ替えたものを転置行列と呼び、$$A^T$$と書きます。  
$$\left\(  
  \begin{array}{cccc}  
    a_{11} & a_{12} & \ldots & a_{1n} \  
    a_{21} & a_{22} & \ldots & a_{2n} \  
    \vdots & \vdots & \ddots & \vdots \  
    a_{m1} & a_{m2} & \ldots & a_{mn}  
  \end{array}  
  \right\)^T =  
  \left\(  
    \begin{array}{cccc}  
      a_{11} & a_{21} & \ldots & a_{m1} \  
      a_{1n} & a_{22} & \ldots & a_{m2} \  
      \vdots & \vdots & \ddots & \vdots \  
      a_{1n} & a_{2n} & \ldots & a_{mn}  
    \end{array}  
    \right\)


$$
### 奇异矩阵 

首先，看这个矩阵是不是方阵（即行数和列数相等的矩阵。若行数和列数不相等，那就谈不上奇异矩阵和非奇异矩阵）。 然后，再看此矩阵的行列式$$|A|$$是否等于0，若等于0，称矩阵$$A$$为奇异矩阵；若不等于0，称矩阵 $$A$$ 为非奇异矩阵。 
同时，由$$|A|\neq 0$$可知矩阵A可逆，这样可以得出另外一个重要结论:**可逆矩阵就是非奇异矩阵，非奇异矩阵也是可逆矩阵**。　
如果A为奇异矩阵，则$$AX=0$$有无穷解，$$AX=b$$有无穷解或者无解。如果A为非奇异矩阵，则$$AX=0$$有且只有唯一零解，$$AX=b$$有唯一解。

### 正定矩阵 

埃尔米特矩阵（英语：Hermitian matrix，又译作厄米矩阵），也称自伴随矩阵，是共轭对称的方阵。埃尔米特矩阵中每一个第i行第j列的元素都与第j行第i列的元素的共轭相等。
实对称矩阵是埃尔米特矩阵的特例。

一个$$n×n$$的实对称矩阵$$M$$是正定的，当且仅当对于所有的非零实系数向量$$z$$，都有$$z^TMz > 0$$。其中$$z^T$$表示$$z$$的转置。

### 0.4.5 ベクトルの内積とノルム

2つのn次元実ベクトル $${\bf a}$$ と $${\bf b}$$ について内積 $${\bf a}\cdot {\bf b}$$ という値が定義されています。
$$ {\bf a} \cdot {\bf b} = a_1b_1 + a_2b_2 + \dots + a_n+b_n
$$


と表されます。

n次元実ベクトル $${\bf a}$$ について、ノルムという値が定義されています。  
$$||{\bf a} || = \sqrt{\sum_{i=1}^n a_i^2}$$

### 0.4.6 特征值

正方行列$$A$$について、  
$$A {\bf x} = \alpha {\bf x}$$  
が成り立つとき、$$\alpha$$を$$A$$の固有値、$${\bf x}$$ を$$A$$の $$\alpha$$に関する固有ベクトルと呼びます。  
一般来说，2×2的**非奇异矩阵**如果有两个相异的特征值，就有两个线性无关的特征向量。在这种情况下，对于特征向量，线性变换仅仅改变它们的长度，而不改变它们的方向（除了反转以外），而对于其它向量，长度和方向都可能被矩阵所改变。如果特征值的模大于1，特征向量的长度将被拉伸，而如果特征值的模小于1，特征向量的长度就将被压缩。如果特征值小于0，特征向量将会被翻转。

