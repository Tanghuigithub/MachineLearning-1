# SMO

1. 设训练样本数目为$N$，原始观测空间的维数为$m$。支持向量机所对应的二次规划问题往往使用**SMO**(Sequential Minimal Optimization)算法求解。SMO算法不断地求解仅涉及两个优化变量的二次规划问题。设给定训练样本为$\{(\mathbf x_i,y_i)\}_{i=1}^N$，对偶问题为：
$$
D(\alpha)=\sum _{i=1}^N \alpha_i-\frac12 \sum _{i=1}^N \sum _{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf x_i^{\mathrm T} \mathbf x_j \\
s.t. \sum _{i=1}^N \alpha_i y_i=0, \alpha_i>0, i=1,\dots,N
$$。
具体地，在每次迭代时，选中两个优化变量$\alpha_{i^*}$和$\alpha_{j^*}$，同时保持其它变量固定，求解关于$\alpha_{i^*}$和$\alpha_{j^*}$的二次规划问题。请结合线性支持向量机（分为线性可分和线性不可分情况），给出在每次迭代过程中关于两个变量$\alpha_{i^*}$和$\alpha_{j^*}$的解。

SVM使用一种非线性映射，把原训练数据映射到较高的维。在新的维上，搜索最佳分离超平面，两个类的数据总可以被超平面分开。

## 1. 线性可分

要找到具有“最大间隔”的划分超平面，即
$$
\max_{w,b}\frac 2{||w||}\\
s.t. y_i(w^\mathrm{T}x_i+b)\geq 1,i=1,2,\dots,N.
$$
等价于
$$
\max_{w,b}\frac 12{||w||}^2\\
s.t. y_i(w^\mathrm{T}x_i+b)\geq 1,i=1,2,\dots,N.
$$


## 2. 线性不可分



线性可分问题的支持向量机学习方法，对线性不可分训练数据是不适用的，为了满足函数间隔大于1的约束条件，可以对每个样本$(\mathbf x_i,y_i)$引进一个松弛变量$ξ_i≥0$，使函数间隔加上松弛变量大于等于1：
$$
y_i(w \cdot x_i+b)\geq 1-ξ_i
$$
目标函数为
$$
\frac 12 ||w||^2 +C\sum_{j=1}^N ξ_i
$$
其中$C>0$为惩罚参数。
因此，最小化目标函数也就是使$\frac12||w||^2$尽量小，同时使误分类点的个数尽量小。线性不可分的线性支持向量机问题变成下面的凸二次规划问题：
$$
\min_{w,b,\xi}\frac12||w||^2+C\sum_{i=1}^N\xi_i\\
s.t.  y_i(w \cdot x_i +b)\geq 1-\xi_i,i=1,2,\dots,N,\xi_i\geq0,i=1,2,\dots,N
$$

## 3. 对偶问题
使用拉格朗日乘子法得到其“对偶问题”：
$$
L(w,b,\alpha)=\frac 12{||w||}^2+\sum_{i=1}^N\alpha_i(1-y_i(w^\mathrm{T}x_i+b))
$$
其中$\alpha=(\alpha_1;\alpha_2;\dots;\alpha_m)$。令$L(w,b,\alpha)$中的$w$和$b$的偏导为0可得

$$
w=\sum_{i=1}^N\alpha_iy_ix_i,\\
0=\sum_{i=1}^N\alpha_iy_i.
$$
即可将$L(w,b,\alpha)$中的$w$和$b$消去，得到对偶问题
$$
D(\alpha)=\sum _{i=1}^N \alpha_i-\frac12 \sum _{i=1}^N \sum _{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf x_i^{\mathrm T} \mathbf x_j \\
s.t. \sum _{i=1}^N \alpha_i y_i=0, \alpha_i>0, i=1,\dots,N
$$


## 4. SMO算法

约束$\sum _{i=1}^N \alpha_i y_i=0$重写为：
$$
\alpha_{i^*}y_{i^*} +\alpha_{j^*}y_{j^*}=c,\alpha_{i^*}\geq 0,\alpha_{j^*}\geq 0\\
c=-\sum _{k \neq i^*,j^*}\alpha_k y_k
$$
用上式中的$c$消去$D(\alpha)$中的$\alpha _j$：
$$
\alpha_{j^*}=(c-\alpha_{i^*}y_{i^*})y_{j^*}
$$
假设$i^*=1$，$j^*=2$：
$$
\alpha_2=(c-\alpha_1y_1)y_2=\gamma-s\alpha_1
$$
$\gamma$为常数，$s=y_1y_2$。


$$
K_{ij}=K(\mathbf x_i,\mathbf x_i),f(x_i)=\sum_{j=1}^N y_j\alpha_jK_{ij}+b,\\
v_i=f(x_i)-\sum_{j=1}^2y_j\alpha_jK_{ij}-b
$$
固定$\alpha_1$、$\alpha_2$,得到：
$$
D(\alpha)=\alpha_1+\alpha_2-\frac 12K_{11}\alpha_1^2-\frac 12K_{22}{\alpha_2}^2-y_{1}y_{2}K_{12}\alpha_1\alpha_2-y_1\alpha_1v_1-y_2\alpha_2v_2+constant
$$
取$\alpha_1$为变量，则得到一个关于$\alpha_1$的单变量二次规划问题，仅有的约束是$\alpha_1\geq 0$：

$$
D(\alpha_1)=\alpha_1+\gamma-s\alpha_1-\frac 12K_{11}\alpha_1^2-\frac 12K_{22}(\gamma-s\alpha_1)^2-sK_{12}\alpha_1(\gamma-s\alpha_1)\\
-y_1\alpha_1v_1-y_2(\gamma-s\alpha_1)v_2+constant
$$

对$\alpha_1$求偏导以求得最大值，有
$$
\frac {\partial W(\alpha_1)}{\partial \alpha_1}=1-s-K_{11}\alpha_1+sK_{22}\gamma-K_{22}\alpha_1-sK_{12}\gamma+2K_{12}\alpha_1-y_1v_1+y_1v_2=0
$$
由此可得：
$$
\alpha_1^{new}=\frac{y_1(y_1-y_2+y_2\gamma(K_{22}-K_{12})+v_2-v_1)}{K_{11}+K_{22}-2K_{12}}
$$
根据$v$的定义，展开$v$得到：
$$
v_2-v_1=f(x_2)-f(x_1)-\alpha_1y_1K_{12}-y_2\alpha_2K_{22}+y_1\alpha_1K_{11}+\alpha_2 y_2K_{12}
$$
规定误差项$E_i=f(x_i)-y_i$，取$\gamma=s\alpha_1^{old}+\alpha_2^{old}$，并规定$\eta=K_{11}+K_{22}-2K_{12}$，
上式可化简为：
$$
\alpha_1^{new}=\alpha_1^{old}+\frac{y_1(E_2-E_1)}\eta
$$
再考虑限制条件$0\leq \alpha_i \leq c$，$(\alpha_1,\alpha_2)$的取值只能为直线$\alpha_1y_1+\alpha_2y_2=\gamma$落在$[0,C]\times [0,C]$矩形中的部分。
![此处输入图片的描述][1]
因此需要检查$\alpha_2^{new}$的值以确认这个值落在约束区间之内：
$$
\alpha_1^{new,clipped}= \left\{ \begin{array}{ll}
H, & \textrm{$\alpha_1^{new}>H$}\\
\alpha_1^{new}, & \textrm{ $L\leq \alpha_1^{new} \leq H $}\\
L, & \textrm{$\alpha_1^{new}< L$}
\end{array} \right.
$$
其中
$$
\cases
{
L=max(0,\alpha_1-\alpha_2),H=max(C,C+\alpha_1+\alpha_2)&$y_1 \neq y_2$\\
L=max(0,\alpha_1+\alpha_2-C),H=max(C,\alpha_1-\alpha_2)&$y_1 = y_2$
}
$$
假设$s=y_1y_2$，则新的$\alpha_2^{new}$为
$$
\alpha_2^{new}=\alpha_2^{old}+s(\alpha_1^{old}-\alpha_1^{new,clipped})
$$


  [1]: http://ww2.sinaimg.cn/mw690/6d96cc41gw1et9udg0andj20ev06gt90.jpg