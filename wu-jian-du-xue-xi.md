# 无监督学习


NNML第十一周及接下来好几周的课程可以说是进入了hard模式，由于平常对Hopfield网络和受限波尔兹曼机接触得就较少，理解起来总是很多地方弄不明白，必须得查阅中文辅助资料才能略通一二。前面课程的内容多多少少都在其他资料中反复见到，唯独这部分可以说是零起点，也算是看到了“新世界”。

- Hopfield Network：单元的值是二元的
    - global energy function
$$
E = -\sum_i s_i b_i - \sum_{i < j} s_i s_j w_{ij}
$$
    - 逐个地调整二元值，只为了找到energy minimum
- Hopfield Network with hidden units
    - energy反映interpretation的好坏
    - search：escape from local minimum
        - noise
        -  “simulated annealing” 
        -  thermal equilibrium ：configuration的概率分布settle down
    - learn
        - 基于最大似然
- Restricted Boltzmann Machine
    - 无向概率图模型
    - deep belief network：
        - 混合图模型，既有有向连接，也有无向连接
        - 多个隐藏层
    - deep Boltzman machine
        -  有多层隐变量的无向图模型
$$
P(\mathrm v =v,\mathrm h=h)=\frac1Z\exp(-E(v,h))\\
E(v,h)=-b^\top v-c^\top h-v^\top wh
$$
    - 从概率计算中可以看出，这个用来归一化的常量$$Z$$是无法计算的。这样归一化后的联合概率分布$$P(v)$$也无法计算。

RBM也相当于一个编解码器 :
编码： 
1. 输入编码前的样本x； 
2. 根据x的值计算概率$$p(h=1|v)$$，其中v的取值就是x的值； 
3. 按照均匀分布产生一个0到1之间的随机数，如果它小于$$p(h=1|v)$$，y的取值就是1，否则就是0； 
4. 得到编码后的样本y。 

解码： 
1. 输入解码前的样本y； 
2. 根据y的值计算概率$$p(v=1|h)$$，其中h的取值就是y的值； 
3. 按照均匀分布产生一个0到1之间的随机浮点数，如果它小于$$p(v=1|h)$$，v的取值就是1，否则就是0； 
4. 得到解码后的样本x。 

## Reference

- Hopfield Network：[ http://page.mi.fu-berlin.de/rojas/neural/chapter/K13.pdf ]
- [ http://www.scholarpedia.org/article/Boltzmann_machine ]