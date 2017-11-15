# 预处理

NNML第六周印象最深的是从误差曲面的角度理解数据预处理，甚至是PCA。如果误差曲面是椭圆的话，最速下降方向往往不指向最小值，这样一来，很可能在偏离很大的方向走了很大一步，这种情况下较大的学习率还会导致震荡。
在使用最速下降时，对输入进行transfer和scale：

 - transfer：训练集的输入向量均值为0
 - ![屏幕快照 2017-06-20 16.35.18.png-350.1kB][2]
 - scale：
![屏幕快照 2017-06-20 21.04.47.png-292.5kB][3]
 - Decorrelate the input(PCA)：去掉特征值较小的主成分，剩下的除以对应的特征值的平方根，对于线性单元来说，这样就把椭圆的误差曲面转变成了圆形。而对于圆形的误差曲面，梯度下降方向指向最小值。

  [2]: http://static.zybuluo.com/sixijinling/rj5lx1v3zojfa5ezdueye0no/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-06-20%2016.35.18.png
  [3]: http://static.zybuluo.com/sixijinling/rbn7pnyl3ib9en3xv86i0tqj/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-06-20%2021.04.47.png