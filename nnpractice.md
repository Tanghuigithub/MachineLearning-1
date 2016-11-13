# 第3章 神经网络编程

---


## TensorFlow

TensorFlow是一个开源 C ++ / Python 软件库，用于使用数据流图的数值计算，尤其是深度神经网络。它是由谷歌创建的。在设计方面，它最类似于 Theano，但比  Caffe 或 Keras 更低级。

## 举例：Minist handwritten

## 关于纹理基元和特征抽取

### CIELAB
![此处输入图片的描述][3]
CIE L*a*b*（CIELAB）是慣常用來描述人眼可見的所有顏色的最完備的色彩模型，具有視覺上的均勻性。名稱是由國際照明委員會（Commission Internationale d'Eclairage的首字母是CIE)提出的。

### SLIC Superpixel 超像素
![此处输入图片的描述][4]
使像素聚类为五维颜色和图像层，用来生成简洁整齐的超像素,对像素迭代聚类
$(L,A,B)+(x,y)$
CIELAB color values
It is not possible to simply use the Euclidean distance in this 5D space without normalizing the spatial distances.
In order to cluster pixels in this 5D space, we therefore introduce a new distance measure that considers superpixel size. Using it, we enforce color similarity as well as pixel proximity in this 5D space such that the expected cluster sizes and their spatial extent are approximately equal.
#### 1. 距离测量方法

输入一个超像素数目的参数$K$。那么对于一张$N$个像素的图像来说，每个超像素大小约为$\frac{N}{K}$个像素。那么，每两个相邻的超像素块之间的距离为$s=\sqrt{\frac{N}{K}}$。
算法开始时，我们选择聚类的中心$C_k =\{l_k,a_k,b_k,x_k,y_k\}$,$k$属于$[1,K]$。每个超像素的面积大约为$s^2$（近似于超像素的面积）。我们可以安全地假设：像素在聚类中心的$(2s)^2$范围内。这个范围就是每个聚类中心的**搜寻范围**。

**欧式距离**在**亮度**空间内小距离是很有意义的。如果空间像素距离超过了颜色距离，那么图像上的边界不再有效（文章的意思是这样是不对的）。因此，在5D空间中取代简单的欧式距离，我们采用$d_s$如下：
$
d_s=d_{lab}+\frac{m}{s}d_{xy}
$
其中$d_s$是$lab$距离和归一化后的$xy$距离之和。其中变量$m$用来控制超像素的紧密度。$m$的值在$[1,20]$之间。在接下来的文章中，我们统一选择$m=10$。这个数值既能在感官经验上满足颜色距离最大化，又能很好的在**颜色相似度**和**空间相似度**的平衡。

### Histogram intersection(直方图交叉核,Pyramid Match Kernel)

假设图像或其他数据的特征可以构成直方图，根据直方图间距的不同可以得到多种类型的直方图：

![此处输入图片的描述][5]
 a)里的y和z代表两种数据分布，三幅图代表三层金字塔，每一层里有间距相等的虚线，意思和我之前说的2cm，4cm的宽度一样。可以看到红点蓝点的位置是固定的，但是根据直方图宽度的不同可以划到不同的直方图里，如(b)所示。(c)图就是L的计算结果，是通过(b)里两种直方图取交集得来的，不过直方图的高度忽略不计，只计算交集后的数目，(c)图每个图的下方都给出了交集数目，比如x0=2,x1=4,x2=3（原图里是5，是不是错了？）。
L得到了，就算N就是通过，也就是通过Ni=Li-Li-1得到（看公式是能取负数的，比如上图里的N0=2，N1=2，N2=-1）。



  [1]: http://www.gumpcs.com/wp-content/uploads/2015/07/070915_0246_1.png
  [2]: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
  [3]: http://4.bp.blogspot.com/-GJsnhAWlYqY/U0erTUv5p0I/AAAAAAAAA3Q/Ac6SeCEYgts/s1600/LAB_COLOR1.png
  [4]: http://ivrl.epfl.ch/files/content/sites/ivrg/files/research/images/RK_SLICSuperpixels/intro_pics/54082_combo.jpg
  [5]: http://img.blog.csdn.net/20140408110748640