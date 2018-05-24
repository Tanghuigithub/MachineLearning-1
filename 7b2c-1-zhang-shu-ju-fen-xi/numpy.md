# list

找list中最大元素的索引：

```
aa = [1,2,3,4,5]
aa.index(max(aa))
```

# numpy

## 中位数

```
#求数组a的中位数
np.median(a)

#求数组a的四分位数
np.percentile(a, [25, 50, 75])
```

## 基本矩阵操作 


```
## 增加维度：
x2 = x[:, np.newaxis]
## 变形
>>> a = np.arange(10).reshape(2, 5)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9])
## 拍平
a.flatten()
## 逻辑判断
a = np.array(randn(4, 4))
b = 1 * (a > 0) # 返回0、1矩阵
a_list = a.ravel().tolist() # 返回list
```
## 找极值+索引
### 寻找矩阵中最大的几个元素的index

```
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
```

linspace 线段

## 寻址、索引和遍历：

```
>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> x[1:7:2]
array([1, 3, 5])
>>> x[-2:10]
array([8, 9])
>>> x[-3:3:-1]
array([7, 6, 5, 4])
```

## 打印

完整打印：

```
numpy.set_printoptions(threshold='nan')
```

