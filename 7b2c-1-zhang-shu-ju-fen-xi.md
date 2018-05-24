你应该对Python的科学计算包和深度学习包有一定了解，这些包包含但不限于numpy, scipy, scikit-learn, pandas...

## Anaconda

安装使用：conda install keras会自动安装其他依赖包

多个python环境：

```
conda info --envs
source activate py27
```

# sickit-learn

## train_test_split
把array或者matrice随机划分为train和test

- 参数stratify： 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样。 
A:B:C=1:2:3 
split后，train和test中，都是A:B:C=1:2:3 
将stratify=X就是按照X中的比例分配 
将stratify=y就是按照y中的比例分配 
一般都是=y

```
from sklearn.model_selection import train_test_split

pat_train, pat_test, labs_train, labs_test = train_test_split(
    pat_train, pat_labs, stratify=pat_labs, test_size=test_size, random_state=12345)
```

# Jupyter

远程访问服务器上的jupyter需设置密码：

```
pip install jupyter --user
jupyter notebook --generate-config

In [1]: from notebook.auth import passwd
In [2]: passwd() sha1:4ebe51569f06:e2030b8480f6b1f07d920fdbf9851101b5ce45da
Enter password: 
Verify password: 

vi ~/.jupyter/jupyter_notebook_config.py
进行如下修改：
c.NotebookApp.ip='*' # 就是设置所有ip皆可访问
c.NotebookApp.password = u'sha:73...刚才复制的那个密文'
c.NotebookApp.open_browser = False # 禁止自动打开浏览器
c.NotebookApp.port =8888 #随便指定一个端口
```

sha1:6770b9dec97f:a6fe99d12b24f296d2fabf3edfdb55b62484469e
运行：
```
jupyter notebook --ip=192.168.8.150 --port=xxxx
```
[安装多个版本python的kernel][1]

编码问题：

```
# 有中文
XXX.decode('utf-8')
```

# pandas

## 写csv
```
data_lines = []
label_lines = ['pid','seriesUID','studyUID']

data_lines.append([pid,seriesUID,studyUID])
label_file = pd.DataFrame(columns=label_lines, data=data_lines)
```
## 统计

```
label_file.groupby(['类型']).size()
```
## 逻辑运算

使用符号`|`和`&`，记得加`（）`
```
calc_label = label_file[(label_file[u'病灶分布']=='') | (label_file[u'钙化形态']=='') | (label_file[u'良恶性']=='')]
```

## 遍历csv

```
df = pd.DataFrame(pd.read_csv('../val.csv'))
for iter_index in df.index:
    filename = df.loc[iter_index,'filepath']
    label = df.loc[iter_index,'label']
```

## 合并表格

纵向拼接：

```
dfs = []
mass_train = pd.read_csv(csv_dir+'Mass-Training-Description.csv')
mass_test = pd.read_csv(csv_dir+'Mass-Test-Description.csv')
calc_train = pd.read_csv(csv_dir+'Calc-Training-Description.csv')
calc_test = pd.read_csv(csv_dir+'Calc-Test-Description.csv')

dfs.append(mass_train)
dfs.append(mass_test)
dfs.append(calc_train)
dfs.append(calc_test)

concate_frame = pd.concat(dfs)
concate_frame = concate_frame.set_index(['patient_id', 'left or right breast', 'image view'])
concate_frame.sort_index(inplace=True)
concate_frame
```

横向匹配+拼接：

```
pd.merge(df2,df1,how='left',on=['key1','key2'])
```

## 去重：

```
data = data.drop_duplicates()
```

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

# Scipy

## 众数

## Reference

- [Pandas & Seaborn - A guide to handle & visualize data in Python][2]


  [1]: https://www.jianshu.com/p/e140c5c97938
  [2]: https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/