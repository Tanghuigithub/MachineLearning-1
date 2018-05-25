# sickit-learn

## train_test_split

把array或者matrice随机划分为train和test

- 参数stratify： 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样。 
A:B:C=1:2:3 
split后，train和test中，都是A:B:C=1:2:3 
将stratify=X就是按照X中的比例分配 
将stratify=y就是按照y中的比例分配 
一般都是=y

```python
from sklearn.model_selection import train_test_split

pat_train, pat_test, labs_train, labs_test = train_test_split(
    pat_train, pat_labs, stratify=pat_labs, test_size=test_size, random_state=12345)
```

