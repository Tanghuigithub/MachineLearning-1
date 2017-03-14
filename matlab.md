# Matlab



* eye（）标准矩阵
* randn（） 高斯分布
* hist （）直方图
* size\(a,1\) row
* size\(a,2\) colum
* length\(vector\)
* clear A
* A\(1,10\)
* save hello.mat A 
* save hello.txt A -ascii
* A\(:,2\)
* A\(:\) 全部元素变成向量
* C = \[A B\]
* C = \[A; B\]

## Computation

* A.\*B 对应元素相乘
* A.^2
* log
* exp
* abs
* a‘ 转置
* \[val, index\] = max\(A\)
* a&lt;3
* find\(a&lt;3\)
* magic\(3\)
* sum\(A, 1\) 输出1xn
* prod\(a\)
* floor\(a\) 取整，取下限
* ceil\(a\) 取整，取上限
* max\(rand\(3\), rand\(3\)\)
* max\(A, \[ \], 1\) = max\(A\)
* flipud\(eye\(9\)\) 左右翻转
* pinv\(A\) 矩阵求逆

## Plotting Data

* plot\(x, y, 'r'\)
* hold on
* xlabel\('xlabel'\)
* ylabel\('ylabel'\)
* legend\('sin','cos'\)
* title\('my plot'\)
* print -dpng 'myplot.png'
* figure\(1\)
* subplot\(1,2,1\)
* axis\(\[0.5 1 -1 1\]\)
* clf
* imagesc\(A\), colorbar, colormap grey

## Control Statement

* indecis = 1:10

## Vectorization



