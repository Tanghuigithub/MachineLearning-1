# CRF

假设图$$G=(V,E)$$，其中$$V={X_1,X_2,\dots,X_N}$$，全局观测为$$I$$。使用Gibbs分布，$$(I,X)$$可被模型为CRF

$$
P(X=x|I)=\frac 1{Z(I)}exp(-E(x|I))
$$
$$ 
E(x)=\sum _i \varphi(x_i)+\sum _{i<j} \varphi_p(x_i.x_j)
$$

$$\varphi_p(x_i.x_j)$$是对$$i$$、$$j$$同时分类成$$x_i$$、$$x_j$$的能量。

### Gibbs Distribution