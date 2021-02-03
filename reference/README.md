# Sparse PCA

+ **Sparse Principal Component Analysis**
> 基于Lasso的想法采用重构主成分和一范数相结合的方式构建目标函数，从而使得负载矩阵稀疏，由于主成分未知，之后采用self-contained的方式构建目标函数，当未知参数$\alpha$和$\beta$相同时，目标函数能够通过PCA最小重构误差进行解释。

+ **A Selective Overview of Sparse Principal Component Analysis**
> 介绍了从PCA不同角度出发的稀疏PCA方式，本质上都是在获得稀疏的主成分，在目标函数和约束上有所差异。