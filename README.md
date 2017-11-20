# ann
simple artificial neural network

+ 具体数学原理参见[博客](https://heloowird.github.io/2017/03/08/diff_errors_of_neural_network/) 推导很详尽。

+ 根目录下的py文件为原始实现，advanced\_version使用numpy实现。

+ Iris数据集上效果：
	+ sigmoid: 
		+ 1000轮: 0.984375
		+ 2000轮: 0.992188
	+ tanh:
		+ 1000轮: 0.992188
		+ 2000轮: 0.992188
+ Iris数据集上，tanh收敛更快
