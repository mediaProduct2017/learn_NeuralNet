# learn_NeuralNet

前面几层的activation function一般用logistic function来计算，最后一层可以用logistic function来计算概率（没有归一化的概率，一类一类来看，哪一类的概率最大就预测是哪一类），也可以用softmax function来计算概率（归一化的概率，统一来看，哪一类的概率最大就预测是哪一类）。

## 1. Cost function and logistic regression

![Cost function and logistic regression](images/logistic.png)

## 2. Forward propagtion, cost function and softmax regression

当使用softmax regression作forward propagation时，最后一个layer是使用softmax function来计算，如果是k个分类，最后一层就有k个neuron，每个neuron的值就是exp（hx），然后将k个neuron的值进行归一化处理（除以k个neuron值的加和），k个neuron的值就变成了是每个分类的概率（k个概率的和为1）。最后，概率最大的那个neuron对应的分类作为预测的分类。

当使用softmax regression作拟合时，所使用的cost function一般用cross entropy cost function，对于softmax function算出的k个neuron的概率值，只有实际分类对应的那个neuron上的概率值会被保留，然后log求和。比如，总共有5个分类，某个实际分类y1i = 列向量[1, 0, 0, 0, 0]，k个neuron的概率值的向量y2i = 列向量[p1, p2, p3, p4, p5]，两个向量的内积（点乘）或者y1i转置后叉乘logy2i，得到一个值yi，最终的cost function就是把m个样本的yi加起来，最后取负值（相反数）。
