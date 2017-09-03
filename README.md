# learn_NeuralNet

前面几层的activation function一般用logistic function来计算，最后一层可以用logistic function来计算概率（没有归一化的概率，一类一类来看，哪一类的概率最大就预测是哪一类），也可以用softmax function来计算概率（归一化的概率，统一来看，哪一类的概率最大就预测是哪一类）。

## 1. Cost function and logistic regression

![Cost function and logistic regression](images/logistic.png)

## 2. Forward propagtion, cost function and softmax regression

当使用softmax regression作forward propagation时，最后一个layer是使用softmax function来计算，如果是k个分类，最后一层就有k个neuron，每个neuron的值就是exp（hx），然后将k个neuron的值进行归一化处理（除以k个neuron值的加和），k个neuron的值就变成了是每个分类的概率（k个概率的和为1）。最后，概率最大的那个neuron对应的分类作为预测的分类。

当使用softmax regression作拟合时，所使用的cost function一般用cross entropy cost function，对于softmax function算出的k个neuron的概率值，只有实际分类对应的那个neuron上的概率值会被保留，然后log求和。比如，总共有5个分类，某个实际分类y1i = 列向量[1, 0, 0, 0, 0]，k个neuron的概率值的向量y2i = 列向量[p1, p2, p3, p4, p5]，两个向量的内积（点乘）或者y1i转置后叉乘logy2i，得到一个值yi，最终的cost function就是把m个样本的yi加起来，最后取负值（相反数）。

![Cost function and softmax regression](images/softmax.png)

C个feature，每个feature的维度是d（对于图像识别，是C个分类，d个像素点）。W.x是线性函数的矩阵形式。

## 3. Wholly linked neural network

此处使用的neural network是wholly linked neural network，没有额外的assumption，完全根据数据来拟合系数参数，是理论上最正确的一种neural network，但因为所需数据较多，计算量较大，实用价值较小。

## 4. Back propagation

最终的output的error是预测值（对真实值）的偏离，但这种偏离不只是由最后一层neuron造成的，而是由多层neuron累积而成的，所以，每一层neuron都存在其预测值的偏离，而这种预测值的偏离误差是可以用后一层的预测值的偏离误差计算出来的，计算公式就是back propagation的公式，这个公式在数学上是可以证明的。

## 5. Function and strength

neural network的强大之处在于，多个线性boundary的结合可以模拟任何复杂的非线性的boundary，所以通过多层多个neuron的设置，可以用neural network逼近任何复杂的函数，模拟该函数。

线性规划的意义和neural network的意义是类似的，虽然实际情况是非线性的，但在线性规划中，通过设置多条直线的boundary，可以模拟非线性情况，所以，线性规划在运筹学中的应用才能如此广泛。

## 6. Neural net用于连续值的建模

Neural net不只可以用于分类（无序类或有序类），也可以用于连续值的预测。

传统上，连续值的建模只是使用简单的linear regresssion. 实际上，可以使用neural net来抓住问题中的非线性特征，在最后一层output中，可以使用$f(x)=x$这个简单的activation function（之前各层的activation function仍然使用常用的sigmoid function），就可以得到连续的预测值，而且预测值的范围可以从无穷小到无穷大，不再局限于（0，1）。
