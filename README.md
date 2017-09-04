# learn_NeuralNet

前面几层的activation function一般用logistic function来计算，最后一层可以用logistic function来计算概率（没有归一化的概率，一类一类来看，哪一类的概率最大就预测是哪一类），也可以用softmax function来计算概率（归一化的概率，统一来看，哪一类的概率最大就预测是哪一类）。

## 1. Cost function and logistic regression

![Cost function and logistic regression](images/logistic.png)

## 2. Forward propagtion, cost function and softmax regression

当使用softmax regression作forward propagation时，最后一个layer是使用softmax function来计算，如果是k个分类，最后一层就有k个neuron，每个neuron的值就是exp（hx），然后将k个neuron的值进行归一化处理（除以k个neuron值的加和），k个neuron的值就变成了是每个分类的概率（k个概率的和为1）。最后，概率最大的那个neuron对应的分类作为预测的分类。

当使用softmax regression作拟合时，所使用的cost function一般用cross entropy cost function (it is called cross entropy error or cross entropy cost)，对于softmax function算出的k个neuron的概率值，只有实际分类对应的那个neuron上的概率值会被保留，然后log求和。比如，总共有5个分类，某个实际分类y1i = 列向量[1, 0, 0, 0, 0]，k个neuron的概率值的向量y2i = 列向量[p1, p2, p3, p4, p5]，两个向量的内积（点乘）或者y1i转置后叉乘logy2i，得到一个值yi，最终的cost function就是把m个样本的yi加起来，最后取负值（相反数）。

![Cost function and softmax regression](images/softmax.png)

C个feature，每个feature的维度是d（对于图像识别，是C个分类，d个像素点）。W.x是线性函数的矩阵形式。

## 3. Wholly linked neural network

此处使用的neural network是wholly linked neural network，没有额外的assumption，完全根据数据来拟合系数参数，是理论上最正确的一种neural network，但因为所需数据较多，计算量较大，实用价值较小。

## 4. Back propagation

最终的output的error是预测值（对真实值）的偏离，但这种偏离不只是由最后一层neuron造成的，而是由多层neuron累积而成的，所以，每一层neuron都存在其预测值的偏离，而这种预测值的偏离误差是可以用后一层的预测值的偏离误差计算出来的，计算公式就是back propagation的公式，这个公式在数学上是可以证明的。

在计算back propagation时，最重要的搞清楚error和error term的算法。对于output layer, error就是实际值减去预测值，error term等于error乘以output layer的activation function derivative. 对于hidden layer, error可以由后一层的error term与两层之间的weights的矩阵乘法得到，error term同样等于error乘以hidden layer的activation function derivative. Gradient descent过程中weights的变化值，可以由前一层的输出值与后一层的error term之间的矩阵乘法得到。

forward propagation: calculate the prediction function; forward pass, calculate output from input; related - activation function, prediction function

back propagation: calculate the gradient for gradient descent of loss function; back pass, calculate error term of layers from output error; related - error term, error function, loss function, cost function, objective function

## 5. Function and strength

neural network的强大之处在于，多个线性boundary的结合可以模拟任何复杂的非线性的boundary，所以通过多层多个neuron的设置，可以用neural network逼近任何复杂的函数，模拟该函数。

线性规划的意义和neural network的意义是类似的，虽然实际情况是非线性的，但在线性规划中，通过设置多条直线的boundary，可以模拟非线性情况，所以，线性规划在运筹学中的应用才能如此广泛。

## 6. Neural net用于连续值的建模

Neural net不只可以用于分类（无序类或有序类），也可以用于连续值的预测。

传统上，连续值的建模只是使用简单的linear regresssion. 实际上，可以使用neural net来抓住问题中的非线性特征，在最后一层output中，可以使用$f(x)=x$这个简单的activation function（之前各层的activation function仍然使用常用的sigmoid function），就可以得到连续的预测值，而且预测值的范围可以从无穷小到无穷大，不再局限于（0，1）。

## 7. Dummy variable 

The rank feature is categorical, the numbers don't encode any sort of relative values. Rank 2 is not twice as much as rank 1, rank 3 is not 1.5 more than rank 2. Instead, we need to use dummy variables to encode rank, splitting the data into four new columns encoded with ones or zeros. Rows with rank 1 have one in the rank 1 dummy column, and zeros in all other columns. Rows with rank 2 have one in the rank 2 dummy column, and zeros in all other columns. And so on.

## 8. Requirements of gradient decent

### (1). Input data

We'll also need to standardize the GRE and GPA data, which means to scale the values such they have zero mean and a standard deviation of 1 (For normal distribution, 68% data are within the range of one standard deviation). This is necessary because the sigmoid function squashes really small and really large inputs. The gradient of really small and large inputs is zero, which means that the gradient descent step will go to zero too. Since the GRE and GPA values are fairly large, we have to be really careful about how we initialize the weights or the gradient descent steps will die off and the network won't train. Instead, if we standardize the data, we can initialize the weights easily and everyone is happy.

### (2). weights

First, you'll need to initialize the weights. We want these to be small such that the input to the sigmoid is in the linear region near 0 and not squashed at the high and low ends. It's also important to initialize them randomly so that they all have different starting values and diverge, breaking symmetry. So, we'll initialize the weights from a normal distribution centered at 0. A good value for the scale is 1/√n，where n is the number of input units. This keeps the input to the sigmoid low for increasing numbers of input units.

### (3). learning rate

To make learning rate between 0.01 and 0.1 (也可能是0.1到1), we use Mean Square Error instead of Sum Square Error. 在实际建模过程中，learning rate是从大往小试，如果使用的是Mean Square Error，一般从1开始试起。

### (4). activation function and vanishing gradient

The maximum derivative of the sigmoid function is 0.25, so the errors in the output layer get reduced by at least 75%, and errors in the hidden layer are scaled down by at least 93.75%! You can see that if you have a lot of layers, using a sigmoid activation function will quickly reduce the weight steps to tiny values in layers near the input. This is known as the vanishing gradient problem. Later in the course you'll learn about other activation functions that perform better in this regard and are more commonly used in modern network architectures.

### (5). local minimum

Since the weights will just go where ever the gradient takes them, they can end up where the error is low, but not the lowest. These spots are called local minima. If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum.

There are methods to avoid this, such as using [momentum](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum)

### (6). cost function or loss function

We can use Mean Square Error for regression (continuous prediction) and cross entropy error for classification.

Mean Square Error corresponds to Sum Square Error. 

Cross Entropy Error corresponds to number of prediction mistankes, number of false positives, number of false negatives.

## 9. neural net和deep learning所解决的问题

长期以来，统计模型和很多机器学习模型的问题在于，总是尝试在线性模型的框架内解决问题。世界是非线性的，线性模型当然是没法很好的解决问题的，这时候，统计模型并没有直面非线性这个问题，而是在线性模型上加上了一个随机项（比如linear mixed model），随机性当然是模型不够准确的一个原因，但绝对不是最主要的原因（非线性是主要原因），把主要的研究精力放在随机性上，大方向就错掉了。

还有一些机器学习模型，比如support vector machine，虽然可以引入非线性，但因为函数结构不是很灵活，并不能逼近、模拟所有的非线性函数，所以效果也一般。

neural net和deep learning的威力在于，通过选择不同的连接假设和拓扑结构（比如RNN、CNN和wholy linked network就有着不同的连接假设）、不同的层数、每一层不同的node数、不同的activation function，几乎可以逼近、模拟所有的非线性函数，所以效果非常好。

一般的统计模型和机器学习模型，在建模时，是需要做feature selection的，也就是需要人为的去考虑模型机制，否则效果很差。对于deep learning，做feature selection的话当然效果也是更好的，但是，只要数据足够，不去人为的做feature selection也可以达到很好的效果，本质上是不再人为的去做feature selection，而是由程序去做feature selection.

对于线性模型，有一个好处是，容易解释自变量对因变量的影响，如果系数为正，就是正的影响，如果系数为负，就是负的影响，如果系数很接近0，就表明影响很小。
