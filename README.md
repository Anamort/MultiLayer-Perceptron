# MultiLayer-Perceptron
In this project, Multi-Layer Perceptron approach for machine learning has been implemented using the Backpropagation Algorithm. For the implementation, Java Programming Language is used.
In the project, a user can create and use her/his external dataset including attribute (feature) count, class count, name of classes, and the data as shown in following figure.

![The structure and syntax of dataset used in the project](https://i.hizliresim.com/kO3qQv.png)

The Multi-Layer Perceptron has been implemented based on two approaches: (1) the output node is single and its value is between 0 and 1; (2) there are multiple output nodes based on the class count. For example, considering the dataset in Figure 1, there are two classes: tested_positive and tested_negative. In the first approach, if the single output node would be close to the zero, it indicates negative value. On the other hand, in second approach, there are two output nodes due to class count. If the output value is negative value as in first approach, one of the output nodes related with that class name would close to one. Likewise, the other node related with tested_positive would close to zero. If one can use three or more classes, the second approach should be used, otherwise first approach would be more appropriate. In the implementation, these two approaches were implemented separate classes namely “MultilayerPerceptron.java” and “SingleClassVersion.java”.

![Hidden Layer Effect](https://i.hizliresim.com/RnLWZo.png)

In the project, to display results, the XOR dataset shown in Figure above was used since it is exploited prevalently in literature considering Backpropagation Algorithm. Moreover, first approach (single output node) was used for this data. For the evaluation of the algorithm, three cases are considered:

* the effect of the number of the hidden layers
* the effect of the number of the nodes in the hidden layer
* the effect of the learning rate

For the evaluation of the effect of the number of hidden layers, hidden layer count 1, 2, and 4 were examined respectively holding the number of the nodes in the layers as 1. Following figure shows the error between different hidden layer counts using 3000 epochs. Note that, since the errors of the layer counts 1 and 2 are very close to each other, it is seen as a single line in the figure. It is clearly observed that when the number of the layers increases, the error decreases.

![Hidden Layer Effect](https://i.hizliresim.com/lO2MDp.png)

Considering the effect of the number of the nodes in the hidden layer, the hidden layer count is fixed to one. 4, 8, and 16 nodes in the layers are examined regarding the error and 3000 epochs. The results in the figure below shows that if the node count in the hidden layer rises, the error reduces.

![The effect of the node count in the hidden layer](https://i.hizliresim.com/bBAndY.png)

Finally, different learning rates including 0.05, 0.25, 0.5, 1, and 5 were analyzed considering the error of the output node in the Figure 5.

![Evaluation of Different Learning Rates for XOR Dataset](https://i.hizliresim.com/azJVbR.png)
