# Extreme learning machines (in construction)

[Extreme learning machines](http://www.ntu.edu.sg/home/egbhuang/) are feedforward neural network for classification, regression, clustering, sparse approximation, compression or feature learning with a single layer or multi layers of hidden nodes, where the parameters of hidden nodes (not just the weights connecting inputs to hidden nodes) need not be tuned. These hidden nodes can be randomly assigned and never updated (i.e. they are random projection but with nonlinear transforms), or can be inherited from their ancestors without being changed. In most cases, the output weights of hidden nodes are usually learned in a single step, which essentially amounts to learning a linear model.


This framework is a simple interface based on Keras. The idea here is to abstract all necessary knowledge in Keras without losing its flexibility to build networks.

### Dependencies
* [numpy](http://www.numpy.org/)
* [TensorFlow](https://www.tensorflow.org/install/)
* [Keras](https://keras.io/)


### TODO (class and examples)
* Binary Classification (in construction)
* Regression
* Cluster