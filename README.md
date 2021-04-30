# NNClass
A simple bit of code for training classification neural networks.

## Installation

Install from `pip3`:

```bash
pip3 install --user NNClass
```

Or by cloning this repository:

```bash
#clone the repo
git clone https://github.com/mattkjames7/NNClass
cd NNClass

#Either create a wheel and use pip: (X.X.X should be replaced with the current version)
python3 setup.py bdist_wheel
pip3 install --user dists/NNClass-X.X.X-py3-none-any.whl

#Or by using setup.py directly
python3 setup.py install --user
```



## Usage

Start by training training a network:

```python
import NNClass as nnc

#create the network, defining the activation functions and the number of nodes in each layer
net = nnc.NNClass(s,AF='sigmoid',Output='softmax')

#note that s should be a list, where each element denotes the number of nodes in each layer

#input training data
net.AddData(X,y)
#Input matrix X should be of the shape (m,n) - where m is the number of samples and n is the number of input features
#Output hypothesis matrix y should either be
# an array (m,) of integers corresponding to class
# or matrix (m,k) of one-hot labels

#optionally add validation and test data
net.AddValidationData(Xv,yv)
#Note that validation data is ignored if kfolds > 1 during training
net.AddTestData(Xt,yt)

#Train the network 
net.Train(nEpoch,kfolds=k)
#nEpoch is the number of training epochs
#kfolds is the number of kfolds to do - if kfolds > 1 then the training data are split 
#into kfold sets, each of which has a turn at being the validation set. This results in
#kfold networks being trained in total (net.model)
#see docstring net.Train? to see more options

```

After training, the cost function may be plotted:

```python
net.PlotCost(k=k)
```

We can use the network on other data:

```python
#X in this case is a new matrix
y = net.Predict(X)
```

The networks can be saved and reloaded:

```python
#save
net.Save(fname='networkname.bin')

#reload
net = nnc.LoadANN(fname='networkname.bin')
```

Running ```mnist = nnc.Test()``` will perform a test on the code, by training a neural network to classify a set of hand-written digits (0-9) from the MNIST dataset (https://deepai.org/dataset/mnist). The function will then plot out the cost, accuracy and an example of a classified digit, e.g.:

![](cost.png)

![](accuracy.png)

![](digit.png)

The 10,000 sample MNIST data can be accessed using the `NNClass.MNIST` object.