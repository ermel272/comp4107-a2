Question 2

Develop a feed forward RBF neural network in python that classifies the images found in the MNIST dataset. You are to train your neural network using backpropagation. You should use gaussian functions as your radial basis functions. You must show that you have:

Used K-means to design the hidden layer in your network.
Performed K-fold cross correlation.
Investigated the performance of your neural network for different sizes of hidden layer.

# Setup
Ensure that your current working directory is '<project root>/q2'.

## Installing Dependencies
```
  pip install -r requirements.txt
```

# Running

```
  python a2-q2.py
```

Note that the value of K is tunable as a global variable in a2-q2.py, as well as the learning rate.

# Testing
Several trained neural networks have been stored in the assets directory. They can be accessed from `interactive-brain.py`

Run `python -i interactive-brain.py` to interact.

This will give you access to the following utility functions:

show(index) - Displays the image at the input index in the training set.

test2(index) - Uses the network to test identification of the image at param index in the test set.

test(index) - Uses the network to test identification of the image at param index in the training set.

accuracy() - Outputs network accuracy information.

test2_some(start, end) - Gives an accuracy measure from the start index to an end index in the test set.

test_some(start, end) - Gives an accuracy measure from the start index to an end index in the train set.