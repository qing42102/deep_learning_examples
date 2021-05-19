Builds deep learning and machine learnig algorithms using NumPy and TensorFlow. 

Includes logistic regression, neural network, convolutional neural network, long short term memory recurrent network, long term recurrent convolutional network. 

## Logistic Regression

- Implements multi-class logistic regression using NumPy for classifying MNIST digits 1-5. 
- Includes logistic loss function, gradient of loss function, and steepest gradient descent optimization calculation. 
- Training data consists of 25112 images and testing data consists of 4982 images. 
- Achieves a classification accuracy of 93.5% on the training data and 94.2% on the testing data. 

## Neural Network

- Implements a 2-layer 100 neurons dense neural network using TensorFlow for classifying MNIST digits 0-9. 
- Includes cross-entropy loss function, backpropagation of gradient of all layers, and stochastic gradient descent optimization calculation. 
- Training data consists of 50000 images and testing data consists of 5000 images.
- Achieves a classification accuracy of 97.95% on training data and 96.64% on testing data.

## Convolutional Neural Network

- Implements a convolutional neural network using TensorFlow for classifying CIFAR-10. 
- Contains 3 convolutional layers with max pooling, batch normalization, and dropout. 
- Achieves a classification accuracy of 78.06% on training data and 76.02% on testing data.

## Long Short Term Memory Recurrent Network

- Implements a LSTM recurrent network combined with a CNN and a MLP using TensorFlow. AlexNet is used for the CNN. The overall model has 7867955 trainable parameters.
- The model is used for regression of 3D humanbody poses in videos. The output is the x, y, z coordinates of human joints. 
- Training data consists of 5964 videos of 8 frames 224×224×3 images and testing data consists of 1368 videos of 8 frames of 224×224×3 images. 
- Achieves an Euclidean distance error of 59.5020 mm for the training data and 74.5818 mm for the testing data. 

## Long Term Recurrent Convolutional Network

- Based on the paper https://arxiv.org/abs/1411.4389. 
- Implements a LSTM recurrent network combined with a CNN using TensorFlow. ResNet is used for the CNN. The overall model has 3317643 trainable parameters.
- The model is used to classify human activities in videos. 
- Training data consists of 5454 videos of 30 frames 64×64×3 images and testing data consists of 1818 videos of 30 frames of 64×64×3 images. 
- Achieves a classification accuracy of 99.39% on training data and 97.69% on testing data. 
