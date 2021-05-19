# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
import time

# %%
def load_images(path: str) -> list:
    '''
    Load images from a directory. Normalize the image to [0, 1]. Return a list of images array
    '''
    imgs = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() == ".jpg":
            file_name = os.path.join(path,f)
            
            # Convert the image to an array of floats normalized to [0, 1]
            image = Image.open(file_name).convert("F")
            image_array = np.array(image.getdata())/255.

            imgs.append(image_array)

    return imgs

train_dataset = load_images("train_data/")
test_dataset = load_images("test_data/")

os.chdir("labels/")
train_label = np.loadtxt("train_label.txt", dtype=int)
test_label = np.loadtxt("test_label.txt", dtype=int)
os.chdir("..")

num_features = len(train_dataset[0])
num_classes = len(np.unique(test_label))

batch_size = 50

train_label = tf.one_hot(train_label, depth=num_classes)
test_label = tf.one_hot(test_label, depth=num_classes)
train_dataset = tf.convert_to_tensor(train_dataset, dtype=tf.float32)
test_dataset = tf.convert_to_tensor(test_dataset, dtype=tf.float32)

print(test_dataset.shape, test_label.shape)

# %%
def loss(y_output: tf.Tensor, y_label: tf.Tensor) -> float:
    '''
    Cross-entropy loss
    y_output and y_label are num_data x num_classes tensors
    '''

    total_loss = tf.tensordot(y_label, tf.math.log(y_output), axes=2)
    
    return -total_loss.numpy()/y_output.shape[0]

def classification_accuracy(y_output: tf.Tensor, y_label: tf.Tensor) -> float:
    '''
    Classification accuracy for the predicted and true labels
    '''
    # Revert the one-hot vector
    y_label = tf.where(tf.equal(y_label, 1))[:, 1]

    # Select the largest probability
    y_output = tf.argmax(y_output, axis=1)

    correct = tf.cast(y_output == y_label, tf.float32)
    accuracy = tf.math.reduce_sum(correct).numpy()/y_output.shape[0]

    return accuracy*100

def digit_accuracy(y_output: tf.Tensor, y_label: tf.Tensor, num_classes: int):
    '''
    Classification accuracy for each of the digits
    '''    
    # Revert the one-hot vector
    y_label = tf.where(tf.equal(y_label, 1))[:, 1]

    # Select the largest probability
    y_output = tf.argmax(y_output, axis=1)
    
    for i in range(num_classes):
        y_i = y_label[y_label==i]
        y_output_i = y_output[y_label==i]

        correct = tf.cast(y_output_i == y_i, tf.float32)
        accuracy = tf.math.reduce_sum(correct).numpy()/y_output_i.shape[0]

        print("Digit", i, "accuracy:", accuracy*100)
    
    print("\n")

def backward_propagation_output_hidden(y_output: tf.Tensor, y_label: tf.Tensor, W3: tf.Variable, H2: tf.Tensor):
    '''
    Back propagation from output to hidden layer
    y_output and y_label are num_data x num_classes tensors
    W3 is a 100 x num_classes tensor
    H2 is a num_data x 100 tensor
    '''

    num_data = y_output.shape[0]

    # Gradient of the cross-entropy loss
    loss_grad = -tf.divide(y_label, y_output)

    # z is the forward calculation of the second hidden layer
    # sigma is the softmax function
    # Calculate the off diaognals and then the diagonal elements 
    dsigma_dz = tf.einsum("ij, ik -> ijk", -y_output, y_output)
    dsigma_dz = tf.linalg.set_diag(dsigma_dz, tf.multiply(y_output, 1-y_output))

    # \frac{d \sigma(z)}{dz} \nabla y
    dsigma_dz_loss_grad = tf.einsum("ijk, ik -> ij", dsigma_dz, loss_grad)

    H2_grad = tf.matmul(dsigma_dz_loss_grad, tf.transpose(W3))
    W3_grad = tf.matmul(tf.transpose(H2), dsigma_dz_loss_grad)
    b3_grad = tf.math.reduce_sum(dsigma_dz_loss_grad, axis=0)

    # H2_grad should be num_data x 100
    # W3_grad should be 100 x num_classes
    # b3_grad should be num_classes
    return H2_grad, W3_grad/num_data, b3_grad/num_data

def backward_propagation_hidden_hidden(H1: tf.Tensor, H2_grad: tf.Tensor, W2: tf.Variable, b2: tf.Variable):
    '''
    Back propagation from hidden to hidden layer
    H1 is a num_data x 100 tensor
    H2_grad is a num_data x 100 tensor
    W2 is a 100x100 tensor
    b2 is a 100x1 tensor
    '''

    num_data = H1.shape[0]

    # The forward calculation of the first hidden layer
    z = tf.matmul(H1, W2) + tf.transpose(b2[:, None])

    # \frac{d \phi(z)}{dz} \nabla H_2
    # phi is the activation function
    dphi_dz_H2_grad = tf.multiply(tf.cast(z>0, tf.float32), H2_grad)

    H1_grad = tf.matmul(dphi_dz_H2_grad, tf.transpose(W2))
    W2_grad = tf.matmul(tf.transpose(H1), dphi_dz_H2_grad)
    b2_grad = tf.math.reduce_sum(dphi_dz_H2_grad, axis=0)

    # H1_grad should be num_data x 100
    # W2_grad should be 100x100
    # b2_grad should be 100x1
    return H1_grad, W2_grad/num_data, b2_grad/num_data

def backward_propagation_hidden_input(X: tf.Tensor, H1_grad: tf.Tensor, W1: tf.Variable, b1: tf.Variable):
    '''
    Back propagation from hidden to input layer
    X is a num_data x num_features tensor
    H1_grad is a num_data x 100 tensor
    W1 is a num_features x 100 tensor
    b1 is a 100x1 tensor
    '''

    num_data = X.shape[0]

    # The forward calculation of the input layer
    z = tf.matmul(X, W1) + tf.transpose(b1[:, None])

    # \frac{d \phi(z)}{dz} \nabla H_1
    # phi is the activation function
    dphi_dz_H1_grad = tf.multiply(tf.cast(z>0, tf.float32), H1_grad)

    W1_grad = tf.matmul(tf.transpose(X), dphi_dz_H1_grad)
    b1_grad = tf.math.reduce_sum(dphi_dz_H1_grad, axis=0)

    # W1_grad should be num_features x 100
    # b1_grad should be 100x1
    return W1_grad/num_data, b1_grad/num_data

def neural_network_model(train_dataset, test_dataset, train_label, test_label):
    train_batches = tf.data.Dataset.from_tensor_slices((train_dataset, train_label)).batch(batch_size)

    # Neural network layers for forward propagation
    hidden1 = tf.keras.layers.Dense(100, activation = 'relu')
    hidden2 = tf.keras.layers.Dense(100, activation='relu')
    output = tf.keras.layers.Dense(10, activation='softmax')
    
    training_accuracy_list = []
    testing_accuracy_list = []
    training_loss_list = []
    testing_loss_list = []

    epoch = 0
    while epoch < 10:
        epoch += 1
        print(epoch)

        stepsize = 0.1*1/epoch
        optimizer = tf.keras.optimizers.SGD(learning_rate=stepsize)

        start = time.time()
        for x, y in train_batches:
            H1 = hidden1(x)
            H2 = hidden2(H1)
            y_output = output(H2)

            W1 = hidden1.trainable_weights[0]
            b1 = hidden1.trainable_weights[1]
            W2 = hidden2.trainable_weights[0]
            b2 = hidden2.trainable_weights[1]
            W3 = output.trainable_weights[0]
            b3 = output.trainable_weights[1]
            
            H2_grad, W3_grad, b3_grad = backward_propagation_output_hidden(y_output, y, W3, H2)
            H1_grad, W2_grad, b2_grad = backward_propagation_hidden_hidden(H1, H2_grad, W2, b2)
            W1_grad, b1_grad = backward_propagation_hidden_input(x, H1_grad, W1, b1)

            # Apply the gradient to each parameter
            optimizer.apply_gradients(zip([W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad], 
                                            [W1, b1, W2, b2, W3, b3]))
        
        print(time.time()-start)

        train_output = output(hidden2(hidden1(train_dataset)))
        training_accuracy = classification_accuracy(train_output, train_label)
        training_loss = loss(train_output, train_label)

        test_output = output(hidden2(hidden1(test_dataset)))
        testing_accuracy = classification_accuracy(test_output, test_label)
        testing_loss = loss(test_output, test_label)
        
        training_accuracy_list.append(training_accuracy)
        testing_accuracy_list.append(testing_accuracy)
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)

        print("Training Accuracy:", training_accuracy, "Training Loss:", training_loss)
        print("Testing Accuracy:", testing_accuracy, "Testing Loss:", testing_loss)

    print("\n")
    digit_accuracy(test_output, test_label, num_classes)

    W_optimal = [W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy(), W3.numpy(), b3.numpy()]

    return W_optimal, training_accuracy_list, testing_accuracy_list, training_loss_list, testing_loss_list

# %%
tf.random.set_seed(0)

results = neural_network_model(train_dataset, test_dataset, train_label, test_label)
W_optimal, training_accuracy, testing_accuracy, training_loss, testing_loss = results

epoch = np.arange(len(training_accuracy))

plt.figure(figsize=(8, 5))
plt.plot(epoch, training_accuracy, label="Training accuracy")
plt.plot(epoch, testing_accuracy, label="Testing accuracy")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epoch, training_loss, label="Training loss")
plt.plot(epoch, testing_loss, label="Testing loss")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend()
plt.show()

filehandler = open("nn_parameters.txt","wb") 
pickle.dump(W_optimal, filehandler, protocol=2) 
filehandler.close() 

# %%
