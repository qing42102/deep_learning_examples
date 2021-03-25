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

train_imgs = load_images("train_data/")
test_imgs = load_images("test_data/")

os.chdir("labels/")
train_label = np.loadtxt("train_label.txt", dtype=int)
test_label = np.loadtxt("test_label.txt", dtype=int)
os.chdir("..")

num_features = len(train_imgs[0])
num_classes = 10

# Samples per iteration
batch_size = 50

train_label = tf.one_hot(train_label, depth=num_classes, dtype=tf.float32)
test_label = tf.one_hot(test_label, depth=num_classes, dtype=tf.float32)
train_imgs = tf.convert_to_tensor(train_imgs, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_label)).batch(batch_size)
test_dataset = tf.convert_to_tensor(test_imgs, dtype=tf.float32)

print(test_dataset.shape, test_label.shape)

# %%
def loss(y_output: tf.Tensor, y_label: tf.Tensor) -> float:
    '''
    Cross-entropy loss
    y_output is a MxK tensor
    y_label is a MxK tensor
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

def backward_propagation_output_hidden(y_output: tf.Tensor, y_label: tf.Tensor, W3: np.ndarray, H2: tf.Tensor):
    '''
    Back propagation from output to hidden layer
    y_output is a MxK tensor
    y_label is a MxK tensor
    W3 is a 100xK tensor
    H2 is a Mx100 tensor
    '''

    num_data = y_output.shape[0]
    num_output = y_output.shape[1]
    prev_num_output = H2.shape[1]

    loss_grad = -tf.divide(y_label, y_output)

    # H2_grad should be Mx100
    # W3_grad should be 100x10
    # b3_grad should be 10x1
    H2_grad = np.zeros((num_data, prev_num_output))
    W3_grad = tf.zeros((prev_num_output, num_output))
    b3_grad = tf.zeros((num_output))

    # Sum over each data point
    for i in range(num_data):
        # Calculate the off diaognals elements and then the diagonal
        dsigma_dz = tf.matmul(-y_output[i][:, None], y_output[i][None, :])
        dsigma_dz = tf.linalg.set_diag(dsigma_dz, y_output[i]*(1-y_output[i]))

        dz_dW3 = np.zeros((prev_num_output, num_output, num_output))
        # Set the diagonal of the 2nd and 3rd axis
        dz_dW3[:, np.eye(dz_dW3.shape[1], dtype=bool)] = H2[i][:, None]
        dz_dW3 = tf.convert_to_tensor(dz_dW3, dtype=tf.float32)

        dsigma_dz_loss_grad = tf.matmul(dsigma_dz, loss_grad[i][:, None])
        H2_grad[i] = tf.matmul(W3, dsigma_dz_loss_grad).numpy()[:, 0]
        W3_grad += tf.matmul(dz_dW3, dsigma_dz_loss_grad)[:, :, 0]
        b3_grad += dsigma_dz_loss_grad[:, 0]

    H2_grad = tf.convert_to_tensor(H2_grad, dtype=tf.float32) 

    return H2_grad/num_data, W3_grad/num_data, b3_grad/num_data

def backward_propagation_hidden_hidden(H1: tf.Tensor, H2_grad: tf.Tensor, W2: np.ndarray, b2: np.ndarray):
    '''
    Back propagation from hidden to hidden layer
    H1 is a Mx100 tensor
    H2_grad is a 50x100 tensor
    W2 is a 100x100 tensor
    b2 is a 100x1 tensor
    '''

    num_data = H1.shape[0]
    num_output = H2_grad.shape[1]
    prev_num_output = H1.shape[1]

    z = tf.matmul(H1, W2) + tf.transpose(b2[:, None])

    # H1_grad should be Mx100
    # W2_grad should be 100x100
    # b2_grad should be 100x1
    H1_grad = np.zeros((num_data, prev_num_output))
    W2_grad = tf.zeros((prev_num_output, num_output))
    b2_grad = tf.zeros((num_output))

    # Sum over each data point
    for i in range(num_data):
        dphi_dz = tf.zeros((num_output, num_output))
        # Set the diagonal based on the sign of z[i]
        dphi_dz = tf.linalg.set_diag(dphi_dz, tf.cast(z[i]>0, tf.float32))

        dz_dW2 = np.zeros((prev_num_output, num_output, num_output))
        # Set the diagonal of the 2nd and 3rd axis
        dz_dW2[:, np.eye(dz_dW2.shape[1], dtype=bool)] = H1[i][:, None]
        dz_dW2 = tf.convert_to_tensor(dz_dW2, dtype=tf.float32)

        dphi_dz_H2_grad = tf.matmul(dphi_dz, H2_grad[i][:, None])
        H1_grad[i] = tf.matmul(W2, dphi_dz_H2_grad).numpy()[:, 0]
        W2_grad += tf.matmul(dz_dW2, dphi_dz_H2_grad)[:, :, 0]
        b2_grad += dphi_dz_H2_grad[:, 0]

    H1_grad = tf.convert_to_tensor(H1_grad, dtype=tf.float32)

    return H1_grad/num_data, W2_grad/num_data, b2_grad/num_data

def backward_propagation_hidden_input(X: tf.Tensor, H1_grad: tf.Tensor, W1: np.ndarray, b1: np.ndarray):
    '''
    Back propagation from hidden to input layer
    X is a Mx784 tensor
    H1_grad is a 50x100 tensor
    W1 is a 784x100 tensor
    b1 is a 100x1 tensor
    '''
    
    num_data = X.shape[0]
    num_output = H1_grad.shape[1]
    prev_num_output = X.shape[1]

    z = tf.matmul(X, W1) + tf.transpose(b1[:, None])

    # W1_grad should be 784x100
    # b1_grad should be 100x1
    W1_grad = tf.zeros((prev_num_output, num_output))
    b1_grad = tf.zeros((num_output))

    # Sum over each data point
    for i in range(num_data):
        dphi_dz = tf.zeros((num_output, num_output))
        # Set the diagonal based on the sign of z[i]
        dphi_dz = tf.linalg.set_diag(dphi_dz, tf.cast(z[i]>0, tf.float32))

        dz_dW1 = np.zeros((prev_num_output, num_output, num_output))
        # Set the diagonal of the 2nd and 3rd axis
        dz_dW1[:, np.eye(dz_dW1.shape[1], dtype=bool)] = X[i][:, None]
        dz_dW1 = tf.convert_to_tensor(dz_dW1, dtype=tf.float32)

        dphi_dz_H1_grad = tf.matmul(dphi_dz, H1_grad[i][:, None])
        W1_grad += tf.matmul(dz_dW1, dphi_dz_H1_grad)[:, :, 0]
        b1_grad += dphi_dz_H1_grad[:, 0]

    return W1_grad/num_data, b1_grad/num_data

def neural_network_model(train_dataset, test_dataset, test_label):
    '''
    X is a MxN tensor
    y is a MxK tensor
    '''

    # Get the entire training dataset as one tensor
    entire_train_x = tf.concat([x for x, y in train_dataset], axis=0)
    entire_train_y = tf.concat([y for x, y in train_dataset], axis=0)

    # Neural network layers for forward propagation
    hidden1 = tf.keras.layers.Dense(100, activation = 'relu')
    hidden2 = tf.keras.layers.Dense(100, activation='relu')
    output = tf.keras.layers.Dense(10, activation='softmax')
    
    training_accuracy_list = []
    testing_accuracy_list = []
    training_loss_list = []
    testing_loss_list = []

    epoch = 0
    while epoch < 1:
        epoch += 1

        stepsize = 1/epoch
        optimizer = tf.keras.optimizers.SGD(learning_rate=stepsize)

        iteration = 0
        for x, y in train_dataset:
            iteration += 1
            print(iteration)
            
            # Forward propagation
            H1 = hidden1(x)
            H2 = hidden2(H1)
            y_output = output(H2)

            W1 = hidden1.trainable_weights[0]
            b1 = hidden1.trainable_weights[1]
            W2 = hidden2.trainable_weights[0]
            b2 = hidden2.trainable_weights[1]
            W3 = output.trainable_weights[0]
            b3 = output.trainable_weights[1]
            
            start = time.time()
            H2_grad, W3_grad, b3_grad = backward_propagation_output_hidden(y_output, y, W3, H2)
            H1_grad, W2_grad, b2_grad = backward_propagation_hidden_hidden(H1, H2_grad, W2, b2)
            W1_grad, b1_grad = backward_propagation_hidden_input(x, H1_grad, W1, b1)
            print(time.time()-start)
            
            # Apply the gradient to each parameter
            optimizer.apply_gradients(zip([W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad], [W1, b1, W2, b2, W3, b3]))

            if iteration % 10 == 0:
                train_output = output(hidden2(hidden1(entire_train_x)))
                training_accuracy = classification_accuracy(train_output, entire_train_y)
                training_loss = loss(train_output, entire_train_y)

                test_output = output(hidden2(hidden1(test_dataset)))
                testing_accuracy = classification_accuracy(test_output, test_label)
                testing_loss = loss(test_output, test_label)
                
                training_accuracy_list.append(training_accuracy)
                testing_accuracy_list.append(testing_accuracy)
                training_loss_list.append(training_loss)
                testing_loss_list.append(testing_loss)

                print("Training Accuracy:", training_accuracy, "Training Loss:", training_loss)
                print("Testing Accuracy:", testing_accuracy, "Testing Loss:", testing_loss)

    digit_accuracy(test_output, test_label, num_classes)

    W_optimal = [W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy(), W3.numpy(), b3.numpy()]

    return W_optimal, training_accuracy_list, testing_accuracy_list, training_loss_list, testing_loss_list

# %%

tf.random.set_seed(0)

results = neural_network_model(train_dataset, test_dataset, test_label)
W_optimal, training_accuracy, testing_accuracy, training_loss, testing_loss = results

iteration = np.arange(len(training_accuracy)*10, step=10)

plt.figure(figsize=(8, 5))
plt.plot(iteration, training_accuracy, label="Training accuracy")
plt.plot(iteration, testing_accuracy, label="Testing accuracy")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(iteration, training_loss, label="Training loss")
plt.plot(iteration, testing_loss, label="Testing loss")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend()
plt.show()

filehandler = open("nn_parameters.txt","wb") 
pickle.dump(W_optimal, filehandler, protocol=2) 
filehandler.close() 
