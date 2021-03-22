# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle

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

os.chdir("deep_learning_examples/")
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

    total_loss = 0.0
    for i in range(y_output.shape[0]):
        loss = tf.matmul(tf.transpose(y_label[i]), tf.math.log(y_output[i]))
        total_loss += loss
    
    return -total_loss/y_output.shape[0]

def classification_accuracy(y_output: tf.Tensor, y_label: tf.Tensor) -> float:
    '''
    Classification accuracy for the predicted and true labels
    '''
    
    accuracy = tf.math.reduce_sum(y_output == y_label)/y_output.shape[0]
    return accuracy*100

def backward_propagation_output_hidden(y_output: tf.Tensor, y_label: tf.Tensor, W3: np.ndarray, H2: tf.Tensor):
    '''
    y_output is a MxK tensor
    y_label is a MxK tensor
    W3 is a 100xK tensor
    H2 is a Mx100 tensor
    '''

    num_data = y_output.shape[0]
    num_output = y_output.shape[1]
    prev_num_output = H2.shape[1]

    loss_grad = -tf.divide(y_label, y_output)

    dsigma_dz = np.zeros((num_data, num_output, num_output))
    for i in range(dsigma_dz.shape[0]):
        for j in range(dsigma_dz.shape[1]):
            for k in range(dsigma_dz.shape[2]):
                if j == k:
                    dsigma_dz[i][j][k] = y_output[i][j]*(1-y_output[i][j])
                else:
                    dsigma_dz[i][j][k] = -y_output[i][j]*y_output[i][k]
    dsigma_dz = tf.convert_to_tensor(dsigma_dz, dtype=tf.float32)

    dz_dW3 = np.zeros((num_data, prev_num_output, num_output, num_output))
    for i in range(dz_dW3.shape[0]):
        for j in range(dz_dW3.shape[2]):
            dz_dW3[i, :, j, j] = H2[i].numpy()
    dz_dW3 = tf.convert_to_tensor(dz_dW3, dtype=tf.float32)

    dsigma_dz_loss_grad = tf.matmul(dsigma_dz, loss_grad[:, :, None])
    H2_grad = tf.einsum("ij, kjl -> ki", W3, dsigma_dz_loss_grad)
    W3_grad = tf.einsum("ijkl, ilm -> jk", dz_dW3, dsigma_dz_loss_grad)
    b3_grad = tf.math.reduce_sum(dsigma_dz_loss_grad, axis=0)

    return H2_grad/num_data, W3_grad/num_data, b3_grad/num_data

def backward_propagation_hidden_hidden(H1: tf.Tensor, H2_grad: tf.Tensor, W2: np.ndarray, b2: np.ndarray):
    '''
    H1 is a Mx100 tensor
    H2_grad is a 50x100 tensor
    W2 is a 100x100 tensor
    b2 is a 100x1 tensor
    '''

    num_data = H1.shape[0]
    num_output = H2_grad.shape[1]
    prev_num_output = H1.shape[1]

    z = tf.matmul(H1, W2) + tf.transpose(b2[:, None])

    dphi_dz = np.zeros((num_data, num_output, num_output))
    for i in range(dphi_dz.shape[0]):
        for j in range(dphi_dz.shape[1]):
            if z[i][j] > 0:
                dphi_dz[i][j][j] = 1
    dphi_dz = tf.convert_to_tensor(dphi_dz, dtype=tf.float32)

    dz_dW2 = np.zeros((num_data, prev_num_output, num_output, num_output))
    for i in range(dz_dW2.shape[0]):
        for j in range(dz_dW2.shape[2]):
            dz_dW2[i, :, j, j] = H1[i].numpy()
    dz_dW2 = tf.convert_to_tensor(dz_dW2, dtype=tf.float32)

    dphi_dz_H2_grad = tf.matmul(dphi_dz, H2_grad[:, :, None])
    H1_grad = tf.einsum("ij, kjl -> ki", W2, dphi_dz_H2_grad)
    W2_grad = tf.einsum("ijkl, ilm -> jk", dz_dW2, dphi_dz_H2_grad)
    b2_grad = tf.math.reduce_sum(dphi_dz_H2_grad, axis=0)

    return H1_grad/num_data, W2_grad/num_data, b2_grad/num_data

def backward_propagation_hidden_input(X: tf.Tensor, H1_grad: tf.Tensor, W1: np.ndarray, b1: np.ndarray):
    '''
    X is a Mx784 tensor
    H1_grad is a 50x100 tensor
    W1 is a 784x100 tensor
    b1 is a 100x1 tensor
    '''
    
    num_data = X.shape[0]
    num_output = H1_grad.shape[1]
    prev_num_output = X.shape[1]

    z = tf.matmul(X, W1) + tf.transpose(b1[:, None])

    dphi_dz = np.zeros((num_data, num_output, num_output))
    for i in range(dphi_dz.shape[0]):
        for j in range(dphi_dz.shape[1]):
            if z[i][j] > 0:
                dphi_dz[i][j][j] = 1
    dphi_dz = tf.convert_to_tensor(dphi_dz, dtype=tf.float32)

    dz_dW1 = np.zeros((num_data, prev_num_output, num_output, num_output))
    for i in range(dz_dW1.shape[0]):
        for j in range(dz_dW1.shape[2]):
            dz_dW1[i, :, j, j] = X[i].numpy()
    dz_dW1 = tf.convert_to_tensor(dz_dW1, dtype=tf.float32)

    dphi_dz_H1_grad = tf.matmul(dphi_dz, H1_grad[:, :, None])
    W1_grad = tf.einsum("ijkl, ilm -> jk", dz_dW1, dphi_dz_H1_grad)
    b1_grad = tf.math.reduce_sum(dphi_dz_H1_grad, axis=0)

    return W1_grad/num_data, b1_grad/num_data

def neural_network_model(train_dataset, test_dataset, step_size: float):
    '''
    X is a MxN tensor
    y is a MxK tensor
    '''

    hidden1 = tf.keras.layers.Dense(100, activation = 'relu')
    hidden2 = tf.keras.layers.Dense(100, activation='relu')
    output = tf.keras.layers.Dense(10, activation='softmax')

    optimizer = tf.keras.optimizers.SGD(learning_rate=step_size)
    
    epoch = 0
    while epoch < 2:
        epoch += 1
        for x, y in train_dataset:
            H1 = hidden1(x)
            H2 = hidden2(H1)
            y_output = output(H2)

            W1 = hidden1.get_weights()[0]
            b1 = hidden1.get_weights()[1]
            W2 = hidden2.get_weights()[0]
            b2 = hidden2.get_weights()[1]
            W3 = output.get_weights()[0]
            b3 = output.get_weights()[1]

            H2_grad, W3_grad, b3_grad = backward_propagation_output_hidden(y_output, y, W3, H2)
            H1_grad, W2_grad, b2_grad = backward_propagation_hidden_hidden(H1, H2_grad, W2, b2)
            W1_grad, b1_grad = backward_propagation_hidden_input(x, H1_grad, W1, b1)

            optimizer.apply_gradients(zip([W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad], [W1, b1, W2, b2, W3, b3]))

            hidden1.set_weights([W1, b1])
            hidden2.set_weights([W2, b2])
            output.set_weights([W3, b3])

        test_x, test_y = zip(*test_dataset)
        test_output = output(hidden2(hidden1(test_x)))
        testing_accuracy = classification_accuracy(test_output, test_y)

        print("Testing Accuracy:", testing_accuracy)

    return 

# %%

tf.random.set_seed(0)

# Initialize the weight vectors including the bias 
neural_network_model(train_dataset, test_dataset, step_size=10**-2)


