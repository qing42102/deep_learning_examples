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

train_label = tf.one_hot(train_label-1, depth=num_classes, dtype=tf.float32)
test_label = tf.one_hot(test_label-1, depth=num_classes, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_label)).batch(batch_size)
test_dataset = tf.convert_to_tensor(test_imgs, dtype=tf.float32)

print(test_dataset.shape, test_label.shape)

# %%
def loss(y_output: tf.Tensor, y_label: tf.Tensor):
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

def backward_propagation_output_hidden(y_output: tf.Tensor, y_label: tf.Tensor, W3: np.ndarray, H2: tf.Tensor):
    '''
    z2 is a Mx100 tensor
    z1 is a Mx100 tensor
    W3 is a 100x10 tensor
    '''
    loss_grad = -tf.divide(y_label, y_output)

    dsigma_dz = np.zeros((y_output.shape[0], y_output.shape[1], y_output.shape[1]))
    for i in range(dsigma_dz.shape[0]):
        for j in range(dsigma_dz.shape[1]):
            for k in range(dsigma_dz.shape[2]):
                if j == k:
                    dsigma_dz[i][j][k] = y_output[i][j]*(1-y_output[i][j])
                else:
                    dsigma_dz[i][j][k] = -y_output[i][j]*y_output[i][k]
    dsigma_dz = tf.convert_to_tensor(dsigma_dz, dtype=tf.float32)

    dz_dW3 = np.zeros((y_output.shape[0], W3.shape[0], y_output.shape[1], y_output.shape[1]))
    for i in range(dz_dW3.shape[0]):
        for k in range(dz_dW3.shape[2]):
            for l in range(dz_dW3.shape[3]):
                if l == k:
                    dz_dW3[i, :, k, l] = H2[i].numpy()
    dz_dW3 = tf.convert_to_tensor(dz_dW3, dtype=tf.float32)

    dsigma_dz_loss_grad = tf.matmul(dsigma_dz, loss_grad[:, :, None])
    H2_grad = tf.matmul(W3, dsigma_dz_loss_grad)
    W3_grad = tf.einsum("ijkl, lm -> jk", dz_dW3, dsigma_dz_loss_grad)
    b3_grad = dsigma_dz_loss_grad

    return H2_grad/y_output.shape[0], W3_grad/y_output.shape[0], b3_grad/y_output.shape[0]

def backward_propagation_hidden_hidden(y_output: tf.Tensor, y_label: tf.Tensor, z2: tf.Tensor, z1: tf.Tensor, W3, W2):
    '''
    z2 is a Mx100 tensor
    z1 is a Mx100 tensor
    '''

    return 

def neural_network_model(X, y, step_size: float):
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

            hidden1.set_weights()
            hidden2.set_weights()
            output.set_weights()

    return 

# %%

# Initialize the weight vectors including the bias 
neural_network_model(train_dataset, train_label, step_size=10**-2)


