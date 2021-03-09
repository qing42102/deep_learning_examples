# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
# import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

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
            
            # Conver the image to an array of floats normalized to [0, 1]
            image = Image.open(file_name).convert("F")
            image_array = np.array(image.getdata())/255.

            imgs.append(image_array)

    return imgs

train_imgs = np.array(load_images("train_data/"))
test_imgs = np.array(load_images("test_data/"))

os.chdir("labels/")
train_label = np.loadtxt("train_label.txt")
test_label = np.loadtxt("test_label.txt")
os.chdir("..")

print(train_imgs.shape, train_label.shape)
print(test_imgs.shape, test_label.shape)

# %%
def softmax_func(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    '''
    Softmax function for calculating the posterior probability
    x should be (N+1)x1
    Return a 1xK array
    '''
    x = x[:, None]
    return np.exp(x.T @ W)/np.sum(np.exp(x.T @ W))

def logistic_loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
    '''
    Logistic regression cross-entropy loss
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    total_loss = 0.0
    for i in range(X.shape[0]):
        # Create the Kx1 binary vector with 1‐of‐K encoding
        t = np.zeros(W.shape[1])
        t[int(y[i])-1] = 1

        log_likelihood = t @ np.log(softmax_func(X[i], W)).T
        total_loss += log_likelihood[0]
    
    return -total_loss/X.shape[0]

def logistic_loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Calculate the gradient for each class
    Return a (N+1)xK matrix
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    total_grad = np.zeros(shape=(W.shape))
    for i in range(X.shape[0]):
        # Create the Kx1 binary vector with 1‐of‐K encoding
        t = np.zeros(W.shape[1])
        t[int(y[i])-1] = 1

        y_diff = t-softmax_func(X[i], W)[0]
        total_grad += X[i][:, None] @ y_diff[None, :]
    
    return -total_grad/X.shape[0]

def classification_accuracy(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
    '''
    Classification accuracy for the predicted and true labels
    '''
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    accuracy = 0.0
    for i in range(X.shape[0]):
        # Select the largest probability
        y_pred = np.argmax(softmax_func(X[i], W))+1
        if y_pred == y[i]:
            accuracy += 1/X.shape[0]

    return accuracy*100

def gradient_descent(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray,\
                         W: np.ndarray, tolerance: float):
    '''
    Steepest gradient descent with a stepsize of inverse square root of iteration number
    The stopping condition is the residual of the gradient

    X should be Mx(N+1)
    W should be (N+1)xK
    y should be Mx1
    '''

    training_accuracy_list = []
    testing_accuracy_list = []
    training_loss_list = []
    testing_loss_list = []

    grad = logistic_loss_grad(train_X, W, train_y)

    # Calculate the residual of the gradient
    res = np.linalg.norm(grad)

    iteration = 1
    while res > tolerance:
        alpha = 1/np.sqrt(iteration)
        W = W - alpha*grad

        grad = logistic_loss_grad(train_X, W, train_y)
        res = np.linalg.norm(grad)

        training_accuracy = classification_accuracy(train_X, W, train_y)
        training_loss = logistic_loss(train_X, W, train_y)

        testing_accuracy = classification_accuracy(test_X, W, test_y)
        testing_loss = logistic_loss(test_X, W, test_y)

        training_accuracy_list.append(training_accuracy)
        testing_accuracy_list.append(testing_accuracy)
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)

        print(iteration)
        print("Norm of gradient:", res)
        print("Training Accuracy:", training_accuracy, "Training Loss:", training_loss)
        print("Testing Accuracy:", testing_accuracy, "Testing Loss:", testing_loss)
        print("\n")

        iteration += 1

    return training_accuracy_list, testing_accuracy_list, training_loss_list, testing_loss_list, W


# %%

num_features = train_imgs.shape[1]
num_classes = 5

# Initialize the weight vectors including the bias 
W = np.zeros(shape=(num_features+1, num_classes))

results = gradient_descent(train_X=train_imgs, train_y=train_label, test_X=test_imgs, \
                            test_y=test_label, W=W, tolerance=10**-1)

training_accuracy, testing_accuracy, training_loss, testing_loss, W_optimal = results
iteration = np.arange(len(training_accuracy))

plt.figure(figsize=(8, 5))
plt.plot(iteration, training_accuracy, label="Training accuracy")
plt.plot(iteration, testing_accuracy, label="Testing accuracy")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(iteration, training_loss, label="Training loss")
plt.plot(iteration, testing_loss, label="Testing loss")
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.show()

for i in range(W_optimal.shape[1]):
    plt.imshow(W_optimal[:, i].reshape(28,28))
    plt.colorbar()
    plt.show()


# %%
