# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
# import tensorflow as tf
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
def softmax_func(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    '''
    Softmax function for calculating the posterior probability
    X should be Mx(N+1)
    Return a MxK matrix
    '''
    exp = np.exp(X @ W)
    sum = np.sum(np.exp(X @ W), axis=1)
    return exp/sum[:, None]

def logistic_loss(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
    '''
    Logistic regression cross-entropy loss
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    log_likelihood = np.log(softmax_func(X, W))
    total_loss = 0.0
    for i in range(X.shape[0]):
        # Create the Kx1 binary vector with 1‐of‐K encoding
        t = np.zeros(W.shape[1])
        t[int(y[i])-1] = 1

        total_loss += t @ log_likelihood[i, :].T
    
    return -total_loss/X.shape[0]

def logistic_loss_grad(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Calculate the gradient for each class
    Return a (N+1)xK matrix
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Create the Kx1 binary vector with 1‐of‐K encoding
    t = np.zeros((y.shape[0], W.shape[1]))
    t[np.arange(y.size), y.astype(int)-1] = 1

    y_diff = t-softmax_func(X, W)
    total_grad = X.T @ y_diff
    
    return -total_grad/X.shape[0]

def classification_accuracy(X: np.ndarray, W: np.ndarray, y: np.ndarray) -> float:
    '''
    Classification accuracy for the predicted and true labels
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Select the largest probability
    y_pred = np.argmax(softmax_func(X, W), axis=1)+1
    
    accuracy = np.sum(y_pred == y)/X.shape[0]

    return accuracy*100

def digit_accuracy(X: np.ndarray, W: np.ndarray, y: np.ndarray):
    '''
    Classification accuracy for each of the digits
    '''
    # Add the bias coefficient to the data
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Select the largest probability
    y_pred = np.argmax(softmax_func(X, W), axis=1)+1
    
    for i in range(W.shape[1]):
        y_i = y[y==i+1]
        y_pred_i = y_pred[y==i+1]
        accuracy = np.sum(y_pred_i == y_i)/y_i.shape[0]
        print("Digit", i+1, "accuracy:", accuracy)
    
    print("\n")


def gradient_descent(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray,\
                         W: np.ndarray, tolerance: float):
    '''
    Steepest gradient descent with a stepsize of inverse square root of iteration number
    The stopping condition is the residual of the gradient and a maximum iteration number of 200

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
    while res > tolerance and iteration != 200:
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
                            test_y=test_label, W=W, tolerance=10**-2)

training_accuracy, testing_accuracy, training_loss, testing_loss, W_optimal = results
iteration = np.arange(len(training_accuracy))

print("Training digits")
digit_accuracy(X=train_imgs, W=W_optimal, y=train_label)
print("Testing digits")
digit_accuracy(X=test_imgs, W=W_optimal, y=test_label)

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

for i in range(num_classes):
    plt.imshow(W_optimal[:num_features, i].reshape(28,28))
    plt.colorbar()
    plt.show()

filehandler = open("multiclass_parameters.txt","wb")
pickle.dump(W_optimal, filehandler)
filehandler.close()



# %%
