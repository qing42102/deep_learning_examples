# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
def digit_accuracy(y_output: tf.Tensor, y_label: tf.Tensor, num_classes: int):
    '''
    Classification accuracy for each of the categories
    '''    

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    y_label = y_label[:, 0]

    # Select the largest probability
    y_output = tf.argmax(y_output, axis=1)

    for i in range(num_classes):
        y_i = y_label[y_label==i]
        y_output_i = y_output[y_label==i]

        correct = tf.cast(y_output_i == y_i, tf.float32)
        accuracy = tf.math.reduce_sum(correct).numpy()/y_output_i.shape[0]

        print(class_names[i], "accuracy:", accuracy*100)
    
    print("\n")

# %%
tf.random.set_seed(0)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
train_images, test_images = train_images/255.0, test_images/255.0
train_images = (train_images-train_images.mean())/train_images.std()
test_images = (test_images-test_images.mean())/test_images.std()

num_classes = len(np.unique(test_labels))

train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)

# %%
# Create the CNN model with 3 convolutional layers with max pooling
# Batch normalization and dropout is added between convolutional layers
# Output the probability for 10 classes
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', \
    input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00045),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

results = model.fit(train_images, train_labels, epochs=50, batch_size=50,
                    validation_data=(test_images, test_labels), verbose=2)

# %%
model.save("CNN")

test_output = model.predict(test_images)
digit_accuracy(test_output, test_labels, num_classes)

# Normalize the filters of the first convolutional layer
filters, biases = model.layers[0].get_weights()
filters = (filters - filters.min()) 

# Plot all 32 filters in 8 columns and 4 rows
fig = plt.figure(figsize=(10, 40))
for i in range(filters.shape[3]):
    filter_i = filters[:, :, :, i]
    ax = fig.add_subplot(filters.shape[3], 8, i+1)
    ax.set_title("Filter" + str(i+1), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(filter_i)
fig.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(results.history['accuracy'], label='Training Accuracy')
plt.plot(results.history['val_accuracy'], label = 'Testing Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(results.history['loss'], label='Training Loss')
plt.plot(results.history['val_loss'], label = 'Testing Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.show()

# %%
