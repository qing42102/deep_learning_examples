# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def class_accuracy(y_actual: tf.Tensor, y_pred: tf.Tensor):
    '''
    Classification accuracy for each of the categories
    '''
    # Select the largest probability
    predicted_labels = tf.argmax(y_pred, axis=1)

    # y_actual = tf.cast(y_actual[:, 0], tf.int64)

    class_accuracy_list = np.zeros(num_classes)
    for i in range(num_classes):
        y_actual_i = y_actual[y_actual==i]
        y_pred_i = predicted_labels[y_actual==i]

        correct = tf.cast(y_pred_i == y_actual_i, tf.float32)
        accuracy = tf.math.reduce_sum(correct).numpy()/y_pred_i.shape[0]

        class_accuracy_list[i] = accuracy*100
    
    return class_accuracy_list

# %%
tf.random.set_seed(0)
np.random.seed(0)

data_file = open('youtube_action_train_data_part1.pkl', 'rb')
train_data1, train_labels1 = pickle.load(data_file)
data_file.close()
data_file = open('youtube_action_train_data_part2.pkl', 'rb')
train_data2, train_labels2 = pickle.load(data_file)
data_file.close()

data = np.concatenate((train_data1, train_data2))
label = np.concatenate((train_labels1, train_labels2))
del train_data1, train_data2, train_labels1, train_labels2
print(data.shape)
print(label.shape)

train_data, test_data, train_label, test_label = train_test_split(data, label, train_size=0.75, random_state=0)
print(train_data.shape, test_data.shape)

train_data = tf.convert_to_tensor(train_data/255.0, dtype=tf.float32)
test_data = tf.convert_to_tensor(test_data/255.0, dtype=tf.float32)
train_label = tf.convert_to_tensor(train_label, dtype=tf.int64)
test_label = tf.convert_to_tensor(test_label, dtype=tf.int64)

label_names = ["basketball shooting", "biking", "diving", "golf swing", "horseback riding", 
            "soccer juggling", "swinging", "tennis swinging", "trampoline jumping", 
            "volleyball spiking", "walking with a dog"]
num_classes = len(label_names)

# %%
# ResNet CNN
def resnet_block(input_data: tf.Tensor, filters: int, stride: int=1) -> tf.Tensor:
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=stride, activation='relu', 
                            padding='same')(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters, kernel_size=3, activation=None, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    if stride != 1:
        x_skip = keras.layers.Conv2D(filters, kernel_size=1, strides=stride, 
                                    padding='same')(input_data)
        x_skip = keras.layers.BatchNormalization()(x_skip)
        x = keras.layers.Add()([x, x_skip])
        x = keras.layers.Activation('relu')(x)
    else:
        x = keras.layers.Add()([x, input_data])
        x = keras.layers.Activation('relu')(x)

    return x

resnet_input = keras.Input(train_data.shape[2:], dtype=tf.float32)
x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, activation='relu')(resnet_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)
x = resnet_block(input_data=x, filters=64)
x = resnet_block(input_data=x, filters=64)
x = resnet_block(input_data=x, filters=128, stride=2)
x = resnet_block(input_data=x, filters=128)
x = resnet_block(input_data=x, filters=256, stride=2)
x = resnet_block(input_data=x, filters=256)
resnet_output = keras.layers.GlobalAveragePooling2D()(x)
resnet_model = keras.Model(resnet_input, resnet_output)

# %%
# LSTM RNN
rnn_model = keras.Sequential()
rnn_model.add(keras.layers.LSTM(256, return_sequences=True, 
                                activity_regularizer=keras.regularizers.L2(0.01)))
rnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))
rnn_model.add(keras.layers.GlobalAvgPool1D())

input = keras.Input(train_data.shape[1:], dtype=tf.float32)
output = rnn_model(keras.layers.TimeDistributed(resnet_model)(input))

final_model = keras.Model(inputs=input, outputs=output)
resnet_model.summary()
rnn_model.summary()

final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2.5*10**-4),
                    loss=keras.losses.SparseCategoricalCrossentropy(), 
                    metrics = ["accuracy"])

results = final_model.fit(train_data, train_label, epochs=10, batch_size=16,
                        validation_data=(test_data, test_label), verbose=2)
                    
final_model.save("LRCN")

# %%
predicted_probability = final_model.predict(test_data)
print(class_accuracy(test_label, predicted_probability))

mat = confusion_matrix(test_label, tf.argmax(predicted_probability, axis=1))
plt.figure(1, figsize=(10, 10))
sns.heatmap(mat.T, square=True, annot=True, cmap='BuPu', fmt='g', cbar=False,
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('True labels', fontsize=14)
plt.ylabel('Predicted labels', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.figure(2, figsize=(8, 5))
plt.plot(results.history['accuracy'], label='Training Accuracy')
plt.plot(results.history['val_accuracy'], label = 'Testing Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(3, figsize=(8, 5))
plt.plot(results.history['loss'], label='Training Loss')
plt.plot(results.history['val_loss'], label = 'Testing Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()

# %%
