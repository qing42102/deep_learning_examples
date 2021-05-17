# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# %%
def mean_per_joint_position_error(y_actual: tf.Tensor, y_pred: tf.Tensor):
    '''
    y_pred and y_actual are 4 dimensional tensors
    batch_size x num_frames x num_joints x 3
    '''    
    euclidean_dist = tf.norm(y_actual-y_pred, ord="euclidean", axis=3)
    
    return tf.math.reduce_mean(euclidean_dist)*1000

# %%
training_data = np.load('videoframes_clips_train.npy')
validation_data = np.load('videoframes_clips_valid.npy')
training_label = np.load('joint_3d_clips_train.npy')
validation_label = np.load('joint_3d_clips_valid.npy')

training_data = tf.convert_to_tensor(training_data/255.0, dtype=tf.float32)
validation_data = tf.convert_to_tensor(validation_data/255.0, dtype=tf.float32)
training_label = tf.convert_to_tensor(training_label, dtype=tf.float32)
validation_label = tf.convert_to_tensor(validation_label, dtype=tf.float32)

num_frames = training_data.shape[1]
num_joints = training_label.shape[2]
num_coordinates = 3

# %%
tf.random.set_seed(0)

# AlexNet CNN
cnn_model = keras.Sequential()
cnn_model.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'))
cnn_model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2))
cnn_model.add(keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu', padding='same'))
cnn_model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2))
cnn_model.add(keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'))
cnn_model.add(keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'))
cnn_model.add(keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
cnn_model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2))
cnn_model.add(keras.layers.Flatten())
cnn_model.add(keras.layers.Dense(512, activation='relu'))
cnn_model.add(keras.layers.Dropout(0.5, seed=0))

# LSTM RNN
rnn_model = keras.Sequential()
rnn_model.add(keras.layers.LSTM(256, return_sequences=True))

# Multilayer perceptron
mlp_model = keras.Sequential()
mlp_model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=keras.regularizers.L2(0.01)))
mlp_model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=keras.regularizers.L2(0.01)))
mlp_model.add(keras.layers.Dense(num_joints*num_coordinates, activation=None))
mlp_model.add(keras.layers.Reshape((num_frames, num_joints, num_coordinates)))

input = keras.Input(training_data.shape[1:], dtype=tf.float32)
output = mlp_model(rnn_model(keras.layers.TimeDistributed(cnn_model)(input)))

final_model = keras.Model(inputs=input, outputs=output)
cnn_model.summary()
rnn_model.summary()
mlp_model.summary()

final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2.5*10**-4),
            loss=keras.losses.MeanSquaredError(), 
            metrics = [mean_per_joint_position_error])

results = final_model.fit(training_data, training_label, epochs=15, batch_size=16,
                    validation_data=(validation_data, validation_label), verbose=2)

# To load the model, use "model = keras.models.load_model("LSTM", compile=False)"
# Only use model.predict() and not model.evaluate()
final_model.save("LSTM")

# %%
plt.figure(1, figsize=(8, 5))
plt.plot(results.history['mean_per_joint_position_error'], label='Training MPJPE')
plt.plot(results.history['val_mean_per_joint_position_error'], label = 'Testing MPJPE')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Per Joint Position Error (mm)', fontsize=14)
plt.legend(fontsize=14)
plt.show()

plt.figure(2, figsize=(8, 5))
plt.plot(results.history['loss'], label='Training Loss')
plt.plot(results.history['val_loss'], label = 'Testing Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()
# %%
