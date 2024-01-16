import matplotlib.pyplot as plt
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape)

# plt.imshow(x_train[1])
# plt.show()

x_train, x_test = x_train/255.0, x_test/255.0

## Regular NN
# model_NN = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(32,32,3)),
#     # tf.keras.layers.Dense(128, activation='sigmoid'),
#     # tf.keras.layers.Dense(128, activation='sigmoid'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(100, activation='softmax')
# ])

# model_NN.compile(
#     optimizer = 'adam',
#     loss = 'sparse_categorical_crossentropy',
#     metrics = ['accuracy']
# )

# model_NN.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), batch_size = 128)

## CNN
model_CNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), strides = (1,1)),
    tf.keras.layers.AveragePooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides = (1,1)),
    tf.keras.layers.AveragePooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', strides = (1,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    # tf.keras.layers.Conv2D(128, (3,3), activation='relu', strides = (1,1)),
    # tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_CNN.summary()
model_CNN.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'] 
)

# model_CNN.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test), batch_size = 128)

history_CNN = model_CNN.fit(x_train, y_train, epochs = 20, validation_data = (x_test, y_test), batch_size = 128)
plt.plot(history_CNN.history['accuracy'], label = 'accuracy')
plt.plot(history_CNN.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc = 'lower right')
plt.show()