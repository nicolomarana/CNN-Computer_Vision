# Fix seed for reproducibility of problem
from reproduce import *
setup(seed_value=42)

# Import Library
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt


# Download data from keras.datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

''' 
    Data Dimension Analysis

    x_train.shape = (60.000, 28, 28)
    y_train.shape = (60.000,)

    x_test.shape = (10.000, 28, 28)
    y_test.shape = (10.000,)

'''


# Reshape dimension of x_train and x_test
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


# in mnist dataset are present 10 different classes
num_classes = 10


# Apply conversion of y transforming in One-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Structure of CNN model
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(28, 28, 1)))
model.add(Conv2D(28, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(14, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Print model summary for see number of parameter in this case the constrain is < 10.000.
print(model.summary())


# Define hyper-paramenter
batch = 256
epoch = 10


# Fit Model
history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, verbose=1, validation_data=(x_test, y_test))


# Plot accuracy on train e validation
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Plot loss on train e validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()