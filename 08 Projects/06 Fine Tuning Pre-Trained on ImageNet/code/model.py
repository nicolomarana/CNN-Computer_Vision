from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt
import datetime


'''
    This function vgg16(weight_path=None)
    
    - Create structure as VGG16
    - Import weight of VGG16 trained on "ImageNet"
    
    - return the model vgg16 with weights
    
'''


def vgg16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


''' 
    This function vgg16_edit(path):
    
    - Create a model with structure of vgg16 through vgg16()
    - Freeze layer of this model
    - Remove a defined number of layer
    - Add new layers
 
    - return the model based on vgg16 but modified

    
'''


def vgg16_edit(path):
    model = vgg16(path)

    # Freezing Vgg16 layers
    for layer in model.layers[:37]:
        layer.trainable = False

    # number of element to delete
    number_delete_layer = 5

    for i in range(number_delete_layer):
        model.pop()

    # Add new layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    return model


'''
    This function training(model, loss, optimizer, train, dev, epoch)
    
    - compile the model
    - set a start timer
    - fit the model
    - set a stop timer
    - print time of execution
    - plot result
   
'''


def training(model, loss, optimizer, train, dev, epoch):

    # Compile model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Start timer
    time_start = datetime.datetime.now()

    # Fit model
    result = model.fit_generator(train,
                                 steps_per_epoch=5,
                                 epochs=epoch,
                                 validation_data=dev,
                                 validation_steps=5)

    # Stop timer
    time_stop = datetime.datetime.now()

    # Print time
    print("Execution time: ", (time_stop - time_start).total_seconds())

    # Show Plot
    show_history(result, 'acc', 'val_acc', 'accuracy', 'epoch', 'train', 'validation', 1)
    show_history(result, 'loss', 'val_loss', 'loss', 'epoch', 'train', 'validation')

    return model


'''
    This function show_history(result, measure1, measure2, metrics, unit, set1, set2)
    
    - plot measure1 = e.g. acc
    - plot measure2 = e.g. val_acc
    - metrics e.g. accuracy
    - unit e.g. epoch
    - set1 e.g. training set
    - set2 e.g. validation set

'''


def show_history(result, measure1='', measure2='', metrics='', unit='', set1='', set2='', acc=None):
    plt.plot(result.history[measure1])
    plt.plot(result.history[measure2])
    axes = plt.gca()
    axes.set_xlim([0, epochs])
    axes.set_ylim([0, acc])

    plt.ylabel(metrics)
    plt.xlabel(unit)
    plt.legend([set1, set2], loc='upper left')
    plt.show()


''' 
    This function testing(model, test):
    
    - evaluate model
    - print result

'''


def testing(model, test, batch):

    score = model.evaluate_generator(test, batch)

    print(score[0], 'loss')
    print(score[1], 'accuracy')

    return

