from keras.preprocessing.image import ImageDataGenerator

''' 
    This function data_generator(path, shuffle)
    
    - Create an instance of ImageDataGenerator with different parametrer
    - Create from directory of data a data generator
    
    - Return a generator of data
'''


def data_generator(path, shuffle=True):
    generator = ImageDataGenerator(rescale=1. / 255)
    data = generator.flow_from_directory(path,
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='categorical',
                                         shuffle=shuffle)
    return data
