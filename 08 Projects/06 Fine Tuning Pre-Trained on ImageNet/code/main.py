from seed import *
setup(seed_value=42)
from preprocessing import *
from model import *
from keras.optimizers import Adam


''' Directory '''
path_train = 'dataset/training_set/'
path_dev = 'dataset/dev_set/'
path_test = 'dataset/test_set/'
path_weights = 'weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


''' Hyper-parameter '''
batch_size = 64
epochs = 30
loss = 'categorical_crossentropy'
lr = 0.0005
decay = lr / epochs
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
optimizer = adam


''' Create Generator for train, dev, test '''
train_generator = data_generator(path_train, shuffle=True)

dev_generator = data_generator(path_dev, shuffle=True)

test_generator = data_generator(path_test, shuffle=False)


''' Print statistics about data'''
class_dictionary = train_generator.class_indices
print("Class-Index:", class_dictionary)


''' Create model based on vgg16 edited'''
model = vgg16_edit(path_weights)


''' Training model'''
training(model, loss, optimizer, train_generator, dev_generator, epochs)


''' Testing model'''
testing(model, test_generator, batch=100)
