from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


from keras import backend as K
import pandas as pd
import tensorflow

nb_train_samples = 400
nb_validation_samples = 100
epochs = 18
batch_size = 16


from warnings import filterwarnings
filterwarnings('ignore')
img_width, img_height = 720, 480
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
train_set = train_datagen.flow_from_directory('output/train',
                                             target_size=(720,480),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set
test_set = test_datagen.flow_from_directory('output/val',
                                           target_size=(720,480),
                                           batch_size = 32,
                                           class_mode='binary',
                                           shuffle=False)
#Test Set /no output available
test_set1 = test_datagen.flow_from_directory('DataSet/Test Images',
                                            target_size=(720,480),
                                            batch_size=32,
                                            shuffle=False)#
model.fit_generator(train_set,steps_per_epoch=nb_train_samples // batch_size,epochs=epochs, validation_data=test_set,validation_steps=nb_validation_samples // batch_size)
#Some Helpful Instructions:

#finetune you network parameter in last by using low learning rate like 0.00001
model.save('hyperparameter1.h5')



#from tensorflow.keras.models import load_model
#model = load_model('LETSDOTHIS.h5')

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
results = []
test = pd.read_csv('test.csv')
length_test = len(test['Image_File'])
testDF = pd.read_csv('test.csv')

for i in range(length_test):
    img1 = image.load_img('DataSet/Test Images/'+ test['Image_File'][i], target_size=(720, 480))
    img = image.img_to_array(img1)
    img = img / 255
    # create a batch of size 1 [N,H,W,C]
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, batch_size=None, steps=1)  # gives all class prob.
    if (prediction[:, :] > 0.5):
        testDF['Class'][i] = "Small"
    else:
        testDF['Class'][i] = "Large"
    print(i)
print(testDF)
testDF.to_csv('hyperparameterTWEAK.csv')


