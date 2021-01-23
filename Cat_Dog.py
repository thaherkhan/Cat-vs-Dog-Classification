from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join




img_width = 150
img_height = 150

train_data_dir = '/content/drive/My Drive/python code/Cat_Dog/train'
validation_data_dir = '/content/drive/My Drive/python code/Cat_Dog/valid'
train_samples = len(train_data_dir)
validation_samples = len(validation_data_dir)
epochs = 10
batch_size = 20

input_shape = (img_width, img_height, 3)



model = Sequential()
# Conv2D : Two dimenstional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.summary()


import keras
from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=.0001), metrics=['accuracy'])


train_datagen = ImageDataGenerator( rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True )


test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory( train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')


print(train_generator.class_indices)

imgs, labels = next(train_generator)

from skimage import io

def imshow(image_RGB):
  io.imshow(image_RGB)
  io.show()
  
  
import matplotlib.pyplot as plt
%matplotlib inline
image_batch,label_batch = train_generator.next()

print(len(image_batch))
for i in range(0,len(image_batch)):
    image = image_batch[i]
    print(label_batch[i])
    imshow(image)
    
    
validation_generator = test_datagen.flow_from_directory( validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')


epochs = 20
history = model.fit_generator(train_generator, steps_per_epoch=train_samples // batch_size, epochs=epochs,validation_data=validation_generator,validation_steps=validation_samples // batch_size)


import matplotlib.pyplot as plt
%matplotlib inline

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



predict_dir_path='/content/drive/My Drive/python code/Cat_Dog/test1'
onlyfiles = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]
print(onlyfiles)



# predicting images
from keras.preprocessing import image

img = image.load_img('/content/drive/My Drive/python code/Cat_Dog/test1/85.jpg',target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

plt.imshow(img)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
classes = classes[0][0]
print(classes)
if classes == 0:
    print('cat')
else:
    print('dog')
