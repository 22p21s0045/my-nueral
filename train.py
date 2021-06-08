from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Activation, Flatten, Dense

train_datagen = ImageDataGenerator(

rescale=1. /255,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True)

train_generator =train_datagen.flow_from_directory("plane_train",

target_size=(128,128),

batch_size=30,
color_mode='grayscale',

class_mode='categorical')

test_generator =train_datagen.flow_from_directory("plane_test",
color_mode='grayscale',

target_size=(128,128),

batch_size=30,

class_mode='categorical')
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(128,128,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(4, activation='sigmoid'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',

optimizer='adam',

metrics=['accuracy'])
model.fit(

train_generator,

epochs=5000,

steps_per_epoch=100/30,

validation_data=train_generator,

validation_steps=100/30

)

model.save('train_plane.h5')
print(train_generator.class_indices)
