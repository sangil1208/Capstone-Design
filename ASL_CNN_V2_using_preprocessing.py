from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
import cv2
import numpy as np

groups_folder_path = 'asl_alphabet_train/asl_alphabet_train/'
categories = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I",
              "J", "K", "L", "M", "N", "nothing", "O","P", "Q", "R",
              "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]
 
num_classes = len(categories)

X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle=True)
print(len(Y_train))
print(len(Y_test))

model = Sequential()
model.add(Convolution2D(16, (3, 3), padding ='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
  
model.add(Convolution2D(64, (3, 3),  activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))

model.summary()
  
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30)
 
model.save('3rd.h5')


