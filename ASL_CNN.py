import tensorflow as tf
import pathlib
import splitfolders
from tensorflow.keras import datasets,layers, models
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

batch_size = 150

# 트레이닝 데이터 증식 코드
# train_generator = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

train_generator = ImageDataGenerator(rescale=1./255)
val_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1./255)

# 데이터셋 트레이닝, 검증, 테스트 8:1:1로 나누는 코드
#splitfolders.ratio('E:/Study/2022-1학기/캡스톤디자인/hand_recog/asl_alphabet_train/asl_alphabet_train', output = 'E:/Study/2022-1학기/캡스톤디자인/hand_recog', seed=77, ratio=(0.8, 0.1, 0.1))

train_gen = train_generator.flow_from_directory(
    'E:/Study/2022-1학기/캡스톤디자인/hand_recog/asl_train',
    target_size=(200, 200),
    batch_size = batch_size,
    class_mode = 'categorical')

val_gen = val_generator.flow_from_directory(
    'E:/Study/2022-1학기/캡스톤디자인/hand_recog/asl_test',
    target_size=(200, 200),
    batch_size = batch_size,
    class_mode = 'categorical')

print(len(val_gen))
test_gen = test_generator.flow_from_directory(
    'E:/Study/2022-1학기/캡스톤디자인/hand_recog/asl_val',
    target_size=(200, 200),
    batch_size = batch_size,
    class_mode = 'categorical')


# CNN 모델 생성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(29, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 학습 실행
model.fit_generator(
    train_gen,
    steps_per_epoch=16,
    epochs=50,
    validation_data=val_gen,
    validation_steps=10
)

model.save_weights('first_try.h5')
