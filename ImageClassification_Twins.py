import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger

import pprint, os
import numpy as np
from PIL import Image

N_CATEGORIES=2
BATCH_SIZE=32
TRAIN_DIR="data/training"
VALIDATION_DIR='data/validation'
MODE_FILE_PREFIX='data/models/vgg16_fine_imageclassify_twins'

LEARNING_RATE=0.0001
MOMENTUM=0.9

def model_definition_and_training():
  # -----------------------------------------------------------------------------------
  # Model definition
  # -----------------------------------------------------------------------------------
  base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

  # 新たなFC層を追加
  top_model= Sequential()
  top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(N_CATEGORIES,activation='softmax'))

  model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

  # VGG16の14層までの重みを固定
  for layer in model.layers[:15]:
    layer.trainable=False
  #

  model.compile(optimizer=SGD(lr=LEARNING_RATE,momentum=MOMENTUM), loss='categorical_crossentropy', metrics=['accuracy'])

  # -----------------------------------------------------------------------------------
  # Training data preprocessing
  # -----------------------------------------------------------------------------------
  train_datagen=ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

  train_generator=train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

  validation_datagen=ImageDataGenerator(rescale=1.0/255)

  validation_generator=validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(224,224), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

  hist=model.fit_generator(train_generator, epochs=200, verbose=1,                         validation_data=validation_generator, callbacks=[CSVLogger(MODE_FILE_PREFIX+'.csv')])

  #save weights
  model.save(MODE_FILE_PREFIX+'.h5')

  return model
#

def main():
  # 既存のモデルが存在する場合はロードし、存在しない場合は新たに学習する。
  if os.path.exists(MODE_FILE_PREFIX+'.h5'):
    model = load_model(MODE_FILE_PREFIX+'.h5')
  else:
    model = model_definition_and_training()
  #
  model.summary()

  img_pil = tf.keras.preprocessing.image.load_img(
    "data/test/2020-05-04 11.06.20.jpg", target_size=(224,224)
  )
  img_pil.show()

  img = tf.keras.applications.vgg16.preprocess_input(
    tf.keras.preprocessing.image.img_to_array(img_pil)[tf.newaxis]
  )

  label = ['Fuuka', 'Honoka']
  predict = model.predict(img)
  score = np.max(predict)
  pred_label = label[np.argmax(predict[0])]
  print('name:',pred_label)
  print('score:',score)  
#

if __name__ == "__main__":
  main()
#