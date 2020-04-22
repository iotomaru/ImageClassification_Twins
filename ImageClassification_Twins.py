import tensorflow as tf
import pprint
from PIL import Image

def main():
  model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
  model.summary()
  
  img_pil = tf.keras.preprocessing.image.load_img(
    "data/test/doll.jpg", target_size=(224,224)
  )
  img_pil.show()

  img = tf.keras.applications.vgg16.preprocess_input(
    tf.keras.preprocessing.image.img_to_array(img_pil)[tf.newaxis]
  )
  predict = model.predict(img)

  result = tf.keras.applications.vgg16.decode_predictions(predict, top=5)
  pprint.pprint(result)
#

if __name__ == "__main__":
  main()
#