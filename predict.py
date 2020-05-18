

import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
import keras
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from keras.preprocessing import image

epochs = 10
num_classes = 10
rows = 28
cols = 28
colour_channels = 1

# load mnist data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Training requires data set to a 4 dimensional shape
# MNIST by default is (60000, 28, 28)
# Reshaping by adding an extra dimension with the value 1
# This extra dimension indicates the number of colour channels as 1
# 1 colour channel = grayscale 
train_images = train_images.reshape(train_images.shape[0], rows, cols, 1)
test_images = test_images.reshape(test_images.shape[0], rows, cols, 1)

# width x height x num colour channels
# 28 x 28 x 1
input_shape = (rows, cols, colour_channels)

# Normalise images between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Conv network
def network():
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))

  return model



def compile_net(net): 
  net.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return net

def fit_network(compiled_net):

  network_history = compiled_net.fit(train_images, train_labels, epochs=epochs, 
                      validation_data=(test_images, test_labels))
  
  return network_history

# load an existing model
def load_model(model_name):
  return tf.keras.models.load_model(model_name)


def make_prediction(trained_model, img_to_predict):

  probability_model = tf.keras.Sequential([trained_model, 
                                          tf.keras.layers.Softmax()])

  predictions = probability_model.predict(img_to_predict)

  pred = np.argmax(predictions)
  print("prediction: ", pred)
  print(predictions)
  return pred

# plot drawn image in matplotlib
def plot_image(file_name):
  img = Image.open(file_name)
  img_show = array_to_img(img)
  plt.imshow(img_show)
  plt.show()


# load the drawn image
# convert to a numpy object with 4 dimensions
# convert from an RGB image to a grayscale image
# As only a single image is being gussed the shape is:
# (1, 28, 28, 1)
# The initial '1' indicates that there is only a single image
# row, col, colour_channels is the individual shape othe image
def load_and_convert_image(file_name):
  img = Image.open(file_name)

  # plot_image(file_name)

  img = np.asarray(img)
  img = tf.image.rgb_to_grayscale(img)

  img = np.asarray(img)
  img = img.reshape(1, rows, cols, colour_channels)
  
  return img

def model_exists(model_name):
  try:
    model = load_model(model_name)
    return True
  except:
    return False

def train():
  net = network()
  compiled_net = compile_net(net)
  trained_network = fit_network(compiled_net)

  compiled_net.save('training_model')
  return compiled_net


def run_predict():
  """
  if a model already exists, use that model to predict the number
  if not, train a new model then use this model to predict the number
  return the prediction
  """
  img = load_and_convert_image('drawing.jpeg')

  pre_trained_model = 'training_model'
  if (model_exists(pre_trained_model)):
    trained_model = load_model(pre_trained_model)
    guess = make_prediction(trained_model, img)
  
  else:
    trained_model = train()
    guess = make_prediction(trained_model, img)
    
  return guess

def main():
  run_predict()
  
if __name__ == "__main__":
    main()