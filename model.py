import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow
from PIL import ImageOps, Image

# Define base directory
base = 'C:/Users/Hello/OneDrive/Desktop/project'

# Load and compile the model
model = tf.keras.models.load_model(f'{base}/CovidTest.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

def image_pre(path):
    """ Preprocess the image for model prediction. """
    print(path)
    size = (128, 128)
    image = Image.open(path)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    data = image_array.reshape((-1, 128, 128, 1)) / 255.0
    return data

def predict(data):
    """ Predict using the model and return rounded output. """
    prediction = model.predict(data)
    return np.round(prediction[0][0])

# Example usage (make sure to provide a valid image path)
# image_data = image_pre('path_to_image.jpg')
# result = predict(image_data)
# print(f'Prediction: {result}')
