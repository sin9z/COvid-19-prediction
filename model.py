import keras
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

base = 'C:\Aaishni Study Courses\IIT Bombay courses\D-E placement courses\Data Science\Covid 19 detection using CT scans Project\Project'
model = keras.models.load_model(f'{base}\CovidTest.h5')
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


def image_pre(path):
    print(path)
    data = np.ndarray(shape=(1,128,128,1),dtype=np.float32)
    size = (128,128)
    image = Image.open(path)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image,size,Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    data = image_array.reshape((-1,128,128,1))/255
    return data

def predict(data):
    prediction = model.predict(data)
    return np.round(prediction[0][0])