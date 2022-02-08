from PIL import Image
import numpy as np 
import streamlit as st
from keras.models import load_model
import tensorflow as tf


def SSIMLoss(y_true, y_pred):
  y_true = tf.cast(y_true,tf.float32)
  y_pred = tf.cast(y_pred,tf.float32)
  ssimloss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))
  return ssimloss

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image.reshape(1,128,128,1)
    image = image/255

    return image[0]

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
col1, col2 = st.columns(2)

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    col1.header("Input Image")
    col1.image(img)
    col1.write(img.shape)
    autoencoder = load_model('models/v0-1.h5')
    pred = autoencoder.predict(img)
    col2.header("reconstructed image")
    pred = np.array(pred).reshape(128,128,1)
    col2.image(pred)
    col2.write(img.shape)
    COL1, COL2,COL3,COL4 = st.columns(4)
    loss = SSIMLoss(img,pred)
    label = 'SSIM Loss value: {:.3f}'

    COL2.write(label.format(loss))


else:
    st.write("Make sure you image is in JPG/PNG Format.")


 

