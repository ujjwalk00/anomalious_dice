from PIL import Image
import numpy as np 
import streamlit as st
from keras.models import load_model


# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image




# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
col1, col2 = st.columns(2)

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    col1.header("Input Image")
    col1.image(img)

    autoencoder = load_model("../assets/autoencoder_pkl.h5")

    col2.header("reconstructed image")
    col2.image(img)
else:
    st.write("Make sure you image is in JPG/PNG Format.")


 

