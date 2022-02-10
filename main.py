from PIL import Image
import numpy as np
import streamlit as st
from keras.models import load_model
import tensorflow as tf
import cv2
import os

# hyper-parameters
ROOT_FOLDER = ""
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train_set")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test_set")
PROCESSED_DATA = os.path.join(ROOT_FOLDER, "processed_data")
TEMPLATE_FOLDER = os.path.join(PROCESSED_DATA, "templates")


def match_template(
    input_path,
    template_path,
    category,
    path=True,
    method="pixel_count",
    debug_mode=False,
):
    """
    function that matches the image to a template.
    :input_path: str path to image
    :template_path: str path to template
    :method: str identifier of what method to use for matching
    :debug_mode: decides whether to return additional debugging data
    :return: an error variable dependent on the chosen method and debugging info or None depending on mode.
    """

    # sometimes the category is given as as

    category_converter = {
        "00": 1,
        "01": 2,
        "02": 2,
        "03": 3,
        "04": 3,
        "05": 4,
        "06": 5,
        "07": 6,
        "08": 6,
        "09": 6,
        "10": 6,
    }
    if category in list(category_converter.keys()):
        category = category_converter[category]

        stored_parameters = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 25}

    else:
        category = int(category)

        stored_parameters = {1: 35, 2: 25, 3: 50, 4: 75, 5: 75, 6: 100}

    if path:
        # load in image and template
        sample_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    else:
        sample_image = input_path

    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if method == "pixel_count":
        # important for this method is that the oprder of subtracting does matter.
        # because a 2 will have holes in the same spots as a 5 and will register a false positive

        thresh = stored_parameters[category]

        diff = template_image - sample_image
        errors = (diff > thresh).sum()

        if debug_mode:
            return errors, sample_image, template_image

        return errors, None, None

    elif method == "MSE":
        # iThe next method uses MSE to calculate the distances between to images

        mse = np.square(np.subtract(template_image, sample_image)).mean()

        if debug_mode:
            return mse, sample_image, template_image

        return mse, None, None


def numpy_model(image, path=True, thresholds=None):
    """
    predicts whether a dice is an anomaly or not
    :image: str path to image file
    :thresholds: dict containg custom thresholds foe each category, load it in from thresholds.pickle
    :return: return a predictions overall, and a prediction list as to what class it may belong
    """

    all_templates = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"]

    predicted = 0
    predictions = []

    if not thresholds:
        thresholds = {
            1: 62.02780473883087,
            2: 60.430025740950214,
            3: 66.30725609998866,
            4: 69.93280870737878,
            5: 74.30941474250847,
            6: 82.75002536741725,
        }

    for idx, template in enumerate(all_templates):
        # this should link to a folder containg the templates.
        template_path = os.path.join(TEMPLATE_FOLDER, template)

        errors, _, _ = match_template(
            image, template_path, path=path, category=1, method="MSE"
        )
        thresh = thresholds[idx + 1]

        if errors > thresh:
            predictions.append(1)
        else:
            predictions.append(0)

    if sum(predictions) >= 6:
        predicted = 1

    return predicted, predictions


def SSIMLoss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ssimloss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return ssimloss


# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image.reshape(1, 128, 128, 1)
    image = image / 255

    return image[0]


def load_image_opencv(pillow_img):
    open_cv_image = np.array(pillow_img)
    return open_cv_image


# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=["jpg", "png"])
col1, col2 = st.columns(2)

# Checking the Format of the page
if uploadFile is not None:
    #   st.write(type(uploadFile))

    # Perform your Manupilations (In my Case applying Filters)
    img_input = load_image(uploadFile)
    img_opencv = load_image_opencv(uploadFile)
    col1.header("Input Image")
    col1.image(img_input)
    col1.write(img_input.shape)
    cnn_model = load_model("models/preprocessed_data_cnn_model.h5")
    autoencoder = load_model("models/v0-1.h5")
    pred = autoencoder.predict(img_input)

    col2.header("reconstructed image")
    pred = np.array(pred).reshape(128, 128, 1)
    prediction = cnn_model.predict(np.array(img_input).reshape(1, 128, 128, 1))
    col2.image(pred)
    col2.write(pred.shape)
    COL1, COL2, COL3, COL4 = st.columns(4)
    loss = SSIMLoss(img_input, pred)
    label = "SSIM Loss value: {:.3f}"
    label_classes = "Predicted Class: {}"
    classes = np.argmax(prediction, axis=1)

    COL2.write(label.format(loss))
    COL2.write(label_classes.format(classes[0] + 1))

    CNN_COL1, CNN_COL2 = st.columns(2)
    img = Image.open(f"processed_data/templates/{classes[0]+1}.png")
    CNN_COL1.header("input image")
    CNN_COL1.image(img_input)
    CNN_COL2.header("predicted_image")
    CNN_COL2.image(img)
    img = np.array(img).reshape(128, 128, 1)
    img = img / 255

    loss2 = SSIMLoss(img_input, img)
    label = "SSIM Loss value: {:.3f}"
    label_classes = "Predicted Class: {}"
    COL1, COL2, COL3, COL4 = st.columns(4)
    COL2.write(label.format(loss2))
    label = "Ratio loss values: {:.2f}"

    COL2.write(label.format(loss / loss2))
    result = ""
    if loss / loss2 < 0.70:
        result = "anomaly"
    else:
        result = "normal"

    COL2.write(result)

    np_prediction = numpy_model(img_opencv, path=False)
    st.write("predicting with numpy model")

    st.write(np_prediction[0])  # returns 0 for normal and 1 for anomaly


else:
    st.write("Make sure you image is in JPG/PNG Format.")
