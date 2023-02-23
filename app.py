import streamlit as st
from PIL import Image

def save_uploaded(uploaded_img):

    return True

def extract_features(img_path, model):
    return True

def recommend(feature):
    return True

st.title('Which bollywood celebrity are you?')

# step 1: user uploads file
uploaded_img = st.file_uploader('Choose an image')

if uploaded_img is not None:
    # save img in directory
    if save_uploaded(uploaded_img):
        # load image
        display_image = Image.open(uploaded_img)
        st.image(display_image)
        # extract features
        features = extract_features()
        # find closest vector image and display
        index = recommend(features)
