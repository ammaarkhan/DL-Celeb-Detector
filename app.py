import os
import cv2
import streamlit as st
from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity


detector = MTCNN()
vggModel = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded(uploaded_img):
    try:
        with open(os.path.join('uploads', uploaded_img.name), 'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    # cv2.imshow('output', face)
    # cv2.waitKey(0)

    # extract features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, features):
    similarity = []
    for x in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[x].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

    return index_pos


st.title('Which bollywood celebrity are you?')

# step 1: user uploads file
uploaded_img = st.file_uploader('Choose an image')

if uploaded_img is not None:
    # save img in directory
    if save_uploaded(uploaded_img):
        # load image
        display_image = Image.open(uploaded_img)

        # extract features
        features = extract_features(os.path.join('uploads', uploaded_img.name), vggModel, detector)
        # st.text(features)
        # st.text(features.shape)

        # find closest vector image and display
        index = recommend(feature_list, features)

        #display
        predicted_actor = " ".join(filenames[index].split('/')[1].split('_'))
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header('Seems like ' + predicted_actor)
            st.image(filenames[index], width=250)
