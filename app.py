import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model_path = 'resnet_model.h5'
emotion_model = load_model(model_path)

def preprocess_image(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

def predict_emotion(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    prediction = emotion_model.predict(preprocessed_image)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    return predicted_emotion

def main():
    # Streamlit UI
    st.set_page_config(page_title="Facial Emotion Recognition")
    st.markdown("<h1 style='text-align: center;'>Facial Emotion Recognition App</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        # image = cv2.imread(uploaded_file.name)
        image_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction when button is clicked
        if st.button("Predict Emotion"):
            predicted_emotion = predict_emotion(image)
            st.markdown(f"<h3>Predicted Emotion</h3>", unsafe_allow_html=True)
            st.info(f"{predicted_emotion}")

if __name__ == "__main__":
    main()
