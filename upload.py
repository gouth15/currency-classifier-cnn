import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment
import io
import base64

model = load_model("model\cnn_model2.h5")

def predict_currency(image):

    pil_image = Image.open(image)
    image_np = np.array(pil_image).astype(np.uint8)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_image = image_np[y:y + h, x:x + w]
    cropped_image = cv2.resize(cropped_image, (150, 150))
    cropped_image = np.array(cropped_image)
    cropped_image = np.expand_dims(cropped_image, axis=0)

    predictions = model.predict(cropped_image)
    predicted_class_index = np.argmax(predictions)

    class_labels = ["10", "100", "20", "200", "2000", "50", "500"]

    predicted_class = class_labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    return predicted_class, probability

st.title("Currency Recognition System")

uploaded_file = st.file_uploader("Choose a currency image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    predicted_class, probability = predict_currency(uploaded_file)

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Probability: {probability}")

    text = f"Currency Detected is {predicted_class} Rupees"
    tts = gTTS(text)
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)  

    audio_base64 = base64.b64encode(audio_data.getvalue()).decode('utf-8')

    html_code = f'''
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    '''

    st.markdown(html_code, unsafe_allow_html=True)