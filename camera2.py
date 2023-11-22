import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import pyttsx3
import io
import base64

try:
    model = load_model("model\cnn_model2.h5")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

capture_image = st.camera_input(label="Capture Currency")

if capture_image is not None:
    st.image(capture_image, caption="Captured Image", use_column_width=True)

    bytes_data = capture_image.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    cvImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pil_image = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
    resized_image = pil_image.resize((150, 150))
    st.image(resized_image, caption="Resized Image", use_column_width=True)
    resized_image_array = np.array(resized_image)
    input_image = np.expand_dims(resized_image_array, axis=0)

    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    class_labels = ['10', '100', '20', '200', '2000', '50', '500']

    predicted_class = class_labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    confidence_threshold = 0.5

    if probability >= confidence_threshold:
        st.write(f'Predicted Class: {predicted_class}')
        st.write(f'Probability: {probability}')
    else:
        st.warning("Model confidence is low. Please try capturing the image again.")

    text = f'{predicted_class} Rupees'

    engine = pyttsx3.init()
    engine.say(text)

    audio_data = io.BytesIO()
    engine.save_to_file(text, 'temp_audio.mp3')
    engine.runAndWait()
    
    with open('temp_audio.mp3', 'rb') as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    html_code = f'''
    <audio controls autoplay>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    '''

    st.markdown(html_code, unsafe_allow_html=True)
