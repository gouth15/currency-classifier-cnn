# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2
# from PIL import Image
# from gtts import gTTS
# import os
# import pygame

# model = load_model("new_cnn_model.h5")

# def predict_currency(image_np):

#     # Resize the image to the expected input size of the model
#     image_np = cv2.resize(image_np, (150, 150))
    
#     image_np = np.array(image_np)
#     image_np = np.expand_dims(image_np, axis=0)

#     predictions = model.predict(image_np)
#     predicted_class_index = np.argmax(predictions)

#     class_labels = ["10", "100", "20", "200", "2000", "50", "500"]

#     predicted_class = class_labels[predicted_class_index]
#     probability = predictions[0][predicted_class_index]

#     return predicted_class, probability

# def capture_and_recognize():
#     cap = cv2.VideoCapture(0)
#     cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

#     pygame.mixer.init()  # Initialize the pygame mixer
#     pygame.mixer.music.set_volume(1.0)

#     while True:
#         ret, frame = cap.read()
#         cv2.imshow("Live Feed", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord(' '):  # Press spacebar to capture frame
#             image_path = "captured_frame.jpg"
#             cv2.imwrite(image_path, frame)
#             print("Image Captured.")

#             predicted_class, probability = predict_currency(frame)
#             print(f"Predicted Class: {predicted_class}")
#             print(f"Probability: {probability}")

#             text = f"Currency Detected is {predicted_class} Rupees"
#             tts = gTTS(text, lang='en')

#             audio_path = "output_audio.mp3"
#             tts.save(audio_path)

#             pygame.mixer.music.load(audio_path)
#             pygame.mixer.music.play()

#         elif key == 27:  # Press 'Esc' to exit
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     pygame.mixer.quit()

# if __name__ == "__main__":
#     capture_and_recognize()


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import streamlit as st 

model = load_model("new_cnn_model.h5")

capture_image = st.camera_input(label="Capture Currency")

if capture_image is not None:
    bytes_data = capture_image.read()

    nparr = np.frombuffer(bytes_data, np.uint8)
    cvImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    st.write(cvImage.shape)

    # Convert captured image to PIL Image
    pil_image = Image.fromarray(cvImage)

    # Resize the image
    resized_image = pil_image.resize((150, 150))

    # Convert back to numpy array
    resized_image_array = np.array(resized_image)

    # Expand dimensions to match the model input shape
    input_image = np.expand_dims(resized_image_array, axis=0)

    # Make predictions
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    class_labels = ['10', '100', '20', '200', '2000', '50', '500']

    predicted_class = class_labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    st.write(f'Predicted Class: {predicted_class}')
    st.write(f'Probability: {probability}')

