import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import pyttsx3

try:
    model = load_model("cnn_model2.h5")
except Exception as e:
    print(f"Error loading the model: {str(e)}")

class_labels = ['10', '100', '20', '200', '2000', '50', '500']

def predict_currency(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    resized_image = pil_image.resize((150, 150))
    resized_image_array = np.array(resized_image)
    input_image = np.expand_dims(resized_image_array, axis=0)

    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    predicted_class = class_labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]

    return predicted_class, probability

def display_with_filled_background(image, denomination):
    text = f'Denomination: {denomination} Rs'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    background_coordinates = ((10, 30), (10 + text_size[0] + 2, 30 - text_size[1] - 5))

    cv2.rectangle(image, background_coordinates[0], background_coordinates[1], (255, 0, 0), -1)
    cv2.putText(image, text, (10, 30 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow('Currency Prediction', image)

cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

while True:
    ret, frame = cap.read()

    # Perform prediction on the captured frame
    predicted_class, probability = predict_currency(frame)

    confidence_threshold = 0.5

    if probability >= confidence_threshold:
        # Display denomination with blue filled background on the camera frame
        display_with_filled_background(frame, predicted_class)

        # Speak the detected currency
        engine.say(f"Detected denomination: {predicted_class} Rupees")
        engine.runAndWait()
    else:
        cv2.putText(frame, 'Low confidence. Please try again.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Currency Prediction', frame)

    # Display the frame and check for the 'q' key to quit
    cv2.imshow('Currency Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
