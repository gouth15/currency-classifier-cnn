import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pyttsx3

try:
    model = load_model("model\cnn_model2.h5")
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

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

captured_frame = None
prediction_made = False

while True:
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1)
    
    if key == ord(' '):  
        captured_frame = frame.copy()  
        prediction_made = False  
        print("Frame captured. Press 'q' to quit or 'c' to make a prediction.")
    
    if key == ord('c') and captured_frame is not None and not prediction_made:
        predicted_class, probability = predict_currency(captured_frame)
        print(f"Predicted denomination: {predicted_class} with probability: {probability}")
        engine.say(f"Currency detected {predicted_class} rupees.")
        engine.runAndWait()
        prediction_made = True

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
