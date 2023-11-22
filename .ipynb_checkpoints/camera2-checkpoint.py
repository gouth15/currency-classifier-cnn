import streamlit as st
import cv2
import numpy as np

# Load the cascade classifier for currency detection
currency_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect currency boundaries in the frame
def detect_currency(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    currencies = currency_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in currencies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return image

# Streamlit app
st.title("Currency Boundary Detection")

# Accessing the camera feed
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    st.error('Unable to load camera.')

# Continuously read frames from the camera
while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        st.error('Failed to capture the frame.')
        break

    # Detect currency boundaries
    frame_with_currency = detect_currency(frame)

    # Display the frames
    st.image(frame_with_currency, channels="BGR", use_column_width=True)

    # Stop the loop if the 'Stop' button is clicked
    if st.button('Stop'):
        break

# Release the camera and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()