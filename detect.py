import cv2
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to plot an image
def plot_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
img = cv2.imread('predict.jpg')
plot_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original Image')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plot_image(gray, 'Grayscale Image')

# Apply thresholding to the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plot_image(thresh, 'Thresholded Image')

# Apply dilation to the image
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations=1)
plot_image(dilation, 'Dilated Image')

# Find contours in the image
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find the text
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img[y:y + h, x:x + w]
    plot_image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), 'ROI')
    text = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
    if text:
        print("Denomination: ", text)
