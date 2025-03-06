import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
f = cv2.imread('/Users/ayusapurboo/Downloads/MD AYUS_20028296/MD AYUS_20028296 (SF)/xray_dataset/test/6/T6.png', 0)

# Check if the image was loaded successfully
if f is not None:
    # Perform CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    f_clahe = clahe.apply(f)

    # Display the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(f_clahe, cmap='gray')
    plt.title('CLAHE Image')
    plt.axis('off')

    plt.show()
else:
    print("Error: Image not found or could not be loaded.")