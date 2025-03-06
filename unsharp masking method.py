import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
f = cv2.imread('/Users/ayusapurboo/Downloads/MD AYUS_20028296/MD AYUS_20028296 (SF)/xray_dataset/test/3/T3.png', 0)

# Check if the image was loaded successfully
if f is not None:
    # Perform the inversion
    f = f / 255
    plt.imshow(f, cmap='gray')
    plt.axis('off')
    plt.show()
else:
    print("Error: Image not found or could not be loaded.")

# blur image
f_blur = cv2.GaussianBlur(src=f, 
                          ksize=(31,31), 
                          sigmaX=0, 
                          sigmaY=0)

plt.imshow(f_blur, cmap='gray'); plt.axis('off'); plt.show()

# mask
g_mask = f - f_blur
plt.imshow(g_mask, cmap='gray'); plt.axis('off'); plt.show()

# unsharp masking
k = 5
g = f + k*g_mask
plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()

g = np.clip(g, 0, 1)
plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()