import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# Load the image in grayscale mode
image_path = '/Users/ayusapurboo/Downloads/FYP Final/xray_dataset/test/3/T3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        # Geometric Transformations
        iaa.Fliplr(0.5),  # Flip horizontally 50% of the time
        iaa.Flipud(0.2),  # Flip vertically 20% of the time
        iaa.Affine(rotate=(-25, 25)),  # Rotate images by -25 to +25 degrees
        iaa.Affine(scale=(0.8, 1.2)),  # Scale images to 80-120% of original size

        # Intensity Transformations
        iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
        iaa.ContrastNormalization((0.8, 1.2)),  # Contrast adjustment
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add Gaussian noise

        # Elastic Deformations
        iaa.ElasticTransformation(alpha=50, sigma=5),  # Elastic deformation
    ])

    # Create 10 augmented images from the CLAHE enhanced image
    augmented_images = seq(images=[image_clahe] * 10)  

    # Create red box outlines on each augmented image
    height, width = image_clahe.shape
    num_boxes = 5  # Number of boxes to draw on each augmented image
    box_size = 20  # Size of the box

    for i in range(len(augmented_images)):
        for _ in range(num_boxes):
            # Random coordinates for the boxes
            x = np.random.randint(0, width - box_size)
            y = np.random.randint(0, height - box_size)

            # Draw a red box outline (BGR format)
            cv2.rectangle(augmented_images[i], (x, y), (x + box_size, y + box_size), (0, 0, 255), 2)  # Red color in BGR

    # Display original, CLAHE enhanced, and augmented images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(image_clahe, cmap='gray')
    plt.title('CLAHE Enhanced Image')
    plt.axis('off')

    for i, aug_image in enumerate(augmented_images):
        plt.subplot(3, 4, i + 3)
        plt.imshow(aug_image, cmap='gray')
        plt.title(f'Augmented Image {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()