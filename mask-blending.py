import cv2
import numpy as np

# Load images
img1 = cv2.imread('Image-Blending-With-Python/img/personday.jpeg')
img2 = cv2.imread('Image-Blending-With-Python/img/night.jpeg')

# Ensure images loaded
if img1 is None:
    raise ValueError("Image 1 not found!")
if img2 is None:
    raise ValueError("Image 2 not found!")

# Resize images to match img1's size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Create a simple horizontal gradient mask
mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.float32)

# Left half = 1, right half = 0 (before blur)
cv2.rectangle(mask, (0, 0), (int(img1.shape[1] * 0.5), img1.shape[0]), 1.0, -1)

# Blur for smooth transition
mask = cv2.GaussianBlur(mask, (51, 51), 0)

# Convert mask to 3 channels
mask_3d = cv2.merge([mask, mask, mask])

# Convert images to float32 for blending
img1_float = img1.astype(np.float32) / 255.0
img2_float = img2.astype(np.float32) / 255.0

# Blend the images using the mask
blended = (img1_float * mask_3d) + (img2_float * (1 - mask_3d))
blended = (blended * 255).astype(np.uint8)

# Display windows
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.imshow('Mask', (mask * 255).astype(np.uint8))
cv2.imshow('Blended Result', blended)

cv2.waitKey(0)
cv2.destroyAllWindows()
