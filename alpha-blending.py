import cv2
import os
print(os.getcwd())
#import two images
img1 = cv2.imread(r'C:\Users\User\OneDrive\Desktop\web practical\Image-Blending-With-Python\img\personday.jpeg')
img2 = cv2.imread(r'C:\Users\User\OneDrive\Desktop\web practical\Image-Blending-With-Python\img\night.jpeg')


#make sure the have the same size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# chose the apha value
alpha = 0.3
beta = 1 - alpha

#perform the blending
blended = cv2.addWeighted(img1, alpha, img2, beta, 0)

#now display the result
cv2.imshow("Blended Image", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()