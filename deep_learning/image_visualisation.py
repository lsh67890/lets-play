# Ways to visualise images

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = './tvxq.jpg'

# opening the image in an array
img_pil = Image.open(path)
img = np.array(img_pil)
print(img.shape) # 3 represents RGB
print(np.min(img), np.max(img)) # checking the range

# visualise the image with a histogram
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
plt.close()

# showing the image
plt.imshow(img)
plt.show()
plt.close()

# convert to black-white
img_pil = Image.open(path).convert('L')
img_bw = np.array(img_pil)
print(img_bw.shape) # 3 has disappeared, meaning it's grayscale
plt.imshow(img_bw, 'gray')
plt.show()
plt.close()

# showing in red-blue or jet
plt.imshow(img_bw, 'RdBu')
plt.show()
plt.close()
plt.imshow(img_bw, 'jet')
plt.show()
plt.close()

# adding Colorbar
plt.imshow(img_bw, 'jet')
plt.colorbar()
plt.show()
plt.close()

# resizing the image
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
plt.close()

# opening the second image
path = './tvxq1.jpg'
img2_pil = Image.open(path)
img2 = np.array(img2_pil)
plt.imshow(img2)
plt.show()
plt.close()

# adding the second image with opencv
import cv2
img1 = cv2.resize(img, (300, 200))
img2 = cv2.resize(img2, (300, 200))
plt.imshow(img1)
plt.imshow(img2, alpha=0.5) # alpha represents opacity
plt.show()
plt.close()

# subplot with title
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('TVXQ MIROTIC')
plt.imshow(img1)
plt.subplot(222)
plt.title('TVXQ HUG')
plt.imshow(img2)
plt.show()
plt.close()
