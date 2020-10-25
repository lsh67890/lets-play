import os
from glob import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

path = 'dataset/mnist_png/training/'
label_nums = os.listdir(path)
nums_dataset = []

for n in label_nums:
    data_per_class = os.listdir(path + n)
    nums_dataset.append(len(data_per_class))

# open with plt
plt.bar(label_nums, nums_dataset)
plt.title('Nr of dataset per class')
#plt.show()
plt.close()

# open with PIL
img_pil = Image.open(path + '/0/1.png')
img = np.array(img_pil)
#plt.imshow(img, 'gray')
plt.show()
plt.close()

# open with tensorflow
gfile = tf.io.read_file(path + '0/1.png')
img = tf.io.decode_image(gfile)
print(img.shape)
plt.imshow(img[:, :, 0], 'gray')
#plt.show()
plt.close()

# make a label
def make_label(path): #full path for the img
    class_name = path.split('/')[-2]
    label = int(class_name)
    return label

# get size of the data image
imgs = os.listdir(path+'0/')
data_path = []
for i in imgs:
    data_path.append(path+'0/'+i)
heights = []
widths = []
for path in data_path:
    img_pil = Image.open(path)
    img = np.array(img_pil)
    h, w = img.shape
    heights.append(h)
    widths.append(w)
# np.unique(heights) to check unique numbers only, which is 28 in this case
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.hist(heights)
plt.title('height')
plt.axvline(np.mean(heights), color='r', linestyle='dashed', linewidth=2) # always good to check by visualising data!
plt.subplot(122)
plt.hist(widths)
plt.title('width')
plt.axvline(np.mean(widths), color='r', linestyle='dashed', linewidth=2)
#plt.show()
plt.close()

# understanding data training
glob_paths = glob('dataset/cifar/train/*.png') # os.listdir only fetches the filename, not the filepath

def read_img(path):
    gfile = tf.io.read_file(path)
    img = tf.io.decode_image(gfile, dtype=tf.float32) # img.shape should be [32, 32, 3]
    return img

plt.imshow(read_img(glob_paths[0]))
#plt.show()
plt.close()

def make_batch(batch_paths):
    batch_img = []
    for path in batch_paths:
        img = read_img(path)
        batch_img.append(img)
    return tf.convert_to_tensor(batch_img) # batch should be 4-Dimensional: (batch_size, height, width, channel)

batch_imgs = make_batch(glob_paths[:8])
batch_size = 16
for step in range(4):
    batch_imgs = make_batch((glob_paths[step * batch_size: (step+1) * batch_size]))
    plt.imshow(batch_imgs[0]) #show first img of the batch
    #plt.show()
    plt.close()