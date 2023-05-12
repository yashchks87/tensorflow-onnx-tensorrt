import tensorflow as tf
import numpy as np
import multiprocessing as mp
import cv2

def read_imgs(data):
    img, shape = data[0], data[1]
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize = (shape[0], shape[1]), interpolation = cv2.INTER_LINEAR)
    return img

def img_batches(dataset, total_size, image_shape = (512, 512, 3)):
    imgs = np.zeros((total_size, image_shape[0], image_shape[1], image_shape[2]))
    count = 0
    for img_path in dataset:
        i = read_imgs((img_path, image_shape))
        imgs[count] = i
        count += 1
    return imgs

def img_batches_mp(dataset, image_shape = (512, 512, 3)):
    final_data = [(dataset[x], image_shape) for x in range(len(dataset))]
    with mp.Pool(5) as p:
        returns = np.array(p.map(read_imgs, final_data))
    return returns


