
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
import time

# CONST
# Initialize Constant
flags = tf.app.flags
FLAGS = flags.FLAGS

# Nvida's camera format
flags.DEFINE_integer('img_h', 66, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

# Fix random seed for reproducibility
np.random.seed(42)

# Path
data_dir = os.path.abspath('./epochs')
out_dir = os.path.abspath('./output')
model_dir = os.path.abspath('./models')


def preprocess(img, color_mode='RGB'):
    # Chop off 1/2 from the top and cut bottom 150px(which contains the head of car)
    img_height = 66
    img_width = 200
    ratio = img_height / img_width
    h1, h2 = int(img.shape[0] / 2), img.shape[0] - 150
    w = (h2 - h1) / ratio
    padding = int(round((img.shape[1] - w) / 2))
    img = img[h1:h2, padding:-padding]
    # Resize the image
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    if color_mode == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Image Normalization
    #img = img / 255.
    return img


def load_data(mode, color_mode='RGB', flip=True):

    data_dir = os.path.abspath('./epochs')
    if mode == 'train':
        epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif mode == 'test':
        epochs = [10]
    else:
        print('choose mode')

    imgs = []
    wheels = []
    # extract image and steering data
    for epoch_id in epochs:
        y_epoch = []
        vid_path = os.path.join(data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
        # cap_video = cv2.VideoCapture(vid_path)
        # frame_count = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        cap = cv2.VideoCapture(vid_path)
        csv_path = os.path.join(data_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
        rows = pd.read_csv(csv_path)
        y_epoch = rows['wheel'].values
        wheels.extend(y_epoch)

        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = preprocess(img, color_mode)
            imgs.append(img)

        cap.release()
        # assert len(imgs) == len(wheels)
        
    if mode == 'train' and flip:
        augmented_imgs = []
        augmented_measurements = []
        for image, measurement in zip(imgs, wheels):
            augmented_imgs.append(image)
            augmented_measurements.append(measurement)
            # 水平翻转
            flipped_image = cv2.flip(image, 1)
            flipped_measurement = float(measurement) * -1.0
            augmented_imgs.append(flipped_image)
            augmented_measurements.append(flipped_measurement)

        X_train = np.array(augmented_imgs)
        y_train = np.array(augmented_measurements)
        y_train = np.reshape(y_train,(len(y_train),1))

    else:
        X_train = np.array(imgs)
        y_train = np.array(wheels)
        y_train = np.reshape(y_train,(len(y_train),1))

    return X_train, y_train



def loss_histroy(model, epochs):
    plt.plot(model.history['loss'], label="loss")
    plt.plot(model.history['val_loss'], label="val_loss")
    
    # plt.title('model loss')
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epochs', fontsize=12)
    plt.legend(loc='best', shadow=True)
    # plt.legend(labels=['loss', 'val_loss'], loc='best')
    plt.xlim((0,epochs))
    plt.xticks(np.arange(0, epochs+1, 2))
    plt.grid()
    plt.show()
