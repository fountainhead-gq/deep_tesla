
#!/usr/bin/env python
import sys
import os
import time
import subprocess as sp
import itertools
# CV
import cv2
# Model
import numpy as np
# Tools
import utils

# Parameters
import params  # you can modify the content of params.py

img_height = 66
img_width = 200
img_channels = 3


# Test epoch
epoch_ids = [10]
# Load model
model = utils.get_model()



def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    # Chop off 1/2 from the top and cut bottom 150px(which contains the head
    # of car)
    ratio = img_height / img_width
    h1, h2 = int(img.shape[0] / 2), img.shape[0] - 150
    w = (h2 - h1) / ratio
    padding = int(round((img.shape[1] - w) / 2))
    img = img[h1:h2, padding:-padding]
    # Resize the image
    img = cv2.resize(img, (img_width, img_height),
                     interpolation=cv2.INTER_AREA)
    # Image Normalization
    #img = img / 255.
    return img


# Process video
for epoch_id in epoch_ids:
    print('---------- processing video for epoch {} ----------'.format(epoch_id))
    vid_path = utils.join_dir(params.data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
    assert os.path.isfile(vid_path)
    
    # frame_count = utils.frame_count(vid_path)
    cap = cv2.VideoCapture(vid_path) 
    
    # New codes 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # import test data: epoch 10
    imgs_test, wheels_test = params.load_data('test')
    imgs_test = np.array(imgs_test)
    # imgs_test = imgs_test.astype('float32')
    # imgs_test /= 255

    
    machine_steering = []

    print('performing inference...')
    time_start = time.time()

    # for frame_id in range(frame_count):
    #    ret, img = cap.read()
    #    assert ret
    # ## you can modify here based on your model
    #    img = img_pre_process(img)
    #    img = img[None,:,:,:]
    #    deg = float(model.predict(img, batch_size=1))
    #    machine_steering.append(deg)

        
    cap.release()    
    
    # add below code
    machine_steering = model.predict(imgs_test, batch_size=128, verbose=0)

    
    fps = frame_count / (time.time() - time_start)

    print('completed inference, total frames: {}, average fps: {} Hz'.format(frame_count, round(fps, 1)))

    print('performing visualization...')
    utils.visualize(epoch_id, machine_steering, params.out_dir, verbose=True, frame_count_limit=None)
