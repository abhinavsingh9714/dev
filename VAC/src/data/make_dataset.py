# -*- coding: utf-8 -*-
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
from PIL import Image
#from keras import optimizers
import numpy as np
import os, glob
import cv2


def mse(imageA, imageB):
    # Compute the mean squared error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def video_capturing_function(data_dir, folder_new):
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        print("folder_path:",folder_path)
        if os.path.isdir(folder_path):
            for video_name in os.listdir(folder_path):
                if video_name.endswith(".avi"):
                    video_read_path = os.path.join(folder_path, video_name)
                    cap = cv2.VideoCapture(video_read_path)

                    frame_dir_path = os.path.join(folder_new, folder_name)
                    print("frame_dir_path:",frame_dir_path)
                    if not os.path.exists(frame_dir_path):
                        os.makedirs(frame_dir_path)

                    train_write_file = os.path.join(frame_dir_path, video_name.split(".")[0])
                    if not os.path.exists(train_write_file):
                        os.makedirs(train_write_file)

                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    skip_frames = 0
                    capture_frames = 10
                    frameRate = max(1, (frame_count - skip_frames) // capture_frames)

                    count = 0
                    last_frame = None
                    while cap.isOpened() and count < capture_frames:
                        frame_index = skip_frames + frameRate * count
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        if last_frame is not None:
                            if mse(frame_grey, last_frame) < 1.0:  # Threshold for frame similarity
                                continue  # Skip similar frame

                        last_frame = frame_grey.copy()

                        filename = "frame%d.jpg" % count
                        cv2.imwrite(os.path.join(train_write_file, filename), frame_grey)
                        count += 1

                    cap.release()

    return print("Done - All frames written in the Folder")

no_frames = 10

def data_load_function_10frames(directory):
    frames = []
    actionlabels = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        print(f"now at:", folder_path) # check if iterating across all folders?
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):

                    # Initialize a list for video data for each subfolder

                    vid_data = []
                    for l in range(no_frames):
                        frame_file = 'frame%d.jpg' % l
                        frame_path = os.path.join(subfolder_path, frame_file)
                        if os.path.exists(frame_path):
                            image = Image.open(frame_path)
                            image = image.resize((80, 100), Image.ANTIALIAS)
                            datu = np.asarray(image)
                            normu_dat = datu / 255
                            vid_data.append(normu_dat)

                    # Append the video data to the frames list if it contains 10 frames

                    if len(vid_data) == no_frames:
                        frames.append(np.array(vid_data))
                        actionlabels.append(folder_name)
    return np.array(frames), np.array(actionlabels)
