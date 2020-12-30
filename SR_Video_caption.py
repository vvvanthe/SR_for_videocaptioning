import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
import glob
from utils import play_video,sample_scaled_frames,create_video
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

inpath="./Downscale14/"
outpath="./Upscale14/"

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)

for i, video in enumerate(sorted(glob.glob(inpath+"*.avi"))):
    video_id = video
    name = os.path.basename(video)
    frame_list, frame_count, fps, size = sample_scaled_frames(video, stride=5, scale_factor=1)
    filepath = outpath + name
    frame_list = frame_list.astype(np.float32)
    frames = []
    for frame in frame_list:
        frame_in = tf.expand_dims(frame, 0)
        frame_out = model(frame_in)
        frame_out = tf.clip_by_value(frame_out, 0, 255)
        frame_out = frame_out.numpy()
        frame_out = frame_out.astype(dtype=np.uint8)
        frame_out = tf.squeeze(frame_out)
        frames.append(frame_out)
    out_frame = np.array(frames)
    create_video(out_frame, filepath, fps, downscale=False)



play_video(outpath+ 'zzit5b_-ukg_5_20.avi')
play_video(inpath+ 'zzit5b_-ukg_5_20.avi')

