import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
from utils import play_video,sample_scaled_frames,create_video


os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"



inpath="./Downscale12/"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)

frame_list, frame_count, fps, size=sample_scaled_frames(inpath+'eyhzdC936uk_15_27.avi',stride=4,scale_factor=1)
frame_list=frame_list.astype(np.float32)

out_frame1=model(frame_list)
out_frame1=tf.clip_by_value(out_frame1, 0, 255)
out_frame=out_frame1.numpy()
out_frame=out_frame.astype(dtype=np.int)
dim=(out_frame.shape[2],out_frame.shape[1])
plt.imshow(out_frame[0])
plt.show()


#video = cv2.VideoWriter('videotest.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, dim)
#video.write(out_frame[0])
#video.release()
#cv2.destroyAllWindows()

#create_video(out_frame,'videotest.avi',fps/10,dim)
play_video('./YouTubeClips/eyhzdC936uk_15_27.avi')