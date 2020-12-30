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

frame_list1, frame_count, fps, size=sample_scaled_frames(inpath+'zzit5b_-ukg_5_20.avi',stride=1,scale_factor=1)
plt.imshow(frame_list1[0]) #cv2.cvtColor(frame_list1[0],cv2.COLOR_BGR2RGB)
plt.show()
frame_list=frame_list1.astype(np.float32)

print(type(frame_list1[0][1,1,1]))
print(frame_list1[0][1,1])
'''
out_frame1=model(frame_list)
out_frame1=tf.clip_by_value(out_frame1, 0, 255)
out_frame=out_frame1.numpy()
out_frame=out_frame.astype(dtype=np.int)
plt.imshow(out_frame[0])
plt.show()
'''
#

frames=[]
for frame in frame_list:
    frame_in=tf.expand_dims(frame, 0)
    frame_out=model(frame_in)
    frame_out=tf.clip_by_value(frame_out, 0, 255)
    frame_out = frame_out.numpy()
    frame_out = frame_out.astype(dtype=np.uint8)
    frame_out=tf.squeeze(frame_out)
    frames.append(frame_out)

out_frame=np.array(frames)

plt.imshow(out_frame[0])
plt.show()



create_video(out_frame,'videotest.avi',fps,downscale=False)
play_video('videotest.avi')




#frame_list2, frame_count, fps, size=sample_scaled_frames('videotest.avi',stride=1,scale_factor=1)

#plt.imshow(frame_list2[0])
#plt.show()

#pic=frame_list2[0]



#print( frame_list2[0])
#print(frame_list2[0].shape)



