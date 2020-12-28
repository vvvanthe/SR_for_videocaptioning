import cv2
import numpy as np
import os
import tqdm
import glob
from utils import play_video,sample_scaled_frames,create_video
inpath="./YouTubeClips/"
outpath="./Downscale31/"



for i, video in enumerate(sorted(glob.glob("./YouTubeClips/*.avi"))):
    if i>=1300:
        video_id = video
        name=os.path.basename(video)
        frame_list, frame_count, fps, size=sample_scaled_frames(video,stride=1,scale_factor=3)
        filepath=outpath+name
        create_video(frame_list, filepath, fps,size)


play_video(outpath+'f9_bP219ehQ_63_70.avi')
#frame_list, frame_count, fps, size=sample_scaled_frames(outpath+'f9_bP219ehQ_63_70.avi',stride=1,scale_factor=2)

