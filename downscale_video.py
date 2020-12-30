import cv2
import numpy as np
import os
import tqdm
import glob
from utils import play_video,sample_scaled_frames,create_video
inpath="./YouTubeClips/"
outpath="./Downscale14/"
outpath2="./Upscale14/"
'''

for i, video in enumerate(sorted(glob.glob("./YouTubeClips/*.avi"))):
    if i>=1900:#1300:
        video_id = video
        name=os.path.basename(video)
        frame_list, frame_count, fps, size=sample_scaled_frames(video,stride=1,scale_factor=4)
        filepath=outpath+name
        create_video(frame_list, filepath, fps,downscale=True)
'''
#play_video(inpath+'zzit5b_-ukg_5_20.avi')
#play_video(outpath+'zzit5b_-ukg_5_20.avi')
#frame_list, frame_count, fps, size=sample_scaled_frames(outpath+'f9_bP219ehQ_63_70.avi',stride=1,scale_factor=2)

play_video(outpath+ 'zCf8NWJ8kzA_47_52.avi')
play_video(outpath2+ 'zCf8NWJ8kzA_47_52.avi')
'''
for i, video in enumerate(sorted(glob.glob(outpath2+"*.avi"))):
    name = os.path.basename(video)
    filepath = outpath2 + name
    play_video(filepath)
'''