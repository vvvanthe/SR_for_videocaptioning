import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def play_video(path):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def create_video(allframes,filepath,fps,downscale):
    dim = (allframes.shape[2], allframes.shape[1])
    #print(size)
    video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
    for frame in allframes:
        if downscale==True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def sample_scaled_frames(video_path,stride,scale_factor):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass
    frames = []
    frame_count = 0
    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(w /scale_factor)
    height = int(h /scale_factor)
    dim = (width, height)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        if (scale_factor!=1):
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            frames.append(resized)
        else:
            frames.append(frame)
        frame_count += 1

    indices = list(range(0, frame_count , stride))
    #print(indices)
    frames = np.array(frames)
    frame_list = frames[indices]
    fps = cap.get(cv2.CAP_PROP_FPS)


    return frame_list, frame_count,fps,dim


