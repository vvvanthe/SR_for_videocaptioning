import cv2
import numpy as np
import os

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

def create_video(allframes,filepath,fps,size):


    #print(size)
    video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for frame in allframes:
        #print(frame.shape)
        #print(frame)
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
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frames.append(resized)
        frame_count += 1

    indices = list(range(8, frame_count - 7, stride))
    #print(indices)
    frames = np.array(frames)
    frame_list = frames[indices]
    fps = cap.get(cv2.CAP_PROP_FPS)

    size=(int(w),int(h))
    return frame_list, frame_count,fps,dim

