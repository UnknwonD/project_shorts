import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import threading
import queue

def preprocess_video_every_3_seconds(video_path: str, frame_size: tuple, frame_rate=3):
    """
    Extracts frames every 3 seconds from a video file, resizing them to frame_size and converting to grayscale.
    
    Args:
    video_path (str): Path to the video file.
    frame_size (tuple): Size (height, width) to resize frames.
    frame_rate (int): Number of frames to extract per second within the 3-second window.

    Returns:
    List[numpy.ndarray]: List of sequences, where each sequence is a numpy array of shape (num_frames, height, width, 1).
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * 3)
    target_frames = int(frame_rate * 3)
    sequences = []

    def read_frames(q):
        while True:
            success, frame = vidcap.read()
            if not success:
                q.put(None)
                break
            q.put(frame)

    frame_queue = queue.Queue(maxsize=100)
    threading.Thread(target=read_frames, args=(frame_queue,)).start()

    while True:
        frames = []
        for _ in range(interval_frames):
            frame = frame_queue.get()
            if frame is None:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            gray_frame = gray_frame.astype(np.float32) / 255.0
            frames.append(gray_frame)
        
        if len(frames) < interval_frames:
            break
        
        sequences.append(np.array(frames[:target_frames]))
    
    vidcap.release()
    return np.array(sequences)
