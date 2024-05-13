import os
import cv2
import numpy as np
import json
from tqdm import tqdm

def preprocess_video_every_3_seconds(video_path:str, frame_size:tuple, block_nums:int, frame_rate=3):
    """
    Extracts frames every 3 seconds from a video file, resizing them to frame_size and converting to grayscale.
    
    Args:
    video_path (str): Path to the video file.
    frame_size (tuple): Size (height, width) to resize frames.
    block_nums (int) : Total count for three-seconds-blocks
    frame_rate (int): Number of frames to extract per second within the 3-second window.

    Returns:
    List[numpy.ndarray]: List of sequences, where each sequence is a numpy array of shape (num_frames, height, width, 1).
    """

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 3)

    sequences = []
    while True:
        frames = []
        for _ in range(interval):
            success, frame = vidcap.read()
            if not success:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, axis=-1)  # 채널 수 늘려줌
            gray_frame = gray_frame.astype(np.float32) / 255.0 
            frames.append(gray_frame)

        if len(frames) == 0:
            break
        
        if len(frames) >= frame_rate : 
            sequences.append(np.array(frames[:frame_rate * 3]))  # 모든 frame이 3초단위로 들어갈 수 있도록 제어
        
        if len(sequences) > block_nums:
            break

    vidcap.release()
    return np.array(sequences[:-1])


def parse_annotations(annotations:list):
    """
    Extracts Every Annotation from json label file
    
    Args:
    annotations(List): List of Dictionary for annotations label with highlight and represent

    Returns:
    Dict: Whether each block is Highlight or not
    """
    highlight_map = {}
    
    for annot in annotations:
        block_num = annot['highlight']
        for num in block_num:
            highlight_map[num] = 1
            
    ret = [0] * len(highlight_map)
    for i, item in enumerate(highlight_map.items()):
        ret[item] = 1
                
    return highlight_map