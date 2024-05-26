import os
import numpy as np
import pandas as pd

from process_audio import extract_audio, preprocess_audio
from process_video import preprocess_video_every_3_seconds

from tensorflow.keras.models import load_model


import numpy as np


def pipeline_video(video_path:str):
    ####### 1. 사용자가 입력한 비디오 불러오기
    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    
    ####### 2. 사용자가 입력한 비디오 전처리하기
    # 1) 비디오 분리
    # 2) 분리된 비디오 전처리

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    print(len(video))

    ####### 3. 모델 불러오기
    # 1) 비디오 모델 load

    video_model = load_model("video_model.h5")

    ####### 4. 모델 추론
    # 1) 비디오 모델 Inference

    video_output = video_model.predict(video)
    print("##################video######################")
    print(video_output)
    print("#############################################")
    final_predictions = np.argmax(video_output, axis=1)
    
    ####### 6. Return
    return final_predictions

video_path = "full_Video.mp4"
test_data = pipeline_video(video_path)

print(test_data)

